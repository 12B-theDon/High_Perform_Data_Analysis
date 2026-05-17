#include "cartographer_parallel/assignment_GPU_v2.h"

#include <algorithm>
#include <chrono>
#include <cstdio>

#include <cuda_runtime.h>

// GPU v2 annotation:
// 이 파일은 GPU v1에 warp-aware reduction을 추가한 버전이다.
//
// GPU v1 대비 추가/변경된 점:
//   1. v1의 device memory reuse, grid/scan cache, CUDA event reuse, CPU fallback은 유지한다.
//   2. block reduction을 shared-memory tree reduction에서 warp shuffle 기반 reduction으로
//      바꾸었다.
//   3. 256 threads/block = 8 warps/block 구조를 명시하고, 각 warp가 먼저 자기 내부 합을
//      __shfl_down_sync로 줄인다.
//   4. 각 warp의 lane 0만 shared memory에 warp sum을 쓰고, 첫 번째 warp가 8개의 warp sum을
//      다시 줄여 candidate 하나의 score를 만든다.
//
// 기대 효과:
//   - v1의 reduction은 모든 단계마다 shared memory 접근과 __syncthreads()가 반복된다.
//   - v2는 warp 내부 합산을 register shuffle로 처리하므로 synchronization/shared memory
//     overhead가 줄어든다.
//   - 단, grid[y * w + x]의 불규칙 global memory load는 그대로 남아 있으므로, v2는
//     reduction overhead를 줄이는 최적화이지 memory access pattern을 근본적으로 바꾸는
//     최적화는 아니다.

namespace cartographer_parallel {
namespace {

// Jetson Nano(Tegra X1)의 CUDA warp size는 32이다.
// 256 threads/block을 사용하므로 candidate 하나는 8개의 warp가 나누어 처리한다.
constexpr int kWarpSize = 32;
constexpr int kThreadsPerBlock = 256;
constexpr int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;

bool CheckCuda(const cudaError_t status, const char* const message) {
  if (status == cudaSuccess) return true;

  std::fprintf(stderr, "[score_all GPU v2] %s failed: %s\n", message,
               cudaGetErrorString(status));
  std::fflush(stderr);
  return false;
}

template <typename T>
bool EnsureDeviceBuffer(T** const ptr, size_t* const capacity_bytes,
                        const size_t required_bytes,
                        const char* const message) {
  // v1과 동일하게, 이미 충분히 큰 device buffer가 있으면 재할당하지 않는다.
  if (*capacity_bytes >= required_bytes) return true;

  cudaFree(*ptr);
  *ptr = nullptr;
  *capacity_bytes = 0;

  if (required_bytes == 0) return true;

  if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(ptr), required_bytes),
                 message)) {
    return false;
  }
  *capacity_bytes = required_bytes;
  return true;
}

void ScoreAllCpuFallback(const std::vector<unsigned char>& grid, const int w,
                         const int h, const std::vector<int>& px,
                         const std::vector<int>& py,
                         const std::vector<int>& cx,
                         const std::vector<int>& cy,
                         std::vector<float>* const score) {
  // v2에서도 small candidate case는 GPU보다 CPU가 빠를 수 있으므로 fallback을 유지한다.
  const int n = static_cast<int>(std::min(cx.size(), cy.size()));
  const int p = static_cast<int>(std::min(px.size(), py.size()));
  score->assign(n, 0.0f);

  const bool valid =
      w > 0 && h > 0 && p > 0 &&
      grid.size() >= static_cast<size_t>(w) * static_cast<size_t>(h);
  if (!valid) return;

  const unsigned char* const grid_data = grid.data();
  const int* const px_data = px.data();
  const int* const py_data = py.data();
  const int* const cx_data = cx.data();
  const int* const cy_data = cy.data();
  float* const score_data = score->data();

  const float inv_norm = 1.0f / (255.0f * static_cast<float>(p));
  const unsigned int uw = static_cast<unsigned int>(w);
  const unsigned int uh = static_cast<unsigned int>(h);

  for (int i = 0; i < n; ++i) {
    const int cxi = cx_data[i];
    const int cyi = cy_data[i];
    int sum = 0;

    for (int j = 0; j < p; ++j) {
      const int x = px_data[j] + cxi;
      const int y = py_data[j] + cyi;
      if (static_cast<unsigned int>(x) < uw &&
          static_cast<unsigned int>(y) < uh) {
        sum += grid_data[y * w + x];
      }
    }

    score_data[i] = static_cast<float>(sum) * inv_norm;
  }
}

struct ScoreAllGpuState {
  // v1에서 도입한 persistent state를 그대로 사용한다.
  // v2의 새 최적화는 이 state가 아니라 kernel reduction 방식에 있다.
  unsigned char* d_grid = nullptr;
  int* d_px = nullptr;
  int* d_py = nullptr;
  int* d_cx = nullptr;
  int* d_cy = nullptr;
  float* d_score = nullptr;

  size_t grid_capacity = 0;
  size_t px_capacity = 0;
  size_t py_capacity = 0;
  size_t cx_capacity = 0;
  size_t cy_capacity = 0;
  size_t score_capacity = 0;

  const unsigned char* cached_grid = nullptr;
  size_t cached_grid_bytes = 0;
  int cached_w = 0;
  int cached_h = 0;

  const int* cached_px = nullptr;
  const int* cached_py = nullptr;
  int cached_p = 0;

  cudaEvent_t kernel_start = nullptr;
  cudaEvent_t kernel_stop = nullptr;
  bool events_ready = false;

  ~ScoreAllGpuState() {
    cudaFree(d_grid);
    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_cx);
    cudaFree(d_cy);
    cudaFree(d_score);
    if (events_ready) {
      cudaEventDestroy(kernel_start);
      cudaEventDestroy(kernel_stop);
    }
  }

  bool EnsureEvents() {
    if (events_ready) return true;
    if (!CheckCuda(cudaEventCreate(&kernel_start), "cudaEventCreate start") ||
        !CheckCuda(cudaEventCreate(&kernel_stop), "cudaEventCreate stop")) {
      return false;
    }
    events_ready = true;
    return true;
  }

  bool EnsureCapacity(const size_t grid_bytes, const size_t point_bytes,
                      const size_t candidate_bytes,
                      const size_t score_bytes) {
    return EnsureDeviceBuffer(&d_grid, &grid_capacity, grid_bytes,
                              "cudaMalloc d_grid") &&
           EnsureDeviceBuffer(&d_px, &px_capacity, point_bytes,
                              "cudaMalloc d_px") &&
           EnsureDeviceBuffer(&d_py, &py_capacity, point_bytes,
                              "cudaMalloc d_py") &&
           EnsureDeviceBuffer(&d_cx, &cx_capacity, candidate_bytes,
                              "cudaMalloc d_cx") &&
           EnsureDeviceBuffer(&d_cy, &cy_capacity, candidate_bytes,
                              "cudaMalloc d_cy") &&
           EnsureDeviceBuffer(&d_score, &score_capacity, score_bytes,
                              "cudaMalloc d_score");
  }
};

__inline__ __device__ int WarpReduceSum(int value) {
  // warp 내부 32개 thread의 값을 register shuffle로 합산한다.
  // shared memory를 거치지 않기 때문에 v1의 tree reduction보다 가볍다.
  value += __shfl_down_sync(0xffffffff, value, 16);
  value += __shfl_down_sync(0xffffffff, value, 8);
  value += __shfl_down_sync(0xffffffff, value, 4);
  value += __shfl_down_sync(0xffffffff, value, 2);
  value += __shfl_down_sync(0xffffffff, value, 1);
  return value;
}

__global__ void ScoreAllKernelWarpReduce(
    const unsigned char* const grid, const int w, const int h,
    const int* const px, const int* const py, const int* const cx,
    const int* const cy, const int n, const int p, const float inv_norm,
    float* const score) {
  // v2 kernel도 blockIdx.x = candidate 구조는 유지한다.
  // 차이는 partial sum을 합치는 reduction 단계가 warp-aware라는 점이다.
  const int candidate = blockIdx.x;
  const int tid = threadIdx.x;
  if (candidate >= n) return;

  const int cxi = cx[candidate];
  const int cyi = cy[candidate];
  const unsigned int uw = static_cast<unsigned int>(w);
  const unsigned int uh = static_cast<unsigned int>(h);

  int local_sum = 0;
  for (int j = tid; j < p; j += blockDim.x) {
    const int x = px[j] + cxi;
    const int y = py[j] + cyi;

    if (static_cast<unsigned int>(x) < uw &&
        static_cast<unsigned int>(y) < uh) {
      local_sum += grid[y * w + x];
    }
  }

  const int lane = tid & (kWarpSize - 1);
  const int warp_id = tid / kWarpSize;
  // Step 1: 각 warp가 자기 warp 내부의 local_sum을 먼저 합친다.
  local_sum = WarpReduceSum(local_sum);

  __shared__ int warp_sums[kWarpsPerBlock];
  if (lane == 0) {
    // Step 2: warp마다 대표 thread(lane 0) 하나만 shared memory에 결과를 쓴다.
    warp_sums[warp_id] = local_sum;
  }
  __syncthreads();

  int block_sum = 0;
  if (warp_id == 0) {
    // Step 3: 첫 번째 warp가 8개의 warp sum을 다시 합쳐 block 전체 합을 만든다.
    block_sum = lane < kWarpsPerBlock ? warp_sums[lane] : 0;
    block_sum = WarpReduceSum(block_sum);
    if (lane == 0) {
      score[candidate] = static_cast<float>(block_sum) * inv_norm;
    }
  }
}

}  // namespace

void score_all_GPU_v2(const std::vector<unsigned char>& grid, const int w,
                      const int h, const std::vector<int>& px,
                      const std::vector<int>& py,
                      const std::vector<int>& cx,
                      const std::vector<int>& cy,
                      std::vector<float>* const score) {
  if (score == nullptr) return;

  const auto total_t0 = std::chrono::high_resolution_clock::now();

  const int n = static_cast<int>(std::min(cx.size(), cy.size()));
  const int p = static_cast<int>(std::min(px.size(), py.size()));
  score->assign(n, 0.0f);

  const bool valid =
      w > 0 && h > 0 && n > 0 && p > 0 &&
      grid.size() >= static_cast<size_t>(w) * static_cast<size_t>(h);
  if (!valid) return;

  const int kMinGpuCandidates = 1024;
  // v1과 같은 policy: candidate가 적으면 GPU 병렬성이 부족하므로 CPU fallback을 쓴다.
  if (n < kMinGpuCandidates) {
    const auto cpu_t0 = std::chrono::high_resolution_clock::now();
    ScoreAllCpuFallback(grid, w, h, px, py, cx, cy, score);
    const auto cpu_t1 = std::chrono::high_resolution_clock::now();
    const double cpu_ms =
        std::chrono::duration<double, std::milli>(cpu_t1 - cpu_t0).count();

    static int cpu_call_count = 0;
    static double cpu_total_ms = 0.0;
    ++cpu_call_count;
    cpu_total_ms += cpu_ms;
    if (cpu_call_count <= 5 || cpu_call_count % 10 == 0) {
      std::printf("[score_all GPU v2 CPU fallback] calls=%d avg=%.6f ms "
                  "last=%.6f ms candidates=%d scan_points=%d threshold=%d\n",
                  cpu_call_count, cpu_total_ms / cpu_call_count, cpu_ms, n, p,
                  kMinGpuCandidates);
      std::fflush(stdout);
    }
    return;
  }

  const size_t grid_bytes = grid.size() * sizeof(unsigned char);
  const size_t point_bytes = static_cast<size_t>(p) * sizeof(int);
  const size_t candidate_bytes = static_cast<size_t>(n) * sizeof(int);
  const size_t score_bytes = static_cast<size_t>(n) * sizeof(float);

  static ScoreAllGpuState state;

  const auto alloc_t0 = std::chrono::high_resolution_clock::now();
  if (!state.EnsureCapacity(grid_bytes, point_bytes, candidate_bytes,
                            score_bytes)) {
    return;
  }
  const auto alloc_t1 = std::chrono::high_resolution_clock::now();

  const auto h2d_t0 = std::chrono::high_resolution_clock::now();
  // v1과 동일하게 grid/scan point는 cache하고, candidate arrays(cx/cy)는 매 호출 복사한다.
  // branch-and-bound 단계마다 candidate set은 바뀌지만 grid와 scan은 자주 재사용되기 때문이다.
  if (state.cached_grid != grid.data() ||
      state.cached_grid_bytes != grid_bytes || state.cached_w != w ||
      state.cached_h != h) {
    if (!CheckCuda(cudaMemcpy(state.d_grid, grid.data(), grid_bytes,
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy grid HtoD")) {
      return;
    }
    state.cached_grid = grid.data();
    state.cached_grid_bytes = grid_bytes;
    state.cached_w = w;
    state.cached_h = h;
  }

  if (state.cached_px != px.data() || state.cached_py != py.data() ||
      state.cached_p != p) {
    if (!CheckCuda(cudaMemcpy(state.d_px, px.data(), point_bytes,
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy px HtoD") ||
        !CheckCuda(cudaMemcpy(state.d_py, py.data(), point_bytes,
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy py HtoD")) {
      return;
    }
    state.cached_px = px.data();
    state.cached_py = py.data();
    state.cached_p = p;
  }

  if (!CheckCuda(cudaMemcpy(state.d_cx, cx.data(), candidate_bytes,
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy cx HtoD") ||
      !CheckCuda(cudaMemcpy(state.d_cy, cy.data(), candidate_bytes,
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy cy HtoD")) {
    return;
  }
  const auto h2d_t1 = std::chrono::high_resolution_clock::now();

  if (!state.EnsureEvents()) return;

  const int threads = kThreadsPerBlock;
  const int blocks = n;
  const float inv_norm = 1.0f / (255.0f * static_cast<float>(p));

  CheckCuda(cudaEventRecord(state.kernel_start), "cudaEventRecord start");
  ScoreAllKernelWarpReduce<<<blocks, threads>>>(
      state.d_grid, w, h, state.d_px, state.d_py, state.d_cx, state.d_cy, n, p,
      inv_norm, state.d_score);
  const cudaError_t launch_status = cudaGetLastError();
  CheckCuda(cudaEventRecord(state.kernel_stop), "cudaEventRecord stop");
  CheckCuda(cudaEventSynchronize(state.kernel_stop),
            "cudaEventSynchronize stop");

  float kernel_ms = 0.0f;
  CheckCuda(cudaEventElapsedTime(&kernel_ms, state.kernel_start,
                                 state.kernel_stop),
            "cudaEventElapsedTime");

  if (!CheckCuda(launch_status, "ScoreAllKernelWarpReduce launch")) return;

  const auto d2h_t0 = std::chrono::high_resolution_clock::now();
  if (!CheckCuda(cudaMemcpy(score->data(), state.d_score, score_bytes,
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy score DtoH")) {
    return;
  }
  const auto d2h_t1 = std::chrono::high_resolution_clock::now();

  const auto total_t1 = std::chrono::high_resolution_clock::now();

  const double alloc_ms =
      std::chrono::duration<double, std::milli>(alloc_t1 - alloc_t0).count();
  const double h2d_ms =
      std::chrono::duration<double, std::milli>(h2d_t1 - h2d_t0).count();
  const double d2h_ms =
      std::chrono::duration<double, std::milli>(d2h_t1 - d2h_t0).count();
  const double total_ms =
      std::chrono::duration<double, std::milli>(total_t1 - total_t0).count();

  static int call_count = 0;
  static double total_sum_ms = 0.0;
  static double alloc_sum_ms = 0.0;
  static double h2d_sum_ms = 0.0;
  static double kernel_sum_ms = 0.0;
  static double d2h_sum_ms = 0.0;

  ++call_count;
  total_sum_ms += total_ms;
  alloc_sum_ms += alloc_ms;
  h2d_sum_ms += h2d_ms;
  kernel_sum_ms += kernel_ms;
  d2h_sum_ms += d2h_ms;

  if (call_count <= 5 || call_count % 10 == 0) {
    std::printf("[score_all GPU v2] calls=%d avg_total=%.6f ms "
                "last_total=%.6f ms avg_alloc=%.6f ms avg_h2d=%.6f ms "
                "avg_kernel=%.6f ms avg_d2h=%.6f ms candidates=%d "
                "scan_points=%d threshold=%d warps_per_block=%d\n",
                call_count, total_sum_ms / call_count, total_ms,
                alloc_sum_ms / call_count, h2d_sum_ms / call_count,
                kernel_sum_ms / call_count, d2h_sum_ms / call_count, n, p,
                kMinGpuCandidates, kWarpsPerBlock);
    std::fflush(stdout);
  }
}

}  // namespace cartographer_parallel
