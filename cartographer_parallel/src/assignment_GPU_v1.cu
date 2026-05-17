#include "cartographer_parallel/assignment_GPU_v1.h"

#include <algorithm>
#include <chrono>
#include <cstdio>

#include <cuda_runtime.h>

// GPU v1 annotation:
// 이 파일은 GPU v0에서 관측된 overhead를 줄이기 위한 첫 번째 최적화 버전이다.
//
// GPU v0 대비 추가/변경된 점:
//   1. ScoreAllGpuState를 두어 device memory를 매 호출마다 해제하지 않고 재사용한다.
//      Why:
//        v0 profiling에서 cudaMalloc/cudaFree가 매 score_all() 호출마다 반복되어
//        kernel 계산과 별개인 고정 overhead가 크게 누적되었다.
//      Expected:
//        한 번 확보한 device buffer를 재사용하여 allocation/free 시간을 거의 0에 가깝게 줄이고,
//        전체 avg_total 시간을 낮춘다.
//   2. grid와 scan point(px, py)는 host pointer와 size를 기준으로 cache한다.
//      Why:
//        branch-and-bound 과정에서는 candidate set(cx, cy)은 자주 바뀌지만,
//        같은 scan frame 안에서 occupancy grid와 scan point는 반복 사용될 수 있다.
//      Expected:
//        큰 grid HtoD copy와 px/py copy를 반복하지 않아 data transfer 시간이 줄어든다.
//   3. CUDA event도 한 번 만든 뒤 재사용한다.
//      Why:
//        event 생성/삭제도 CUDA runtime API 호출이며, 작은 kernel에서는 무시하기 어려운 overhead가 된다.
//      Expected:
//        profiling용 kernel timing은 유지하면서 event create/destroy 비용을 줄인다.
//   4. candidate 수가 너무 작은 경우(n < 1024)는 GPU를 쓰지 않고 CPU fallback을 쓴다.
//      depth 4처럼 candidate가 4개 정도이면 kernel launch/copy overhead가 실제 계산보다
//      크기 때문이다.
//      Expected:
//        GPU가 불리한 small-candidate case에서는 CPU v1 수준의 시간을 유지하고,
//        GPU는 candidate가 충분히 많을 때만 사용한다.
//
// Kernel mapping은 v0와 동일하다:
//   - blockIdx.x  = candidate index
//   - threadIdx.x = scan point worker
//   - 256 threads/block = Jetson Nano 기준 8 warps/block
//
// 즉 v1의 핵심은 kernel 수식 최적화가 아니라, data stream과 memory lifetime을 정리해서
// 반복 호출 비용을 줄이는 것이다.

namespace cartographer_parallel {
namespace {

bool CheckCuda(const cudaError_t status, const char* const message) {
  if (status == cudaSuccess) return true;

  std::fprintf(stderr, "[score_all GPU v1] %s failed: %s\n", message,
               cudaGetErrorString(status));
  std::fflush(stderr);
  return false;
}

template <typename T>
bool EnsureDeviceBuffer(T** const ptr, size_t* const capacity_bytes,
                        const size_t required_bytes,
                        const char* const message) {
  // 기존 device buffer가 충분히 크면 재할당하지 않는다.
  // score_all()은 반복 호출되므로, 이 조건문이 v1의 중요한 overhead 절감 지점이다.
  // Why:
  //   필요한 크기가 이전 호출보다 작거나 같으면 같은 GPU memory를 다시 쓸 수 있다.
  // Expected:
  //   candidate 수가 조금씩 변해도 매번 cudaMalloc을 호출하지 않아 runtime API overhead가 줄어든다.
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
  // 작은 candidate set에서는 GPU 병렬성이 충분히 나오지 않는다.
  // 이 fallback은 CPU v1 스타일의 raw pointer 접근을 사용하여 small case를 처리한다.
  // Why:
  //   candidate가 4개나 961개 정도이면 GPU block 수가 적고, copy/launch 비용이 계산량보다 커질 수 있다.
  // Expected:
  //   작은 depth case에서 GPU v0처럼 느려지는 문제를 피하고, CPU v1과 비슷한 성능을 얻는다.
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
  // v1에서 새로 추가한 persistent GPU state.
  // 함수가 다시 호출되어도 static object가 살아 있으므로 device buffer와 CUDA event를
  // 계속 재사용할 수 있다.
  // Why:
  //   ROS bag 실행 중 score_all()은 수천 번 이상 호출되므로 per-call resource management가
  //   전체 실행 시간에 큰 영향을 준다.
  // Expected:
  //   반복 호출 구조에서 allocation/free/event setup 비용을 한 번 또는 드문 재할당으로 제한한다.
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

  // px/py는 같은 scan frame 안에서 여러 candidate set에 대해 반복 사용될 수 있으므로
  // pointer와 point count가 바뀌지 않으면 다시 복사하지 않는다.
  // Why:
  //   px/py는 모든 candidate에서 공통으로 쓰이는 scan geometry이고, candidate search 단계마다
  //   자주 재사용된다.
  // Expected:
  //   HtoD copy 횟수를 줄여 GPU v0의 transfer overhead를 줄인다.
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
    // CUDA event 생성도 호출마다 하면 API overhead가 된다.
    // v1에서는 한 번 만든 event를 kernel timing에 계속 재사용한다.
    // Expected:
    //   kernel_ms 측정은 유지하면서 cudaEventCreate/cudaEventDestroy가 avg_total에
    //   반복해서 들어가는 것을 막는다.
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

__global__ void ScoreAllKernel(const unsigned char* const grid, const int w,
                               const int h, const int* const px,
                               const int* const py, const int* const cx,
                               const int* const cy, const int n, const int p,
                               const float inv_norm, float* const score) {
  // v1 kernel 자체는 v0와 같은 one-block-per-candidate 구조이다.
  // v1의 성능 개선은 주로 wrapper의 memory reuse/cache/fallback에서 나온다.
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

  extern __shared__ int partial[];
  partial[tid] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    score[candidate] = static_cast<float>(partial[0]) * inv_norm;
  }
}

}  // namespace

void score_all_GPU_v1(const std::vector<unsigned char>& grid, const int w,
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
  // candidate가 1024개보다 적으면 GPU launch/copy overhead가 더 클 수 있으므로
  // CPU fallback을 사용한다. 이 threshold는 Jetson Nano profiling 결과를 보며 조정한다.
  // Why:
  //   GPU는 block 수가 충분해야 SM을 채울 수 있는데, small candidate case는 병렬성이 부족하다.
  // Expected:
  //   depth 4/3 같은 작은 case에서 GPU v0보다 훨씬 낮은 avg_total을 얻는다.
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
      std::printf("[score_all GPU v1 CPU fallback] calls=%d avg=%.6f ms "
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
  // grid는 occupancy map 전체라서 HtoD copy 크기가 가장 크다.
  // 같은 grid pointer/size/w/h이면 이미 device에 있다고 보고 copy를 생략한다.
  // Why:
  //   nvprof trace에서 grid HtoD copy는 반복적으로 나타나는 큰 transfer 중 하나였다.
  // Expected:
  //   grid가 바뀌지 않는 동안에는 이 copy를 제거하여 data movement 비용을 낮춘다.
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
    // scan point 배열도 같은 scan이면 반복 copy하지 않는다.
    // Expected:
    //   px/py transfer가 candidate set마다 반복되는 것을 막아 h2d 시간을 줄인다.
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

  const int threads = 256;
  const int blocks = n;
  const size_t shared_bytes = static_cast<size_t>(threads) * sizeof(int);
  const float inv_norm = 1.0f / (255.0f * static_cast<float>(p));

  CheckCuda(cudaEventRecord(state.kernel_start), "cudaEventRecord start");
  ScoreAllKernel<<<blocks, threads, shared_bytes>>>(
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

  if (!CheckCuda(launch_status, "ScoreAllKernel launch")) return;

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
    std::printf("[score_all GPU v1] calls=%d avg_total=%.6f ms "
                "last_total=%.6f ms avg_alloc=%.6f ms avg_h2d=%.6f ms "
                "avg_kernel=%.6f ms avg_d2h=%.6f ms candidates=%d "
                "scan_points=%d threshold=%d\n",
                call_count, total_sum_ms / call_count, total_ms,
                alloc_sum_ms / call_count, h2d_sum_ms / call_count,
                kernel_sum_ms / call_count, d2h_sum_ms / call_count, n, p,
                kMinGpuCandidates);
    std::fflush(stdout);
  }
}

}  // namespace cartographer_parallel
