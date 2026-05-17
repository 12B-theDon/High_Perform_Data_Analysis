#include "cartographer_parallel/assignment_GPU.h"

#include <algorithm>
#include <chrono>
#include <cstdio>

#include <cuda_runtime.h>

// GPU v0 annotation:
// 이 파일은 score_all()을 CUDA로 옮긴 첫 번째 기본 구현이다.
// CPU v1의 계산식은 유지하고, 병렬화 구조만 GPU에 맞게 바꾸었다.
//
// 이전 CPU 구현 대비 추가된 점:
//   1. candidate 하나를 CUDA block 하나에 대응시킨다.
//      Why:
//        score_all()의 출력은 candidate마다 score 하나이므로 candidate 간 의존성이 없다.
//        그래서 candidate 단위는 GPU block으로 나누기 가장 자연스럽다.
//      Expected:
//        candidate 수가 많아질수록 여러 block이 동시에 실행되어 CPU single-thread보다
//        높은 병렬성을 얻을 수 있다.
//   2. block 내부의 thread들이 scan point들을 나누어 처리한다.
//      Why:
//        한 candidate의 score는 여러 scan point의 grid 값을 합산한 결과이다.
//        scan point loop를 thread들에게 분배하면 candidate 하나 내부에서도 병렬성이 생긴다.
//      Expected:
//        scan point 수가 1000개 이상일 때 한 block의 256개 thread가 합산 작업을 나누어
//        CPU의 inner loop보다 빠르게 처리할 수 있다.
//   3. 각 thread의 partial sum을 shared memory에 저장한 뒤 block reduction으로
//      candidate 하나의 최종 score를 만든다.
//      Why:
//        각 thread가 계산한 부분합을 하나의 score로 합쳐야 하므로 block-level reduction이 필요하다.
//      Expected:
//        global memory나 atomic add를 사용하지 않고 block 내부 shared memory에서 합산하여
//        candidate당 하나의 score를 안정적으로 생성한다.
//   4. 매 score_all 호출마다 cudaMalloc, HtoD copy, kernel launch, DtoH copy,
//      cudaFree를 수행한다.
//      Why:
//        첫 CUDA 버전에서는 correctness와 profiling 기준점을 먼저 확보하기 위해
//        가장 단순하고 추적하기 쉬운 data flow를 사용했다.
//      Expected:
//        어떤 단계(cudaMalloc, copy, kernel, cudaFree)가 병목인지 nvprof로 분리해서 볼 수 있다.
//
// 이 버전의 목적은 "GPU에서 같은 계산을 올바르게 수행하는 기준점"을 만드는 것이다.
// 따라서 memory reuse나 small-candidate fallback 같은 최적화는 아직 넣지 않았다.
// profiling 결과에서는 kernel 자체보다 반복적인 할당/복사 비용과 작은 candidate 수에서의
// GPU under-utilization이 크게 보일 수 있다.

namespace cartographer_parallel {
namespace {

bool CheckCuda(const cudaError_t status, const char* const message) {
  if (status == cudaSuccess) return true;

  std::fprintf(stderr, "[score_all GPU] %s failed: %s\n", message,
               cudaGetErrorString(status));
  std::fflush(stderr);
  return false;
}

// Step 2. Define the CUDA kernel.
//
// Parallelization mapping:
//   - blockIdx.x  = candidate index.
//   - threadIdx.x = scan-point worker inside one candidate.
//
// With 256 threads per block and a warp size of 32 on Jetson Nano,
// each candidate is processed by 8 warps.
//
// 한국어 설명:
//   - blockIdx.x는 후보 위치(candidate)를 의미한다.
//   - threadIdx.x는 해당 candidate 안에서 scan point 일부를 담당한다.
//   - thread들은 j = tid, tid + blockDim.x, ... 방식으로 scan point를 나누어
//     grid[y * w + x]를 읽는다.
//   - Why: scan point 수 p가 thread 수보다 클 수 있으므로 strided loop가 필요하다.
//   - Expected: 모든 scan point가 빠짐없이 처리되고, thread workload가 비교적 균등하게 나뉜다.
//   - grid 접근은 여전히 불규칙하므로, GPU로 옮겨도 global memory load 병목은
//     완전히 사라지지 않는다.
__global__ void ScoreAllKernel(const unsigned char* const grid, const int w,
                               const int h, const int* const px,
                               const int* const py, const int* const cx,
                               const int* const cy, const int n, const int p,
                               const float inv_norm, float* const score) {
  // Step 3. Map one CUDA block to one candidate.
  const int candidate = blockIdx.x;
  const int tid = threadIdx.x;
  if (candidate >= n) return;

  const int cxi = cx[candidate];
  const int cyi = cy[candidate];
  const unsigned int uw = static_cast<unsigned int>(w);
  const unsigned int uh = static_cast<unsigned int>(h);

  // Step 4. Each thread processes scan points with a strided loop.
  int local_sum = 0;
  for (int j = tid; j < p; j += blockDim.x) {
    const int x = px[j] + cxi;
    const int y = py[j] + cyi;

    if (static_cast<unsigned int>(x) < uw &&
        static_cast<unsigned int>(y) < uh) {
      local_sum += grid[y * w + x];
    }
  }

  // Step 5. Store each thread's partial sum in shared memory.
  // Why:
  //   thread마다 계산한 local_sum을 candidate 하나의 최종 sum으로 합쳐야 한다.
  // Expected:
  //   shared memory reduction을 사용하면 global atomic보다 단순하고 예측 가능한
  //   block-level 합산이 가능하다.
  extern __shared__ int partial[];
  partial[tid] = local_sum;
  __syncthreads();

  // Step 6. Block-level reduction: combine thread partial sums into one sum.
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }

  // Step 7. One block produces one score value for one candidate.
  if (tid == 0) {
    score[candidate] = static_cast<float>(partial[0]) * inv_norm;
  }
}

}  // namespace

// Step 1. CPU-callable GPU version of score_all().
//
// This wrapper runs on the CPU. It prepares device memory, copies inputs to the
// GPU, launches the kernel, copies scores back, and releases device memory.
//
// 한국어 설명:
//   v0 wrapper는 가장 단순한 data stream 구조를 사용한다.
//   Host vector -> Device buffer -> CUDA kernel -> Host score vector 순서로
//   매 호출마다 전체 데이터를 이동시킨다. 구현은 명확하지만, 같은 grid와 scan point를
//   반복해서 복사하므로 ROS bag 실행처럼 score_all()이 수천 번 호출되는 상황에서는
//   copy/allocation overhead가 누적된다.
void score_all_GPU(const std::vector<unsigned char>& grid, const int w,
                   const int h, const std::vector<int>& px,
                   const std::vector<int>& py, const std::vector<int>& cx,
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

  const size_t grid_bytes = grid.size() * sizeof(unsigned char);
  const size_t point_bytes = static_cast<size_t>(p) * sizeof(int);
  const size_t candidate_bytes = static_cast<size_t>(n) * sizeof(int);
  const size_t score_bytes = static_cast<size_t>(n) * sizeof(float);

  unsigned char* d_grid = nullptr;
  int* d_px = nullptr;
  int* d_py = nullptr;
  int* d_cx = nullptr;
  int* d_cy = nullptr;
  float* d_score = nullptr;

  // Step 8. Allocate device memory.
  const auto alloc_t0 = std::chrono::high_resolution_clock::now();
  if (!CheckCuda(cudaMalloc(&d_grid, grid_bytes), "cudaMalloc d_grid") ||
      !CheckCuda(cudaMalloc(&d_px, point_bytes), "cudaMalloc d_px") ||
      !CheckCuda(cudaMalloc(&d_py, point_bytes), "cudaMalloc d_py") ||
      !CheckCuda(cudaMalloc(&d_cx, candidate_bytes), "cudaMalloc d_cx") ||
      !CheckCuda(cudaMalloc(&d_cy, candidate_bytes), "cudaMalloc d_cy") ||
      !CheckCuda(cudaMalloc(&d_score, score_bytes), "cudaMalloc d_score")) {
    cudaFree(d_grid);
    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_cx);
    cudaFree(d_cy);
    cudaFree(d_score);
    return;
  }
  const auto alloc_t1 = std::chrono::high_resolution_clock::now();

  // Step 9. Copy input arrays from host memory to device memory.
  const auto h2d_t0 = std::chrono::high_resolution_clock::now();
  if (!CheckCuda(cudaMemcpy(d_grid, grid.data(), grid_bytes,
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy grid HtoD") ||
      !CheckCuda(cudaMemcpy(d_px, px.data(), point_bytes,
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy px HtoD") ||
      !CheckCuda(cudaMemcpy(d_py, py.data(), point_bytes,
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy py HtoD") ||
      !CheckCuda(cudaMemcpy(d_cx, cx.data(), candidate_bytes,
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy cx HtoD") ||
      !CheckCuda(cudaMemcpy(d_cy, cy.data(), candidate_bytes,
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy cy HtoD")) {
    cudaFree(d_grid);
    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_cx);
    cudaFree(d_cy);
    cudaFree(d_score);
    return;
  }
  const auto h2d_t1 = std::chrono::high_resolution_clock::now();

  cudaEvent_t kernel_start;
  cudaEvent_t kernel_stop;
  if (!CheckCuda(cudaEventCreate(&kernel_start), "cudaEventCreate start") ||
      !CheckCuda(cudaEventCreate(&kernel_stop), "cudaEventCreate stop")) {
    cudaFree(d_grid);
    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_cx);
    cudaFree(d_cy);
    cudaFree(d_score);
    return;
  }

  // Step 10. Launch the CUDA kernel.
  //
  // blocks = number of candidates.
  // threads = 256 threads per block = 8 warps per candidate.
  const int threads = 256;
  const int blocks = n;
  const size_t shared_bytes = static_cast<size_t>(threads) * sizeof(int);
  const float inv_norm = 1.0f / (255.0f * static_cast<float>(p));

  CheckCuda(cudaEventRecord(kernel_start), "cudaEventRecord start");
  ScoreAllKernel<<<blocks, threads, shared_bytes>>>(d_grid, w, h, d_px, d_py,
                                                    d_cx, d_cy, n, p,
                                                    inv_norm, d_score);
  const cudaError_t launch_status = cudaGetLastError();
  CheckCuda(cudaEventRecord(kernel_stop), "cudaEventRecord stop");
  CheckCuda(cudaEventSynchronize(kernel_stop), "cudaEventSynchronize stop");

  float kernel_ms = 0.0f;
  CheckCuda(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop),
            "cudaEventElapsedTime");
  cudaEventDestroy(kernel_start);
  cudaEventDestroy(kernel_stop);

  if (!CheckCuda(launch_status, "ScoreAllKernel launch")) {
    cudaFree(d_grid);
    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_cx);
    cudaFree(d_cy);
    cudaFree(d_score);
    return;
  }

  // Step 11. Copy output scores from device memory back to host memory.
  const auto d2h_t0 = std::chrono::high_resolution_clock::now();
  if (!CheckCuda(cudaMemcpy(score->data(), d_score, score_bytes,
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy score DtoH")) {
    cudaFree(d_grid);
    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_cx);
    cudaFree(d_cy);
    cudaFree(d_score);
    return;
  }
  const auto d2h_t1 = std::chrono::high_resolution_clock::now();

  // Step 12. Free device memory for this first CUDA version.
  const auto free_t0 = std::chrono::high_resolution_clock::now();
  cudaFree(d_grid);
  cudaFree(d_px);
  cudaFree(d_py);
  cudaFree(d_cx);
  cudaFree(d_cy);
  cudaFree(d_score);
  const auto free_t1 = std::chrono::high_resolution_clock::now();

  const auto total_t1 = std::chrono::high_resolution_clock::now();

  const double alloc_ms =
      std::chrono::duration<double, std::milli>(alloc_t1 - alloc_t0).count();
  const double h2d_ms =
      std::chrono::duration<double, std::milli>(h2d_t1 - h2d_t0).count();
  const double d2h_ms =
      std::chrono::duration<double, std::milli>(d2h_t1 - d2h_t0).count();
  const double free_ms =
      std::chrono::duration<double, std::milli>(free_t1 - free_t0).count();
  const double total_ms =
      std::chrono::duration<double, std::milli>(total_t1 - total_t0).count();

  static int call_count = 0;
  static double total_sum_ms = 0.0;
  static double alloc_sum_ms = 0.0;
  static double h2d_sum_ms = 0.0;
  static double kernel_sum_ms = 0.0;
  static double d2h_sum_ms = 0.0;
  static double free_sum_ms = 0.0;

  ++call_count;
  total_sum_ms += total_ms;
  alloc_sum_ms += alloc_ms;
  h2d_sum_ms += h2d_ms;
  kernel_sum_ms += kernel_ms;
  d2h_sum_ms += d2h_ms;
  free_sum_ms += free_ms;

  if (call_count <= 5 || call_count % 10 == 0) {
    std::printf("[score_all GPU v0] calls=%d avg_total=%.6f ms "
                "last_total=%.6f ms avg_alloc=%.6f ms avg_h2d=%.6f ms "
                "avg_kernel=%.6f ms avg_d2h=%.6f ms avg_free=%.6f ms "
                "candidates=%d scan_points=%d\n",
                call_count, total_sum_ms / call_count, total_ms,
                alloc_sum_ms / call_count, h2d_sum_ms / call_count,
                kernel_sum_ms / call_count, d2h_sum_ms / call_count,
                free_sum_ms / call_count, n, p);
    std::fflush(stdout);
  }
}

}  // namespace cartographer_parallel
