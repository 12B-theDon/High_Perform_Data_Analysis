#include "cartographer_parallel/cuda_assignment.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>

namespace cartographer_parallel {
namespace {

bool CheckCuda(const cudaError_t error, const char* const expression) {
  if (error == cudaSuccess) return true;
  std::fprintf(stderr, "[score_all CUDA error] %s failed: %s\n", expression,
               cudaGetErrorString(error));
  std::fflush(stderr);
  return false;
}

__global__ void ScoreAllKernel(const unsigned char* const grid, const int w,
                               const int h, const int* const px,
                               const int* const py, const int* const cx,
                               const int* const cy, const int n, const int p,
                               float* const score) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  int sum = 0;
  for (int j = 0; j < p; ++j) {
    const int x = px[j] + cx[i];
    const int y = py[j] + cy[i];
    if (x >= 0 && x < w && y >= 0 && y < h) {
      sum += grid[y * w + x];
    }
  }
  score[i] = static_cast<float>(sum) / (255.0f * static_cast<float>(p));
}

}  // namespace

void score_all_CUDA(const std::vector<unsigned char>& grid, const int w,
                    const int h, const std::vector<int>& px,
                    const std::vector<int>& py, const std::vector<int>& cx,
                    const std::vector<int>& cy,
                    std::vector<float>* const score) {
  if (score == nullptr) return;
  const auto t0 = std::chrono::high_resolution_clock::now();

  const int n = std::min(cx.size(), cy.size());
  const int p = std::min(px.size(), py.size());
  score->assign(n, 0.0f);

  const bool valid_input =
      w > 0 && h > 0 && n > 0 && p > 0 &&
      grid.size() >= static_cast<size_t>(w) * static_cast<size_t>(h);

  if (valid_input) {
    cudaStream_t stream = 0;  // default stream
    const size_t grid_bytes =
        static_cast<size_t>(w) * static_cast<size_t>(h) *
        sizeof(unsigned char);
    const size_t point_bytes = static_cast<size_t>(p) * sizeof(int);
    const size_t candidate_bytes = static_cast<size_t>(n) * sizeof(int);
    const size_t score_bytes = static_cast<size_t>(n) * sizeof(float);

    unsigned char* d_grid = nullptr;
    int* d_px = nullptr;
    int* d_py = nullptr;
    int* d_cx = nullptr;
    int* d_cy = nullptr;
    float* d_score = nullptr;

    bool cuda_ok = true;
    cuda_ok = CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_grid),
                                   grid_bytes),
                        "cudaMalloc d_grid") &&
              cuda_ok;
    cuda_ok = CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_px),
                                   point_bytes),
                        "cudaMalloc d_px") &&
              cuda_ok;
    cuda_ok = CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_py),
                                   point_bytes),
                        "cudaMalloc d_py") &&
              cuda_ok;
    cuda_ok =
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_cx), candidate_bytes),
                  "cudaMalloc d_cx") &&
        cuda_ok;
    cuda_ok =
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_cy), candidate_bytes),
                  "cudaMalloc d_cy") &&
        cuda_ok;
    cuda_ok =
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_score), score_bytes),
                  "cudaMalloc d_score") &&
        cuda_ok;

    if (cuda_ok) {
      cuda_ok =
          CheckCuda(cudaMemcpyAsync(d_grid, grid.data(), grid_bytes,
                                    cudaMemcpyHostToDevice, stream),
                    "cudaMemcpyAsync grid H2D") &&
          cuda_ok;
      cuda_ok =
          CheckCuda(cudaMemcpyAsync(d_px, px.data(), point_bytes,
                                    cudaMemcpyHostToDevice, stream),
                    "cudaMemcpyAsync px H2D") &&
          cuda_ok;
      cuda_ok =
          CheckCuda(cudaMemcpyAsync(d_py, py.data(), point_bytes,
                                    cudaMemcpyHostToDevice, stream),
                    "cudaMemcpyAsync py H2D") &&
          cuda_ok;
      cuda_ok =
          CheckCuda(cudaMemcpyAsync(d_cx, cx.data(), candidate_bytes,
                                    cudaMemcpyHostToDevice, stream),
                    "cudaMemcpyAsync cx H2D") &&
          cuda_ok;
      cuda_ok =
          CheckCuda(cudaMemcpyAsync(d_cy, cy.data(), candidate_bytes,
                                    cudaMemcpyHostToDevice, stream),
                    "cudaMemcpyAsync cy H2D") &&
          cuda_ok;
    }

    if (cuda_ok) {
      const int threads_per_block = 256;
      const int blocks = (n + threads_per_block - 1) / threads_per_block;
      ScoreAllKernel<<<blocks, threads_per_block, 0, stream>>>(
          d_grid, w, h, d_px, d_py, d_cx, d_cy, n, p, d_score);
      cuda_ok =
          CheckCuda(cudaGetLastError(), "ScoreAllKernel launch") && cuda_ok;
      cuda_ok =
          CheckCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize kernel") &&
          cuda_ok;
    }

    if (cuda_ok) {
      cuda_ok =
          CheckCuda(cudaMemcpyAsync(score->data(), d_score, score_bytes,
                                    cudaMemcpyDeviceToHost, stream),
                    "cudaMemcpyAsync score D2H") &&
          cuda_ok;
      cuda_ok =
          CheckCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize D2H") &&
          cuda_ok;
    }

    if (d_grid != nullptr) {
      cuda_ok = CheckCuda(cudaFree(d_grid), "cudaFree d_grid") && cuda_ok;
    }
    if (d_px != nullptr) {
      cuda_ok = CheckCuda(cudaFree(d_px), "cudaFree d_px") && cuda_ok;
    }
    if (d_py != nullptr) {
      cuda_ok = CheckCuda(cudaFree(d_py), "cudaFree d_py") && cuda_ok;
    }
    if (d_cx != nullptr) {
      cuda_ok = CheckCuda(cudaFree(d_cx), "cudaFree d_cx") && cuda_ok;
    }
    if (d_cy != nullptr) {
      cuda_ok = CheckCuda(cudaFree(d_cy), "cudaFree d_cy") && cuda_ok;
    }
    if (d_score != nullptr) {
      cuda_ok = CheckCuda(cudaFree(d_score), "cudaFree d_score") && cuda_ok;
    }
    (void)cuda_ok;
  }

  const auto t1 = std::chrono::high_resolution_clock::now();
  const double ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();

  static int call_count = 0;
  static double total_ms = 0.0;
  ++call_count;
  total_ms += ms;

  if (call_count <= 5 || call_count % 10 == 0) {
    std::printf("[score_all CUDA default stream] calls=%d avg=%.6f ms "
                "last=%.6f ms candidates=%d scan_points=%d\n",
                call_count, total_ms / call_count, ms, n, p);
    std::fflush(stdout);
  }
}

}  // namespace cartographer_parallel
