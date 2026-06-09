#include "cartographer_parallel/cuda_assignment.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>

namespace cartographer_parallel {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;
constexpr int kPointTile = 256;

bool CheckCuda(const cudaError_t error, const char* const expression) {
  if (error == cudaSuccess) return true;
  std::fprintf(stderr, "[score_all CUDA ver1 error] %s failed: %s\n",
               expression, cudaGetErrorString(error));
  std::fflush(stderr);
  return false;
}

template <typename T>
bool EnsureCapacity(T** const ptr, size_t* const capacity,
                    const size_t required, const char* const name) {
  if (*capacity >= required) return true;
  if (*ptr != nullptr) {
    if (!CheckCuda(cudaFree(*ptr), name)) return false;
    *ptr = nullptr;
    *capacity = 0;
  }
  if (required == 0) return true;
  if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(ptr),
                            required * sizeof(T)),
                 name)) {
    return false;
  }
  *capacity = required;
  return true;
}

struct DeviceScoreCache {
  unsigned char* grid = nullptr;
  int* px = nullptr;
  int* py = nullptr;
  int* cx = nullptr;
  int* cy = nullptr;
  float* score = nullptr;

  size_t grid_capacity = 0;
  size_t px_capacity = 0;
  size_t py_capacity = 0;
  size_t cx_capacity = 0;
  size_t cy_capacity = 0;
  size_t score_capacity = 0;

  const unsigned char* host_grid_ptr = nullptr;
  size_t host_grid_size = 0;
  int host_grid_w = 0;
  int host_grid_h = 0;

  ~DeviceScoreCache() {
    if (grid != nullptr) cudaFree(grid);
    if (px != nullptr) cudaFree(px);
    if (py != nullptr) cudaFree(py);
    if (cx != nullptr) cudaFree(cx);
    if (cy != nullptr) cudaFree(cy);
    if (score != nullptr) cudaFree(score);
  }
};

__device__ int WarpReduceSum(int value) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__global__ void ScoreAllKernelVer1(const unsigned char* const grid,
                                   const int w, const int h,
                                   const int* const px,
                                   const int* const py,
                                   const int* const cx,
                                   const int* const cy, const int n,
                                   const int p, float* const score) {
  __shared__ int s_px[kPointTile];
  __shared__ int s_py[kPointTile];

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_in_block = threadIdx.x / kWarpSize;
  const int candidate =
      blockIdx.x * kWarpsPerBlock + warp_in_block;

  int sum = 0;
  int offset_x = 0;
  int offset_y = 0;
  if (candidate < n) {
    offset_x = cx[candidate];
    offset_y = cy[candidate];
  }

  for (int base = 0; base < p; base += kPointTile) {
    const int remaining = p - base;
    const int tile_count = remaining < kPointTile ? remaining : kPointTile;

    if (threadIdx.x < tile_count) {
      s_px[threadIdx.x] = px[base + threadIdx.x];
      s_py[threadIdx.x] = py[base + threadIdx.x];
    }

    __syncthreads();

    if (candidate < n) {
      for (int j = lane; j < tile_count; j += kWarpSize) {
        const int x = s_px[j] + offset_x;
        const int y = s_py[j] + offset_y;
        if (x >= 0 && x < w && y >= 0 && y < h) {
          sum += grid[y * w + x];
        }
      }
    }

    __syncthreads();
  }

  if (candidate < n) {
    const int warp_sum = WarpReduceSum(sum);
    if (lane == 0) {
      score[candidate] =
          static_cast<float>(warp_sum) / (255.0f * static_cast<float>(p));
    }
  }
}

}  // namespace

void score_all_CUDA_ver1(const std::vector<unsigned char>& grid, const int w,
                         const int h, const std::vector<int>& px,
                         const std::vector<int>& py,
                         const std::vector<int>& cx,
                         const std::vector<int>& cy,
                         std::vector<float>* const score) {
  if (score == nullptr) return;
  const auto t0 = std::chrono::high_resolution_clock::now();

  const int n = std::min(cx.size(), cy.size());
  const int p = std::min(px.size(), py.size());
  score->assign(n, 0.0f);

  float h2d_ms = 0.0f;
  float kernel_ms = 0.0f;
  float d2h_ms = 0.0f;
  bool grid_uploaded = false;

  const bool valid_input =
      w > 0 && h > 0 && n > 0 && p > 0 &&
      grid.size() >= static_cast<size_t>(w) * static_cast<size_t>(h);

  if (valid_input) {
    static DeviceScoreCache cache;
    cudaStream_t stream = 0;  // default stream

    const size_t grid_elems = static_cast<size_t>(w) * static_cast<size_t>(h);
    const size_t point_elems = static_cast<size_t>(p);
    const size_t candidate_elems = static_cast<size_t>(n);

    bool cuda_ok = true;
    cuda_ok = EnsureCapacity(&cache.grid, &cache.grid_capacity, grid_elems,
                             "cudaMalloc cached grid") &&
              cuda_ok;
    cuda_ok = EnsureCapacity(&cache.px, &cache.px_capacity, point_elems,
                             "cudaMalloc cached px") &&
              cuda_ok;
    cuda_ok = EnsureCapacity(&cache.py, &cache.py_capacity, point_elems,
                             "cudaMalloc cached py") &&
              cuda_ok;
    cuda_ok = EnsureCapacity(&cache.cx, &cache.cx_capacity,
                             candidate_elems, "cudaMalloc cached cx") &&
              cuda_ok;
    cuda_ok = EnsureCapacity(&cache.cy, &cache.cy_capacity,
                             candidate_elems, "cudaMalloc cached cy") &&
              cuda_ok;
    cuda_ok = EnsureCapacity(&cache.score, &cache.score_capacity,
                             candidate_elems, "cudaMalloc cached score") &&
              cuda_ok;

    cudaEvent_t h2d_start = nullptr;
    cudaEvent_t h2d_stop = nullptr;
    cudaEvent_t kernel_stop = nullptr;
    cudaEvent_t d2h_stop = nullptr;
    if (cuda_ok) {
      cuda_ok = CheckCuda(cudaEventCreate(&h2d_start),
                          "cudaEventCreate h2d_start") &&
                cuda_ok;
      cuda_ok = CheckCuda(cudaEventCreate(&h2d_stop),
                          "cudaEventCreate h2d_stop") &&
                cuda_ok;
      cuda_ok = CheckCuda(cudaEventCreate(&kernel_stop),
                          "cudaEventCreate kernel_stop") &&
                cuda_ok;
      cuda_ok = CheckCuda(cudaEventCreate(&d2h_stop),
                          "cudaEventCreate d2h_stop") &&
                cuda_ok;
    }

    if (cuda_ok) {
      cuda_ok =
          CheckCuda(cudaEventRecord(h2d_start, stream), "record h2d_start") &&
          cuda_ok;

      const bool grid_changed =
          cache.host_grid_ptr != grid.data() ||
          cache.host_grid_size != grid_elems || cache.host_grid_w != w ||
          cache.host_grid_h != h;
      if (grid_changed) {
        cuda_ok = CheckCuda(cudaMemcpyAsync(
                                cache.grid, grid.data(),
                                grid_elems * sizeof(unsigned char),
                                cudaMemcpyHostToDevice, stream),
                            "cudaMemcpyAsync cached grid H2D") &&
                  cuda_ok;
        if (cuda_ok) {
          cache.host_grid_ptr = grid.data();
          cache.host_grid_size = grid_elems;
          cache.host_grid_w = w;
          cache.host_grid_h = h;
          grid_uploaded = true;
        }
      }

      cuda_ok =
          CheckCuda(cudaMemcpyAsync(cache.px, px.data(),
                                    point_elems * sizeof(int),
                                    cudaMemcpyHostToDevice, stream),
                    "cudaMemcpyAsync px H2D") &&
          cuda_ok;
      cuda_ok =
          CheckCuda(cudaMemcpyAsync(cache.py, py.data(),
                                    point_elems * sizeof(int),
                                    cudaMemcpyHostToDevice, stream),
                    "cudaMemcpyAsync py H2D") &&
          cuda_ok;
      cuda_ok =
          CheckCuda(cudaMemcpyAsync(cache.cx, cx.data(),
                                    candidate_elems * sizeof(int),
                                    cudaMemcpyHostToDevice, stream),
                    "cudaMemcpyAsync cx H2D") &&
          cuda_ok;
      cuda_ok =
          CheckCuda(cudaMemcpyAsync(cache.cy, cy.data(),
                                    candidate_elems * sizeof(int),
                                    cudaMemcpyHostToDevice, stream),
                    "cudaMemcpyAsync cy H2D") &&
          cuda_ok;

      cuda_ok =
          CheckCuda(cudaEventRecord(h2d_stop, stream), "record h2d_stop") &&
          cuda_ok;
    }

    if (cuda_ok) {
      const int blocks =
          (n + kWarpsPerBlock - 1) / kWarpsPerBlock;
      ScoreAllKernelVer1<<<blocks, kThreadsPerBlock, 0, stream>>>(
          cache.grid, w, h, cache.px, cache.py, cache.cx, cache.cy, n, p,
          cache.score);
      cuda_ok =
          CheckCuda(cudaGetLastError(), "ScoreAllKernelVer1 launch") &&
          cuda_ok;
      cuda_ok =
          CheckCuda(cudaEventRecord(kernel_stop, stream),
                    "record kernel_stop") &&
          cuda_ok;
    }

    if (cuda_ok) {
      cuda_ok =
          CheckCuda(cudaMemcpyAsync(score->data(), cache.score,
                                    candidate_elems * sizeof(float),
                                    cudaMemcpyDeviceToHost, stream),
                    "cudaMemcpyAsync score D2H") &&
          cuda_ok;
      cuda_ok =
          CheckCuda(cudaEventRecord(d2h_stop, stream), "record d2h_stop") &&
          cuda_ok;
      cuda_ok =
          CheckCuda(cudaStreamSynchronize(stream),
                    "cudaStreamSynchronize default stream") &&
          cuda_ok;
    }

    if (cuda_ok) {
      cudaEventElapsedTime(&h2d_ms, h2d_start, h2d_stop);
      cudaEventElapsedTime(&kernel_ms, h2d_stop, kernel_stop);
      cudaEventElapsedTime(&d2h_ms, kernel_stop, d2h_stop);
    }

    if (h2d_start != nullptr) cudaEventDestroy(h2d_start);
    if (h2d_stop != nullptr) cudaEventDestroy(h2d_stop);
    if (kernel_stop != nullptr) cudaEventDestroy(kernel_stop);
    if (d2h_stop != nullptr) cudaEventDestroy(d2h_stop);
  }

  const auto t1 = std::chrono::high_resolution_clock::now();
  const double ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();

  static int call_count = 0;
  static double total_ms = 0.0;
  ++call_count;
  total_ms += ms;

  if (call_count <= 5 || call_count % 10 == 0) {
    std::printf("[score_all CUDA ver1 cached tiled] calls=%d avg=%.6f ms "
                "last=%.6f ms h2d=%.6f ms kernel=%.6f ms d2h=%.6f ms "
                "grid_upload=%d candidates=%d scan_points=%d\n",
                call_count, total_ms / call_count, ms, h2d_ms, kernel_ms,
                d2h_ms, grid_uploaded ? 1 : 0, n, p);
    std::fflush(stdout);
  }
}

}  // namespace cartographer_parallel
