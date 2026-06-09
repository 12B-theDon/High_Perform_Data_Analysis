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
  std::fprintf(stderr, "[score_all CUDA ver6 error] %s failed: %s\n",
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
  int* flat_px = nullptr;
  int* flat_py = nullptr;
  int* flat_cell = nullptr;
  int* point_offsets = nullptr;
  int* point_counts = nullptr;
  int* cand_offsets = nullptr;
  int* cand_counts = nullptr;
  int* cx = nullptr;
  int* cy = nullptr;
  int* candidate_cell = nullptr;
  unsigned char* full_inside = nullptr;
  float* score = nullptr;

  size_t grid_capacity = 0;
  size_t flat_px_capacity = 0;
  size_t flat_py_capacity = 0;
  size_t flat_cell_capacity = 0;
  size_t point_offsets_capacity = 0;
  size_t point_counts_capacity = 0;
  size_t cand_offsets_capacity = 0;
  size_t cand_counts_capacity = 0;
  size_t cx_capacity = 0;
  size_t cy_capacity = 0;
  size_t candidate_cell_capacity = 0;
  size_t full_inside_capacity = 0;
  size_t score_capacity = 0;

  const unsigned char* host_grid_ptr = nullptr;
  size_t host_grid_size = 0;
  int host_grid_w = 0;
  int host_grid_h = 0;

  bool events_created = false;
  cudaEvent_t h2d_start = nullptr;
  cudaEvent_t h2d_stop = nullptr;
  cudaEvent_t kernel_stop = nullptr;
  cudaEvent_t d2h_stop = nullptr;

  ~DeviceScoreCache() {
    if (grid != nullptr) cudaFree(grid);
    if (flat_px != nullptr) cudaFree(flat_px);
    if (flat_py != nullptr) cudaFree(flat_py);
    if (flat_cell != nullptr) cudaFree(flat_cell);
    if (point_offsets != nullptr) cudaFree(point_offsets);
    if (point_counts != nullptr) cudaFree(point_counts);
    if (cand_offsets != nullptr) cudaFree(cand_offsets);
    if (cand_counts != nullptr) cudaFree(cand_counts);
    if (cx != nullptr) cudaFree(cx);
    if (cy != nullptr) cudaFree(cy);
    if (candidate_cell != nullptr) cudaFree(candidate_cell);
    if (full_inside != nullptr) cudaFree(full_inside);
    if (score != nullptr) cudaFree(score);
    if (events_created) {
      cudaEventDestroy(h2d_start);
      cudaEventDestroy(h2d_stop);
      cudaEventDestroy(kernel_stop);
      cudaEventDestroy(d2h_stop);
    }
  }
};

bool EnsureEvents(DeviceScoreCache* const cache) {
  if (cache->events_created) return true;
  if (!CheckCuda(cudaEventCreate(&cache->h2d_start),
                 "cudaEventCreate cached h2d_start")) {
    return false;
  }
  if (!CheckCuda(cudaEventCreate(&cache->h2d_stop),
                 "cudaEventCreate cached h2d_stop")) {
    return false;
  }
  if (!CheckCuda(cudaEventCreate(&cache->kernel_stop),
                 "cudaEventCreate cached kernel_stop")) {
    return false;
  }
  if (!CheckCuda(cudaEventCreate(&cache->d2h_stop),
                 "cudaEventCreate cached d2h_stop")) {
    return false;
  }
  cache->events_created = true;
  return true;
}

double BytesToMiB(const size_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

__device__ int WarpReduceSumVer6(int value) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__global__ void ScoreAllKernelVer6Batched(
    const unsigned char* __restrict__ const grid, const int w, const int h,
    const int* __restrict__ const flat_px,
    const int* __restrict__ const flat_py,
    const int* __restrict__ const flat_cell,
    const int* __restrict__ const point_offsets,
    const int* __restrict__ const point_counts,
    const int* __restrict__ const cand_offsets,
    const int* __restrict__ const cand_counts,
    const int* __restrict__ const cx, const int* __restrict__ const cy,
    const int* __restrict__ const candidate_cell,
    const unsigned char* __restrict__ const full_inside,
    const int scan_count, float* __restrict__ const score) {
  __shared__ int s_px[kPointTile];
  __shared__ int s_py[kPointTile];
  __shared__ int s_cell[kPointTile];

  const int scan = blockIdx.y;
  if (scan >= scan_count) return;

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_in_block = threadIdx.x / kWarpSize;
  const int local_candidate =
      blockIdx.x * kWarpsPerBlock + warp_in_block;
  const int candidate_count = __ldg(&cand_counts[scan]);
  const bool valid_candidate = local_candidate < candidate_count;
  const int point_count = __ldg(&point_counts[scan]);
  const int point_offset = __ldg(&point_offsets[scan]);
  const int candidate =
      valid_candidate ? __ldg(&cand_offsets[scan]) + local_candidate : 0;

  int sum = 0;
  int offset_x = 0;
  int offset_y = 0;
  int candidate_cell_offset = 0;
  bool candidate_full_inside = false;
  if (valid_candidate) {
    offset_x = __ldg(&cx[candidate]);
    offset_y = __ldg(&cy[candidate]);
    candidate_cell_offset = __ldg(&candidate_cell[candidate]);
    candidate_full_inside = __ldg(&full_inside[candidate]) != 0;
  }

  for (int base = 0; base < point_count; base += kPointTile) {
    const int remaining = point_count - base;
    const int tile_count = remaining < kPointTile ? remaining : kPointTile;

    if (threadIdx.x < tile_count) {
      const int point_index = point_offset + base + threadIdx.x;
      s_px[threadIdx.x] = __ldg(&flat_px[point_index]);
      s_py[threadIdx.x] = __ldg(&flat_py[point_index]);
      s_cell[threadIdx.x] = __ldg(&flat_cell[point_index]);
    }

    __syncthreads();

    if (valid_candidate) {
      if (candidate_full_inside) {
        for (int j = lane; j < tile_count; j += kWarpSize) {
          sum += __ldg(&grid[s_cell[j] + candidate_cell_offset]);
        }
      } else {
        for (int j = lane; j < tile_count; j += kWarpSize) {
          const int x = s_px[j] + offset_x;
          const int y = s_py[j] + offset_y;
          if (x >= 0 && x < w && y >= 0 && y < h) {
            sum += __ldg(&grid[y * w + x]);
          }
        }
      }
    }

    __syncthreads();
  }

  if (valid_candidate) {
    const int warp_sum = WarpReduceSumVer6(sum);
    if (lane == 0) {
      score[candidate] = static_cast<float>(warp_sum) /
                         (255.0f * static_cast<float>(point_count));
    }
  }
}

}  // namespace

void score_all_CUDA_ver6_batched(
    const std::vector<unsigned char>& grid, const int w, const int h,
    const std::vector<int>& flat_px, const std::vector<int>& flat_py,
    const std::vector<int>& flat_cell,
    const std::vector<int>& point_offsets,
    const std::vector<int>& point_counts,
    const std::vector<int>& cand_offsets, const std::vector<int>& cand_counts,
    const std::vector<int>& cx, const std::vector<int>& cy,
    const std::vector<int>& candidate_cell,
    const std::vector<unsigned char>& full_inside,
    std::vector<float>* const score) {
  if (score == nullptr) return;
  const auto t0 = std::chrono::high_resolution_clock::now();

  const int n = static_cast<int>(
      std::min({cx.size(), cy.size(), candidate_cell.size(),
                full_inside.size()}));
  const int scan_count =
      static_cast<int>(std::min(point_offsets.size(), point_counts.size()));
  const int cand_scan_count =
      static_cast<int>(std::min(cand_offsets.size(), cand_counts.size()));
  const int total_points =
      static_cast<int>(std::min({flat_px.size(), flat_py.size(),
                                 flat_cell.size()}));
  score->assign(n, 0.0f);

  int max_candidates_per_scan = 0;
  for (int count : cand_counts) {
    max_candidates_per_scan = std::max(max_candidates_per_scan, count);
  }
  int full_inside_count = 0;
  for (int i = 0; i < n; ++i) {
    if (full_inside[i] != 0) ++full_inside_count;
  }

  float h2d_ms = 0.0f;
  float kernel_ms = 0.0f;
  float d2h_ms = 0.0f;
  bool grid_uploaded = false;

  const size_t grid_elems = static_cast<size_t>(w) * static_cast<size_t>(h);
  const size_t point_elems = static_cast<size_t>(total_points);
  const size_t scan_elems = static_cast<size_t>(scan_count);
  const size_t candidate_elems = static_cast<size_t>(n);

  const bool valid_input =
      w > 0 && h > 0 && n > 0 && scan_count > 0 &&
      scan_count == cand_scan_count && total_points > 0 &&
      max_candidates_per_scan > 0 && grid.size() >= grid_elems;

  if (valid_input) {
    static DeviceScoreCache cache;
    cudaStream_t stream = 0;

    bool cuda_ok = true;
    cuda_ok = EnsureCapacity(&cache.grid, &cache.grid_capacity, grid_elems,
                             "cudaMalloc cached grid") &&
              cuda_ok;
    cuda_ok = EnsureCapacity(&cache.flat_px, &cache.flat_px_capacity,
                             point_elems, "cudaMalloc cached flat_px") &&
              cuda_ok;
    cuda_ok = EnsureCapacity(&cache.flat_py, &cache.flat_py_capacity,
                             point_elems, "cudaMalloc cached flat_py") &&
              cuda_ok;
    cuda_ok = EnsureCapacity(&cache.flat_cell, &cache.flat_cell_capacity,
                             point_elems, "cudaMalloc cached flat_cell") &&
              cuda_ok;
    cuda_ok =
        EnsureCapacity(&cache.point_offsets, &cache.point_offsets_capacity,
                       scan_elems, "cudaMalloc cached point_offsets") &&
        cuda_ok;
    cuda_ok =
        EnsureCapacity(&cache.point_counts, &cache.point_counts_capacity,
                       scan_elems, "cudaMalloc cached point_counts") &&
        cuda_ok;
    cuda_ok =
        EnsureCapacity(&cache.cand_offsets, &cache.cand_offsets_capacity,
                       scan_elems, "cudaMalloc cached cand_offsets") &&
        cuda_ok;
    cuda_ok = EnsureCapacity(&cache.cand_counts, &cache.cand_counts_capacity,
                             scan_elems, "cudaMalloc cached cand_counts") &&
              cuda_ok;
    cuda_ok = EnsureCapacity(&cache.cx, &cache.cx_capacity, candidate_elems,
                             "cudaMalloc cached cx") &&
              cuda_ok;
    cuda_ok = EnsureCapacity(&cache.cy, &cache.cy_capacity, candidate_elems,
                             "cudaMalloc cached cy") &&
              cuda_ok;
    cuda_ok = EnsureCapacity(&cache.candidate_cell,
                             &cache.candidate_cell_capacity, candidate_elems,
                             "cudaMalloc cached candidate_cell") &&
              cuda_ok;
    cuda_ok = EnsureCapacity(&cache.full_inside,
                             &cache.full_inside_capacity, candidate_elems,
                             "cudaMalloc cached full_inside") &&
              cuda_ok;
    cuda_ok = EnsureCapacity(&cache.score, &cache.score_capacity,
                             candidate_elems, "cudaMalloc cached score") &&
              cuda_ok;
    cuda_ok = cuda_ok && EnsureEvents(&cache);

    if (cuda_ok) {
      cuda_ok =
          CheckCuda(cudaEventRecord(cache.h2d_start, stream),
                    "record cached h2d_start") &&
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
          CheckCuda(cudaMemcpyAsync(cache.flat_px, flat_px.data(),
                                    point_elems * sizeof(int),
                                    cudaMemcpyHostToDevice, stream),
                    "cudaMemcpyAsync flat_px H2D") &&
          cuda_ok;
      cuda_ok =
          CheckCuda(cudaMemcpyAsync(cache.flat_py, flat_py.data(),
                                    point_elems * sizeof(int),
                                    cudaMemcpyHostToDevice, stream),
                    "cudaMemcpyAsync flat_py H2D") &&
          cuda_ok;
      cuda_ok =
          CheckCuda(cudaMemcpyAsync(cache.flat_cell, flat_cell.data(),
                                    point_elems * sizeof(int),
                                    cudaMemcpyHostToDevice, stream),
                    "cudaMemcpyAsync flat_cell H2D") &&
          cuda_ok;
      cuda_ok =
          CheckCuda(cudaMemcpyAsync(cache.point_offsets, point_offsets.data(),
                                    scan_elems * sizeof(int),
                                    cudaMemcpyHostToDevice, stream),
                    "cudaMemcpyAsync point_offsets H2D") &&
          cuda_ok;
      cuda_ok =
          CheckCuda(cudaMemcpyAsync(cache.point_counts, point_counts.data(),
                                    scan_elems * sizeof(int),
                                    cudaMemcpyHostToDevice, stream),
                    "cudaMemcpyAsync point_counts H2D") &&
          cuda_ok;
      cuda_ok =
          CheckCuda(cudaMemcpyAsync(cache.cand_offsets, cand_offsets.data(),
                                    scan_elems * sizeof(int),
                                    cudaMemcpyHostToDevice, stream),
                    "cudaMemcpyAsync cand_offsets H2D") &&
          cuda_ok;
      cuda_ok =
          CheckCuda(cudaMemcpyAsync(cache.cand_counts, cand_counts.data(),
                                    scan_elems * sizeof(int),
                                    cudaMemcpyHostToDevice, stream),
                    "cudaMemcpyAsync cand_counts H2D") &&
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
          CheckCuda(cudaMemcpyAsync(cache.candidate_cell,
                                    candidate_cell.data(),
                                    candidate_elems * sizeof(int),
                                    cudaMemcpyHostToDevice, stream),
                    "cudaMemcpyAsync candidate_cell H2D") &&
          cuda_ok;
      cuda_ok =
          CheckCuda(cudaMemcpyAsync(cache.full_inside, full_inside.data(),
                                    candidate_elems * sizeof(unsigned char),
                                    cudaMemcpyHostToDevice, stream),
                    "cudaMemcpyAsync full_inside H2D") &&
          cuda_ok;

      cuda_ok =
          CheckCuda(cudaEventRecord(cache.h2d_stop, stream),
                    "record cached h2d_stop") &&
          cuda_ok;
    }

    if (cuda_ok) {
      const int blocks_x =
          (max_candidates_per_scan + kWarpsPerBlock - 1) / kWarpsPerBlock;
      const dim3 blocks(blocks_x, scan_count);
      ScoreAllKernelVer6Batched<<<blocks, kThreadsPerBlock, 0, stream>>>(
          cache.grid, w, h, cache.flat_px, cache.flat_py, cache.flat_cell,
          cache.point_offsets, cache.point_counts, cache.cand_offsets,
          cache.cand_counts, cache.cx, cache.cy, cache.candidate_cell,
          cache.full_inside, scan_count, cache.score);
      cuda_ok =
          CheckCuda(cudaGetLastError(), "ScoreAllKernelVer6Batched launch") &&
          cuda_ok;
      cuda_ok =
          CheckCuda(cudaEventRecord(cache.kernel_stop, stream),
                    "record cached kernel_stop") &&
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
          CheckCuda(cudaEventRecord(cache.d2h_stop, stream),
                    "record cached d2h_stop") &&
          cuda_ok;
      cuda_ok =
          CheckCuda(cudaStreamSynchronize(stream),
                    "cudaStreamSynchronize default stream") &&
          cuda_ok;
    }

    if (cuda_ok) {
      cudaEventElapsedTime(&h2d_ms, cache.h2d_start, cache.h2d_stop);
      cudaEventElapsedTime(&kernel_ms, cache.h2d_stop, cache.kernel_stop);
      cudaEventElapsedTime(&d2h_ms, cache.kernel_stop, cache.d2h_stop);
    }
  }

  const auto t1 = std::chrono::high_resolution_clock::now();
  const double ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();

  static int call_count = 0;
  static double total_ms = 0.0;
  ++call_count;
  total_ms += ms;

  if (call_count <= 10 || call_count % 100 == 0) {
    const double cuda_measured_ms =
        static_cast<double>(h2d_ms) + static_cast<double>(kernel_ms) +
        static_cast<double>(d2h_ms);
    const double host_overhead_ms = ms - cuda_measured_ms;
    const int blocks_x = max_candidates_per_scan > 0
                             ? (max_candidates_per_scan + kWarpsPerBlock - 1) /
                                   kWarpsPerBlock
                             : 0;
    const int total_blocks = blocks_x * scan_count;
    const double grid_mib =
        BytesToMiB(grid_elems * sizeof(unsigned char));
    const double point_mib = BytesToMiB(point_elems * sizeof(int) * 3);
    const double scan_meta_mib = BytesToMiB(scan_elems * sizeof(int) * 4);
    const double candidate_meta_mib =
        BytesToMiB(candidate_elems * (sizeof(int) * 3 + sizeof(unsigned char)));
    const double score_mib = BytesToMiB(candidate_elems * sizeof(float));
    const double full_inside_ratio =
        n > 0 ? static_cast<double>(full_inside_count) / static_cast<double>(n)
              : 0.0;

    std::printf("[score_all CUDA ver6 batched profile] calls=%d avg=%.6f ms "
                "last=%.6f ms h2d=%.6f ms kernel=%.6f ms d2h=%.6f ms "
                "host_overhead=%.6f ms scans=%d candidates=%d "
                "scan_points=%d full_inside=%d full_inside_ratio=%.6f "
                "max_candidates_per_scan=%d blocks_x=%d total_blocks=%d "
                "threads_per_block=%d grid_upload=%d grid_mib=%.6f "
                "point_mib=%.6f scan_meta_mib=%.6f candidate_meta_mib=%.6f "
                "score_mib=%.6f\n",
                call_count, total_ms / call_count, ms, h2d_ms, kernel_ms,
                d2h_ms, host_overhead_ms, scan_count, n, total_points,
                full_inside_count, full_inside_ratio, max_candidates_per_scan,
                blocks_x, total_blocks, kThreadsPerBlock,
                grid_uploaded ? 1 : 0, grid_mib, point_mib, scan_meta_mib,
                candidate_meta_mib, score_mib);
    std::fflush(stdout);
  }
}

}  // namespace cartographer_parallel
