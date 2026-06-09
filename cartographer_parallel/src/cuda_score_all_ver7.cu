#include "cartographer_parallel/cuda_assignment.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace cartographer_parallel {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kDefaultChunkSize = 65536;

bool CheckCuda(const cudaError_t error, const char* const expression) {
  if (error == cudaSuccess) return true;
  std::fprintf(stderr, "[score_all CUDA ver7 error] %s failed: %s\n",
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
  int* point_offsets = nullptr;
  int* point_counts = nullptr;
  int* cand_offsets = nullptr;
  int* cand_counts = nullptr;
  int* cx = nullptr;
  int* cy = nullptr;
  int* scan_min_x = nullptr;
  int* scan_max_x = nullptr;
  int* scan_min_y = nullptr;
  int* scan_max_y = nullptr;
  float* score = nullptr;

  size_t grid_capacity = 0;
  size_t flat_px_capacity = 0;
  size_t flat_py_capacity = 0;
  size_t point_offsets_capacity = 0;
  size_t point_counts_capacity = 0;
  size_t cand_offsets_capacity = 0;
  size_t cand_counts_capacity = 0;
  size_t cx_capacity = 0;
  size_t cy_capacity = 0;
  size_t scan_min_x_capacity = 0;
  size_t scan_max_x_capacity = 0;
  size_t scan_min_y_capacity = 0;
  size_t scan_max_y_capacity = 0;
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
    if (point_offsets != nullptr) cudaFree(point_offsets);
    if (point_counts != nullptr) cudaFree(point_counts);
    if (cand_offsets != nullptr) cudaFree(cand_offsets);
    if (cand_counts != nullptr) cudaFree(cand_counts);
    if (cx != nullptr) cudaFree(cx);
    if (cy != nullptr) cudaFree(cy);
    if (scan_min_x != nullptr) cudaFree(scan_min_x);
    if (scan_max_x != nullptr) cudaFree(scan_max_x);
    if (scan_min_y != nullptr) cudaFree(scan_min_y);
    if (scan_max_y != nullptr) cudaFree(scan_max_y);
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

int ChunkSizeFromEnv() {
  const char* const env = std::getenv("CUDA_SCORE_CHUNK_SIZE");
  if (env == nullptr) return kDefaultChunkSize;
  const int value = std::atoi(env);
  return value > 0 ? value : kDefaultChunkSize;
}

__global__ void ScoreAllKernelVer7Batched(
    const unsigned char* __restrict__ const grid, const int w, const int h,
    const int* __restrict__ const flat_px,
    const int* __restrict__ const flat_py,
    const int* __restrict__ const point_offsets,
    const int* __restrict__ const point_counts,
    const int* __restrict__ const cand_offsets,
    const int* __restrict__ const cand_counts,
    const int* __restrict__ const cx, const int* __restrict__ const cy,
    const int* __restrict__ const scan_min_x,
    const int* __restrict__ const scan_max_x,
    const int* __restrict__ const scan_min_y,
    const int* __restrict__ const scan_max_y, const int scan_count,
    const int chunk_begin, float* __restrict__ const score) {
  const int scan = blockIdx.y;
  if (scan >= scan_count) return;

  const int local_candidate =
      chunk_begin + blockIdx.x * blockDim.x + threadIdx.x;
  const int candidate_count = __ldg(&cand_counts[scan]);
  if (local_candidate >= candidate_count) return;

  const int candidate = __ldg(&cand_offsets[scan]) + local_candidate;
  const int point_count = __ldg(&point_counts[scan]);
  const int point_offset = __ldg(&point_offsets[scan]);
  if (point_count <= 0) {
    score[candidate] = 0.0f;
    return;
  }

  const int offset_x = __ldg(&cx[candidate]);
  const int offset_y = __ldg(&cy[candidate]);
  const bool full_inside =
      __ldg(&scan_min_x[scan]) + offset_x >= 0 &&
      __ldg(&scan_max_x[scan]) + offset_x < w &&
      __ldg(&scan_min_y[scan]) + offset_y >= 0 &&
      __ldg(&scan_max_y[scan]) + offset_y < h;

  int sum = 0;
  if (full_inside) {
    for (int i = 0; i < point_count; ++i) {
      const int point_index = point_offset + i;
      const int x = __ldg(&flat_px[point_index]) + offset_x;
      const int y = __ldg(&flat_py[point_index]) + offset_y;
      sum += __ldg(&grid[y * w + x]);
    }
  } else {
    for (int i = 0; i < point_count; ++i) {
      const int point_index = point_offset + i;
      const int x = __ldg(&flat_px[point_index]) + offset_x;
      const int y = __ldg(&flat_py[point_index]) + offset_y;
      if (x >= 0 && x < w && y >= 0 && y < h) {
        sum += __ldg(&grid[y * w + x]);
      }
    }
  }

  score[candidate] =
      static_cast<float>(sum) / (255.0f * static_cast<float>(point_count));
}

}  // namespace

void score_all_CUDA_ver7_batched(
    const std::vector<unsigned char>& grid, const int w, const int h,
    const std::vector<int>& flat_px, const std::vector<int>& flat_py,
    const std::vector<int>& point_offsets,
    const std::vector<int>& point_counts,
    const std::vector<int>& cand_offsets, const std::vector<int>& cand_counts,
    const std::vector<int>& cx, const std::vector<int>& cy,
    const std::vector<int>& scan_min_x, const std::vector<int>& scan_max_x,
    const std::vector<int>& scan_min_y, const std::vector<int>& scan_max_y,
    std::vector<float>* const score) {
  if (score == nullptr) return;
  const auto t0 = std::chrono::high_resolution_clock::now();

  const int n = static_cast<int>(std::min(cx.size(), cy.size()));
  const int scan_count =
      static_cast<int>(std::min(point_offsets.size(), point_counts.size()));
  const int cand_scan_count =
      static_cast<int>(std::min(cand_offsets.size(), cand_counts.size()));
  const int bounds_scan_count = static_cast<int>(
      std::min({scan_min_x.size(), scan_max_x.size(), scan_min_y.size(),
                scan_max_y.size()}));
  const int total_points = static_cast<int>(std::min(flat_px.size(),
                                                     flat_py.size()));
  score->assign(n, 0.0f);

  int max_candidates_per_scan = 0;
  for (int count : cand_counts) {
    max_candidates_per_scan = std::max(max_candidates_per_scan, count);
  }

  float h2d_ms = 0.0f;
  float kernel_ms = 0.0f;
  float d2h_ms = 0.0f;
  bool grid_uploaded = false;
  int chunk_size = ChunkSizeFromEnv();
  int chunks = 0;

  const size_t grid_elems = static_cast<size_t>(w) * static_cast<size_t>(h);
  const size_t point_elems = static_cast<size_t>(total_points);
  const size_t scan_elems = static_cast<size_t>(scan_count);
  const size_t candidate_elems = static_cast<size_t>(n);

  const bool valid_input =
      w > 0 && h > 0 && n > 0 && scan_count > 0 &&
      scan_count == cand_scan_count && scan_count == bounds_scan_count &&
      total_points > 0 && max_candidates_per_scan > 0 &&
      grid.size() >= grid_elems;

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
    cuda_ok = EnsureCapacity(&cache.scan_min_x, &cache.scan_min_x_capacity,
                             scan_elems, "cudaMalloc cached scan_min_x") &&
              cuda_ok;
    cuda_ok = EnsureCapacity(&cache.scan_max_x, &cache.scan_max_x_capacity,
                             scan_elems, "cudaMalloc cached scan_max_x") &&
              cuda_ok;
    cuda_ok = EnsureCapacity(&cache.scan_min_y, &cache.scan_min_y_capacity,
                             scan_elems, "cudaMalloc cached scan_min_y") &&
              cuda_ok;
    cuda_ok = EnsureCapacity(&cache.scan_max_y, &cache.scan_max_y_capacity,
                             scan_elems, "cudaMalloc cached scan_max_y") &&
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
          CheckCuda(cudaMemcpyAsync(cache.scan_min_x, scan_min_x.data(),
                                    scan_elems * sizeof(int),
                                    cudaMemcpyHostToDevice, stream),
                    "cudaMemcpyAsync scan_min_x H2D") &&
          cuda_ok;
      cuda_ok =
          CheckCuda(cudaMemcpyAsync(cache.scan_max_x, scan_max_x.data(),
                                    scan_elems * sizeof(int),
                                    cudaMemcpyHostToDevice, stream),
                    "cudaMemcpyAsync scan_max_x H2D") &&
          cuda_ok;
      cuda_ok =
          CheckCuda(cudaMemcpyAsync(cache.scan_min_y, scan_min_y.data(),
                                    scan_elems * sizeof(int),
                                    cudaMemcpyHostToDevice, stream),
                    "cudaMemcpyAsync scan_min_y H2D") &&
          cuda_ok;
      cuda_ok =
          CheckCuda(cudaMemcpyAsync(cache.scan_max_y, scan_max_y.data(),
                                    scan_elems * sizeof(int),
                                    cudaMemcpyHostToDevice, stream),
                    "cudaMemcpyAsync scan_max_y H2D") &&
          cuda_ok;

      cuda_ok =
          CheckCuda(cudaEventRecord(cache.h2d_stop, stream),
                    "record cached h2d_stop") &&
          cuda_ok;
    }

    if (cuda_ok) {
      for (int chunk_begin = 0; chunk_begin < max_candidates_per_scan;
           chunk_begin += chunk_size) {
        const int this_chunk =
            std::min(chunk_size, max_candidates_per_scan - chunk_begin);
        const int blocks_x =
            (this_chunk + kThreadsPerBlock - 1) / kThreadsPerBlock;
        const dim3 blocks(blocks_x, scan_count);
        ScoreAllKernelVer7Batched<<<blocks, kThreadsPerBlock, 0, stream>>>(
            cache.grid, w, h, cache.flat_px, cache.flat_py,
            cache.point_offsets, cache.point_counts, cache.cand_offsets,
            cache.cand_counts, cache.cx, cache.cy, cache.scan_min_x,
            cache.scan_max_x, cache.scan_min_y, cache.scan_max_y, scan_count,
            chunk_begin, cache.score);
        ++chunks;
        cuda_ok =
            CheckCuda(cudaGetLastError(), "ScoreAllKernelVer7Batched launch") &&
            cuda_ok;
        if (!cuda_ok) break;
      }
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
    const int blocks_x = chunk_size > 0
                             ? (std::min(chunk_size,
                                         std::max(max_candidates_per_scan, 0)) +
                                kThreadsPerBlock - 1) /
                                   kThreadsPerBlock
                             : 0;
    const int total_blocks = blocks_x * scan_count * std::max(chunks, 1);
    const double grid_mib = BytesToMiB(grid_elems * sizeof(unsigned char));
    const double pxpy_mib = BytesToMiB(point_elems * sizeof(int) * 2);
    const double scan_meta_mib = BytesToMiB(scan_elems * sizeof(int) * 8);
    const double cxcy_mib = BytesToMiB(candidate_elems * sizeof(int) * 2);
    const double score_mib = BytesToMiB(candidate_elems * sizeof(float));

    std::printf("[score_all CUDA ver7 batched profile] calls=%d avg=%.6f ms "
                "last=%.6f ms h2d=%.6f ms kernel=%.6f ms d2h=%.6f ms "
                "host_overhead=%.6f ms scans=%d candidates=%d "
                "scan_points=%d max_candidates_per_scan=%d chunk_size=%d "
                "chunks=%d blocks_x=%d total_blocks=%d threads_per_block=%d "
                "grid_upload=%d grid_mib=%.6f pxpy_mib=%.6f "
                "scan_meta_mib=%.6f cxcy_mib=%.6f score_mib=%.6f\n",
                call_count, total_ms / call_count, ms, h2d_ms, kernel_ms,
                d2h_ms, host_overhead_ms, scan_count, n, total_points,
                max_candidates_per_scan, chunk_size, chunks, blocks_x,
                total_blocks, kThreadsPerBlock, grid_uploaded ? 1 : 0,
                grid_mib, pxpy_mib, scan_meta_mib, cxcy_mib, score_mib);
    std::fflush(stdout);
  }
}

}  // namespace cartographer_parallel
