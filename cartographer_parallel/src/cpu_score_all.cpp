#include "cartographer_parallel/cpu_assignment.h"

#include <algorithm>
#include <chrono>
#include <cstdio>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cartographer_parallel {

void score_all_cpu_parallel(const std::vector<unsigned char>& grid, const int w,
                            const int h, const std::vector<int>& px,
                            const std::vector<int>& py,
                            const std::vector<int>& cx,
                            const std::vector<int>& cy,
                            std::vector<float>* const score) {
  if (score == nullptr) return;
  const auto t0 = std::chrono::high_resolution_clock::now();

  const int n = static_cast<int>(std::min(cx.size(), cy.size()));
  const int p = static_cast<int>(std::min(px.size(), py.size()));
  score->assign(n, 0.0f);

  if (w > 0 && h > 0 && p > 0 &&
      grid.size() >= static_cast<size_t>(w) * static_cast<size_t>(h)) {
    const unsigned char* const grid_ptr = grid.data();
    const int* const px_ptr = px.data();
    const int* const py_ptr = py.data();
    const int* const cx_ptr = cx.data();
    const int* const cy_ptr = cy.data();
    float* const score_ptr = score->data();
    const float inv_norm = 1.0f / (255.0f * static_cast<float>(p));

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
      const int cand_x = cx_ptr[i];
      const int cand_y = cy_ptr[i];
      int sum = 0;

#pragma omp simd reduction(+ : sum)
      for (int j = 0; j < p; ++j) {
        const int x = px_ptr[j] + cand_x;
        const int y = py_ptr[j] + cand_y;
        if (x >= 0 && x < w && y >= 0 && y < h) {
          sum += static_cast<int>(
              grid_ptr[static_cast<size_t>(y) * static_cast<size_t>(w) +
                       static_cast<size_t>(x)]);
        }
      }

      score_ptr[i] = static_cast<float>(sum) * inv_norm;
    }
  }

  const auto t1 = std::chrono::high_resolution_clock::now();
  const double ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();

  static int call_count = 0;
  static double total_ms = 0.0;
  ++call_count;
  total_ms += ms;

#ifdef _OPENMP
  const int threads = omp_get_max_threads();
#else
  const int threads = 1;
#endif

  if (call_count <= 5 || call_count % 10 == 0) {
    std::printf("[score_all CPU openmp_simd] calls=%d avg=%.6f ms "
                "last=%.6f ms candidates=%d scan_points=%d "
                "threads=%d duplicate_compaction=0 scan_compression=0\n",
                call_count,
                total_ms / static_cast<double>(call_count),
                ms,
                n,
                p,
                threads);
    std::fflush(stdout);
  }
}

}  // namespace cartographer_parallel
