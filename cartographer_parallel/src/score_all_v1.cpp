#include "cartographer_parallel/assignment.h"

#include <algorithm>
#include <chrono>
#include <cstdio>

namespace cartographer_parallel {

void make_cand(const int min_x, const int max_x, const int min_y,
               const int max_y, const int step, std::vector<int>* const cx,
               std::vector<int>* const cy) {
  if (cx == nullptr || cy == nullptr || step <= 0) return;
  for (int x = min_x; x <= max_x; x += step) {
    for (int y = min_y; y <= max_y; y += step) {
      cx->push_back(x);
      cy->push_back(y);
    }
  }
}

// v1: single-thread CPU optimization.
//
// Why:
// perf showed that the inner scan-point loop is the hottest part of score_all().
// The main bottleneck is still the grid memory load, but the baseline also does
// repeated vector indexing, repeated candidate loads, and a division for every
// score. v1 reduces these extra costs without changing the algorithm.
//
// Changes and expected effects:
// - Use vector::data() raw pointers.
//   Expected: reduce repeated std::vector operator[] overhead in the hot loop.
// - Load cx[i] and cy[i] once per candidate as cxi/cyi.
//   Expected: avoid reloading the same candidate offset for every scan point.
// - Precompute inv_norm and use multiplication instead of division.
//   Expected: make final score normalization cheaper.
// - Use unsigned bounds checks.
//   Expected: combine x >= 0 and x < w style checks into fewer comparisons.
//
// Overall expectation:
// Keep the same single-thread behavior but reduce instruction overhead around
// the random grid access hotspot.
void score_all(const std::vector<unsigned char>& grid, const int w,
               const int h, const std::vector<int>& px,
               const std::vector<int>& py, const std::vector<int>& cx,
               const std::vector<int>& cy, std::vector<float>* const score) {
  if (score == nullptr) return;
  const auto t0 = std::chrono::high_resolution_clock::now();

  const int n = std::min(cx.size(), cy.size());
  const int p = std::min(px.size(), py.size());
  score->assign(n, 0.0f);
  const bool valid =
      w > 0 && h > 0 && p > 0 && grid.size() >= static_cast<size_t>(w * h);

  if (valid) {


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
            static_cast<unsigned int>(y) < uh){
              sum += grid_data[y * w + x];
        }
      }
      score_data[i] = static_cast<float>(sum) * inv_norm;
    }
  }

  const auto t1 = std::chrono::high_resolution_clock::now();
  const double ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();

  static int call_count = 0;
  static double total_ms = 0.0;
  ++call_count;
  total_ms += ms;

  if (call_count <= 5 || call_count % 10 == 0) {
    std::printf("[score_all v1] calls=%d avg=%.6f ms last=%.6f ms "
                "candidates=%d scan_points=%d\n",
                call_count, total_ms / call_count, ms, n, p);
    std::fflush(stdout);
  }
}

}  // namespace cartographer_parallel
