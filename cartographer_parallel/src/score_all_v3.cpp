#include "cartographer_parallel/assignment.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

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

// v3: precompute scan-point grid offsets.
//
// Why:
// perf showed that the grid access line is expensive. Part of that line is the
// address calculation y * w + x. v3 tests whether moving some of that index
// calculation outside the candidate loop can reduce inner-loop work.
//
// Changes and expected effects:
// - Precompute scan_offsets[j] = py[j] * w + px[j] once per score_all call.
//   Expected: reuse each scan point's base grid offset across all candidates.
// - Compute candidate_offset = cy[i] * w + cx[i] once per candidate.
//   Expected: convert y * w + x into offset_data[j] + candidate_offset.
// - Use grid_data[offset_data[j] + candidate_offset] for the load.
//   Expected: reduce repeated arithmetic in the hottest loop.
//
// Overall expectation:
// Reduce index arithmetic cost when there are many candidates. Limitation: this
// does not remove the random grid memory access itself, and scan_offsets must be
// prepared every call.
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

    std::vector<int> scan_offsets(p);
    for (int j = 0; j < p; ++j) {
      scan_offsets[j] = py_data[j] * w + px_data[j];
    }
    const int* const offset_data = scan_offsets.data();

    for (int i = 0; i < n; ++i) {
      const int cxi = cx_data[i];
      const int cyi = cy_data[i];
      const int candidate_offset = cyi * w + cxi;

      int sum = 0;

      for (int j = 0; j < p; ++j) {
        const int x = px_data[j] + cxi;
        const int y = py_data[j] + cyi;

        if (static_cast<unsigned int>(x) < uw &&
            static_cast<unsigned int>(y) < uh) {
          sum += grid_data[offset_data[j] + candidate_offset];
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
    std::printf("[score_all v3] calls=%d avg=%.6f ms last=%.6f ms "
                "candidates=%d scan_points=%d\n",
                call_count, total_ms / call_count, ms, n, p);
    std::fflush(stdout);
  }
}

}  // namespace cartographer_parallel
