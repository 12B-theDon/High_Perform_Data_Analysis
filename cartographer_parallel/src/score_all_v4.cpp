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

// v4: hybrid of v1 direct indexing and v3 offset indexing.
//
// Why:
// v3 adds scan_offsets preparation overhead. This overhead is not worth it when
// score_all() is called with only a few candidates, which happens often in the
// ROS bag run. v4 tries to avoid that cost for small calls while still using the
// offset idea for larger candidate sets.
//
// Changes and expected effects:
// - If n < 64, use the v1-style direct grid index path.
//   Expected: avoid scan_offsets setup when the candidate count is too small to
//   benefit from offset reuse.
// - If n >= 64, use offset-based indexing.
//   Expected: reduce repeated grid index arithmetic when many candidates reuse
//   the same scan point offsets.
// - Store scan_offsets in a static thread_local vector.
//   Expected: reuse vector capacity across calls and reduce repeated allocation
//   overhead compared with v3.
//
// Overall expectation:
// Combine v1's low overhead for small workloads with v3's offset reuse idea for
// larger workloads. Limitation: the final grid memory access pattern is still
// irregular, so the random memory access bottleneck remains.
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

    if (n < 64) {
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
    } else {
      static thread_local std::vector<int> scan_offsets;
      scan_offsets.resize(p);
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
  }

  const auto t1 = std::chrono::high_resolution_clock::now();
  const double ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();

  static int call_count = 0;
  static double total_ms = 0.0;
  ++call_count;
  total_ms += ms;

  if (call_count <= 5 || call_count % 10 == 0) {
    std::printf("[score_all v4] calls=%d avg=%.6f ms last=%.6f ms "
                "candidates=%d scan_points=%d\n",
                call_count, total_ms / call_count, ms, n, p);
    std::fflush(stdout);
  }
}

}  // namespace cartographer_parallel
