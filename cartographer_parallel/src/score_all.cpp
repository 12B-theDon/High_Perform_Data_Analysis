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

void score_all(const std::vector<unsigned char>& grid, const int w,
               const int h, const std::vector<int>& px,
               const std::vector<int>& py, const std::vector<int>& cx,
               const std::vector<int>& cy, std::vector<float>* const score) {
  if (score == nullptr) return;
  const auto t0 = std::chrono::high_resolution_clock::now();

  const int n = std::min(cx.size(), cy.size());
  const int p = std::min(px.size(), py.size());
  score->assign(n, 0.0f);
  if (w > 0 && h > 0 && p > 0 &&
      grid.size() >= static_cast<size_t>(w * h)) {
    for (int i = 0; i < n; ++i) {
      int sum = 0;
      for (int j = 0; j < p; ++j) {
        const int x = px[j] + cx[i];
        const int y = py[j] + cy[i];
        if (x >= 0 && x < w && y >= 0 && y < h) {
          sum += grid[y * w + x];
        }
      }
      (*score)[i] =
          static_cast<float>(sum) / (255.0f * static_cast<float>(p));
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
    std::printf("[score_all baseline] calls=%d avg=%.6f ms last=%.6f ms "
                "candidates=%d scan_points=%d\n",
                call_count, total_ms / call_count, ms, n, p);
    std::fflush(stdout);
  }
}

}  // namespace cartographer_parallel
