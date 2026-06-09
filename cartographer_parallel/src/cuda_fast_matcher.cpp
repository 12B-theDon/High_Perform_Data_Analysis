#include "cartographer_parallel/fast_matcher.h"

#include "cartographer_parallel/assignment.h"

#ifdef CARTOGRAPHER_PARALLEL_USE_CUDA
#include "cartographer_parallel/cuda_assignment.h"
#endif

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace cartographer_parallel {
namespace {

std::string Trim(const std::string& s) {
  const char* ws = " \t\r\n";
  const std::string::size_type b = s.find_first_not_of(ws);
  if (b == std::string::npos) return "";
  const std::string::size_type e = s.find_last_not_of(ws);
  return s.substr(b, e - b + 1);
}

std::string Unquote(const std::string& s) {
  const std::string v = Trim(s);
  if (v.size() >= 2 &&
      ((v.front() == '"' && v.back() == '"') ||
       (v.front() == '\'' && v.back() == '\''))) {
    return v.substr(1, v.size() - 2);
  }
  return v;
}

std::string Dirname(const std::string& path) {
  const std::string::size_type slash = path.find_last_of('/');
  return slash == std::string::npos ? "." : path.substr(0, slash);
}

bool IsAbs(const std::string& path) {
  return !path.empty() && path[0] == '/';
}

std::string Join(const std::string& dir, const std::string& file) {
  if (file.empty() || IsAbs(file)) return file;
  return dir == "." ? file : dir + "/" + file;
}

std::vector<double> ParseList(std::string v) {
  for (char& c : v) {
    if (c == '[' || c == ']' || c == ',') c = ' ';
  }
  std::istringstream in(v);
  std::vector<double> out;
  double x = 0.0;
  while (in >> x) out.push_back(x);
  return out;
}

std::string PgmToken(std::istream* in) {
  std::string token;
  char c = 0;
  while (in->get(c)) {
    if (std::isspace(static_cast<unsigned char>(c))) continue;
    if (c == '#') {
      in->ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      continue;
    }
    token.push_back(c);
    break;
  }
  while (in->get(c)) {
    if (std::isspace(static_cast<unsigned char>(c))) break;
    if (c == '#') {
      in->ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      break;
    }
    token.push_back(c);
  }
  return token;
}

int ClampInt(const int x, const int lo, const int hi) {
  return std::max(lo, std::min(hi, x));
}

double NormalizeYaw(double yaw) {
  while (yaw > M_PI) yaw -= 2.0 * M_PI;
  while (yaw < -M_PI) yaw += 2.0 * M_PI;
  return yaw;
}

double MsSince(const std::chrono::high_resolution_clock::time_point& start,
               const std::chrono::high_resolution_clock::time_point& end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

#ifdef CARTOGRAPHER_PARALLEL_USE_CUDA
enum class CudaScoreVersion {
  kCpu,
  kBaseline,
  kVer1,
  kVer2,
  kVer3,
  kVer4,
  kVer5,
  kVer6,
  kVer7,
  kVer8,
};

CudaScoreVersion GetCudaScoreVersion() {
  const char* const env = std::getenv("CUDA_SCORE_VERSION");
  const std::string version = env == nullptr ? "ver6" : Trim(env);
  if (version == "cpu") return CudaScoreVersion::kCpu;
  if (version == "baseline" || version == "default" || version == "ver0") {
    return CudaScoreVersion::kBaseline;
  }
  if (version == "ver1") return CudaScoreVersion::kVer1;
  if (version == "ver2") return CudaScoreVersion::kVer2;
  if (version == "ver3") return CudaScoreVersion::kVer3;
  if (version == "ver4") return CudaScoreVersion::kVer4;
  if (version == "ver5") return CudaScoreVersion::kVer5;
  if (version == "ver6") return CudaScoreVersion::kVer6;
  if (version == "ver7") return CudaScoreVersion::kVer7;
  if (version == "ver8") return CudaScoreVersion::kVer8;
  return CudaScoreVersion::kVer6;
}

const char* CudaScoreVersionName(const CudaScoreVersion version) {
  switch (version) {
    case CudaScoreVersion::kCpu:
      return "cpu";
    case CudaScoreVersion::kBaseline:
      return "baseline";
    case CudaScoreVersion::kVer1:
      return "ver1";
    case CudaScoreVersion::kVer2:
      return "ver2";
    case CudaScoreVersion::kVer3:
      return "ver3";
    case CudaScoreVersion::kVer4:
      return "ver4";
    case CudaScoreVersion::kVer5:
      return "ver5";
    case CudaScoreVersion::kVer6:
      return "ver6";
    case CudaScoreVersion::kVer7:
      return "ver7";
    case CudaScoreVersion::kVer8:
      return "ver8";
  }
  return "ver6";
}

int InferCandidateStep(const std::vector<int>& cx,
                       const std::vector<int>& cy) {
  int step = std::numeric_limits<int>::max();
  for (size_t i = 1; i < cx.size() && i < cy.size(); ++i) {
    const int dx = std::abs(cx[i] - cx[i - 1]);
    const int dy = std::abs(cy[i] - cy[i - 1]);
    if (dx > 0) step = std::min(step, dx);
    if (dy > 0) step = std::min(step, dy);
  }
  return step == std::numeric_limits<int>::max() ? 1 : step;
}
#endif

}  // namespace

bool FastMatcher::LoadMap(const std::string& yaml_file) {
  std::ifstream yaml(yaml_file);
  if (!yaml) return false;

  std::string image;
  bool negate = false;
  double occupied_thresh = 0.65;
  double free_thresh = 0.196;
  std::string line;
  while (std::getline(yaml, line)) {
    line = line.substr(0, line.find('#'));
    const std::string::size_type colon = line.find(':');
    if (colon == std::string::npos) continue;
    const std::string key = Trim(line.substr(0, colon));
    const std::string val = Trim(line.substr(colon + 1));
    if (key == "image") {
      image = Join(Dirname(yaml_file), Unquote(val));
    } else if (key == "resolution") {
      res_ = std::stod(val);
    } else if (key == "origin") {
      const std::vector<double> origin = ParseList(val);
      if (origin.size() >= 2) {
        ox_ = origin[0];
        oy_ = origin[1];
      }
    } else if (key == "negate") {
      negate = (val == "1" || val == "true" || val == "True");
    } else if (key == "occupied_thresh") {
      occupied_thresh = std::stod(val);
    } else if (key == "free_thresh") {
      free_thresh = std::stod(val);
    }
  }
  (void)occupied_thresh;
  (void)free_thresh;
  if (image.empty()) return false;

  std::ifstream pgm(image, std::ios::binary);
  if (!pgm) return false;
  const std::string magic = PgmToken(&pgm);
  if (magic != "P5" && magic != "P2") return false;
  w_ = std::stoi(PgmToken(&pgm));
  h_ = std::stoi(PgmToken(&pgm));
  const int max_value = std::stoi(PgmToken(&pgm));
  if (w_ <= 0 || h_ <= 0 || max_value <= 0 || max_value > 255) return false;

  std::vector<unsigned char> pixels(w_ * h_, 0);
  if (magic == "P5") {
    pgm.read(reinterpret_cast<char*>(pixels.data()), pixels.size());
    if (pgm.gcount() != static_cast<std::streamsize>(pixels.size())) {
      return false;
    }
  } else {
    for (unsigned char& pixel : pixels) {
      const std::string token = PgmToken(&pgm);
      if (token.empty()) return false;
      pixel = static_cast<unsigned char>(
          ClampInt(std::stoi(token), 0, max_value));
    }
  }

  map_.assign(w_ * h_, 0);
  for (int i = 0; i < w_ * h_; ++i) {
    const double v = static_cast<double>(pixels[i]) / max_value;
    const double occ = negate ? v : (1.0 - v);
    map_[i] = static_cast<unsigned char>(
        ClampInt(static_cast<int>(std::lround(255.0 * occ)), 0, 255));
  }
  grids_ = MakeGridStack();
  return true;
}

void FastMatcher::SetOptions(const MatchOpt& opt) {
  opt_ = opt;
  if (has_map()) grids_ = MakeGridStack();
}

std::vector<FastMatcher::Scan> FastMatcher::MakeScans(
    const std::vector<float>& xs, const std::vector<float>& ys,
    const Pose2& init, int* const num_ang, double* const step) const {
  double max_range = 3.0 * res_;
  for (size_t i = 0; i < xs.size() && i < ys.size(); ++i) {
    max_range = std::max(max_range,
                         std::hypot(static_cast<double>(xs[i]),
                                    static_cast<double>(ys[i])));
  }

  double angle_step = opt_.angular_step;
  if (angle_step <= 0.0) {
    const double c = 1.0 - (res_ * res_) / (2.0 * max_range * max_range);
    angle_step = 0.999 * std::acos(std::max(-1.0, std::min(1.0, c)));
    if (!std::isfinite(angle_step) || angle_step <= 0.0) angle_step = 0.05;
  }
  const int n_ang = std::max(0, static_cast<int>(
                                   std::ceil(opt_.angular_window / angle_step)));
  const int scan_count = 2 * n_ang + 1;
  if (num_ang) *num_ang = n_ang;
  if (step) *step = angle_step;

  std::vector<Scan> scans(scan_count);
  for (int s = 0; s < scan_count; ++s) {
    const double da = (s - n_ang) * angle_step;
    const double yaw = init.yaw + da;
    const double c = std::cos(yaw);
    const double sn = std::sin(yaw);
    scans[s].x.reserve(xs.size());
    scans[s].y.reserve(xs.size());
    for (size_t i = 0; i < xs.size() && i < ys.size(); ++i) {
      const double wx = init.x + c * xs[i] - sn * ys[i];
      const double wy = init.y + sn * xs[i] + c * ys[i];
      const int mx = static_cast<int>(std::floor((wx - ox_) / res_));
      const int row_bottom = static_cast<int>(std::floor((wy - oy_) / res_));
      const int my = h_ - 1 - row_bottom;
      scans[s].x.push_back(mx);
      scans[s].y.push_back(my);
    }
  }
  return scans;
}

std::vector<FastMatcher::Bounds> FastMatcher::MakeBounds(
    const std::vector<Scan>& scans, const double window,
    const bool full_map) const {
  const int lin = static_cast<int>(std::ceil(window / res_));
  std::vector<Bounds> bounds(scans.size());
  for (size_t s = 0; s < scans.size(); ++s) {
    Bounds b;
    if (full_map) {
      b.min_x = std::numeric_limits<int>::lowest() / 4;
      b.max_x = std::numeric_limits<int>::max() / 4;
      b.min_y = std::numeric_limits<int>::lowest() / 4;
      b.max_y = std::numeric_limits<int>::max() / 4;
    } else {
      b.min_x = -lin;
      b.max_x = lin;
      b.min_y = -lin;
      b.max_y = lin;
      // Local/global-window search should stay centered on the initial pose.
      // Out-of-map scan points are already scored as zero in score_all().
      bounds[s] = b;
      continue;
    }

    for (size_t i = 0; i < scans[s].x.size(); ++i) {
      b.min_x = std::max(b.min_x, -scans[s].x[i]);
      b.max_x = std::min(b.max_x, w_ - 1 - scans[s].x[i]);
      b.min_y = std::max(b.min_y, -scans[s].y[i]);
      b.max_y = std::min(b.max_y, h_ - 1 - scans[s].y[i]);
    }
    bounds[s] = b;
  }
  return bounds;
}

std::vector<FastMatcher::Grid> FastMatcher::MakeGridStack() const {
  const int depth = std::max(1, opt_.branch_depth);
  std::vector<Grid> grids;
  grids.reserve(depth);
  for (int level = 0; level < depth; ++level) {
    const int win = 1 << level;
    Grid g;
    g.w = w_;
    g.h = h_;
    g.win = win;
    g.cell.assign(w_ * h_, 0);
    for (int y = 0; y < h_; ++y) {
      for (int x = 0; x < w_; ++x) {
        unsigned char best = 0;
        for (int dy = 0; dy < win && y + dy < h_; ++dy) {
          for (int dx = 0; dx < win && x + dx < w_; ++dx) {
            best = std::max(best, map_[(y + dy) * w_ + (x + dx)]);
          }
        }
        g.cell[y * w_ + x] = best;
      }
    }
    grids.push_back(g);
  }
  return grids;
}

std::vector<FastMatcher::Cand> FastMatcher::MakeLowCands(
    const std::vector<Bounds>& bounds, const int depth) const {
  const int step = 1 << depth;
  std::vector<Cand> out;
  for (size_t s = 0; s < bounds.size(); ++s) {
    if (bounds[s].min_x > bounds[s].max_x ||
        bounds[s].min_y > bounds[s].max_y) {
      continue;
    }
    std::vector<int> cx;
    std::vector<int> cy;
    make_cand(bounds[s].min_x, bounds[s].max_x, bounds[s].min_y,
              bounds[s].max_y, step, &cx, &cy);
    for (size_t i = 0; i < cx.size(); ++i) {
      Cand c;
      c.scan = static_cast<int>(s);
      c.x = cx[i];
      c.y = cy[i];
      out.push_back(c);
    }
  }
  return out;
}

void FastMatcher::Score(const Grid& grid, const std::vector<Scan>& scans,
                        std::vector<Cand>* const cand) const {
  if (cand == nullptr || cand->empty()) return;
  const auto score_start = std::chrono::high_resolution_clock::now();
  int score_calls = 0;
  size_t scored_candidates = 0;
  size_t max_candidates_per_scan = 0;
#ifdef CARTOGRAPHER_PARALLEL_USE_CUDA
  constexpr size_t kBatchedCandidateThreshold = 1024;
  const CudaScoreVersion cuda_version = GetCudaScoreVersion();
  static bool printed_cuda_version = false;
  if (!printed_cuda_version) {
    std::printf("[CUDA score dispatcher] CUDA_SCORE_VERSION=%s\n",
                CudaScoreVersionName(cuda_version));
    std::fflush(stdout);
    printed_cuda_version = true;
  }
  const bool use_batched =
      (cuda_version == CudaScoreVersion::kVer5 ||
       cuda_version == CudaScoreVersion::kVer6 ||
       cuda_version == CudaScoreVersion::kVer7 ||
       cuda_version == CudaScoreVersion::kVer8) &&
      cand->size() >= kBatchedCandidateThreshold;
  const bool use_ver3_bounds =
      cuda_version == CudaScoreVersion::kVer3 &&
      cand->size() >= kBatchedCandidateThreshold;
  if (!use_batched) {
    for (size_t s = 0; s < scans.size(); ++s) {
      std::vector<int> ids;
      std::vector<int> cx;
      std::vector<int> cy;
      for (size_t i = 0; i < cand->size(); ++i) {
        if ((*cand)[i].scan == static_cast<int>(s)) {
          ids.push_back(i);
          cx.push_back((*cand)[i].x);
          cy.push_back((*cand)[i].y);
        }
      }
      if (ids.empty()) continue;
      ++score_calls;
      scored_candidates += ids.size();
      max_candidates_per_scan = std::max(max_candidates_per_scan, ids.size());
      std::vector<float> score;
      if (use_ver3_bounds) {
        const int min_x = *std::min_element(cx.begin(), cx.end());
        const int max_x = *std::max_element(cx.begin(), cx.end());
        const int min_y = *std::min_element(cy.begin(), cy.end());
        const int max_y = *std::max_element(cy.begin(), cy.end());
        const int step = InferCandidateStep(cx, cy);
        score_all_CUDA_ver3_bounds(grid.cell, grid.w, grid.h, scans[s].x,
                                   scans[s].y, min_x, max_x, min_y, max_y,
                                   step, &score);
      } else if (cuda_version == CudaScoreVersion::kCpu) {
        score_all(grid.cell, grid.w, grid.h, scans[s].x, scans[s].y, cx, cy,
                  &score);
      } else if (cuda_version == CudaScoreVersion::kBaseline) {
        score_all_CUDA(grid.cell, grid.w, grid.h, scans[s].x, scans[s].y, cx,
                       cy, &score);
      } else if (cuda_version == CudaScoreVersion::kVer1) {
        score_all_CUDA_ver1(grid.cell, grid.w, grid.h, scans[s].x,
                            scans[s].y, cx, cy, &score);
      } else if (cuda_version == CudaScoreVersion::kVer2) {
        score_all_CUDA_ver2(grid.cell, grid.w, grid.h, scans[s].x,
                            scans[s].y, cx, cy, &score);
      } else {
        score_all_CUDA_ver4(grid.cell, grid.w, grid.h, scans[s].x,
                            scans[s].y, cx, cy, &score);
      }
      for (size_t i = 0; i < ids.size() && i < score.size(); ++i) {
        (*cand)[ids[i]].score = score[i];
      }
    }
  } else {
    std::vector<int> flat_px;
    std::vector<int> flat_py;
    std::vector<int> flat_cell;
    const bool need_ver6_aux = cuda_version == CudaScoreVersion::kVer6;
    const bool use_sorted_offsets = cuda_version == CudaScoreVersion::kVer8;
    std::vector<int> point_offsets(scans.size(), 0);
    std::vector<int> point_counts(scans.size(), 0);
    std::vector<int> scan_min_x(scans.size(), 0);
    std::vector<int> scan_max_x(scans.size(), -1);
    std::vector<int> scan_min_y(scans.size(), 0);
    std::vector<int> scan_max_y(scans.size(), -1);
    for (size_t s = 0; s < scans.size(); ++s) {
      point_offsets[s] = static_cast<int>(flat_px.size());
      const int point_count =
          static_cast<int>(std::min(scans[s].x.size(), scans[s].y.size()));
      point_counts[s] = point_count;
      if (point_count > 0) {
        int min_x = scans[s].x[0];
        int max_x = scans[s].x[0];
        int min_y = scans[s].y[0];
        int max_y = scans[s].y[0];
        std::vector<int> order;
        if (use_sorted_offsets) {
          order.resize(static_cast<size_t>(point_count));
          for (int i = 0; i < point_count; ++i) {
            order[static_cast<size_t>(i)] = i;
          }
          std::sort(order.begin(), order.end(),
                    [&](const int lhs, const int rhs) {
                      const int lhs_cell =
                          scans[s].y[static_cast<size_t>(lhs)] * grid.w +
                          scans[s].x[static_cast<size_t>(lhs)];
                      const int rhs_cell =
                          scans[s].y[static_cast<size_t>(rhs)] * grid.w +
                          scans[s].x[static_cast<size_t>(rhs)];
                      if (lhs_cell != rhs_cell) return lhs_cell < rhs_cell;
                      if (scans[s].y[static_cast<size_t>(lhs)] !=
                          scans[s].y[static_cast<size_t>(rhs)]) {
                        return scans[s].y[static_cast<size_t>(lhs)] <
                               scans[s].y[static_cast<size_t>(rhs)];
                      }
                      return scans[s].x[static_cast<size_t>(lhs)] <
                             scans[s].x[static_cast<size_t>(rhs)];
                    });
        }
        for (int i = 0; i < point_count; ++i) {
          const int source_index =
              use_sorted_offsets ? order[static_cast<size_t>(i)] : i;
          const int px = scans[s].x[static_cast<size_t>(source_index)];
          const int py = scans[s].y[static_cast<size_t>(source_index)];
          flat_px.push_back(px);
          flat_py.push_back(py);
          if (need_ver6_aux || use_sorted_offsets) {
            flat_cell.push_back(py * grid.w + px);
          }
          min_x = std::min(min_x, px);
          max_x = std::max(max_x, px);
          min_y = std::min(min_y, py);
          max_y = std::max(max_y, py);
        }
        scan_min_x[s] = min_x;
        scan_max_x[s] = max_x;
        scan_min_y[s] = min_y;
        scan_max_y[s] = max_y;
      }
    }

    std::vector<int> cand_offsets(scans.size(), 0);
    std::vector<int> cand_counts(scans.size(), 0);
    for (const Cand& candidate : *cand) {
      if (candidate.scan >= 0 &&
          candidate.scan < static_cast<int>(scans.size())) {
        ++cand_counts[candidate.scan];
      }
    }

    int candidate_offset = 0;
    for (size_t s = 0; s < cand_counts.size(); ++s) {
      cand_offsets[s] = candidate_offset;
      candidate_offset += cand_counts[s];
      if (cand_counts[s] > 0) {
        ++score_calls;
        max_candidates_per_scan =
            std::max(max_candidates_per_scan,
                     static_cast<size_t>(cand_counts[s]));
      }
    }
    scored_candidates = static_cast<size_t>(candidate_offset);

    std::vector<int> ids(scored_candidates, 0);
    std::vector<int> cx(scored_candidates, 0);
    std::vector<int> cy(scored_candidates, 0);
    std::vector<int> candidate_cell(need_ver6_aux ? scored_candidates : 0, 0);
    std::vector<unsigned char> full_inside(need_ver6_aux ? scored_candidates : 0,
                                           0);
    std::vector<int> fill = cand_offsets;
    for (size_t i = 0; i < cand->size(); ++i) {
      const int scan = (*cand)[i].scan;
      if (scan < 0 || scan >= static_cast<int>(scans.size())) continue;
      const int out = fill[scan]++;
      ids[out] = static_cast<int>(i);
      const int offset_x = (*cand)[i].x;
      const int offset_y = (*cand)[i].y;
      cx[out] = offset_x;
      cy[out] = offset_y;
      if (need_ver6_aux) {
        candidate_cell[out] = offset_y * grid.w + offset_x;
        if (point_counts[scan] > 0 &&
            scan_min_x[scan] + offset_x >= 0 &&
            scan_max_x[scan] + offset_x < grid.w &&
            scan_min_y[scan] + offset_y >= 0 &&
            scan_max_y[scan] + offset_y < grid.h) {
          full_inside[out] = 1;
        }
      }
    }

    std::vector<float> score;
    if (cuda_version == CudaScoreVersion::kVer5) {
      score_all_CUDA_ver5_batched(grid.cell, grid.w, grid.h, flat_px, flat_py,
                                  point_offsets, point_counts, cand_offsets,
                                  cand_counts, cx, cy, &score);
    } else {
      if (cuda_version == CudaScoreVersion::kVer6) {
        score_all_CUDA_ver6_batched(grid.cell, grid.w, grid.h, flat_px, flat_py,
                                    flat_cell, point_offsets, point_counts,
                                    cand_offsets, cand_counts, cx, cy,
                                    candidate_cell, full_inside, &score);
      } else if (cuda_version == CudaScoreVersion::kVer8) {
        score_all_CUDA_ver8_batched(
            grid.cell, grid.w, grid.h, flat_px, flat_py, flat_cell,
            point_offsets, point_counts, cand_offsets, cand_counts, cx, cy,
            scan_min_x, scan_max_x, scan_min_y, scan_max_y, &score);
      } else {
        score_all_CUDA_ver7_batched(
            grid.cell, grid.w, grid.h, flat_px, flat_py, point_offsets,
            point_counts, cand_offsets, cand_counts, cx, cy, scan_min_x,
            scan_max_x, scan_min_y, scan_max_y, &score);
      }
    }
    for (size_t i = 0; i < ids.size() && i < score.size(); ++i) {
      (*cand)[ids[i]].score = score[i];
    }
  }
#else
  for (size_t s = 0; s < scans.size(); ++s) {
    std::vector<int> ids;
    std::vector<int> cx;
    std::vector<int> cy;
    for (size_t i = 0; i < cand->size(); ++i) {
      if ((*cand)[i].scan == static_cast<int>(s)) {
        ids.push_back(i);
        cx.push_back((*cand)[i].x);
        cy.push_back((*cand)[i].y);
      }
    }
    if (ids.empty()) continue;
    ++score_calls;
    scored_candidates += ids.size();
    max_candidates_per_scan = std::max(max_candidates_per_scan, ids.size());
    std::vector<float> score;
    score_all(grid.cell, grid.w, grid.h, scans[s].x, scans[s].y, cx, cy,
              &score);
    for (size_t i = 0; i < ids.size() && i < score.size(); ++i) {
      (*cand)[ids[i]].score = score[i];
    }
  }
#endif
  std::sort(cand->begin(), cand->end(),
            [](const Cand& a, const Cand& b) { return a.score > b.score; });
  const auto score_end = std::chrono::high_resolution_clock::now();
  static int score_profile_calls = 0;
  ++score_profile_calls;
  if (score_profile_calls <= 20 || score_profile_calls % 100 == 0) {
    std::printf("[Score profile] calls=%d total=%.6f ms scans=%zu "
                "score_calls=%d input_candidates=%zu scored_candidates=%zu "
                "max_candidates_per_scan=%zu grid=%dx%d\n",
                score_profile_calls, MsSince(score_start, score_end),
                scans.size(), score_calls, cand->size(), scored_candidates,
                max_candidates_per_scan, grid.w, grid.h);
    std::fflush(stdout);
  }
}

void FastMatcher::ScoreRegular(const Grid& grid,
                               const std::vector<Scan>& scans,
                               const std::vector<Bounds>& bounds,
                               const int depth,
                               std::vector<Cand>* const cand) const {
  if (cand == nullptr || cand->empty()) return;
#ifdef CARTOGRAPHER_PARALLEL_USE_CUDA
  const int step = 1 << depth;
  for (size_t s = 0; s < scans.size() && s < bounds.size(); ++s) {
    if (bounds[s].min_x > bounds[s].max_x ||
        bounds[s].min_y > bounds[s].max_y) {
      continue;
    }
    std::vector<float> score;
    score_all_CUDA_ver3_bounds(grid.cell, grid.w, grid.h, scans[s].x,
                               scans[s].y, bounds[s].min_x, bounds[s].max_x,
                               bounds[s].min_y, bounds[s].max_y, step,
                               &score);
    size_t score_index = 0;
    for (Cand& candidate : *cand) {
      if (candidate.scan != static_cast<int>(s)) continue;
      if (score_index < score.size()) {
        candidate.score = score[score_index];
      }
      ++score_index;
    }
  }
  std::sort(cand->begin(), cand->end(),
            [](const Cand& a, const Cand& b) { return a.score > b.score; });
#else
  (void)bounds;
  (void)depth;
  Score(grid, scans, cand);
#endif
}

FastMatcher::Cand FastMatcher::Branch(const std::vector<Grid>& grids,
                                      const std::vector<Scan>& scans,
                                      const std::vector<Bounds>& bounds,
                                      const std::vector<Cand>& cand,
                                      const int depth,
                                      const float min_score) const {
  if (cand.empty()) {
    Cand empty;
    empty.score = 0.0f;
    return empty;
  }
  if (depth == 0) return cand.front();

  Cand best;
  best.score = min_score;
  const int half = 1 << (depth - 1);
  for (const Cand& c : cand) {
    if (c.score <= best.score) break;
    std::vector<Cand> child;
    for (const int dx : {0, half}) {
      if (c.x + dx > bounds[c.scan].max_x) continue;
      for (const int dy : {0, half}) {
        if (c.y + dy > bounds[c.scan].max_y) continue;
        Cand next;
        next.scan = c.scan;
        next.x = c.x + dx;
        next.y = c.y + dy;
        child.push_back(next);
      }
    }
    Score(grids[depth - 1], scans, &child);
    const Cand refined = Branch(grids, scans, bounds, child, depth - 1,
                                best.score);
    if (refined.score > best.score) best = refined;
  }
  return best;
}

CandOut FastMatcher::ToOut(const Cand& cand, const Pose2& init,
                           const int num_ang, const double step) const {
  CandOut out;
  out.x = init.x + cand.x * res_;
  out.y = init.y - cand.y * res_;
  out.yaw = NormalizeYaw(init.yaw + (cand.scan - num_ang) * step);
  out.score = cand.score;
  return out;
}

bool FastMatcher::Match(const std::vector<float>& xs,
                        const std::vector<float>& ys, const Pose2& init,
                        const bool global, MatchOut* const out) const {
  return MatchWithWindow(xs, ys, init,
                         global ? opt_.global_window : opt_.linear_window,
                         global && opt_.full_map_search, out);
}

bool FastMatcher::MatchWithWindow(const std::vector<float>& xs,
                                  const std::vector<float>& ys,
                                  const Pose2& init,
                                  const double window,
                                  const bool full_map,
                                  MatchOut* const out) const {
  if (out == nullptr) return false;
  *out = MatchOut();
  if (!has_map() || xs.empty() || ys.empty()) return false;

  int num_ang = 0;
  double step = 0.0;
  const auto match_start = std::chrono::high_resolution_clock::now();
  const auto make_scans_start = std::chrono::high_resolution_clock::now();
  const std::vector<Scan> scans = MakeScans(xs, ys, init, &num_ang, &step);
  const auto make_scans_end = std::chrono::high_resolution_clock::now();
  const auto make_bounds_start = std::chrono::high_resolution_clock::now();
  const std::vector<Bounds> bounds = MakeBounds(scans, window, full_map);
  const auto make_bounds_end = std::chrono::high_resolution_clock::now();
  const auto make_grid_start = std::chrono::high_resolution_clock::now();
  std::vector<Grid> temp_grids;
  const std::vector<Grid>* grids_ptr = &grids_;
  if (grids_ptr->empty()) {
    temp_grids = MakeGridStack();
    grids_ptr = &temp_grids;
  }
  const auto make_grid_end = std::chrono::high_resolution_clock::now();
  const std::vector<Grid>& grids = *grids_ptr;
  const int max_depth = static_cast<int>(grids.size()) - 1;

  const auto make_low_cands_start = std::chrono::high_resolution_clock::now();
  std::vector<Cand> coarse = MakeLowCands(bounds, max_depth);
  const auto make_low_cands_end = std::chrono::high_resolution_clock::now();
  const size_t coarse_candidate_count = coarse.size();
  const auto score_coarse_start = std::chrono::high_resolution_clock::now();
  Score(grids[max_depth], scans, &coarse);
  const auto score_coarse_end = std::chrono::high_resolution_clock::now();
  if (coarse.empty()) return false;
  const auto branch_start = std::chrono::high_resolution_clock::now();
  const Cand best = Branch(grids, scans, bounds, coarse, max_depth,
                           opt_.min_score);
  const auto branch_end = std::chrono::high_resolution_clock::now();
  out->ok = best.score > opt_.min_score;
  out->score = best.score;
  out->pose = init;
  if (out->ok) {
    const CandOut best_out = ToOut(best, init, num_ang, step);
    out->pose.x = best_out.x;
    out->pose.y = best_out.y;
    out->pose.yaw = best_out.yaw;
  }

  const int n = std::min(opt_.max_cand, static_cast<int>(coarse.size()));
  out->cand.reserve(n);
  for (int i = 0; i < n; ++i) {
    out->cand.push_back(ToOut(coarse[i], init, num_ang, step));
  }
  const auto match_end = std::chrono::high_resolution_clock::now();
  static int match_profile_calls = 0;
  ++match_profile_calls;
  if (match_profile_calls <= 20 || match_profile_calls % 20 == 0) {
    std::printf("[MatchWithWindow profile] calls=%d total=%.6f ms "
                "MakeScans=%.6f ms MakeBounds=%.6f ms MakeGridStack=%.6f ms "
                "MakeLowCands=%.6f ms ScoreCoarse=%.6f ms Branch=%.6f ms "
                "scans=%zu coarse_candidates=%zu max_depth=%d window=%.6f "
                "full_map=%d ok=%d score=%.6f\n",
                match_profile_calls, MsSince(match_start, match_end),
                MsSince(make_scans_start, make_scans_end),
                MsSince(make_bounds_start, make_bounds_end),
                MsSince(make_grid_start, make_grid_end),
                MsSince(make_low_cands_start, make_low_cands_end),
                MsSince(score_coarse_start, score_coarse_end),
                MsSince(branch_start, branch_end), scans.size(),
                coarse_candidate_count, max_depth, window, full_map ? 1 : 0,
                out->ok ? 1 : 0, out->score);
    std::fflush(stdout);
  }
  return out->ok;
}

}  // namespace cartographer_parallel
