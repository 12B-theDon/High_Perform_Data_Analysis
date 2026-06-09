#ifndef CARTOGRAPHER_PARALLEL_CUDA_ASSIGNMENT_H_
#define CARTOGRAPHER_PARALLEL_CUDA_ASSIGNMENT_H_

#include <vector>

namespace cartographer_parallel {

// Assignment baseline: score every candidate against a precomputed grid.
//
// grid: row-major unsigned char map with size w*h.
// px/py: scan endpoint cell coordinates.
// cx/cy: candidate cell offsets.
// score: resized to cx.size(), values are in [0, 1].
//
// This is the function students are expected to parallelize with CUDA.
void score_all_CUDA(const std::vector<unsigned char>& grid, int w, int h,
                    const std::vector<int>& px, const std::vector<int>& py,
                    const std::vector<int>& cx, const std::vector<int>& cy,
                    std::vector<float>* score);

void score_all_CUDA_ver1(const std::vector<unsigned char>& grid, int w, int h,
                         const std::vector<int>& px,
                         const std::vector<int>& py,
                         const std::vector<int>& cx,
                         const std::vector<int>& cy,
                         std::vector<float>* score);

void score_all_CUDA_ver2(const std::vector<unsigned char>& grid, int w, int h,
                         const std::vector<int>& px,
                         const std::vector<int>& py,
                         const std::vector<int>& cx,
                         const std::vector<int>& cy,
                         std::vector<float>* score);

void score_all_CUDA_ver3_bounds(const std::vector<unsigned char>& grid, int w,
                                int h, const std::vector<int>& px,
                                const std::vector<int>& py, int min_x,
                                int max_x, int min_y, int max_y, int step,
                                std::vector<float>* score);

void score_all_CUDA_ver4(const std::vector<unsigned char>& grid, int w, int h,
                         const std::vector<int>& px,
                         const std::vector<int>& py,
                         const std::vector<int>& cx,
                         const std::vector<int>& cy,
                         std::vector<float>* score);

void score_all_CUDA_ver5_batched(
    const std::vector<unsigned char>& grid, int w, int h,
    const std::vector<int>& flat_px, const std::vector<int>& flat_py,
    const std::vector<int>& point_offsets,
    const std::vector<int>& point_counts,
    const std::vector<int>& cand_offsets,
    const std::vector<int>& cand_counts, const std::vector<int>& cx,
    const std::vector<int>& cy, std::vector<float>* score);

void score_all_CUDA_ver6_batched(
    const std::vector<unsigned char>& grid, int w, int h,
    const std::vector<int>& flat_px, const std::vector<int>& flat_py,
    const std::vector<int>& flat_cell,
    const std::vector<int>& point_offsets,
    const std::vector<int>& point_counts,
    const std::vector<int>& cand_offsets,
    const std::vector<int>& cand_counts, const std::vector<int>& cx,
    const std::vector<int>& cy,
    const std::vector<int>& candidate_cell,
    const std::vector<unsigned char>& full_inside,
    std::vector<float>* score);

void score_all_CUDA_ver7_batched(
    const std::vector<unsigned char>& grid, int w, int h,
    const std::vector<int>& flat_px, const std::vector<int>& flat_py,
    const std::vector<int>& point_offsets,
    const std::vector<int>& point_counts,
    const std::vector<int>& cand_offsets,
    const std::vector<int>& cand_counts, const std::vector<int>& cx,
    const std::vector<int>& cy, const std::vector<int>& scan_min_x,
    const std::vector<int>& scan_max_x, const std::vector<int>& scan_min_y,
    const std::vector<int>& scan_max_y, std::vector<float>* score);

void score_all_CUDA_ver8_batched(
    const std::vector<unsigned char>& grid, int w, int h,
    const std::vector<int>& flat_px, const std::vector<int>& flat_py,
    const std::vector<int>& flat_cell,
    const std::vector<int>& point_offsets,
    const std::vector<int>& point_counts,
    const std::vector<int>& cand_offsets,
    const std::vector<int>& cand_counts, const std::vector<int>& cx,
    const std::vector<int>& cy, const std::vector<int>& scan_min_x,
    const std::vector<int>& scan_max_x, const std::vector<int>& scan_min_y,
    const std::vector<int>& scan_max_y, std::vector<float>* score);

}  // namespace cartographer_parallel

#endif  // CARTOGRAPHER_PARALLEL_CUDA_ASSIGNMENT_H_
