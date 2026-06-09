#ifndef CARTOGRAPHER_PARALLEL_CPU_ASSIGNMENT_H_
#define CARTOGRAPHER_PARALLEL_CPU_ASSIGNMENT_H_

#include <vector>

namespace cartographer_parallel {

void score_all_cpu_parallel(const std::vector<unsigned char>& grid, int w,
                            int h, const std::vector<int>& px,
                            const std::vector<int>& py,
                            const std::vector<int>& cx,
                            const std::vector<int>& cy,
                            std::vector<float>* score);

}  // namespace cartographer_parallel

#endif  // CARTOGRAPHER_PARALLEL_CPU_ASSIGNMENT_H_
