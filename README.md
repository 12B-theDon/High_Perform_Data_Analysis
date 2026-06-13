# High Perform Data Analysis - PA02

> **For PA02, follow this README.**

Cartographer Fast Correlative Scan Matcher를 ROS1 환경에서 빌드하고 실행하기 위한 배포용 안내이다. 세부 실행 명령과 profiling 재현 명령은 문서를 분리하였다.

## 문서 구성

- `docs/PA02_SETUP_AND_RUN.md`: Jetson Nano에서 패키지를 배치, 빌드, 실행하는 방법
- `docs/PA02_PROFILING.md`: 수정된 코드의 profiling 출력을 다시 확인하는 방법

## 빠른 실행

CUDA build 후 포함된 map과 bag으로 최종 batch CUDA 구현을 실행한다.

```bash
cd ~/catkin_ws
catkin_make -DBUILD_CUDA_TASK=ON -DBUILD_GPU_TASK=ON -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
export ROS_MASTER_URI=http://localhost:11311
export CUDA_SCORE_VERSION=ver8

roslaunch cartographer_parallel cartographer_parallel_with_bag.launch \
  ns:=student_05 \
  branch_and_bound_depth:=2
```

## profiling 출력만 확인

```bash
cd ~/catkin_ws
source devel/setup.bash
export ROS_MASTER_URI=http://localhost:11311
export CUDA_SCORE_VERSION=ver8

roslaunch cartographer_parallel cartographer_parallel_with_bag.launch \
  ns:=student_05 \
  branch_and_bound_depth:=2 \
  2>&1 \
  | grep --line-buffered -E "CUDA score dispatcher|\\[score_all CUDA|\\[Score profile|\\[MatchWithWindow profile|\\[Node profile|\\[match_time\\]"
```

## 기준 코드

PA02는 PA01 branch의 Fast Correlative Scan Matcher 및 `score_all()` 구현을 기준으로 확장하였다.

```text
https://github.com/12B-theDon/High_Perform_Data_Analysis/tree/PA01
```

## 주요 파일

- `cartographer_parallel/CMakeLists.txt`: CPU/CUDA build option 정의
- `cartographer_parallel/launch/cartographer_parallel_with_bag.launch`: map과 bag을 함께 실행하는 launch file
- `cartographer_parallel/launch/fast_correlative.launch`: matcher node 설정
- `cartographer_parallel/src/cuda_fast_matcher.cpp`: CUDA scoring dispatcher와 matcher profiling 출력
- `cartographer_parallel/src/cpu_fast_matcher.cpp`: CPU/OpenMP 비교용 matcher
- `cartographer_parallel/src/fast_correlative_node_print.cpp`: ROS node 및 callback-level profiling 출력
- `cartographer_parallel/src/cuda_score_all.cu`: 기준 CUDA scoring 구현
- `cartographer_parallel/src/cuda_score_all_ver8.cu`: 최종 batch CUDA scoring 구현
