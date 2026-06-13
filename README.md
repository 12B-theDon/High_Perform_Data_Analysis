# High Perform Data Analysis - PA02

> **For PA02, follow this README.**

Cartographer Fast Correlative Scan Matcher를 ROS1 환경에서 빌드하고 실행하기 위한 배포용 안내이다. 세부 실행 명령과 profiling 재현 명령은 문서를 분리하였다.

## 문서 구성

- `docs/PA02_SETUP_AND_RUN.md`: Jetson Nano에서 패키지를 배치, 빌드, 실행하는 방법
- `docs/PA02_PROFILING.md`: profiling line을 `grep`/`tee`로 저장하고 CSV로 집계하는 방법

## CUDA version mapping

`CUDA_SCORE_VERSION`은 실행할 CUDA scoring 구현을 선택하는 환경 변수이다. 보고서와 실행 문서에서는 내부 version 이름 대신 다음 구현명으로 해석한다.

| selector | 구현명 | 의미 |
|---|---|---|
| `baseline`, `default`, `ver0` | 기준 CUDA | scan마다 개별 CUDA 호출, 매 호출마다 device allocation/copy/free 수행 |
| `ver1` | shmem만 | warp reduction과 shared memory tiling을 적용한 비 batch kernel |
| `ver2` | buffer 재사용 | 기준 kernel 구조를 유지하되 device buffer와 grid upload를 재사용 |
| `ver3` | bounds만 | 정규 candidate grid를 bounds-indexed 방식으로 계산하는 비 batch 구현 |
| `ver4` | kernel 변경 | 비 batch 호출 구조에서 kernel 내부 reduction과 memory read 방식을 변경한 구현 |
| `ver5` | scan batch | 여러 scan의 point/candidate를 flat array로 묶어 batch kernel로 처리 |
| `ver6` | bounds 감소 | scan batch에 `flat_cell`, `candidate_cell`, `full_inside` 기반 bounds check 감소 적용 |
| `ver7` | scan batch 개선 | scan별 min/max bounds로 full-inside candidate를 판정하는 batch 구현 |
| `ver8` | 최종 batch CUDA | sorted point offset과 `flat_cell`을 사용한 최종 bounds 감소 batch 구현 |

6개의 실험 구현은 `kernel 변경`, `buffer 재사용`, `shmem만`, `bounds만`, `scan batch`, `bounds 감소`로 묶어 비교한다. `ver5`와 `ver7`은 scan batch 계열, `ver6`과 `ver8`은 bounds 감소 계열이며, 최종 결과는 `ver8`을 사용한다.

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
