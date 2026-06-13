# PA02 Profiling Reproduction

수정된 코드를 실행했을 때 terminal에서 profiling 결과를 확인하는 방법이다. 결과 해석이 아니라 재현 절차만 정리한다.

## 코드 내 profiling 출력

node를 실행하면 다음 형태의 profiling line이 출력된다.

```text
[CUDA score dispatcher] CUDA_SCORE_VERSION=...
[score_all CUDA ...] calls=... avg=... ms last=... ms ...
[Score profile] calls=... total=... ms scans=... input_candidates=...
[MatchWithWindow profile] calls=... total=... ms MakeScans=... ms ...
[Node profile ...] calls=... total_avg=... ms total_last=... ms ...
[match_time] ... ms
```

출력 의미는 다음과 같다.

- `[score_all CUDA ...]`: CUDA score 호출 단위 시간
- `[Score profile]`: `FastMatcher::Score()` 전체 시간과 candidate 수
- `[MatchWithWindow profile]`: matcher 내부 함수별 시간
- `[Node profile ...]`, `[match_time]`: ROS callback 기준 전체 matching 시간

## profiling line만 화면에 표시

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

## 기준 CUDA와 최종 batch CUDA 비교 실행

```bash
cd ~/catkin_ws
source devel/setup.bash
export ROS_MASTER_URI=http://localhost:11311

for version in baseline ver8; do
  echo "===== CUDA_SCORE_VERSION=${version} ====="
  export CUDA_SCORE_VERSION=${version}

  roslaunch cartographer_parallel cartographer_parallel_with_bag.launch \
    ns:=student_05 \
    branch_and_bound_depth:=2 \
    2>&1 \
    | grep --line-buffered -E "CUDA score dispatcher|\\[score_all CUDA|\\[Score profile|\\[MatchWithWindow profile|\\[Node profile|\\[match_time\\]"

  sleep 3
done
```

## CUDA runtime/API profiling

CUDA kernel/API 호출을 확인할 때 `nvprof`로 동일 launch를 감싼다.

### 기준 CUDA

```bash
cd ~/catkin_ws
source devel/setup.bash
export ROS_MASTER_URI=http://localhost:11311
export CUDA_SCORE_VERSION=baseline

nvprof \
  --profile-child-processes \
  --print-gpu-trace \
  --print-api-trace \
  --unified-memory-profiling off \
  roslaunch cartographer_parallel cartographer_parallel_with_bag.launch \
    ns:=student_05 \
    branch_and_bound_depth:=2
```

### 최종 batch CUDA

```bash
cd ~/catkin_ws
source devel/setup.bash
export ROS_MASTER_URI=http://localhost:11311
export CUDA_SCORE_VERSION=ver8

nvprof \
  --profile-child-processes \
  --print-gpu-trace \
  --print-api-trace \
  --unified-memory-profiling off \
  roslaunch cartographer_parallel cartographer_parallel_with_bag.launch \
    ns:=student_05 \
    branch_and_bound_depth:=2
```

## CUDA kernel metric profiling

kernel metric이 필요하면 다음처럼 실행한다.

```bash
cd ~/catkin_ws
source devel/setup.bash
export ROS_MASTER_URI=http://localhost:11311
export CUDA_SCORE_VERSION=baseline

nvprof \
  --profile-child-processes \
  --metrics achieved_occupancy,sm_efficiency,gld_efficiency,gst_efficiency,dram_read_throughput,dram_write_throughput,warp_execution_efficiency,branch_efficiency,ipc \
  --unified-memory-profiling off \
  roslaunch cartographer_parallel cartographer_parallel_with_bag.launch \
    ns:=student_05 \
    branch_and_bound_depth:=2
```
