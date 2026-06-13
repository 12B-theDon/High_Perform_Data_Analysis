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

## profiling 결과를 txt/log로 저장

실행 결과 전체를 저장하지 않고 profiling에 필요한 line만 `grep`으로 추려서 저장하였다. `tee`를 사용하면 terminal에서 진행 상황을 보면서 동시에 파일로 남길 수 있다.

```bash
cd ~/catkin_ws
source devel/setup.bash
export ROS_MASTER_URI=http://localhost:11311
mkdir -p cuda_logs

export CUDA_SCORE_VERSION=baseline

roslaunch cartographer_parallel cartographer_parallel_with_bag.launch \
  ns:=student_05 \
  branch_and_bound_depth:=2 \
  2>&1 \
  | grep --line-buffered -E "CUDA score dispatcher|\\[score_all CUDA|\\[Score profile|\\[MatchWithWindow profile|\\[Node profile|\\[match_time\\]|ERROR|FATAL|timeout|timed out" \
  | tee cuda_logs/PA02_FM_CUDA_depth2_baseline.txt
```

최종 batch CUDA도 같은 방식으로 `CUDA_SCORE_VERSION`과 파일명만 바꾸어 저장하였다.

```bash
export CUDA_SCORE_VERSION=ver8

roslaunch cartographer_parallel cartographer_parallel_with_bag.launch \
  ns:=student_05 \
  branch_and_bound_depth:=2 \
  2>&1 \
  | grep --line-buffered -E "CUDA score dispatcher|\\[score_all CUDA|\\[Score profile|\\[MatchWithWindow profile|\\[Node profile|\\[match_time\\]|ERROR|FATAL|timeout|timed out" \
  | tee cuda_logs/PA02_FM_CUDA_depth2_ver8.txt
```

반복 측정이나 workload별 측정에서는 파일명에 version, resolution, depth를 포함시켜 나중에 자동 집계할 수 있게 하였다.

```bash
version=ver8
resolution_tag=res0p05
depth=2

export CUDA_SCORE_VERSION=${version}

roslaunch cartographer_parallel cartographer_parallel_with_bag.launch \
  ns:=student_05 \
  branch_and_bound_depth:=${depth} \
  2>&1 \
  | grep --line-buffered -E "CUDA score dispatcher|\\[score_all CUDA|\\[Score profile|\\[MatchWithWindow profile|\\[Node profile|\\[match_time\\]|ERROR|FATAL|timeout|timed out" \
  | tee "cuda_logs/PA02_FM_CUDA_${resolution_tag}_depth${depth}_${version}.log"
```

## txt/log를 CSV로 변환

저장된 `.txt` 또는 `.log` 파일은 `plot_cuda_log_results.py`로 집계하였다. 이 스크립트는 `[score_all CUDA ...]`, `[Score profile]`, `[MatchWithWindow profile]`, `[match_time]` line에서 key-value 값을 읽어 CSV와 Markdown summary를 만든다.

```bash
cd ~/catkin_ws/src/cartographer_parallel

python3 src/plot_cuda_log_results.py \
  ~/catkin_ws/cuda_logs \
  --glob "PA02_FM_CUDA_*.log" \
  --skip-calls 5 \
  --output-dir ~/catkin_ws/cuda_report \
  --csv ~/catkin_ws/cuda_report/summary.csv \
  --report-md ~/catkin_ws/cuda_report/summary.md \
  --no-plots
```

CSV에는 version별 `match_total_mean`, `scorecoarse_mean`, `kernel_mean`, `h2d_mean`, `d2h_mean`, `score_last_mean`, `candidates_median` 등이 저장된다. `--skip-calls 5`는 CUDA context 생성과 초기 buffer 준비가 섞인 초반 호출을 집계에서 제외하기 위해 사용하였다.

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
