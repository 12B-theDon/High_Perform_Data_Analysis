# PA02 Setup and Run

PA02 코드를 Jetson Nano의 ROS1 catkin workspace에서 빌드하고 실행했던 절차를 정리하였다.

## 기준 코드

PA02는 PA01에서 작성한 Fast Correlative Scan Matcher 및 `score_all()` 구현을 기준으로 확장하였다.

```text
https://github.com/12B-theDon/High_Perform_Data_Analysis/tree/PA01
```

## 환경

- Jetson Nano
- Ubuntu 18.04
- ROS Melodic
- CUDA 10.2
- catkin workspace: `~/catkin_ws`

## 패키지 배치

저장소의 ROS package는 `cartographer_parallel/`을 사용하였다.

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src

# repository를 clone한 경우
# git clone <repository-url>
# cp -r High_Perform_Data_Analysis/cartographer_parallel .

# 최종 위치는 다음과 같아야 한다.
# ~/catkin_ws/src/cartographer_parallel
```

## CUDA 버전 빌드

```bash
cd ~/catkin_ws
catkin_make -DBUILD_CUDA_TASK=ON -DBUILD_GPU_TASK=ON -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
```

CUDA 실행 파일은 다음 위치에 생성되는 것을 기준으로 확인하였다.

```bash
~/catkin_ws/devel/lib/cartographer_parallel/fast_correlative_node
```

## CPU 버전 빌드

CPU / OpenMP 비교 실행이 필요할 때만 사용하였다.

```bash
cd ~/catkin_ws
catkin_make -DBUILD_CUDA_TASK=OFF -DBUILD_GPU_TASK=OFF -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
export ROS_MASTER_URI=http://localhost:11311
```

CPU 전용 실행 파일은 다음 위치에 생성되는 것을 기준으로 확인하였다.

```bash
~/catkin_ws/devel/lib/cartographer_parallel/cpu_fast_correlative_node
```

## 기본 실행

포함된 map과 bag을 사용하여 Fast Correlative Scan Matcher를 실행하였다.

```bash
cd ~/catkin_ws
source devel/setup.bash

roslaunch cartographer_parallel cartographer_parallel_with_bag.launch \
  ns:=student_05 \
  branch_and_bound_depth:=2
```

기본 입력은 다음과 같이 사용하였다.

- map: `$(find cartographer_parallel)/maps/0501.yaml`
- bag: `$(find cartographer_parallel)/bags/scan.bag`
- node: `fast_correlative_node`
- initial pose: `x=-2.0`, `y=6.82`, `yaw=-3.0255282583321743`

## 기준 CUDA 실행

```bash
cd ~/catkin_ws
source devel/setup.bash
export CUDA_SCORE_VERSION=baseline

roslaunch cartographer_parallel cartographer_parallel_with_bag.launch \
  ns:=student_05 \
  branch_and_bound_depth:=2
```

## 최종 batch CUDA 실행

```bash
cd ~/catkin_ws
source devel/setup.bash
export CUDA_SCORE_VERSION=ver6

roslaunch cartographer_parallel cartographer_parallel_with_bag.launch \
  ns:=student_05 \
  branch_and_bound_depth:=2
```

## CUDA version mapping

`CUDA_SCORE_VERSION`은 실행할 CUDA scoring 구현을 선택하기 위해 사용한 환경 변수이다. 보고서와 실행 문서에서는 내부 version 이름 대신 다음 구현명으로 해석하였다.

| selector | 구현명 | 의미 |
|---|---|---|
| `baseline`, `default`, `ver0` | 기준 CUDA | scan마다 개별 CUDA 호출, 매 호출마다 device allocation/copy/free 수행 |
| `ver1` | shmem만 | warp reduction과 shared memory tiling을 적용한 비 batch kernel |
| `ver2` | buffer 재사용 | 기준 kernel 구조를 유지하되 device buffer와 grid upload를 재사용 |
| `ver3` | bounds만 | 정규 candidate grid를 bounds-indexed 방식으로 계산하는 비 batch 구현 |
| `ver4` | kernel 변경 | 비 batch 호출 구조에서 kernel 내부 reduction과 memory read 방식을 변경한 구현 |
| `ver5` | scan batch | 여러 scan의 point/candidate를 flat array로 묶어 batch kernel로 처리 |
| `ver6` | 최종 batch CUDA / bounds 감소 | scan batch에 `flat_cell`, `candidate_cell`, `full_inside` 기반 bounds check 감소 적용 |
| `ver7` | scan batch 개선 | scan별 min/max bounds로 full-inside candidate를 판정하는 batch 구현 |
| `ver8` | 추가 sorted-offset 실험 | sorted point offset과 `flat_cell`을 사용한 추가 batch 구현 |

6개의 실험 구현은 `kernel 변경`, `buffer 재사용`, `shmem만`, `bounds만`, `scan batch`, `bounds 감소`로 묶어 비교하였다. `ver5`와 `ver7`은 scan batch 계열, `ver6`과 `ver8`은 bounds 감소 계열로 정리하였다. 최종 결과는 기본 실행값이기도 한 `ver6`을 사용하였다.

중간 구현은 `CUDA_SCORE_VERSION`만 바꾸어 실행하였다.

```bash
export CUDA_SCORE_VERSION=ver1   # shmem만
export CUDA_SCORE_VERSION=ver2   # buffer 재사용
export CUDA_SCORE_VERSION=ver3   # bounds만
export CUDA_SCORE_VERSION=ver4   # kernel 변경
export CUDA_SCORE_VERSION=ver5   # scan batch
export CUDA_SCORE_VERSION=ver6   # 최종 batch CUDA / bounds 감소
```

## workload 변경 실행

workload는 map resolution과 branch-and-bound depth에 의해 달라지도록 설정하였다. 실행 전 map yaml의 `resolution` 값을 바꾼 뒤 같은 launch를 실행하였다.

```bash
MAP_YAML="$(rospack find cartographer_parallel)/maps/0501.yaml"
cp "$MAP_YAML" "${MAP_YAML}.bak"

sed -i -E "s/^resolution:.*/resolution: 0.05/" "$MAP_YAML"

export CUDA_SCORE_VERSION=ver6
roslaunch cartographer_parallel cartographer_parallel_with_bag.launch \
  ns:=student_05 \
  branch_and_bound_depth:=2

cp "${MAP_YAML}.bak" "$MAP_YAML"
rm -f "${MAP_YAML}.bak"
```
