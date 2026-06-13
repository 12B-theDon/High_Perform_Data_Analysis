# PA02 Setup and Run

PA02 코드를 Jetson Nano의 ROS1 catkin workspace에서 빌드하고 실행하는 절차이다.

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

저장소의 ROS package는 `cartographer_parallel/`이다.

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
export ROS_MASTER_URI=http://localhost:11311
```

CUDA 실행 파일은 보통 다음 위치에 생성된다.

```bash
~/catkin_ws/devel/lib/cartographer_parallel/fast_correlative_node
```

## CPU 버전 빌드

CPU / OpenMP 비교 실행이 필요할 때만 사용한다.

```bash
cd ~/catkin_ws
catkin_make -DBUILD_CUDA_TASK=OFF -DBUILD_GPU_TASK=OFF -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
export ROS_MASTER_URI=http://localhost:11311
```

CPU 전용 실행 파일은 다음 위치에 생성된다.

```bash
~/catkin_ws/devel/lib/cartographer_parallel/cpu_fast_correlative_node
```

## 기본 실행

포함된 map과 bag을 사용하여 Fast Correlative Scan Matcher를 실행한다.

```bash
cd ~/catkin_ws
source devel/setup.bash
export ROS_MASTER_URI=http://localhost:11311

roslaunch cartographer_parallel cartographer_parallel_with_bag.launch \
  ns:=student_05 \
  branch_and_bound_depth:=2
```

기본 입력은 다음과 같다.

- map: `$(find cartographer_parallel)/maps/0501.yaml`
- bag: `$(find cartographer_parallel)/bags/scan.bag`
- node: `fast_correlative_node`
- initial pose: `x=-2.0`, `y=6.82`, `yaw=-3.0255282583321743`

## 기준 CUDA 실행

```bash
cd ~/catkin_ws
source devel/setup.bash
export ROS_MASTER_URI=http://localhost:11311
export CUDA_SCORE_VERSION=baseline

roslaunch cartographer_parallel cartographer_parallel_with_bag.launch \
  ns:=student_05 \
  branch_and_bound_depth:=2
```

## 최종 batch CUDA 실행

```bash
cd ~/catkin_ws
source devel/setup.bash
export ROS_MASTER_URI=http://localhost:11311
export CUDA_SCORE_VERSION=ver8

roslaunch cartographer_parallel cartographer_parallel_with_bag.launch \
  ns:=student_05 \
  branch_and_bound_depth:=2
```

## 중간 CUDA 구현 선택

중간 구현을 실행하려면 `CUDA_SCORE_VERSION`만 바꾼다.

```bash
export CUDA_SCORE_VERSION=ver1   # kernel 변경
export CUDA_SCORE_VERSION=ver2   # buffer 재사용
export CUDA_SCORE_VERSION=ver5   # scan batch 계열
export CUDA_SCORE_VERSION=ver8   # 최종 batch CUDA
```

## workload 변경 실행

workload는 map resolution과 branch-and-bound depth에 의해 달라진다. 실행 전에 map yaml의 `resolution` 값을 바꾼 뒤 같은 launch를 실행한다.

```bash
MAP_YAML="$(rospack find cartographer_parallel)/maps/0501.yaml"
cp "$MAP_YAML" "${MAP_YAML}.bak"

sed -i -E "s/^resolution:.*/resolution: 0.05/" "$MAP_YAML"

export CUDA_SCORE_VERSION=ver8
roslaunch cartographer_parallel cartographer_parallel_with_bag.launch \
  ns:=student_05 \
  branch_and_bound_depth:=2

cp "${MAP_YAML}.bak" "$MAP_YAML"
rm -f "${MAP_YAML}.bak"
```
