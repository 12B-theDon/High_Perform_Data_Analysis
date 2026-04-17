import time

import time                     # 실행 시간을 측정하기 위한 time 모듈 import
from functools import wraps     # decorator에서 원래 함수 정보를 유지하기 위한 wraps import

# 실행 시간을 측정하는 decorator 함수
def timefn(fn):
  @wraps(fn)                    # decorator 적용 후에도 함수 이름/metadata 유지
  def measure_time(*args, **kwargs):  # 실제로 호출되는 wrapper 함수
    t1 = time.time()            # 함수 실행 전 현재 시간 기록
    result = fn(*args, **kwargs) # 원래 함수(fn) 실행
    t2 = time.time()            # 함수 실행 후 현재 시간 기록
    print(f"@timefn: {fn.__name__} took {t2 - t1} seconds")  # 실행 시간 출력
    return result               # 원래 함수의 결과 반환
  return measure_time           # wrapper 함수 반환


@timefn                         # timefn decorator 적용 (실행 시간 자동 측정)
def timefnVer_calculate_Julia_set(max_iter, zs, cs):
    output = [0] * len(zs)      # 결과를 저장할 리스트 초기화 (각 점의 iteration 횟수)

    for i in range(len(zs)):    # 모든 complex point(zs)에 대해 반복
      n = 0                     # 현재 point의 iteration 카운트
      z = zs[i]                 # 초기 z 값
      c = cs[i]                 # Julia set 상수 c

      # |z| < 2 이고 iteration이 max_iter보다 작은 동안 반복
      while abs(z) < 2 and n < max_iter:
        z = z * z + c           # Julia iteration: z = z^2 + c
        n += 1                  # iteration 횟수 증가

      output[i] = n             # 해당 point의 iteration 횟수 저장

    return output               # 전체 결과 반환

@timefn
def timefnVer_build_Julia_set(desired_width, max_iterations):
    x1 = -1.8                       # x축 시작값
    x2 = 1.8                        # x축 끝값
    y1 = -1.8                       # y축 시작값
    y2 = 1.8                        # y축 끝값

    c_real = 4                      # c의 실수부 (바로 아래에서 덮어써져 실제로는 사용되지 않음)
    c_real = -0.67172               # c의 실제 실수부
    c_imag = -0.42193               # c의 허수부

    x_step = (x2 - x1) / desired_width   # x축 방향 샘플 간격
    y_step = (y1 - y2) / desired_width   # y축 방향 샘플 간격 (음수)
    x = []                          # x 좌표들을 저장할 리스트
    y = []                          # y 좌표들을 저장할 리스트
    ycoord = y2                     # y는 위쪽 끝에서 시작

    # y 좌표 리스트 생성
    while ycoord > y1:
        y.append(ycoord)            # 현재 y 좌표 추가
        ycoord += y_step            # 다음 y 좌표로 이동 (감소)

    xcoord = x1                     # x는 왼쪽 끝에서 시작

    # x 좌표 리스트 생성
    while xcoord < x2:
        x.append(xcoord)            # 현재 x 좌표 추가
        xcoord += x_step            # 다음 x 좌표로 이동 (증가)

    zs = []                         # 초기 복소수 점 z들을 저장할 리스트
    cs = []                         # 각 점에 대응되는 c 값들을 저장할 리스트

    # 2차원 평면의 모든 (x, y) 조합에 대해 복소수 점 생성
    for ycoord in y:
        for xcoord in x:
            zs.append(complex(xcoord, ycoord))      # 현재 위치를 복소수로 저장
            cs.append(complex(c_real, c_imag))      # 동일한 c 값을 저장

    start_time = time.time()
    output = timefnVer_calculate_Julia_set(max_iterations, zs, cs)  # 모든 점에 대해 Julia 반복 계산
    end_time = time.time()
    secs = end_time - start_time
    print(timefnVer_build_Julia_set.__name__ + " took", secs, "seconds to execute.")

    return output                   # 계산 결과 반환

if __name__ == "__main__":
    timefnVer_build_Julia_set(desired_width=2000, max_iterations=500)	
