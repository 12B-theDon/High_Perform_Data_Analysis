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