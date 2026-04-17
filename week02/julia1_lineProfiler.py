import time

@profile
def calculate_Julia_set(max_iter, zs, cs):
    output = [0] * len(zs)          # 각 점의 반복 횟수를 저장할 결과 리스트 생성

    for i in range(len(zs)):        # 모든 복소수 점에 대해 반복
      n = 0                         # 현재 점의 반복 횟수 초기화
      z = zs[i]                     # 현재 점의 초기 복소수 값
      c = cs[i]                     # 현재 점에 대응되는 상수 c 값

      # |z|가 2보다 작고, 반복 횟수가 최대치보다 작은 동안 반복
      while abs(z) < 2 and n < max_iter:
        z = z * z + c               # Julia 식: z = z^2 + c
        n += 1                      # 반복 횟수 1 증가

      output[i] = n                 # 해당 점의 최종 반복 횟수 저장

    return output                   # 전체 결과 반환


#---------------------------------------------------#

# Julia set을 만들기 위한 복소수 좌표들을 생성하고 계산하는 함수
@profile
def build_Julia_set(desired_width, max_iterations):
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
    output = calculate_Julia_set(max_iterations, zs, cs)  # 모든 점에 대해 Julia 반복 계산
    end_time = time.time()
    secs = end_time - start_time
    print(build_Julia_set.__name__ + " took", secs, "seconds to execute.")
	
    return output                   # 계산 결과 반환


# 이 파일을 직접 실행했을 때만 아래 코드 실행
if __name__ == "__main__":
    build_Julia_set(desired_width=2000, max_iterations=500)

# kernprof -lv julia1_lineProfiler.py