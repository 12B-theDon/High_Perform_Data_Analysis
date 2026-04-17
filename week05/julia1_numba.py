import time
import numpy as np
from numba import jit

x1 = -1.8
x2 = 1.8
y1 = -1.8
y2 = 1.8

c_real = -0.8
c_imag = 0.156


@jit(nopython=True)
def calculate_julia_set(max_iter, zs, cs, output):
    for i in range(len(zs)):        # 모든 복소수 점에 대해 반복
        n = 0                         # 현재 점의 반복 횟수 초기화
        z = zs[i]                     # 현재 점의 초기 복소수 값
        c = cs[i]                     # 현재 점에 대응되는 상수 c 값

        while n < max_iter and (z.real*z.real + z.imag*z.imag) < 4 :
            z = z * z + c               # Julia 식: z = z^2 + c
            n += 1                      # 반복 횟수 1 증가

        output[i] = n                 # 해당 점의 최종 반복 횟수 저장

    return output                   # 전체 결과 반환


def build_julia_set(desired_width, max_iterations):
    x_step = (x2 - x1) / desired_width
    y_step = (y1 - y2) / desired_width
    x = []
    y = []
    ycoord = y2

    while ycoord > y1:
        y.append(ycoord)
        ycoord += y_step

    xcoord = x1
    while xcoord < x2:
        x.append(xcoord)
        xcoord += x_step

    zs = []
    cs = []

    for ycoord in y:
        for xcoord in x:
            zs.append(complex(xcoord, ycoord))
            cs.append(complex(c_real, c_imag))

    output = [0]*len(zs)
    output = np.asarray(output)
    zs = np.asarray(zs)
    cs = np.asarray(cs)
    start_time = time.time()
    output = calculate_julia_set(max_iterations, zs, cs, output)
    print(time.time() - start_time, " sec")
    return output


if __name__ == "__main__":
    build_julia_set(desired_width=2000, max_iterations=500)

# for basic
# 4.441613674163818  sec
