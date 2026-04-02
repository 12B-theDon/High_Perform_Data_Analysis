import numpy as np
cimport numpy as np

# ???????????????/
# Initialize NumPy C-API for this extension module.
#<void>np._import_array
# ???????????????/

def calculate_julia_set(int max_iter, double complex[:] zs, double complex[:] cs):
    cdef unsigned int i,n
    cdef double complex z, c
    cdef int[:] output = np.empty(len(zs), dtype=np.int32)

    for i in range(len(zs)):        # 모든 복소수 점에 대해 반복
      n = 0                         # 현재 점의 반복 횟수 초기화
      z = zs[i]                     # 현재 점의 초기 복소수 값
      c = cs[i]                     # 현재 점에 대응되는 상수 c 값

      # |z|가 2보다 작고, 반복 횟수가 최대치보다 작은 동안 반복
      while n < max_iter and abs(z) < 2 :
        z = z * z + c               # Julia 식: z = z^2 + c
        n += 1                      # 반복 횟수 1 증가

      output[i] = n                 # 해당 점의 최종 반복 횟수 저장

    return output                   # 전체 결과 반환


# cython -a cythonfn_numpy.pyx
