from cython.parallel import prange
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

    length = len(zs)
    with nogil:
      for i in prange(length, schedule="guided"):
        z = zs[i]
        c = cs[i]
        output[i] = 0
        while output[i] < max_iter and (z.real*z.real+z.imag*z.imag) < 4:
          z = z*z + c
          output[i] += 1
    return output                   # 전체 결과 반환


# cython -a cythonfn_openMP.pyx
