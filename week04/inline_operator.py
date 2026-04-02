import numpy as np

def inplace_add():
    array1 = np.random.random((10, 10))
    array2 = np.random.random((10, 10))
    array1 += array2
    return array1

def normal_add():
    array1 = np.random.random((10, 10))
    array2 = np.random.random((10, 10))
    array1 = array1 + array2
    return array1


# python3 -m timeit -n 10000 -r 7 -s "from basic_functions import inplace_add" "inplace_add()"

# python3 -m timeit -n 10000 -r 7 -s "from basic_functions import normal_add" "normal_add()"