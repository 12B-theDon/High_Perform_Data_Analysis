import timeit
from array import array

import numpy

def norm_square_list(vector):
    norm=0
    for v in vector:
        norm += v * v
    return norm

def norm_square_list_comprehension(vector):
    return sum([v * v for v in vector])

def norm_square_array(vector):
    norm=0
    for v in vector:
        norm += v * v
    return norm

def norm_square_numpy(vector):
    return numpy.sum(vector * vector)

def norm_square_numpy_dot(vector):
    return numpy.dot(vector, vector)



vector = list(range(1_000_000))
vector_array = array('l', range(1_000_000))
vector_np = numpy.arange(1_000_000)

# python3 -m timeit -n 10 -r 5 "from basic_functions import norm_square_list; v=list(range(1000000))" "norm_square_list(v)"

# python3 -m timeit -n 10 -r 5 "from basic_functions import norm_square_list_comprehension; v=list(range(1000000))" "norm_square_list_comprehension(v)"

# python3 -m timeit -n 10 -r 5 "from basic_functions import norm_square_array; from array import array; v=array('l', range(1000000))" "norm_square_array(v)"

# python3 -m timeit -n 10 -r 5 "import numpy as np; from basic_functions import norm_square_numpy; v=np.arange(1000000)" "norm_square_numpy(v)"

# python3 -m timeit -n 10 -r 5 "import numpy as np; from basic_functions import norm_square_numpy_dot; v=np.arange(1000000)" "norm_square_numpy_dot(v)"
