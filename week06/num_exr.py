#import ipython_memory_usage .ipython_memory_usage as imu
#import numpy as np
import sys

print(sys.getsizeof(0))
print(sys.getsizeof(1))
print(sys.getsizeof(2**30 -1))
print(sys.getsizeof((2**30)))