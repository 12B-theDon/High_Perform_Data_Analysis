import math
from math import sin
import timeit

import dis

@profile
def builtin_sin(x):
    res = 1
    for _ in range(1000):
        res += math.sin(x)
    return res

@profile
def global_sin(x):
    res = 1
    for _ in range(1000):
        res += sin(x)
    return res

@profile
def local_sin(x, sin = math.sin):
    res = 1
    for _ in range(1000):
        res += sin(x)
    return res

x = 1.0

#python3 -m timeit -n 5 -r 1 -s "import namespace_sin" "namespace_sin.builtin_sin(100)"

#python3 -m timeit -n 5 -r 1 -s "import namespace_sin" "namespace_sin.global_sin(100)"

#python3 -m timeit -n 5 -r 1 -s "import namespace_sin" "namespace_sin.local_sin(100)"

#python3 -m memory_profiler namespace_sin.py
if __name__ == "__main__":
    builtin_sin(x)
    global_sin(x)
    local_sin(x)
#    dis.dis(builtin_sin)
#    dis.dis(global_sin)
#    dis.dis(local_sin)


# kernprof -lv namespace_sin.py
# perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses,task-clock,faults,minor-faults,cs,migrations python3 namespace_sin.py