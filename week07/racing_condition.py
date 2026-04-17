import multiprocessing
import os
import time

def work(accum_count, max_count):
    for n in range(max_count):
        accum_count.value += 1

NUM_PRROCESSES = 4
MAX_CONT_PER_PROCESS = 100_000

accumulated_count = 0
total_expected_count = NUM_PRROCESSES * MAX_CONT_PER_PROCESS
processes = []
accum_count = multiprocessing.Value('i', 0)
for i in range(NUM_PRROCESSES):
    p = multiprocessing.Process(target=work, args=(accum_count, MAX_CONT_PER_PROCESS, ))
    p.start()
    processes.append(p)

for p in processes:
    p.join()

print("Expected count {}".format(total_expected_count))
print("Real count {}".format(accum_count.value))
