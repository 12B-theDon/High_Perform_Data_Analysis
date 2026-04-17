import os
import multiprocessing
from collections import Counter
import ctypes
import numpy as np
from prettytable import PrettyTable

SIZE_ROW, SIZE_COL = 10_000, 40_000 # 3.2e9 bytes

def do_something_worker(idx):
    """Do some work on the shared np array on a row idx"""
    if idx % 1000 == 0:
        print(" {}: with idx {} \n id of local_nparray_process is {} in PID {} "\
              .format(do_something_worker.__name__, idx, id(main_nparray), os.getpid()))
        main_nparray[idx, :] = os.getpid()


if __name__ == '__main__':
    DEFAULT_VALUE = 1
    NBR_OF_PROCESSES = 4

    input("Press a key to prepare shared memory...")

    NUM_ITEMS = SIZE_ROW * SIZE_COL
    shared_array_base = multiprocessing.Array(ctypes.c_double, NUM_ITEMS, lock = False)
    main_nparray = np.frombuffer(shared_array_base, dtype=ctypes.c_double)
    main_nparray = main_nparray.reshape(SIZE_ROW, SIZE_COL)

    assert main_nparray.base.base is shared_array_base
    print("Created shared array with {:,} nbytes".format(main_nparray.nbytes))
    print("Shared array id is {} in PID".format(id(main_nparray), os.getpid()))
    print("Starting with an array of 0 values: ")
    print(main_nparray)
    print()

    input("Press a key to start workers using multiprocessing")
    print()

    pool = multiprocessing.Pool(processes=NBR_OF_PROCESSES)
    pool.map(do_something_worker, range(SIZE_ROW))

    print()
    print("The pid value has been overwritten: ")
    print(main_nparray)
    print()

    input("Press a key to exit...")

    counter = Counter(main_nparray.flat)
    tbl = PrettyTable(["PID", "Count"])
    for pid, count in list(counter.items()):
        tbl.add_row([pid, count])
    print(tbl)
