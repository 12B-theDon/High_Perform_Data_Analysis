import time
import numpy as np
import os
import argparse

def estimate_num_points_in_circle(num_samples):
    np.random.seed()
    xs = np.random.uniform(0, 1, num_samples)
    ys = np.random.uniform(0, 1, num_samples)
    quarter_unit_circle =(xs * xs + ys * ys) <= 1
    num_trials_in_quarter_unit_circle = np.sum(quarter_unit_circle)
    return num_trials_in_quarter_unit_circle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='!')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--processes', action="store_true", default=False)
    args = parser.parse_args()
    if args.processes == True:
        print("Using Processes")
        from multiprocessing import Pool
    else:
        print("Using Threads")
        from multiprocessing.dummy import Pool

    num_parallel_blocks = int(args.num_workers)
    np.random.seed()
    num_samples_in_total = (1e8)
    num_samples_per_worker = int(num_samples_in_total / num_parallel_blocks)
    print("Allocate {} samples per worker".format(num_samples_per_worker))

    #Proceeses
    pool = Pool(processes=num_parallel_blocks)
    pool_input = [num_samples_per_worker] * num_parallel_blocks

    start_time = time.time()
    num_in_circle = pool.map(estimate_num_points_in_circle, pool_input)
    print("Time: {} secs".format(time.time() - start_time))
    pool.close()

    pi_estiamte = float(sum(num_in_circle)) / num_samples_in_total * 4
    print("Estimated PI", pi_estiamte)
    print("PI", np.pi)