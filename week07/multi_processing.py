import math
import random
import time
import multiprocessing
from multiprocessing import Pool
import argparse

FLAG_LAST_DATA = b"FLAG_LAST_DATA"
FLAG_WORKER_FIN = b"FLAG_WORKER_DONE"

def check_prime(candidate_queue, definite_queue):
    while True:
        n = candidate_queue.get()
        time.sleep(random.uniform(0.01, 0.1))
        if n == FLAG_LAST_DATA:
            definite_queue.put(FLAG_WORKER_FIN)
            break
        else:
            if n % 2 == 0:
                continue
            for i in range(3, int(math.sqrt(n) + 1), 2):
                if n % i == 0:
                    break
            else:
                definite_queue.put(n)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    print(args)

    manager = multiprocessing.Manager()
    candidate_queue = manager.Queue()
    definite_queue = manager.Queue()

    primes = []
    for _ in range(args.num_workers):
        p = multiprocessing.Process(target=check_prime,args=(candidate_queue, definite_queue))
        p.start()
    
    t1 = time.time()
    number_range = range(100000000, 101000000)
    
    for candidate_prime in number_range:
        candidate_queue.put(candidate_prime)
    
    for n in range(args.num_workers):
        candidate_queue.put(FLAG_LAST_DATA)
    
    cnt_fin_workers = 0

    while True:
        new_prime = definite_queue.get()
        if new_prime == FLAG_WORKER_FIN:
            print("WORKER {} HAS JUST FINISHED.".format(cnt_fin_workers))
            cnt_fin_workers += 1
            if cnt_fin_workers == args.num_workers:
                break
        else:
            primes.append(new_prime)
    assert cnt_fin_workers == args.num_workers

    print("Took: ", time.time() - t1)
    print(len(primes), primes[:10], primes[-10:])
