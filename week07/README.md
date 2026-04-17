# Week 07: Multiprocessing and Shared Memory

This folder contains small experiments for comparing threads vs. processes, building a multiprocessing pipeline, reproducing race conditions, and sharing NumPy arrays across worker processes.

## Files

- `pi_estimator.py`: single-process Monte Carlo estimation of pi with `1e8` random samples.
- `multi_thread_process.py`: compares thread-based and process-based parallel execution for the same Monte Carlo pi workload.
- `single_pipeline.py`: single-process prime search over `100000000..100999999`.
- `multi_processing.py`: queue-based multiprocessing version of the prime search pipeline.
- `racing_condition.py`: demonstrates a shared counter race condition.
- `racing_condition_lock1.py`: fixes the race condition using `with lock:`.
- `racing_condition_lock2.py`: fixes the race condition using explicit `lock.acquire()` / `lock.release()`.
- `racing_condition_lock3.py`: fixes the race condition with `RawValue` plus an explicit lock.
- `multi_processing_sharedNumpy.py`: demonstrates sharing one large NumPy array across multiple processes.

## Thread vs. Process Pi Estimation

Run with threads:

```bash
python3 multi_thread_process.py --num_workers 4
```

Run with processes:

```bash
python3 multi_thread_process.py --num_workers 4 --processes
```

### Measured Results

All runs used `100,000,000` total samples.

| Mode | Workers | Samples per worker | Time (s) | Estimated pi |
| --- | ---: | ---: | ---: | ---: |
| Processes | 1 | 100,000,000 | 1.6802 | 3.14156224 |
| Processes | 2 | 50,000,000 | 1.0466 | 3.14163384 |
| Processes | 4 | 25,000,000 | 0.8402 | 3.14147816 |
| Processes | 8 | 12,500,000 | 0.8261 | 3.14165896 |
| Threads | 1 | 100,000,000 | 1.6846 | 3.14156868 |
| Threads | 2 | 50,000,000 | 1.4642 | 3.14191872 |
| Threads | 4 | 25,000,000 | 1.3409 | 3.14178488 |
| Threads | 8 | 12,500,000 | 1.2986 | 3.14168660 |

Reference value:

```text
PI = 3.141592653589793
```

### Observation

- Processes scaled much better than threads for this CPU-bound NumPy Monte Carlo workload.
- The best measured time was with `8` processes: about `0.8261 s`.
- Threads improved a little with more workers, but not as much as processes.

## Prime Search Pipeline

Single-process baseline:

```bash
python3 single_pipeline.py
```

Output:

```text
Took: 10.559819221496582
54208 [100000007, 100000037, 100000039, 100000049, 100000073, 100000081, 100000123, 100000127, 100000193, 100000213] [100999889, 100999897, 100999901, 100999903, 100999919, 100999939, 100999949, 100999979, 100999981, 100999993]
```

Multiprocessing version:

```bash
python3 multi_processing.py
```

Output with `4` workers:

```text
Namespace(num_workers=4)
WORKER 0 HAS JUST FINISHED.
WORKER 1 HAS JUST FINISHED.
WORKER 2 HAS JUST FINISHED.
WORKER 3 HAS JUST FINISHED.
Took: 45.41461682319641
54208 [100000007, 100000037, 100000039, 100000049, 100000073, 100000081, 100000123, 100000127, 100000193, 100000213] [100999889, 100999897, 100999901, 100999903, 100999919, 100999939, 100999949, 100999979, 100999981, 100999993]
```

### Observation

- Both programs found the same `54,208` primes in the tested range.
- The multiprocessing pipeline was much slower here (`45.41 s` vs. `10.56 s`).
- This slowdown is expected because the example uses manager queues and also adds an artificial `sleep`, so the process overhead dominates the work.

## Race Condition Example

Run the broken version:

```bash
python3 racing_condition.py
```

Observed result:

```text
Expected count 400000
Real count 154617
```

This shows a classic race condition: multiple processes update the same shared counter without synchronization, so increments are lost.

Run the fixed versions:

```bash
python3 racing_condition_lock1.py
python3 racing_condition_lock2.py
python3 racing_condition_lock3.py
```

Observed results:

```text
Expected count 400000
Real count 400000
```

## Shared NumPy Array Example

Run:

```bash
python3 multi_processing_sharedNumpy.py
```

This program:

- creates a shared NumPy array of shape `10000 x 40000`
- allocates about `3,200,000,000` bytes
- uses a process pool with `4` workers
- writes worker process IDs into selected rows of the shared array

Observed messages included:

```text
Created shared array with 3,200,000,000 nbytes
```

and worker logs such as:

```text
do_something_worker: with idx 0
id of local_nparray_process is ... in PID 13301
```

### Observation

- The shared array is visible from all worker processes.
- Workers write directly into shared memory instead of making private copies.
- The example is interactive and waits for keyboard input before preparing memory, starting workers, and exiting.

## Summary

- Use processes for CPU-bound parallel work when the overhead is justified.
- Threads did not scale as well as processes in the pi experiment.
- Multiprocessing can become slower than a serial version if queueing and coordination overhead are too large.
- Shared writable state needs synchronization.
- Large NumPy arrays can be shared between processes to avoid copying.
