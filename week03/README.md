# Week 03

This folder contains small experiments about Python performance, memory usage, generators, namespaces, and hash table behavior.

All results in this README are based on the `*_result.txt` files in this folder.
Measured values can vary by computer, operating system, and Python version.

## 1. List Indexing Time

File:
[list_timeit.py](/home/mhlee/Documents/High_Performance_Data_Analysis/week03/list_timeit.py)

Result:
- `l[5]`: 26.97 ns per loop
- `l[500]`: 29.98 ns per loop
- `l[500_000]`: 24.08 ns per loop

Conclusion:
- Python list indexing is `O(1)`.
- Access time is almost constant even when the list size grows a lot.

## 2. List Comprehension vs Append Memory Usage

File:
[list_memoryProfiler.py](/home/mhlee/Documents/High_Performance_Data_Analysis/week03/list_memoryProfiler.py)

Result summary from [list_memoryProfiler_result.txt](/home/mhlee/Documents/High_Performance_Data_Analysis/week03/list_memoryProfiler_result.txt):
- List comprehension increment: about 4.0 MiB
- `append()` loop increment: about 2.6 MiB on this machine

Observation:
- Both methods create the same 100,000-element list.
- The measured memory pattern can differ from other examples depending on environment and profiler behavior.

Conclusion:
- Memory profiling results are environment-dependent.
- Do not assume one exact ratio unless it is reproduced on the same machine and Python version.

## 3. Tuple Creation vs List Creation

File:
[tuple_timit.py](/home/mhlee/Documents/High_Performance_Data_Analysis/week03/tuple_timit.py)

Result summary from [tuple_timeit_result.txt](/home/mhlee/Documents/High_Performance_Data_Analysis/week03/tuple_timeit_result.txt):
- `arr_list`: 41.65 ns per loop
- `arr_tuple`: 8.50 ns per loop

Conclusion:
- Tuple literal creation is faster than list literal creation for fixed values.
- One reason is that tuples are immutable and can be handled more efficiently.

## 4. Generator vs List for Fibonacci

File:
[list_generator.py](/home/mhlee/Documents/High_Performance_Data_Analysis/week03/list_generator.py)

Result summary from [list_generator_result.txt](/home/mhlee/Documents/High_Performance_Data_Analysis/week03/list_generator_result.txt):
- `fibonacci_list(100_000)`: 289.58 ms per loop
- `fibonacci_gen(100_000)`: 158.84 ns per loop
- `fibonacci_list(100_000)` memory increment: about 441.3 MiB
- `fibonacci_gen(100_000)` memory increment: about 0.0 MiB

Important note:
- The generator timing here measures generator creation, not full consumption of all yielded values.

Conclusion:
- A generator is a lazy iterator.
- It produces values only when needed, so it can use much less memory than building a full list.

## 5. Namespace Example

File:
[namespace.py](/home/mhlee/Documents/High_Performance_Data_Analysis/week03/namespace.py)

Result summary from [namespace_result.txt](/home/mhlee/Documents/High_Performance_Data_Analysis/week03/namespace_result.txt):
- Inner scope shows `{'a': 30}`
- Outer function scope shows `{'a': 20, 'inner_func': ...}`
- Global scope shows `{'a': 10, ...}`

Conclusion:
- Python keeps separate namespaces for local, enclosing, and global scopes.
- The same variable name can exist with different values in different scopes.

## 6. Namespace Lookup Cost with `sin`

File:
[namespace_sin.py](/home/mhlee/Documents/High_Performance_Data_Analysis/week03/namespace_sin.py)

Result summary from [namespace_sin_result.txt](/home/mhlee/Documents/High_Performance_Data_Analysis/week03/namespace_sin_result.txt):
- `builtin_sin(100)`: 81.2 usec per loop
- `global_sin(100)`: 73.2 usec per loop
- `local_sin(100)`: 54.4 usec per loop

Conclusion:
- Local name lookup is faster than global or module attribute lookup.
- Binding `math.sin` to a local variable reduces repeated lookup overhead inside a loop.

## 7. Hash Table Lookup

File:
[hashTable.py](/home/mhlee/Documents/High_Performance_Data_Analysis/week03/hashTable.py)

Result summary from [hashTable_result.txt](/home/mhlee/Documents/High_Performance_Data_Analysis/week03/hashTable_result.txt):
- First lookup time: 9.267824521999955
- Second lookup time: 0.20851217900008123

Conclusion:
- Hash-table-style lookup is usually very fast.
- Actual lookup time can change a lot depending on the data structure quality, collisions, and implementation details.

## Overall Summary

- List indexing is effectively constant time.
- Tuple literals are faster to create than list literals for fixed data.
- Generators are memory-efficient because they are lazy.
- Local variable lookup can improve performance inside tight loops.
- Memory and timing results should always be interpreted in the context of the machine and Python version used for the test.
