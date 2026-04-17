import argparse
import array
import gc

from memory_profiler import memory_usage


parser = argparse.ArgumentParser(description="Reproduce memory_profiler allocation examples.")
parser.add_argument(
    "--size",
    type=int,
    default=int(1e8),
    help="Number of elements to allocate in each example.",
)
args = parser.parse_args()
size = args.size

shared_globals = {"array": array, "size": size}

label = f"[0] * {size}"
expression = "[0] * size"
baseline = memory_usage(-1, interval=0.1, timeout=1)[0]
peak, result = memory_usage((eval, (expression, shared_globals, {}), {}), interval=0.1, max_usage=True, retval=True)
increment = peak - baseline
print(label)
print(f">> peak memory: {peak:.2f} MiB, increment: {increment:.2f} MiB")
print(">> meaning:")
print("   This creates a list with many references to the same integer object `0`.")
print("   The list needs memory for the element slots, but it does not create separate integer objects for every element.")
print("   That is why the increment is much smaller than the list-comprehension case.")
del result
gc.collect()
print()

label = f"[n for n in range({size})]"
expression = "[n for n in range(size)]"
baseline = memory_usage(-1, interval=0.1, timeout=1)[0]
peak, result = memory_usage((eval, (expression, shared_globals, {}), {}), interval=0.1, max_usage=True, retval=True)
increment = peak - baseline
print(label)
print(f">> peak memory: {peak:.2f} MiB, increment: {increment:.2f} MiB")
print(">> meaning:")
print("   This creates a list and also creates a separate Python integer object for each element.")
print("   So memory is used by both the list slots and the integer objects themselves.")
print("   That is why this case uses the most memory.")
del result
gc.collect()
print()

label = f"array.array('l', range({size}))"
expression = "array.array('l', range(size))"
baseline = memory_usage(-1, interval=0.1, timeout=1)[0]
peak, result = memory_usage((eval, (expression, shared_globals, {}), {}), interval=0.1, max_usage=True, retval=True)
increment = peak - baseline
theoretical = (len(result) * result.itemsize) / (1024 ** 2)
print(label)
print(f">> peak memory: {peak:.2f} MiB, increment: {increment:.2f} MiB")
print(f">> itemsize: {result.itemsize}")
print(">> meaning:")
print("   This stores raw machine integers in a compact contiguous buffer.")
print(f"   Each element uses {result.itemsize} bytes, so the theoretical data size is about {theoretical:.2f} MiB.")
print(f"   The measured increment ({increment:.2f} MiB) is close to that value, so the memory mostly goes to the data itself.")
del result
gc.collect()
print()
