from memory_profiler import profile


@profile
def list_comprehension():
    arr1 = [i for i in range(100_000)]
    return arr1


@profile
def list_append():
    arr2 = []
    for i in range(100_000):
        arr2.append(i)
    return arr2


if __name__ == "__main__":
    list_comprehension()
    list_append()

    print("\nResults can vary by computer and Python version.")
    print("Run with: python3 -m memory_profiler week03/simple_memoryProfiler.py")
