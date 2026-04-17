import timeit


t = timeit.timeit("arr_list = [0,1,2,3,4,5,6,7,8,9]", number=10_000_000)
print(f"arr_list: {t / 10_000_000 * 1_000_000_000:.2f} ns per loop")

t = timeit.timeit("arr_tuple = (0,1,2,3,4,5,6,7,8,9)", number=10_000_000)
print(f"arr_tuple: {t / 10_000_000 * 1_000_000_000:.2f} ns per loop")

print("\nTuple creation is usually faster than list creation for fixed values.")
print("Results can vary by computer and Python version.")
