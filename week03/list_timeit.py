import timeit


l = list(range(10))
t = timeit.timeit("l[5]", globals=globals(), number=10_000_000)
print(f"l[5]: {t / 10_000_000 * 1_000_000_000:.2f} ns per loop")

l = list(range(10_000_000))
t = timeit.timeit("l[500]", globals=globals(), number=10_000_000)
print(f"l[500]: {t / 10_000_000 * 1_000_000_000:.2f} ns per loop")

t = timeit.timeit("l[500_000]", globals=globals(), number=10_000_000)
print(f"l[500_000]: {t / 10_000_000 * 1_000_000_000:.2f} ns per loop")

print("\nConclusion: list indexing is O(1).")
