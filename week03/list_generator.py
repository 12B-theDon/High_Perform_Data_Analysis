import timeit

from memory_profiler import profile


def fibonacci_list(num_items):
    numbers = []
    a, b = 0, 1
    while len(numbers) < num_items:
        numbers.append(a)
        a, b = b, a + b
    return numbers


def fibonacci_gen(num_items):
    a, b = 0, 1
    while num_items:
        yield a
        a, b = b, a + b
        num_items -= 1


@profile
def profile_fibonacci_list():
    return fibonacci_list(100_000)


@profile
def profile_fibonacci_gen():
    return fibonacci_gen(100_000)


if __name__ == "__main__":
    t = timeit.timeit("fibonacci_list(100_000)", globals=globals(), number=1)
    print(f"fibonacci_list(100_000): {t * 1_000:.2f} ms per loop")

    t = timeit.timeit("fibonacci_gen(100_000)", globals=globals(), number=1_000_000)
    print(f"fibonacci_gen(100_000): {t / 1_000_000 * 1_000_000_000:.2f} ns per loop")

    print("\nGenerator: lazy iterator. It produces values only when needed.")
    print("Results can vary by computer and Python version.")
    print("Run memory check with: python3 -m memory_profiler week03/list_generator.py")

    profile_fibonacci_list()
    profile_fibonacci_gen()
