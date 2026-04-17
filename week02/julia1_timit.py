import timeit


if __name__ == "__main__":
    t = timeit.timeit(
        stmt="import julia1; julia1.build_Julia_set(desired_width=1000, max_iterations=300)",
        number=5,
    )

    print(f"Total time for 5 loops: {t:.6f} sec")
    print(f"Average time per loop: {t / 5:.6f} sec")
