import cProfile
import pstats


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.run("import julia1; julia1.build_Julia_set(desired_width=1000, max_iterations=300)")
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats()
