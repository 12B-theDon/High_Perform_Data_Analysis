from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import os
import re
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig_"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_GRID_SIZES = [1024, 2048, 4096, 8192]#, 16384]
DEFAULT_MAX_ITERATIONS = 500
MODULE_PATTERN = "julia1_*.py"
EXCLUDED_FILENAMES = {"julia1_basic copy.py"}
TIME_PATTERN = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*sec")


def discover_julia_files(base_dir: Path) -> list[Path]:
    candidates = []
    for path in sorted(base_dir.glob(MODULE_PATTERN)):
        if path.name in EXCLUDED_FILENAMES:
            continue
        if path.name == Path(__file__).name:
            continue
        candidates.append(path)
    return candidates


def import_module_from_path(module_path: Path, index: int):
    module_name = f"gridscaled_{module_path.stem}_{index}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def benchmark_module(module, grid_size: int, max_iterations: int) -> float:
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        module.build_julia_set(desired_width=grid_size, max_iterations=max_iterations)

    output_text = captured.getvalue()
    match = TIME_PATTERN.search(output_text)
    if match is not None:
        return float(match.group(1))

    raise ValueError(
        f"Could not parse timing output from module {module.__name__!r}: {output_text!r}"
    )


def run_benchmarks(
    module_paths: list[Path], grid_sizes: list[int], max_iterations: int
) -> dict[str, list[float]]:
    results: dict[str, list[float]] = {}

    for index, module_path in enumerate(module_paths):
        module = import_module_from_path(module_path, index)
        times = []
        print(f"Benchmarking {module_path.name}...")
        for grid_size in grid_sizes:
            start = time.perf_counter()
            calc_time = benchmark_module(module, grid_size, max_iterations)
            elapsed = time.perf_counter() - start
            times.append(calc_time)
            print(
                f"  {grid_size:>5} x {grid_size:<5} -> "
                f"calc {calc_time:.3f} s, wall {elapsed:.3f} s"
            )
        results[module_path.stem] = times

    return results


def plot_results(
    grid_sizes: list[int], results: dict[str, list[float]], output_path: Path
) -> None:
    plt.figure(figsize=(11, 7))

    for label, times in results.items():
        line, = plt.plot(
            grid_sizes,
            times,
            marker="o",
            linewidth=2.2,
            markersize=6,
            label=label,
        )
        color = line.get_color()
        plt.annotate(
            f"{times[0]:.3f}",
            (grid_sizes[0], times[0]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            color=color,
            fontsize=9,
            fontweight="bold",
        )
        plt.annotate(
            f"{times[-1]:.3f}",
            (grid_sizes[-1], times[-1]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            color=color,
            fontsize=9,
            fontweight="bold",
        )

    plt.title("Julia Set Calculation Time vs Grid Size")
    plt.xlabel("Grid size")
    plt.ylabel("Calculation time (s)")
    plt.xticks(grid_sizes, [f"{size // 1024}k" for size in grid_sizes])
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark week05 Julia implementations and plot grid scaling."
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=f"Maximum Julia iterations per point (default: {DEFAULT_MAX_ITERATIONS})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("julia_grid_scaling.png"),
        help="Path to save the output plot image.",
    )
    parser.add_argument(
        "--grid-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_GRID_SIZES,
        help="Grid widths to benchmark. Each run uses width x width.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    module_paths = discover_julia_files(base_dir)

    if not module_paths:
        raise FileNotFoundError(f"No files matched {MODULE_PATTERN!r} in {base_dir}")

    print("Modules:")
    for module_path in module_paths:
        print(f"  - {module_path.name}")

    print("\nGrid sizes:")
    for grid_size in args.grid_sizes:
        print(f"  - {grid_size} x {grid_size}")

    results = run_benchmarks(module_paths, args.grid_sizes, args.max_iterations)
    plot_results(args.grid_sizes, results, args.output)

    print(f"\nSaved plot to: {args.output}")


if __name__ == "__main__":
    main()
