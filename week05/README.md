# Week 05 Julia Set Performance Comparison

## Overview

This directory compares several Julia set implementations and measures how the execution time changes as the grid size increases.

The benchmarked implementations are listed below in this order:

1. `no comment`
2. `basic`
3. `pyximport`
4. `type_casting`
5. `equation`
6. `numpy`
7. `openMP`

## Single Run Results at Default Size

The following results were measured from running each script once at its default configuration.

| Implementation | Script | Time (s) |
|---|---|---:|
| `no comment` | `julia1_nocomment.py` | 8.303 |
| `basic` | `julia1_basic.py` | 4.517 |
| `pyximport` | `julia1_pyximport.py` | 4.395 |
| `type_casting` | `julia1_typecasting.py` | 1.221 |
| `equation` | `julia1_typecasting_equation.py` | 0.730 |
| `numpy` | `julia1_numpy.py` | 0.557 |
| `openMP` | `julia1_openMP.py` | 0.085 |

## Grid-Scaling Benchmark

The scaling benchmark was executed with [`drawGridScaled.py`](./drawGridScaled.py).

Attempting to include `16384 \times 16384` caused the process to be terminated because of memory pressure:

```text
Killed
-> Out of Memory Detected, so it has been killed.
```

Therefore, the comparison below uses grid sizes from `1024 \times 1024` to `8192 \times 8192`.

### Calculation Time

This table uses `calc time`, which represents only the Julia set computation time.

| Grid Size | no comment | basic | pyximport | type_casting | equation | numpy | openMP |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1024 x 1024` | 3.404 | 2.060 | 1.855 | 0.555 | 0.310 | 0.223 | 0.070 |
| `2048 x 2048` | 13.067 | 7.865 | 6.607 | 2.239 | 1.330 | 0.854 | 0.158 |
| `4096 x 4096` | 58.225 | 27.970 | 31.488 | 8.752 | 5.388 | 3.164 | 0.491 |
| `8192 x 8192` | 222.280 | 122.926 | 114.941 | 30.128 | 17.842 | 15.780 | 1.682 |

### Wall Time

This table uses `wall time`, which represents the full elapsed runtime.

| Grid Size | no comment | basic | pyximport | type_casting | equation | numpy | openMP |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1024 x 1024` | 3.814 | 2.420 | 2.167 | 0.943 | 0.685 | 0.656 | 0.548 |
| `2048 x 2048` | 14.268 | 9.279 | 7.686 | 3.644 | 2.732 | 2.470 | 1.967 |
| `4096 x 4096` | 62.635 | 32.994 | 36.087 | 14.214 | 10.960 | 9.333 | 7.427 |
| `8192 x 8192` | 240.738 | 141.344 | 134.640 | 50.282 | 38.745 | 39.847 | 26.860 |

## Meaning of `calc time` and `wall time`

Let

\[
T_{\text{calc}} = \text{time spent only inside the Julia set calculation function}
\]

\[
T_{\text{wall}} = \text{total elapsed runtime for the whole program}
\]

In practice:

\[
T_{\text{wall}} \geq T_{\text{calc}}
\]

because `wall time` includes not only the computation itself, but also:

- data preparation
- list or array creation
- type conversion
- memory allocation
- function-call overhead

### Short Summary

- `calc time` = pure computation time
- `wall time` = total runtime observed by the user

## Output Figure

The benchmark plot is saved as:

- [`julia_grid_scaling.png`](./julia_grid_scaling.png)
