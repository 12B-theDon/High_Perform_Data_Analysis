#!/usr/bin/env python3

import argparse
import glob
import os
import re
import statistics
from collections import defaultdict


KEY_VALUE_RE = re.compile(r"([A-Za-z_]+)=(-?\d+(?:\.\d+)?)")
FILE_VER_RE = re.compile(r"_ver(\d+)_", re.IGNORECASE)
LINE_VER_RE = re.compile(r"\[score_all CUDA (ver\d+|default stream)")


def natural_version_key(version):
    match = re.search(r"(\d+)", version)
    if match:
        return (1, int(match.group(1)), version)
    return (0, 0, version)


def parse_values(line):
    values = {}
    for key, value in KEY_VALUE_RE.findall(line):
        if "." in value:
            values[key] = float(value)
        else:
            values[key] = int(value)
    return values


def version_from_file(path):
    name = os.path.basename(path)
    match = FILE_VER_RE.search(name)
    if match:
        return "ver{}".format(match.group(1))
    if "CUDA" in name:
        return "cuda_baseline"
    if "CPU" in name:
        return "cpu_baseline"
    return "unknown"


def version_from_score_line(line, fallback):
    match = LINE_VER_RE.search(line)
    if not match:
        return fallback
    version = match.group(1)
    if version == "default stream":
        return "cuda_baseline"
    return version


def mean(values):
    return sum(values) / len(values) if values else None


def median(values):
    return statistics.median(values) if values else None


def min_value(values):
    return min(values) if values else None


def max_value(values):
    return max(values) if values else None


def field_values(rows, field):
    return [row[field] for row in rows if field in row]


def print_stat_line(label, values, unit="ms"):
    if not values:
        print("{}: n/a".format(label))
        return
    suffix = " {}".format(unit) if unit else ""
    print("{} mean:   {:.6f}{}".format(label, mean(values), suffix))
    print("{} median: {:.6f}{}".format(label, median(values), suffix))
    print("{} min:    {:.6f}{}".format(label, min_value(values), suffix))
    print("{} max:    {:.6f}{}".format(label, max_value(values), suffix))


def add_derived_score_fields(row):
    if all(key in row for key in ("last", "h2d", "kernel", "d2h")):
        row.setdefault(
            "host_overhead",
            row["last"] - row["h2d"] - row["kernel"] - row["d2h"],
        )
    return row


def parse_logs(log_dir, pattern):
    score_rows = defaultdict(list)
    score_files = defaultdict(set)
    score_profile_rows = defaultdict(list)
    match_rows = defaultdict(list)
    all_files = sorted(glob.glob(os.path.join(log_dir, pattern)))

    for path in all_files:
        file_version = version_from_file(path)
        try:
            with open(path, "r", errors="replace") as handle:
                for line in handle:
                    if "[score_all CUDA" in line or "[score_all baseline]" in line:
                        version = version_from_score_line(line, file_version)
                        row = add_derived_score_fields(parse_values(line))
                        if row:
                            row["_file"] = path
                            score_rows[version].append(row)
                            score_files[version].add(path)
                    elif "[Score profile]" in line:
                        row = parse_values(line)
                        if row:
                            row["_file"] = path
                            score_profile_rows[file_version].append(row)
                    elif "[MatchWithWindow profile]" in line:
                        row = parse_values(line)
                        if row:
                            row["_file"] = path
                            match_rows[file_version].append(row)
        except OSError as error:
            print("skip {}: {}".format(path, error))

    return score_rows, score_files, score_profile_rows, match_rows


def filter_rows(rows, skip_calls):
    if skip_calls <= 0:
        return rows
    return [row for row in rows if row.get("calls", 0) > skip_calls]


def summarize_score_rows(version, rows, files, skip_calls):
    used = filter_rows(rows, skip_calls)
    print("\n[{} score_all]".format(version))
    print("files: {}".format(len(files)))
    print("samples: {} (calls > {})".format(len(used), skip_calls))
    if not used:
        return None

    last_values = field_values(used, "last")
    h2d_values = field_values(used, "h2d")
    kernel_values = field_values(used, "kernel")
    d2h_values = field_values(used, "d2h")
    host_overhead_values = field_values(used, "host_overhead")
    candidate_values = field_values(used, "candidates")
    scan_point_values = field_values(used, "scan_points")

    print_stat_line("last", last_values)
    if h2d_values:
        print("h2d mean:          {:.6f} ms".format(mean(h2d_values)))
    if kernel_values:
        print("kernel mean:       {:.6f} ms".format(mean(kernel_values)))
    if d2h_values:
        print("d2h mean:          {:.6f} ms".format(mean(d2h_values)))
    if host_overhead_values:
        print("host overhead mean:{:.6f} ms".format(mean(host_overhead_values)))
    if candidate_values:
        print("candidates median: {:.0f}".format(median(candidate_values)))
        print("candidates max:    {:.0f}".format(max_value(candidate_values)))
    if scan_point_values:
        print("scan_points median:{:.0f}".format(median(scan_point_values)))

    return mean(last_values) if last_values else None


def summarize_profile_rows(version, rows, title, fields):
    print("\n[{} {}]".format(version, title))
    print("samples: {}".format(len(rows)))
    if not rows:
        return
    for field in fields:
        values = field_values(rows, field)
        if values:
            print("{} mean: {:.6f} ms".format(field, mean(values)))


def summarize_comparison(score_means):
    valid = [(version, value) for version, value in score_means.items() if value]
    if len(valid) < 2:
        return
    valid.sort(key=lambda item: item[1])
    best_version, best_mean = valid[0]
    print("\n[score_all comparison]")
    print("best: {} ({:.6f} ms)".format(best_version, best_mean))
    for version, value in sorted(valid, key=lambda item: natural_version_key(item[0])):
        speedup = value / best_mean if best_mean > 0 else 0.0
        print("{} mean last: {:.6f} ms, relative_to_best: {:.3f}x".format(
            version, value, speedup
        ))


def main():
    parser = argparse.ArgumentParser(
        description="Analyze cartographer_parallel CUDA score_all and matcher logs."
    )
    parser.add_argument(
        "log_dir",
        nargs="?",
        default=".",
        help="Directory containing PA02_FM_CUDA_ver*_nvprof_*.log files.",
    )
    parser.add_argument(
        "--glob",
        default="*.log",
        help="Glob pattern inside log_dir. Default: *.log",
    )
    parser.add_argument(
        "--skip-calls",
        type=int,
        default=5,
        help="Ignore score_all samples with calls <= this value. Default: 5",
    )
    args = parser.parse_args()

    score_rows, score_files, score_profile_rows, match_rows = parse_logs(
        args.log_dir, args.glob
    )

    versions = sorted(
        set(score_rows) | set(score_profile_rows) | set(match_rows),
        key=natural_version_key,
    )
    if not versions:
        print("No matching log lines found in {}".format(args.log_dir))
        return

    score_means = {}
    for version in versions:
        if version in score_rows:
            score_means[version] = summarize_score_rows(
                version, score_rows[version], score_files[version], args.skip_calls
            )
        if version in score_profile_rows:
            summarize_profile_rows(
                version,
                score_profile_rows[version],
                "Score profile",
                ("total",),
            )
        if version in match_rows:
            summarize_profile_rows(
                version,
                match_rows[version],
                "MatchWithWindow profile",
                (
                    "total",
                    "MakeScans",
                    "MakeBounds",
                    "MakeGridStack",
                    "MakeLowCands",
                    "ScoreCoarse",
                    "Branch",
                ),
            )

    summarize_comparison(score_means)


if __name__ == "__main__":
    main()
