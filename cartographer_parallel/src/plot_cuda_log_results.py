#!/usr/bin/env python3

import argparse
import csv
import glob
import math
import os
import re
import statistics
from collections import defaultdict

KEY_VALUE_RE = re.compile(r"([A-Za-z_]+)=(-?\d+(?:\.\d+)?)")
FILE_RE = re.compile(
    r"PA02_FM_(?P<mode>CPU|CUDA)(?:_res(?P<res>[0-9p]+))?(?:_depth(?P<depth>\d+))?(?:_(?P<version>baseline|ver\d+|cpu_baseline|cuda_baseline))?(?:_nvprof)?(?:_(?P<run>\d+))?\.log",
    re.IGNORECASE,
)
LINE_VER_RE = re.compile(r"\[score_all CUDA (ver\d+|default stream|baseline)", re.IGNORECASE)
MATCH_TIME_RE = re.compile(r"\[match_time\]\s*([0-9]+(?:\.[0-9]+)?)\s*ms")
ERROR_RE = re.compile(r"error|fatal|timeout|timed out|terminated", re.IGNORECASE)
VERSION_ORDER = ["cpu_baseline", "baseline", "ver1", "ver2", "ver3", "ver4", "ver5", "ver6", "ver7"]


def parse_values(line):
    values = {}
    for key, value in KEY_VALUE_RE.findall(line):
        values[key] = float(value) if "." in value else int(value)
    return values


def decode_resolution(tag):
    if not tag:
        return "default"
    return tag.replace("p", ".")


def file_meta(path):
    name = os.path.basename(path)
    match = FILE_RE.search(name)
    if not match:
        return {
            "case": "default_depthdefault",
            "resolution": "default",
            "depth": "default",
            "version": "unknown",
            "run": None,
        }
    mode = match.group("mode").lower()
    version = (match.group("version") or "").lower()
    if not version:
        version = "cpu_baseline" if mode == "cpu" else "baseline"
    if mode == "cpu" and version == "baseline":
        version = "cpu_baseline"
    if version in ("cuda", "cuda_baseline"):
        version = "baseline"
    if version in ("cpu", "cpu_baseline"):
        version = "cpu_baseline"
    resolution = decode_resolution(match.group("res"))
    depth = match.group("depth") or "default"
    return {
        "case": "res{}_depth{}".format(resolution, depth),
        "resolution": resolution,
        "depth": depth,
        "version": version,
        "run": int(match.group("run")) if match.group("run") else None,
    }


def version_from_score_line(line, fallback):
    match = LINE_VER_RE.search(line)
    if not match:
        return fallback
    version = match.group(1).lower()
    if version in ("default stream", "baseline"):
        return "baseline"
    return version


def mean(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return sum(values) / len(values) if values else None


def median(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.median(values) if values else None


def field(rows, name):
    return [row[name] for row in rows if name in row]


def version_key(version):
    if version in VERSION_ORDER:
        return (0, VERSION_ORDER.index(version))
    match = re.search(r"\d+", version)
    return (1, int(match.group()) if match else 999, version)


def case_key(case):
    match = re.search(r"res([^_]+)_depth(\w+)", case)
    if not match:
        return (999.0, 999, case)
    res = match.group(1)
    depth = match.group(2)
    try:
        res_value = float(res)
    except ValueError:
        res_value = 999.0
    try:
        depth_value = int(depth)
    except ValueError:
        depth_value = 999
    return (res_value, depth_value, case)


def parse_logs(log_dir, pattern):
    grouped = defaultdict(lambda: defaultdict(lambda: {
        "files": set(),
        "error_files": set(),
        "score_rows": [],
        "match_rows": [],
        "score_profile_rows": [],
    }))
    for path in sorted(glob.glob(os.path.join(log_dir, pattern))):
        meta = file_meta(path)
        case = meta["case"]
        file_version = meta["version"]
        seen_error = False
        try:
            with open(path, "r", errors="replace") as handle:
                for line in handle:
                    if ERROR_RE.search(line):
                        seen_error = True
                    if "[score_all CUDA" in line or "[score_all baseline]" in line:
                        version = version_from_score_line(line, file_version)
                        row = parse_values(line)
                        if row:
                            row.update(meta)
                            row["version"] = version
                            grouped[case][version]["score_rows"].append(row)
                    elif "[MatchWithWindow profile]" in line:
                        row = parse_values(line)
                        if row:
                            row.update(meta)
                            grouped[case][file_version]["match_rows"].append(row)
                    elif "[match_time]" in line:
                        match_time = MATCH_TIME_RE.search(line)
                        if match_time:
                            row = {"total": float(match_time.group(1))}
                            row.update(meta)
                            grouped[case][file_version]["match_rows"].append(row)
                    elif "[Score profile]" in line:
                        row = parse_values(line)
                        if row:
                            row.update(meta)
                            grouped[case][file_version]["score_profile_rows"].append(row)
        except OSError as error:
            print("skip {}: {}".format(path, error))
            continue
        grouped[case][file_version]["files"].add(path)
        if seen_error:
            grouped[case][file_version]["error_files"].add(path)
    return grouped


def filter_score_rows(rows, skip_calls):
    if skip_calls <= 0:
        return rows
    return [row for row in rows if row.get("calls", 0) > skip_calls]


def summarize(grouped, skip_calls):
    rows = []
    for case in sorted(grouped, key=case_key):
        for version in sorted(grouped[case], key=version_key):
            data = grouped[case][version]
            score_rows = filter_score_rows(data["score_rows"], skip_calls)
            match_rows = data["match_rows"]
            meta_source = (
                data["score_rows"] or data["match_rows"] or
                data["score_profile_rows"] or [{}]
            )[0]
            summary = {
                "case": case,
                "resolution": meta_source.get("resolution"),
                "depth": meta_source.get("depth"),
                "version": version,
                "files": len(data["files"]),
                "error_files": len(data["error_files"]),
                "score_samples": len(score_rows),
                "match_samples": len(match_rows),
                "match_total_mean": mean(field(match_rows, "total")),
                "scorecoarse_mean": mean(field(match_rows, "ScoreCoarse")),
                "branch_mean": mean(field(match_rows, "Branch")),
                "score_last_mean": mean(field(score_rows, "last")),
                "kernel_mean": mean(field(score_rows, "kernel")),
                "h2d_mean": mean(field(score_rows, "h2d")),
                "d2h_mean": mean(field(score_rows, "d2h")),
                "full_inside_ratio_mean": mean(field(score_rows, "full_inside_ratio")),
                "candidates_median": median(field(score_rows, "candidates")),
            }
            rows.append(summary)
    return rows


def fmt(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return ("{:." + str(digits) + "f}").format(value)


def markdown_table(rows):
    headers = [
        "case", "version", "files", "errors", "match_total_ms",
        "ScoreCoarse_ms", "kernel_ms", "score_last_ms", "full_inside", "candidates_med",
    ]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        values = [
            row["case"],
            row["version"],
            str(row["files"]),
            str(row["error_files"]),
            fmt(row["match_total_mean"]),
            fmt(row["scorecoarse_mean"]),
            fmt(row["kernel_mean"]),
            fmt(row["score_last_mean"]),
            fmt(row["full_inside_ratio_mean"], 4),
            fmt(row["candidates_median"], 0),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def print_markdown_table(rows):
    print(markdown_table(rows))


def write_csv(rows, output_path):
    if not rows:
        return
    fieldnames = [
        "case", "resolution", "depth", "version", "files", "error_files",
        "score_samples", "match_samples", "match_total_mean",
        "scorecoarse_mean", "branch_mean", "score_last_mean", "kernel_mean",
        "h2d_mean", "d2h_mean", "full_inside_ratio_mean", "candidates_median",
    ]
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def write_markdown_report(rows, figure_paths, output_path):
    rel_figures = [os.path.relpath(path, os.path.dirname(output_path))
                   for path in figure_paths]
    with open(output_path, "w") as handle:
        handle.write("# CUDA Fast Matcher Log Summary\n\n")
        handle.write("## Summary Table\n\n")
        handle.write(markdown_table(rows))
        handle.write("\n\n")
        if rel_figures:
            handle.write("## Figures\n\n")
            for path in rel_figures:
                title = os.path.splitext(os.path.basename(path))[0]
                handle.write("![{}]({})\n\n".format(title, path))
        handle.write("## Notes\n\n")
        handle.write("- `score_last_ms` is parsed from `[score_all ... last=]`.\n")
        handle.write("- `kernel_ms`, `h2d_mean`, and `d2h_mean` are available when the CUDA version prints those fields.\n")
        handle.write("- `match_total_ms` is parsed from `[MatchWithWindow profile]` or `[match_time]`.\n")
        handle.write("- `errors` counts log files containing `ERROR`, `FATAL`, `timeout`, `timed out`, or `terminated`.\n")


def require_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as error:
        print("matplotlib unavailable; skip PNG generation: {}".format(error))
        return None


def plot_grouped_bars(plt, rows, metric, ylabel, title, output_path):
    usable = [r for r in rows if r.get(metric) is not None]
    if not usable:
        return False
    cases = sorted({r["case"] for r in usable}, key=case_key)
    versions = [v for v in VERSION_ORDER if any(r["version"] == v for r in usable)]
    width = 0.8 / max(1, len(versions))
    x = list(range(len(cases)))
    fig, ax = plt.subplots(figsize=(max(8, len(cases) * 1.4), 4.8))
    for index, version in enumerate(versions):
        values = []
        for case in cases:
            match = next((r for r in usable if r["case"] == case and r["version"] == version), None)
            values.append(match.get(metric) if match else 0.0)
        offsets = [pos - 0.4 + width / 2 + index * width for pos in x]
        ax.bar(offsets, values, width=width, label=version)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=min(len(versions), 4))
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def plot_ver6_scatter(plt, grouped, output_path):
    xs = []
    ys = []
    colors = []
    labels = []
    case_to_index = {}
    for case in sorted(grouped, key=case_key):
        case_to_index[case] = len(case_to_index)
        data = grouped[case].get("ver6")
        if not data:
            continue
        for row in filter_score_rows(data["score_rows"], 0):
            if "full_inside_ratio" in row and "kernel" in row:
                xs.append(row["full_inside_ratio"])
                ys.append(row["kernel"])
                colors.append(case_to_index[case])
                labels.append(case)
    if not xs:
        return False
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    scatter = ax.scatter(xs, ys, c=colors, cmap="tab10", alpha=0.75, s=28)
    ax.set_title("ver6 full-inside ratio vs kernel time")
    ax.set_xlabel("full_inside_ratio")
    ax.set_ylabel("kernel time (ms)")
    ax.grid(alpha=0.25)
    handles = []
    for case, idx in case_to_index.items():
        if case in labels:
            handles.append(plt.Line2D([0], [0], marker="o", color="w", label=case,
                                      markerfacecolor=scatter.cmap(scatter.norm(idx)), markersize=7))
    if handles:
        ax.legend(handles=handles, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def save_plots(grouped, rows, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt = require_matplotlib()
    if plt is None:
        return []
    outputs = []
    specs = [
        ("match_total_mean", "MatchWithWindow total (ms)", "End-to-end matcher time by case/version", "fig_match_total_by_version.png"),
        ("scorecoarse_mean", "ScoreCoarse (ms)", "Coarse scoring time by case/version", "fig_scorecoarse_by_version.png"),
        ("kernel_mean", "score_all kernel (ms)", "CUDA kernel time by case/version", "fig_kernel_by_version.png"),
    ]
    for metric, ylabel, title, filename in specs:
        path = os.path.join(output_dir, filename)
        if plot_grouped_bars(plt, rows, metric, ylabel, title, path):
            outputs.append(path)
    path = os.path.join(output_dir, "fig_ver6_full_inside_vs_kernel.png")
    if plot_ver6_scatter(plt, grouped, path):
        outputs.append(path)
    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Parse CUDA matcher nvprof logs, print Markdown tables, and save report PNG figures."
    )
    parser.add_argument("log_dir", nargs="?", default="/root/catkin_ws/cuda_logs")
    parser.add_argument("--glob", default="*.log")
    parser.add_argument("--skip-calls", type=int, default=5)
    parser.add_argument("--output-dir", default="cuda_report_figures")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--report-md", default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    grouped = parse_logs(args.log_dir, args.glob)
    rows = summarize(grouped, args.skip_calls)
    if not rows:
        print("No matching rows found in {}".format(args.log_dir))
        return
    print_markdown_table(rows)
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = args.csv or os.path.join(args.output_dir, "summary.csv")
    write_csv(rows, csv_path)
    print("\nSaved table CSV:")
    print("- {}".format(csv_path))
    outputs = []
    if not args.no_plots:
        outputs = save_plots(grouped, rows, args.output_dir)
        if outputs:
            print("\nSaved figures:")
            for output in outputs:
                print("- {}".format(output))
    report_path = args.report_md or os.path.join(args.output_dir, "summary.md")
    write_markdown_report(rows, outputs, report_path)
    print("\nSaved Markdown report:")
    print("- {}".format(report_path))


if __name__ == "__main__":
    main()
