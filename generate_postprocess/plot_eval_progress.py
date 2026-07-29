"""Plot per-task evaluation progress from infer_* result directories.

The default plot uses a compact raw-value view: at most two core metrics per
task, with a secondary y-axis when two metrics are shown.
"""

import argparse
import math
import os
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from summarize_eval_results import (  # noqa: E402
    RESULT_FILES,
    TASK_METRICS,
    collect_result_dir,
)

DISPLAY_PERCENT_METRICS = {"WER", "SIM", "Semitone Accuracy", "IB"}

PLOT_METRICS = OrderedDict((
    ("TTS", (("WER", "down"), ("UTMOSv2", "up"))),
    ("SVS", (("F0", "down"), ("Semitone Accuracy", "up"))),
    ("T2A", (("FD", "down"), ("CLAP", "up"))),
    ("T2M", (("FD", "down"), ("CLAP", "up"))),
    ("SE", (("PESQ", "up"), ("STOI", "up"))),
    ("SR", (("LSD", "down"), )),
    ("V2A", (("IB", "up"), ("Sync", "down"))),
))

DISPLAY_METRIC_NAMES = {
    "Semitone Accuracy": "SA",
    "IB": "ImageBind",
    "Sync": "SYNC",
}

METRIC_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")


@dataclass
class ResultRow:
    iteration: int
    name: str
    path: str
    task_rows: OrderedDict
    source: str
    warnings: list[str]


def parse_iteration(name):
    """Parse iteration number from names like infer_iters_100K_steps_25."""
    m = re.search(r"iters_(\d+(?:\.\d+)?)([KM]?)(?:_|$)", name, re.IGNORECASE)
    if not m:
        m = re.search(r"(\d+(?:\.\d+)?)([KM])(?:_|$)", name, re.IGNORECASE)
    if m:
        value = float(m.group(1))
        suffix = m.group(2).upper()
        if suffix == "K":
            return int(value * 1000)
        if suffix == "M":
            return int(value * 1_000_000)
        return int(value)

    m = re.search(r"(\d{4,})", name)
    if m:
        return int(m.group(1))
    return None


def format_iteration(iteration):
    if iteration % 1_000_000 == 0:
        return f"{iteration // 1_000_000}M"
    if iteration % 1000 == 0:
        return f"{iteration // 1000}K"
    return str(iteration)


def parse_visible_float(value):
    value = str(value).strip().strip("`").replace(",", "")
    if not value or value == "-":
        return None
    if value.endswith("%"):
        value = value[:-1].strip()
    try:
        return float(value)
    except ValueError:
        return None


def empty_task_rows():
    return OrderedDict((task, {}) for task in TASK_METRICS)


def parse_summary_markdown(path):
    task_rows = empty_task_rows()
    current_task = None
    with open(path, "r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if not line.startswith("|"):
                continue

            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) < 3:
                continue
            if cells[0] == "Task" or set(cells[0]) <= {"-"}:
                continue

            task, metric, value = cells[:3]
            if task:
                current_task = task
            task = current_task
            if task not in task_rows or not metric:
                continue

            parsed = parse_visible_float(value)
            if parsed is not None:
                task_rows[task][metric] = parsed
    return task_rows


def scale_raw_task_rows(task_rows):
    scaled = empty_task_rows()
    for task, metric_specs in TASK_METRICS.items():
        for metric, _direction in metric_specs:
            value = task_rows.get(task, {}).get(metric)
            if value is None:
                continue
            if metric in DISPLAY_PERCENT_METRICS:
                value = value * 100
            scaled[task][metric] = value
    return scaled


def merge_task_rows(primary, fallback):
    merged = empty_task_rows()
    for task, metric_specs in TASK_METRICS.items():
        for metric, _direction in metric_specs:
            primary_value = primary.get(task, {}).get(metric)
            fallback_value = fallback.get(task, {}).get(metric)
            if primary_value is not None:
                merged[task][metric] = primary_value
            elif fallback_value is not None:
                merged[task][metric] = fallback_value
    return merged


def has_result_file(result_dir):
    if os.path.isfile(os.path.join(result_dir, "summary.md")):
        return True
    return any(
        os.path.isfile(os.path.join(result_dir, filename))
        for filename in RESULT_FILES.values()
    )


def collect_result_dirs(exp_dir, include_regex=None, exclude_regex=None):
    include = re.compile(include_regex) if include_regex else None
    exclude = re.compile(exclude_regex) if exclude_regex else None

    result_dirs = []
    for name in os.listdir(exp_dir):
        path = os.path.join(exp_dir, name)
        if not os.path.isdir(path) or not name.startswith("infer"):
            continue
        if include and not include.search(name):
            continue
        if exclude and exclude.search(name):
            continue
        iteration = parse_iteration(name)
        if iteration is None or not has_result_file(path):
            continue
        result_dirs.append((iteration, name, path))

    return sorted(result_dirs, key=lambda item: (item[0], item[1]))


def load_result_dir(result_dir, source):
    summary_path = os.path.join(result_dir, "summary.md")
    raw_rows = None
    raw_warnings = []
    summary_rows = None

    if source in {"auto", "summary"} and os.path.isfile(summary_path):
        summary_rows = parse_summary_markdown(summary_path)

    if source in {"auto", "raw"}:
        raw_rows, raw_warnings = collect_result_dir(result_dir)
        raw_rows = scale_raw_task_rows(raw_rows)

    if source == "summary":
        return summary_rows or empty_task_rows(), "summary", []
    if source == "raw":
        return raw_rows or empty_task_rows(), "raw", raw_warnings

    if summary_rows is not None and raw_rows is not None:
        return merge_task_rows(
            summary_rows, raw_rows
        ), "summary+raw", raw_warnings
    if summary_rows is not None:
        return summary_rows, "summary", []
    if raw_rows is not None:
        return raw_rows, "raw", raw_warnings
    return empty_task_rows(), "none", ["No summary.md or result files"]


def dedupe_rows(rows, duplicate_policy):
    if duplicate_policy == "all":
        return rows

    grouped = OrderedDict()
    for row in rows:
        if row.iteration not in grouped:
            grouped[row.iteration] = []
        grouped[row.iteration].append(row)

    deduped = []
    for iteration, group in grouped.items():
        if duplicate_policy == "first":
            deduped.append(group[0])
        elif duplicate_policy == "last":
            deduped.append(group[-1])
        elif duplicate_policy == "error" and len(group) > 1:
            names = ", ".join(row.name for row in group)
            raise ValueError(
                f"Duplicate iteration {format_iteration(iteration)}: {names}"
            )
        else:
            deduped.extend(group)
    return deduped


def collect_rows(exp_dir, args):
    rows = []
    for iteration, name, path in collect_result_dirs(
        exp_dir,
        include_regex=args.include_regex,
        exclude_regex=args.exclude_regex,
    ):
        task_rows, source, warnings = load_result_dir(path, args.source)
        rows.append(
            ResultRow(iteration, name, path, task_rows, source, warnings)
        )
    return dedupe_rows(rows, args.duplicate_policy)


def finite_values(values):
    return [
        value
        for value in values if value is not None and math.isfinite(value)
    ]


def normalize_values(values, direction):
    valid = finite_values(values)
    if not valid:
        return [math.nan for _ in values]

    min_value = min(valid)
    max_value = max(valid)
    if max_value == min_value:
        return [
            0.5 if value is not None and math.isfinite(value) else math.nan
            for value in values
        ]

    normalized = []
    for value in values:
        if value is None or not math.isfinite(value):
            normalized.append(math.nan)
            continue
        if direction == "down":
            normalized.append((max_value - value) / (max_value - min_value))
        else:
            normalized.append((value - min_value) / (max_value - min_value))
    return normalized


def metric_label(metric):
    return DISPLAY_METRIC_NAMES.get(metric, metric)


def legend_label(metric, direction):
    arrow = r"$\downarrow$" if direction == "down" else r"$\uparrow$"
    return f"{metric_label(metric)} ({arrow})"


def build_task_series(rows, task, metric):
    values = [row.task_rows.get(task, {}).get(metric) for row in rows]
    return [value if value is not None else math.nan for value in values]


def set_axis_limits(axis, values):
    valid = finite_values(values)
    if not valid:
        return
    value_range = max(valid) - min(valid)
    pad = value_range * 0.12 if value_range else abs(valid[0]) * 0.05 or 0.05
    axis.set_ylim(min(valid) - pad, max(valid) + pad)


def plot_progress(rows, out_path, title, tasks, width, height_per_task):
    if not rows:
        raise ValueError("No inference result directories found.")

    selected_tasks = [task for task in PLOT_METRICS if task in tasks]
    if not selected_tasks:
        raise ValueError(f"No valid tasks selected: {tasks}")

    x_values = [row.iteration / 1000 for row in rows]
    x_labels = [format_iteration(row.iteration) for row in rows]

    fig_height = max(height_per_task * len(selected_tasks), 4.0)
    fig, axes = plt.subplots(
        len(selected_tasks),
        1,
        figsize=(width, fig_height),
        sharex=True,
        squeeze=False,
    )
    axes = axes.flatten()

    for axis, task in zip(axes, selected_tasks):
        axis.grid(True, alpha=0.3)

        plotted = False
        metric_specs = PLOT_METRICS[task]

        if len(metric_specs) == 1:
            metric, direction = metric_specs[0]
            values = build_task_series(rows, task, metric)
            label = metric_label(metric)

            lines = []
            if finite_values(values):
                lines = axis.plot(
                    x_values,
                    values,
                    marker="o",
                    linewidth=2,
                    markersize=5,
                    color="#1f77b4",
                    label=legend_label(metric, direction),
                )
                axis.set_ylabel(label, fontsize=10, color="#1f77b4")
                axis.tick_params(axis="y", labelcolor="#1f77b4")
                set_axis_limits(axis, values)
                plotted = True

            if lines:
                axis.legend(lines, [line.get_label() for line in lines],
                            fontsize=9, loc="best")

            if not plotted:
                axis.text(
                    0.5,
                    0.5,
                    "No metrics found",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=10,
                )
            continue

        lines = []
        markers = ("o", "s", "^")
        for metric_index, (metric, direction) in enumerate(metric_specs):
            metric_axis = axis if metric_index == 0 else axis.twinx()
            if metric_index > 1:
                metric_axis.spines["right"].set_position(
                    ("outward", 60 * (metric_index - 1))
                )
            values = build_task_series(rows, task, metric)
            if not finite_values(values):
                continue

            color = METRIC_COLORS[metric_index % len(METRIC_COLORS)]
            label = metric_label(metric)
            metric_lines = metric_axis.plot(
                x_values,
                values,
                marker=markers[metric_index % len(markers)],
                linewidth=2,
                markersize=5,
                color=color,
                label=legend_label(metric, direction),
            )
            metric_axis.set_ylabel(label, fontsize=10, color=color)
            metric_axis.tick_params(axis="y", labelcolor=color)
            set_axis_limits(metric_axis, values)
            lines.extend(metric_lines)
            plotted = True

        if lines:
            axis.legend(
                lines, [line.get_label() for line in lines],
                fontsize=9,
                loc="best"
            )

        if not plotted:
            axis.text(
                0.5,
                0.5,
                "No metrics found",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=10,
            )

    axes[-1].set_xlabel("Iteration", fontsize=11)
    axes[-1].set_xticks(x_values)
    axes[-1].set_xticklabels(x_labels)

    if title:
        fig.suptitle(title, fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def format_value(value):
    if value is None:
        return "-"
    if isinstance(value, float) and not math.isfinite(value):
        return "nan"
    return f"{value:.3f}"


def print_table(rows, tasks):
    columns = ["Iteration", "Dir", "Source"]
    metric_columns = []
    for task in PLOT_METRICS:
        if task not in tasks:
            continue
        for metric, _direction in PLOT_METRICS[task]:
            columns.append(f"{task} {metric}")
            metric_columns.append((task, metric))

    print("\t".join(columns))
    for row in rows:
        values = [format_iteration(row.iteration), row.name, row.source]
        for task, metric in metric_columns:
            values.append(
                format_value(row.task_rows.get(task, {}).get(metric))
            )
        print("\t".join(values))

    warnings = []
    for row in rows:
        for warning in row.warnings:
            warnings.append(f"{row.name}: {warning}")
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"- {warning}")


def parse_tasks(value):
    if not value:
        return list(PLOT_METRICS)
    tasks = []
    for item in value.split(","):
        task = item.strip().upper()
        if task:
            tasks.append(task)
    return tasks


def default_output_path(exp_dir):
    return os.path.join(exp_dir, "eval_progress.png")


def main():
    parser = argparse.ArgumentParser(
        description=
        "Plot per-task performance changes over inference iterations.",
    )
    parser.add_argument(
        "exp_dir", help="Experiment directory containing infer_* subdirs"
    )
    parser.add_argument(
        "-o",
        "--output",
        help=
        "Output PNG path. Defaults to eval_progress_<mode>.png in exp_dir.",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "summary", "raw"),
        default="auto",
        help=
        "Read summary.md, raw result files, or use summary with raw fallback.",
    )
    parser.add_argument(
        "--tasks",
        help=
        "Comma-separated tasks to plot, e.g. TTS,T2A,T2M,V2A. Defaults to all tasks.",
    )
    parser.add_argument("--title", help="Figure title")
    parser.add_argument(
        "--width",
        type=float,
        default=16.0,
        help="Figure width in inches.",
    )
    parser.add_argument(
        "--height-per-task",
        type=float,
        default=3.2,
        help="Figure height in inches for each task subplot.",
    )
    parser.add_argument(
        "--include-regex",
        help=
        "Only include infer directories whose basename matches this regex.",
    )
    parser.add_argument(
        "--exclude-regex",
        help="Exclude infer directories whose basename matches this regex.",
    )
    parser.add_argument(
        "--duplicate-policy",
        choices=("last", "first", "all", "error"),
        default="last",
        help=
        "How to handle multiple infer directories with the same iteration.",
    )
    args = parser.parse_args()

    exp_dir = os.path.normpath(args.exp_dir)
    tasks = parse_tasks(args.tasks)
    rows = collect_rows(exp_dir, args)
    out_path = args.output or default_output_path(exp_dir)
    title = args.title or os.path.basename(exp_dir)

    plot_progress(
        rows,
        out_path,
        title,
        tasks,
        width=args.width,
        height_per_task=args.height_per_task,
    )
    print(f"Saved to {out_path}")
    print_table(rows, tasks)


if __name__ == "__main__":
    main()
