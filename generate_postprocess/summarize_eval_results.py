"""Summarize evaluation result directories as Markdown."""

import argparse
import json
import math
import os
import re
from collections import OrderedDict
from statistics import mean

TASK_METRICS = OrderedDict((
    ("TTS", (("WER", "down"), ("SIM", "up"), ("UTMOSv2", "up"))),
    ("SVS", (("F0", "down"), ("Semitone Accuracy", "up"))),
    ("T2A", (("FD", "down"), ("KL", "down"), ("CLAP", "up"))),
    ("T2M", (("FD", "down"), ("KL", "down"), ("CLAP", "up"))),
    ("SE", (("PESQ", "up"), ("STOI", "up"))),
    ("SR", (("LSD", "down"), )),
    ("V2A", (("FD", "down"), ("IB", "up"), ("Sync", "down"))),
))

RESULT_FILES = {
    "TTS": "tts_results.txt",
    "SVS": "svs_results.txt",
    "T2A": "t2a_results.txt",
    "T2M": "t2m_cnn14_results.txt",
    "SE": "se_results.txt",
    "SR": "sr_all_res.txt",
    "V2A": "v2a_results.txt",
}

KEY_VALUE_ALIASES = {
    "FD": ("frechet_distance", ),
    "KL": ("kullback_leibler_divergence_softmax", ),
    "CLAP": ("CLAP_score", ),
    "IB": ("image_bind_score", ),
    "Sync": ("synchformer_score", "Synchformer", "sync"),
    "LSD": ("Average LSD", "avg_lsd", "lsd"),
    "PESQ": ("pesq", ),
    "STOI": ("stoi", ),
}

JSONL_ALIASES = {
    "WER": ("average_wer", "WER"),
    "SIM": ("average_cosine_similarity", "SIM"),
    "UTMOSv2": ("average_utmos_v2", "UTMOSv2"),
    "F0": ("f0", "F0"),
    "Semitone Accuracy": ("semitone", "Semitone Accuracy", "Semitone"),
}


def read_text(path):
    with open(path, "r", encoding="utf-8") as reader:
        return reader.read()


def parse_float(value):
    value = str(value).strip()
    if not value:
        return None

    is_percent = value.endswith("%")
    if is_percent:
        value = value[:-1].strip()

    try:
        parsed = float(value)
    except ValueError:
        return None

    if is_percent:
        return parsed / 100.0
    return parsed


def parse_key_value_file(path):
    metrics = {}
    for line in read_text(path).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed = parse_float(value)
        if parsed is not None:
            metrics[key.strip()] = parsed
    return metrics


def parse_csv_summary_file(path):
    lines = [
        line.strip() for line in read_text(path).splitlines() if line.strip()
    ]
    if len(lines) < 2:
        return {}

    headers = [item.strip() for item in lines[0].split(",")]
    values = [item.strip() for item in lines[1].split(",")]
    metrics = {}
    for key, value in zip(headers, values):
        parsed = parse_float(value)
        if parsed is not None:
            metrics[key] = parsed
    return metrics


def parse_jsonl_summary_file(path, aliases):
    rows = []
    summary = {}
    for line in read_text(path).splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append(row)
        for key, value in row.items():
            if key.startswith("average_"):
                parsed = parse_float(value)
                if parsed is not None:
                    summary[key] = parsed

    metrics = {}
    for metric, candidate_keys in aliases.items():
        for key in candidate_keys:
            if key in summary:
                metrics[metric] = summary[key]
                break
        if metric in metrics:
            continue

        values = []
        for row in rows:
            for key in candidate_keys:
                if key not in row:
                    continue
                parsed = parse_float(row[key])
                if parsed is not None:
                    values.append(parsed)
                    break
        if values:
            metrics[metric] = mean(values)

    return metrics


def pick_metric(metrics, aliases):
    for key in aliases:
        if key in metrics:
            return metrics[key]
    return None


def parse_svs_file(path):
    metrics = parse_jsonl_summary_file(
        path,
        {
            "F0": JSONL_ALIASES["F0"],
            "Semitone Accuracy": JSONL_ALIASES["Semitone Accuracy"],
        },
    )

    tail_metrics = parse_key_value_file(path)
    if "f0" in tail_metrics:
        metrics["F0"] = tail_metrics["f0"]
    if "semitone" in tail_metrics:
        metrics["Semitone Accuracy"] = tail_metrics["semitone"]
    return metrics


def parse_sr_file(path):
    metrics = parse_key_value_file(path)
    if "Average LSD" in metrics:
        return {"LSD": metrics["Average LSD"]}

    avg_lsd_values = re.findall(
        r'"avg_lsd"\s*:\s*([-+0-9.eE]+)', read_text(path)
    )
    if avg_lsd_values:
        return {"LSD": mean(float(value) for value in avg_lsd_values)}
    return {}


def parse_task_metrics(result_dir, task):
    path = os.path.join(result_dir, RESULT_FILES[task])
    if not os.path.isfile(path):
        return {}, [f"Missing {RESULT_FILES[task]}"]

    if task == "TTS":
        return parse_jsonl_summary_file(
            path,
            {
                "WER": JSONL_ALIASES["WER"],
                "SIM": JSONL_ALIASES["SIM"],
                "UTMOSv2": JSONL_ALIASES["UTMOSv2"],
            },
        ), []
    if task == "SVS":
        return parse_svs_file(path), []
    if task == "SE":
        parsed = parse_csv_summary_file(path)
        return {
            "PESQ": pick_metric(parsed, KEY_VALUE_ALIASES["PESQ"]),
            "STOI": pick_metric(parsed, KEY_VALUE_ALIASES["STOI"]),
        }, []
    if task == "SR":
        return parse_sr_file(path), []

    parsed = parse_key_value_file(path)
    metrics = {}
    for metric, _direction in TASK_METRICS[task]:
        metrics[metric] = pick_metric(parsed, KEY_VALUE_ALIASES[metric])
    return metrics, []


def collect_result_dir(result_dir):
    task_rows = OrderedDict()
    warnings = []
    for task in TASK_METRICS:
        metrics, task_warnings = parse_task_metrics(result_dir, task)
        task_rows[task] = metrics
        warnings.extend(f"{task}: {warning}" for warning in task_warnings)
    return task_rows, warnings


def format_value(value, digits, percent=False):
    if value is None:
        return "-"
    if isinstance(value, float) and not math.isfinite(value):
        return "nan"
    if percent:
        return f"{value * 100:.{digits}f}"
    return f"{value:.{digits}f}"


def make_task_table(result_dir, task_rows, digits):
    lines = [
        "| Task | Metric | Value |",
        "| --- | --- | --- |",
    ]
    for task, metric_specs in TASK_METRICS.items():
        metrics = task_rows.get(task, {})
        for index, (metric, direction) in enumerate(metric_specs):
            value = metrics.get(metric)
            percent = metric in ["Semitone Accuracy", "WER", "SIM", "IB"]
            task_cell = task if index == 0 else ""
            lines.append(
                f"| {task_cell} | {metric} | {format_value(value, digits, percent=percent)} |"
            )
    return lines


def make_wide_table(results, digits):
    header = ["Result"]
    metric_columns = []
    for task, metric_specs in TASK_METRICS.items():
        for metric, direction in metric_specs:
            header.append(f"{task} {metric}")
            metric_columns.append((task, metric))

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for result_dir, task_rows, _warnings in results:
        row = [os.path.basename(os.path.normpath(result_dir))]
        for task, metric in metric_columns:
            value = task_rows.get(task, {}).get(metric)
            row.append(
                format_value(
                    value, digits, percent=metric == "Semitone Accuracy"
                )
            )
        lines.append("| " + " | ".join(row) + " |")
    return lines


def make_markdown(result_dirs, digits):
    results = []
    for result_dir in result_dirs:
        task_rows, warnings = collect_result_dir(result_dir)
        results.append((result_dir, task_rows, warnings))

    lines = ["# Evaluation Summary", ""]
    if len(results) == 1:
        result_dir, task_rows, _warnings = results[0]
        lines.extend([
            *make_task_table(result_dir, task_rows, digits),
        ])
    else:
        lines.extend(make_wide_table(results, digits))

    warnings = []
    for result_dir, _task_rows, result_warnings in results:
        for warning in result_warnings:
            warnings.append(f"- `{result_dir}`: {warning}")
    if warnings:
        lines.extend(["", "## Warnings", *warnings])

    lines.append("")
    return "\n".join(lines)


def default_output_path(result_dirs):
    if len(result_dirs) == 1:
        return os.path.join(result_dirs[0], "summary.md")
    return "summary.md"


def write_markdown(markdown, output, result_dirs, output_was_explicit):
    try:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        with open(output, "w", encoding="utf-8") as writer:
            writer.write(markdown)
        return output
    except PermissionError:
        if output_was_explicit or len(result_dirs) != 1:
            raise

        fallback = f"{os.path.basename(os.path.normpath(result_dirs[0]))}_summary.md"
        with open(fallback, "w", encoding="utf-8") as writer:
            writer.write(markdown)
        print(
            f"Could not write to {output}; saved summary to {fallback} instead."
        )
        return fallback


def main():
    parser = argparse.ArgumentParser(
        description="Summarize evaluation results into a Markdown table.",
    )
    parser.add_argument(
        "result_dirs",
        nargs="+",
        help=
        "One or more inference result directories, e.g. experiments/.../infer_iters_100K_steps_25",
    )
    parser.add_argument(
        "-o",
        "--output",
        help=(
            "Output Markdown path. Defaults to summary.md inside the single result dir, "
            "or ./summary.md for multiple dirs. If the single result dir is not writable, "
            "the script falls back to ./<result_dir_name>_summary.md."
        ),
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=3,
        help="Number of digits after the decimal point."
    )
    args = parser.parse_args()

    result_dirs = [os.path.normpath(path) for path in args.result_dirs]
    markdown = make_markdown(result_dirs, args.digits)
    output = write_markdown(
        markdown,
        args.output or default_output_path(result_dirs),
        result_dirs,
        output_was_explicit=bool(args.output),
    )
    print(f"Saved summary to {output}")


if __name__ == "__main__":
    main()
