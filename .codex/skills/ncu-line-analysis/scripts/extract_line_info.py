#!/usr/bin/env python3
"""Extract source-correlated line metrics from an Nsight Compute report."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


PREFERRED_METRICS = (
    "smsp__pcsamp_sample_count",
    "inst_executed",
    "thread_inst_executed",
    "thread_inst_executed_true",
)
STALL_PREFIX = "smsp__pcsamp_warps_issue_stalled_"


def _version_key(path: str) -> tuple[int, ...]:
    match = re.search(r"nsight-compute/([^/]+)/extras/python$", path)
    if not match:
        return ()
    return tuple(int(part) for part in re.findall(r"\d+", match.group(1)))


def import_ncu_report():
    try:
        import ncu_report  # type: ignore

        return ncu_report
    except ModuleNotFoundError:
        pass

    candidates: list[str] = []
    env_dir = os.environ.get("NCU_REPORT_PYTHON_DIR")
    if env_dir:
        candidates.append(env_dir)
    candidates.extend(glob.glob("/opt/nvidia/nsight-compute/*/extras/python"))
    candidates.extend(glob.glob("/usr/local/NVIDIA-Nsight-Compute*/extras/python"))

    for candidate in sorted(set(candidates), key=_version_key, reverse=True):
        if not Path(candidate, "ncu_report.py").is_file():
            continue
        sys.path.insert(0, candidate)
        try:
            import ncu_report  # type: ignore

            return ncu_report
        except ModuleNotFoundError:
            sys.path.pop(0)

    raise SystemExit(
        "error: could not import ncu_report. Set NCU_REPORT_PYTHON_DIR to the "
        "Nsight Compute extras/python directory."
    )


def split_metric_args(values: list[str] | None) -> list[str]:
    if not values:
        return []
    metrics: list[str] = []
    for value in values:
        metrics.extend(part.strip() for part in value.split(",") if part.strip())
    return metrics


def metric_value(metric: Any, idx: int) -> int | float | None:
    value = metric.value(idx)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    return None


def correlation_address(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value, 0)
        except ValueError:
            return None
    return None


def source_line(source_files: dict[str, list[str]], file_name: str, line_no: int) -> str | None:
    lines = source_files.get(file_name)
    if not lines or line_no < 1 or line_no > len(lines):
        return None
    return lines[line_no - 1].rstrip()


def choose_metrics(action: Any, requested: list[str], all_correlated: bool) -> list[str]:
    names = tuple(action.metric_names())
    name_set = set(names)

    if requested:
        missing = [name for name in requested if name not in name_set]
        if missing:
            raise KeyError(", ".join(missing))
        return requested

    correlated: list[str] = []
    for name in names:
        metric = action.metric_by_name(name)
        if metric is not None and metric.has_correlation_ids() and metric.num_instances() > 0:
            correlated.append(name)

    if all_correlated:
        return correlated

    selected = [name for name in PREFERRED_METRICS if name in correlated]
    selected.extend(name for name in correlated if name.startswith(STALL_PREFIX))
    return selected


def short_stall_name(metric_name: str) -> str:
    return metric_name.removeprefix(STALL_PREFIX)


def aggregate_report(args: argparse.Namespace) -> dict[str, Any]:
    ncu_report = import_ncu_report()
    report_path = Path(args.report)
    ctx = ncu_report.load_report(report_path)
    requested_metrics = split_metric_args(args.metric)

    payload: dict[str, Any] = {
        "report": str(report_path),
        "ncu_report_version": ctx.get_version(),
        "actions": [],
        "lines": [],
    }

    selected_actions = 0
    actions_with_imported_source = 0
    missing_source_actions: list[str] = []
    missing_metric_errors: list[str] = []
    skipped_non_address_correlations = 0
    line_map: dict[tuple[int, int, str, int], dict[str, Any]] = {}

    for range_idx, report_range in enumerate(ctx):
        for action_idx, action in enumerate(report_range):
            action_name = action.name()
            if args.kernel and args.kernel not in action_name:
                continue
            selected_actions += 1

            raw_sources = dict(action.source_files())
            imported_sources = {
                file_name: contents.splitlines()
                for file_name, contents in raw_sources.items()
                if contents
            }
            if not imported_sources:
                missing_source_actions.append(f"range {range_idx} action {action_idx} {action_name}")
                continue
            actions_with_imported_source += 1

            try:
                metric_names = choose_metrics(action, requested_metrics, args.all_correlated_metrics)
            except KeyError as exc:
                missing_metric_errors.append(f"{action_name}: {exc}")
                continue

            action_info = {
                "range_index": range_idx,
                "action_index": action_idx,
                "name": action_name,
                "source_files": [
                    {
                        "file": file_name,
                        "imported": bool(contents),
                        "line_count": len(contents.splitlines()) if contents else 0,
                    }
                    for file_name, contents in sorted(raw_sources.items())
                ],
                "metrics_used": metric_names,
            }
            payload["actions"].append(action_info)

            for metric_name in metric_names:
                metric = action.metric_by_name(metric_name)
                if metric is None or not metric.has_correlation_ids():
                    continue
                correlation_ids = metric.correlation_ids()
                if correlation_ids is None:
                    continue
                instance_count = min(metric.num_instances(), correlation_ids.num_instances())
                for idx in range(instance_count):
                    value = metric_value(metric, idx)
                    if value is None or value == 0:
                        continue
                    address = correlation_address(correlation_ids.value(idx))
                    if address is None:
                        skipped_non_address_correlations += 1
                        continue
                    source_info = action.source_info(address)
                    if source_info is None:
                        continue
                    file_name = source_info.file_name()
                    line_no = int(source_info.line())
                    if args.file and args.file not in file_name:
                        continue

                    key = (range_idx, action_idx, file_name, line_no)
                    row = line_map.get(key)
                    if row is None:
                        row = {
                            "range_index": range_idx,
                            "action_index": action_idx,
                            "kernel": action_name,
                            "file": file_name,
                            "line": line_no,
                            "source": source_line(imported_sources, file_name, line_no),
                            "metrics": defaultdict(float),
                            "_pcs": set(),
                            "pc_samples": [],
                        }
                        line_map[key] = row

                    row["metrics"][metric_name] += value
                    if address not in row["_pcs"]:
                        row["_pcs"].add(address)
                        if len(row["pc_samples"]) < args.pc_samples:
                            row["pc_samples"].append(
                                {
                                    "pc": hex(address),
                                    "sass": action.sass_by_pc(address).strip(),
                                }
                            )

    if selected_actions == 0:
        raise SystemExit("error: no actions matched the requested filters")
    if actions_with_imported_source == 0:
        details = "; ".join(missing_source_actions[:5])
        raise SystemExit(
            "error: selected report actions have no imported source content. "
            "Collect with --import-source yes and compile CUDA with -lineinfo. "
            + (f"Examples: {details}" if details else "")
        )
    if missing_metric_errors:
        raise SystemExit("error: requested metrics were not found: " + "; ".join(missing_metric_errors))

    for row in line_map.values():
        metrics = dict(sorted(row["metrics"].items()))
        stall_items = [
            (short_stall_name(name), value)
            for name, value in metrics.items()
            if name.startswith(STALL_PREFIX) and not name.endswith("_not_issued") and value
        ]
        top_stall = None
        if stall_items:
            stall_name, stall_value = max(stall_items, key=lambda item: item[1])
            top_stall = {"name": stall_name, "value": stall_value}
        row_out = {
            key: value
            for key, value in row.items()
            if key not in {"metrics", "_pcs"}
        }
        row_out["pc_count"] = len(row["_pcs"])
        row_out["metrics"] = metrics
        row_out["top_stall"] = top_stall
        payload["lines"].append(row_out)

    if not payload["lines"]:
        raise SystemExit(
            "error: no source-correlated line metric instances were found in selected actions"
        )

    sort_metric = args.sort_metric
    payload["lines"].sort(
        key=lambda row: (
            row["metrics"].get(sort_metric, 0),
            row["metrics"].get("thread_inst_executed", 0),
            row["pc_count"],
        ),
        reverse=True,
    )
    payload["summary"] = {
        "selected_actions": selected_actions,
        "actions_with_imported_source": actions_with_imported_source,
        "line_count": len(payload["lines"]),
        "sort_metric": sort_metric,
        "skipped_non_address_correlations": skipped_non_address_correlations,
    }
    return payload


def write_json(payload: dict[str, Any], output: str) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    if output == "-":
        print(text)
        return
    Path(output).write_text(text + "\n", encoding="utf-8")


def write_csv(payload: dict[str, Any], output: str) -> None:
    metric_names = sorted({name for row in payload["lines"] for name in row["metrics"]})
    fieldnames = [
        "range_index",
        "action_index",
        "kernel",
        "file",
        "line",
        "source",
        "pc_count",
        "top_stall",
        *metric_names,
    ]

    out_file = sys.stdout if output == "-" else open(output, "w", newline="", encoding="utf-8")
    try:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload["lines"]:
            csv_row = {name: row.get(name) for name in fieldnames}
            csv_row["top_stall"] = (
                f"{row['top_stall']['name']}={row['top_stall']['value']}"
                if row.get("top_stall")
                else ""
            )
            for metric_name in metric_names:
                csv_row[metric_name] = row["metrics"].get(metric_name, 0)
            writer.writerow(csv_row)
    finally:
        if out_file is not sys.stdout:
            out_file.close()


def print_summary(payload: dict[str, Any], top: int, sort_metric: str) -> None:
    summary = payload["summary"]
    print(f"Report: {payload['report']}")
    print(
        "Actions: "
        f"{summary['actions_with_imported_source']}/{summary['selected_actions']} "
        "with imported source"
    )
    print(f"Line rows: {summary['line_count']}  sort metric: {sort_metric}")
    print()

    for idx, row in enumerate(payload["lines"][:top], start=1):
        value = row["metrics"].get(sort_metric, 0)
        location = f"{row['file']}:{row['line']}"
        source = (row.get("source") or "").strip()
        if len(source) > 140:
            source = source[:137] + "..."
        stall = ""
        if row.get("top_stall"):
            stall = f"  top_stall={row['top_stall']['name']}:{row['top_stall']['value']:g}"
        print(
            f"{idx:2d}. {sort_metric}={value:g} pcs={row['pc_count']} "
            f"{location} action={row['range_index']}:{row['action_index']} "
            f"kernel={row['kernel']}{stall}"
        )
        if source:
            print(f"    {source}")


def list_metrics(args: argparse.Namespace) -> None:
    ncu_report = import_ncu_report()
    ctx = ncu_report.load_report(Path(args.report))
    seen: set[str] = set()
    for report_range in ctx:
        for action in report_range:
            if args.kernel and args.kernel not in action.name():
                continue
            for name in action.metric_names():
                metric = action.metric_by_name(name)
                if metric is not None and metric.has_correlation_ids() and metric.num_instances() > 0:
                    seen.add(name)
    for name in sorted(seen):
        print(name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract line-level source correlation from an Nsight Compute report."
    )
    parser.add_argument("report", help="Path to .ncu-rep or .ncu-repz")
    parser.add_argument("--kernel", help="Only include actions whose name contains this substring")
    parser.add_argument("--file", help="Only include source files whose path contains this substring")
    parser.add_argument("--metric", action="append", help="Metric to extract. Repeat or comma-separate.")
    parser.add_argument(
        "--all-correlated-metrics",
        action="store_true",
        help="Extract every source-correlated metric instead of the default signal set.",
    )
    parser.add_argument(
        "--sort-metric",
        default="smsp__pcsamp_sample_count",
        help="Metric used for summary ordering.",
    )
    parser.add_argument("--top", type=int, default=20, help="Number of summary rows to print.")
    parser.add_argument("--pc-samples", type=int, default=3, help="Number of PC/SASS samples per line.")
    parser.add_argument("--json", help="Write full JSON payload to this path, or '-' for stdout.")
    parser.add_argument("--csv", help="Write line summary CSV to this path, or '-' for stdout.")
    parser.add_argument("--list-metrics", action="store_true", help="List source-correlated metrics and exit.")
    parser.add_argument("--quiet", action="store_true", help="Do not print the text summary.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_metrics:
        list_metrics(args)
        return 0

    try:
        payload = aggregate_report(args)
    except SystemExit as exc:
        message = str(exc)
        if message:
            print(message, file=sys.stderr)
        return 2

    if args.json:
        write_json(payload, args.json)
    if args.csv:
        write_csv(payload, args.csv)
    if not args.quiet:
        print_summary(payload, args.top, args.sort_metric)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
