#!/usr/bin/env python3
"""Map BSSY/BSYNC SASS instructions onto source control points."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from extract_line_info import correlation_address, import_ncu_report, metric_value


PC_METRICS = (
    "inst_executed",
    "thread_inst_executed",
    "thread_inst_executed_true",
    "smsp__pcsamp_sample_count",
)

SYNC_RE = re.compile(r"\b(BSSY|BSYNC)\s+B(\d+)(?:,\s*(0x[0-9a-fA-F]+))?")
CONTROL_HEAD_RE = re.compile(r"\b(if|for|while|switch)\s*\(|\bdo\b")
TRANSFER_RE = re.compile(r"\b(return|break|continue)\b")
TERNARY_RE = re.compile(r"\?")
CONTROL_KIND = {
    "if": "branch",
    "for": "loop",
    "while": "loop",
    "switch": "switch",
    "do": "loop",
    "return": "transfer",
    "break": "transfer",
    "continue": "transfer",
    "?:": "ternary",
}


class SourceResolver:
    def __init__(self, repo_root: Path | None, imported_sources: dict[str, list[str]]):
        self.repo_root = repo_root
        self.imported_sources = imported_sources
        self._local_cache: dict[str, list[str] | None] = {}
        self._path_cache: dict[str, Path | None] = {}

    def local_path(self, file_name: str) -> Path | None:
        cached = self._path_cache.get(file_name)
        if file_name in self._path_cache:
            return cached

        path = Path(file_name)
        candidates: list[Path] = []
        if path.is_absolute() and path.is_file():
            candidates.append(path)
        if self.repo_root is not None:
            parts = path.parts
            if "nvmolkit" in parts:
                idx = parts.index("nvmolkit")
                candidates.append(self.repo_root.joinpath(*parts[idx + 1 :]))
            if not path.is_absolute():
                candidates.append(self.repo_root / path)

        found = None
        for candidate in candidates:
            if candidate.is_file():
                found = candidate.resolve()
                break
        self._path_cache[file_name] = found
        return found

    def repo_relative(self, file_name: str) -> str | None:
        if self.repo_root is None:
            return None
        path = self.local_path(file_name)
        if path is None:
            return None
        try:
            return path.relative_to(self.repo_root).as_posix()
        except ValueError:
            return None

    def file_key(self, file_name: str | None) -> str | None:
        if file_name is None:
            return None
        return self.repo_relative(file_name) or file_name

    def source_status(self, file_name: str | None) -> str:
        if file_name is None:
            return "no_source"
        if self.repo_relative(file_name) is not None:
            return "repo_source"
        if file_name in self.imported_sources:
            return "imported_external_source"
        return "external_source"

    def lines(self, file_name: str) -> list[str] | None:
        if file_name not in self._local_cache:
            path = self.local_path(file_name)
            if path is None:
                self._local_cache[file_name] = None
            else:
                self._local_cache[file_name] = path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
        return self._local_cache[file_name] or self.imported_sources.get(file_name)

    def line(self, file_name: str | None, line_no: int | None) -> str | None:
        if not file_name or not line_no:
            return None
        lines = self.lines(file_name)
        if not lines or line_no < 1 or line_no > len(lines):
            return None
        return lines[line_no - 1].rstrip()

    def context(
        self, file_name: str | None, line_no: int | None, radius: int
    ) -> list[dict[str, Any]]:
        if not file_name or not line_no or radius <= 0:
            return []
        lines = self.lines(file_name)
        if not lines:
            return []
        start = max(1, line_no - radius)
        end = min(len(lines), line_no + radius)
        return [
            {"line": idx, "source": lines[idx - 1].rstrip()}
            for idx in range(start, end + 1)
        ]


def strip_comments_and_strings(lines: list[str]) -> list[str]:
    stripped: list[str] = []
    in_block_comment = False
    quote: str | None = None
    escaped = False

    for line in lines:
        out: list[str] = []
        idx = 0
        while idx < len(line):
            ch = line[idx]
            nxt = line[idx + 1] if idx + 1 < len(line) else ""

            if in_block_comment:
                if ch == "*" and nxt == "/":
                    in_block_comment = False
                    out.extend("  ")
                    idx += 2
                else:
                    out.append(" ")
                    idx += 1
                continue

            if quote is not None:
                out.append(" ")
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == quote:
                    quote = None
                idx += 1
                continue

            if ch == "/" and nxt == "*":
                in_block_comment = True
                out.extend("  ")
                idx += 2
            elif ch == "/" and nxt == "/":
                out.extend(" " * (len(line) - idx))
                break
            elif ch in {"'", '"'}:
                quote = ch
                out.append(" ")
                idx += 1
            else:
                out.append(ch)
                idx += 1

        stripped.append("".join(out))
    return stripped


def find_matching_token(
    lines: list[str], start_line: int, start_col: int, open_ch: str, close_ch: str
) -> tuple[int, int] | None:
    depth = 0
    for line_idx in range(start_line, len(lines)):
        col = start_col if line_idx == start_line else 0
        while col < len(lines[line_idx]):
            ch = lines[line_idx][col]
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return line_idx, col
            col += 1
    return None


def find_first_body_brace(
    lines: list[str], start_line: int, start_col: int
) -> tuple[int, int] | None:
    for line_idx in range(start_line, min(len(lines), start_line + 24)):
        col = start_col if line_idx == start_line else 0
        segment = lines[line_idx][col:]
        brace = segment.find("{")
        semicolon = segment.find(";")
        if brace >= 0 and (semicolon < 0 or brace < semicolon):
            return line_idx, col + brace
        if semicolon >= 0 and (brace < 0 or semicolon < brace):
            return None
    return None


def find_statement_end(lines: list[str], start_line: int) -> int:
    for line_idx in range(start_line, min(len(lines), start_line + 24)):
        if ";" in lines[line_idx]:
            return line_idx
    return start_line


def skip_blank_lines(lines: list[str], start_line: int) -> int:
    line_idx = start_line
    while line_idx < len(lines) and not lines[line_idx].strip():
        line_idx += 1
    return line_idx


def extend_if_else_span(lines: list[str], end_line: int) -> int:
    next_line = skip_blank_lines(lines, end_line + 1)
    if next_line >= len(lines):
        return end_line
    if not lines[next_line].lstrip().startswith("else"):
        return end_line

    brace = find_first_body_brace(lines, next_line, lines[next_line].find("else") + 4)
    if brace is not None:
        match = find_matching_token(lines, brace[0], brace[1], "{", "}")
        return match[0] if match is not None else end_line
    return find_statement_end(lines, next_line)


def control_condition_text(
    original: list[str], start_line: int, end_line: int, start_col: int = 0
) -> str:
    snippets = []
    for line_idx in range(start_line, min(end_line + 1, len(original))):
        text = original[line_idx]
        if line_idx == start_line:
            text = text[start_col:]
        snippets.append(text.strip())
    return " ".join(part for part in snippets if part)


def make_branch_point(
    file_name: str,
    repo_file: str | None,
    index: int,
    keyword: str,
    start_line: int,
    end_line: int,
    condition_end_line: int,
    source: str,
    condition_text: str,
) -> dict[str, Any]:
    file_key = repo_file or file_name
    return {
        "id": f"{file_key}:{start_line}:{keyword}:{index}",
        "file": file_name,
        "repo_file": repo_file,
        "file_key": file_key,
        "line": start_line,
        "end_line": end_line,
        "kind": CONTROL_KIND[keyword],
        "keyword": keyword,
        "source": source.rstrip(),
        "condition_text": condition_text,
        "condition_end_line": condition_end_line,
        "mapped_bssy_count": 0,
        "mapped_bssy": [],
    }


def scan_source_control_points(
    file_name: str, repo_file: str | None, original_lines: list[str]
) -> list[dict[str, Any]]:
    code_lines = strip_comments_and_strings(original_lines)
    points: list[dict[str, Any]] = []

    for line_idx, code in enumerate(code_lines):
        search_col = 0
        while True:
            match = CONTROL_HEAD_RE.search(code, search_col)
            if match is None:
                break
            keyword = match.group(1) or "do"
            start_line = line_idx
            start_col = match.start()
            condition_end_line = line_idx
            end_line = line_idx

            if keyword == "do":
                brace = find_first_body_brace(code_lines, line_idx, match.end())
                if brace is not None:
                    brace_match = find_matching_token(code_lines, brace[0], brace[1], "{", "}")
                    end_line = brace_match[0] if brace_match is not None else line_idx
                    while_line = skip_blank_lines(code_lines, end_line + 1)
                    if while_line < len(code_lines) and "while" in code_lines[while_line]:
                        end_line = find_statement_end(code_lines, while_line)
                else:
                    end_line = find_statement_end(code_lines, line_idx)
            else:
                paren_col = code_lines[line_idx].find("(", match.start())
                paren_match = find_matching_token(code_lines, line_idx, paren_col, "(", ")")
                if paren_match is not None:
                    condition_end_line = paren_match[0]
                    brace = find_first_body_brace(
                        code_lines, paren_match[0], paren_match[1] + 1
                    )
                    if brace is not None:
                        brace_match = find_matching_token(
                            code_lines, brace[0], brace[1], "{", "}"
                        )
                        end_line = brace_match[0] if brace_match is not None else condition_end_line
                    else:
                        end_line = find_statement_end(code_lines, condition_end_line)
                    if keyword == "if":
                        end_line = extend_if_else_span(code_lines, end_line)

            points.append(
                make_branch_point(
                    file_name=file_name,
                    repo_file=repo_file,
                    index=len(points),
                    keyword=keyword,
                    start_line=start_line + 1,
                    end_line=end_line + 1,
                    condition_end_line=condition_end_line + 1,
                    source=original_lines[start_line] if start_line < len(original_lines) else "",
                    condition_text=control_condition_text(
                        original_lines, start_line, condition_end_line, start_col
                    ),
                )
            )
            search_col = match.end()

        if CONTROL_HEAD_RE.search(code):
            continue

        transfer = TRANSFER_RE.search(code)
        if transfer is not None:
            keyword = transfer.group(1)
            start_line = line_idx
            end_line = find_statement_end(code_lines, line_idx)
            points.append(
                make_branch_point(
                    file_name=file_name,
                    repo_file=repo_file,
                    index=len(points),
                    keyword=keyword,
                    start_line=start_line + 1,
                    end_line=end_line + 1,
                    condition_end_line=start_line + 1,
                    source=original_lines[start_line] if start_line < len(original_lines) else "",
                    condition_text=control_condition_text(
                        original_lines, start_line, start_line, transfer.start()
                    ),
                )
            )
            continue

        ternary = TERNARY_RE.search(code)
        if ternary is not None:
            start_line = line_idx
            end_line = find_statement_end(code_lines, line_idx)
            points.append(
                make_branch_point(
                    file_name=file_name,
                    repo_file=repo_file,
                    index=len(points),
                    keyword="?:",
                    start_line=start_line + 1,
                    end_line=end_line + 1,
                    condition_end_line=start_line + 1,
                    source=original_lines[start_line] if start_line < len(original_lines) else "",
                    condition_text=control_condition_text(
                        original_lines, start_line, start_line, 0
                    ),
                )
            )
    return points


def parse_sync(sass: str) -> dict[str, Any] | None:
    match = SYNC_RE.search(sass)
    if not match:
        return None
    target = int(match.group(3), 16) if match.group(3) else None
    return {"op": match.group(1), "barrier": f"B{match.group(2)}", "target": target}


def selected_metric_names(action: Any) -> list[str]:
    available = set(action.metric_names())
    selected = [name for name in PC_METRICS if name in available]
    if selected:
        return selected

    fallback: list[str] = []
    for name in action.metric_names():
        metric = action.metric_by_name(name)
        if metric is not None and metric.has_correlation_ids() and metric.num_instances() > 0:
            fallback.append(name)
    return fallback


def collect_instructions(
    action: Any, resolver: SourceResolver, file_filter: str | None
) -> tuple[list[dict[str, Any]], int]:
    pc_metrics: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    pcs: set[int] = set()
    skipped_non_address = 0

    for metric_name in selected_metric_names(action):
        metric = action.metric_by_name(metric_name)
        if metric is None or not metric.has_correlation_ids():
            continue
        correlation_ids = metric.correlation_ids()
        if correlation_ids is None:
            continue
        instance_count = min(metric.num_instances(), correlation_ids.num_instances())
        for idx in range(instance_count):
            address = correlation_address(correlation_ids.value(idx))
            if address is None:
                skipped_non_address += 1
                continue
            pcs.add(address)
            value = metric_value(metric, idx)
            if value is not None:
                pc_metrics[address][metric_name] += float(value)

    instructions: list[dict[str, Any]] = []
    for pc in sorted(pcs):
        sass = (action.sass_by_pc(pc) or "").strip()
        if not sass:
            continue
        source_info = action.source_info(pc)
        file_name = source_info.file_name() if source_info is not None else None
        line_no = int(source_info.line()) if source_info is not None else None
        if file_filter and file_name and file_filter not in file_name:
            continue
        instructions.append(
            {
                "pc": pc,
                "pc_hex": hex(pc),
                "sass": sass,
                "file": file_name,
                "repo_file": resolver.repo_relative(file_name) if file_name else None,
                "file_key": resolver.file_key(file_name),
                "line": line_no,
                "source": resolver.line(file_name, line_no),
                "source_status": resolver.source_status(file_name),
                "metrics": dict(sorted(pc_metrics.get(pc, {}).items())),
            }
        )
    return instructions, skipped_non_address


def instruction_ref(inst: dict[str, Any] | None) -> dict[str, Any] | None:
    if inst is None:
        return None
    return {
        "pc": inst["pc_hex"],
        "sass": inst["sass"],
        "file": inst.get("file"),
        "repo_file": inst.get("repo_file"),
        "line": inst.get("line"),
        "source": inst.get("source"),
        "source_status": inst.get("source_status"),
        "metrics": inst.get("metrics", {}),
    }


def find_matching_bsync(
    instructions: list[dict[str, Any]], start_idx: int, barrier: str, target: int | None
) -> int | None:
    pc_values = [inst["pc"] for inst in instructions]
    if target is not None:
        target_idx = bisect.bisect_right(pc_values, target)
        for idx in range(target_idx - 1, start_idx, -1):
            sync = parse_sync(instructions[idx]["sass"])
            if sync and sync["op"] == "BSYNC" and sync["barrier"] == barrier:
                return idx
    for idx in range(start_idx + 1, len(instructions)):
        sync = parse_sync(instructions[idx]["sass"])
        if sync and sync["op"] == "BSYNC" and sync["barrier"] == barrier:
            return idx
    return None


def ensure_branch_points(
    branch_points_by_file: dict[str, list[dict[str, Any]]],
    resolver: SourceResolver,
    file_name: str,
    include_external: bool,
) -> None:
    file_key = resolver.file_key(file_name)
    if file_key is None or file_key in branch_points_by_file:
        return
    repo_file = resolver.repo_relative(file_name)
    if repo_file is None and not include_external:
        branch_points_by_file[file_key] = []
        return
    lines = resolver.lines(file_name)
    if not lines:
        branch_points_by_file[file_key] = []
        return
    branch_points_by_file[file_key] = scan_source_control_points(file_name, repo_file, lines)


def map_instruction_to_branch(
    branch_points_by_file: dict[str, list[dict[str, Any]]],
    inst: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    file_key = inst.get("file_key")
    line_no = inst.get("line")
    if not file_key or not line_no:
        return None, "no_source_line"

    candidates = [
        point
        for point in branch_points_by_file.get(file_key, [])
        if point["line"] <= line_no <= point["end_line"]
    ]
    if not candidates:
        return None, "unmapped_to_source_control_point"

    direct = [
        point
        for point in candidates
        if point["line"] <= line_no <= point["condition_end_line"]
    ]
    if direct:
        direct.sort(key=lambda point: (point["end_line"] - point["line"], point["id"]))
        return direct[0], "control_condition_line"

    candidates.sort(key=lambda point: (point["end_line"] - point["line"], -point["line"]))
    return candidates[0], "inside_control_span"


def build_bssy_row(
    range_idx: int,
    action_idx: int,
    action_name: str,
    instructions: list[dict[str, Any]],
    bssy_idx: int,
    branch_points_by_file: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    bssy_inst = instructions[bssy_idx]
    sync = parse_sync(bssy_inst["sass"])
    assert sync is not None
    bsync_idx = find_matching_bsync(
        instructions, bssy_idx, sync["barrier"], sync["target"]
    )
    branch_point, mapping = map_instruction_to_branch(branch_points_by_file, bssy_inst)

    row = {
        "range_index": range_idx,
        "action_index": action_idx,
        "kernel": action_name,
        "bssy": instruction_ref(bssy_inst),
        "barrier": sync["barrier"],
        "bssy_target": hex(sync["target"]) if sync["target"] is not None else None,
        "bsync": instruction_ref(instructions[bsync_idx]) if bsync_idx is not None else None,
        "branch_point_id": branch_point["id"] if branch_point else None,
        "branch_mapping": mapping,
    }
    if branch_point is not None:
        branch_point["mapped_bssy_count"] += 1
        if len(branch_point["mapped_bssy"]) < 16:
            branch_point["mapped_bssy"].append(
                {
                    "range_index": range_idx,
                    "action_index": action_idx,
                    "kernel": action_name,
                    "pc": bssy_inst["pc_hex"],
                    "sass": bssy_inst["sass"],
                    "line": bssy_inst.get("line"),
                    "mapping": mapping,
                }
            )
    return row


def flatten_branch_points(
    branch_points_by_file: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for file_key in sorted(branch_points_by_file):
        points.extend(
            sorted(
                branch_points_by_file[file_key],
                key=lambda point: (point["line"], point["end_line"], point["id"]),
            )
        )
    return points


def analyze_report(args: argparse.Namespace) -> dict[str, Any]:
    ncu_report = import_ncu_report()
    report_path = Path(args.report)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else None
    ctx = ncu_report.load_report(report_path)

    branch_points_by_file: dict[str, list[dict[str, Any]]] = {}
    bssy_rows: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    selected_actions = 0
    actions_with_imported_source = 0
    skipped_non_address_correlations = 0
    missing_source_actions: list[str] = []

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
            resolver = SourceResolver(repo_root, imported_sources)
            for file_name in imported_sources:
                if args.file and args.file not in file_name:
                    continue
                ensure_branch_points(
                    branch_points_by_file,
                    resolver,
                    file_name,
                    args.include_external,
                )

            instructions, skipped = collect_instructions(action, resolver, args.file)
            skipped_non_address_correlations += skipped
            action_rows_before = len(bssy_rows)

            for idx, inst in enumerate(instructions):
                if inst.get("file"):
                    ensure_branch_points(
                        branch_points_by_file,
                        resolver,
                        inst["file"],
                        args.include_external,
                    )
                sync = parse_sync(inst["sass"])
                if sync and sync["op"] == "BSSY":
                    bssy_rows.append(
                        build_bssy_row(
                            range_idx,
                            action_idx,
                            action_name,
                            instructions,
                            idx,
                            branch_points_by_file,
                        )
                    )

            actions.append(
                {
                    "range_index": range_idx,
                    "action_index": action_idx,
                    "name": action_name,
                    "source_files": [
                        {
                            "file": file_name,
                            "imported": bool(contents),
                            "line_count": len(contents.splitlines()) if contents else 0,
                            "repo_file": resolver.repo_relative(file_name),
                        }
                        for file_name, contents in sorted(raw_sources.items())
                    ],
                    "instruction_count": len(instructions),
                    "bssy_count": len(bssy_rows) - action_rows_before,
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
    if not bssy_rows:
        raise SystemExit("error: no BSSY instructions were found in selected actions")

    branch_points = flatten_branch_points(branch_points_by_file)
    mapped_bssy = sum(1 for row in bssy_rows if row["branch_point_id"])
    repo_bssy = sum(
        1 for row in bssy_rows if (row.get("bssy") or {}).get("source_status") == "repo_source"
    )
    payload = {
        "report": str(report_path),
        "repo_root": str(repo_root) if repo_root else None,
        "ncu_report_version": ctx.get_version(),
        "actions": actions,
        "branch_points": branch_points,
        "bssy_rows": bssy_rows,
        "unmapped_bssy_rows": [
            row for row in bssy_rows if row["branch_point_id"] is None
        ],
        "summary": {
            "selected_actions": selected_actions,
            "actions_with_imported_source": actions_with_imported_source,
            "branch_point_count": len(branch_points),
            "branch_points_with_bssy": sum(
                1 for point in branch_points if point["mapped_bssy_count"]
            ),
            "bssy_count": len(bssy_rows),
            "repo_source_bssy_count": repo_bssy,
            "bssy_mapped_to_branch_points": mapped_bssy,
            "bssy_unmapped_to_branch_points": len(bssy_rows) - mapped_bssy,
            "skipped_non_address_correlations": skipped_non_address_correlations,
        },
    }
    return payload


def write_json(payload: dict[str, Any], output: str) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    if output == "-":
        print(text)
        return
    Path(output).write_text(text + "\n", encoding="utf-8")


def write_branch_csv(payload: dict[str, Any], output: str) -> None:
    fieldnames = [
        "file",
        "repo_file",
        "line",
        "end_line",
        "kind",
        "keyword",
        "source",
        "condition_text",
        "mapped_bssy_count",
        "mapped_bssy_pcs",
    ]
    out_file = sys.stdout if output == "-" else open(output, "w", newline="", encoding="utf-8")
    try:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        for point in payload["branch_points"]:
            row = {name: point.get(name) for name in fieldnames}
            row["mapped_bssy_pcs"] = " ".join(
                item["pc"] for item in point.get("mapped_bssy", [])
            )
            writer.writerow(row)
    finally:
        if out_file is not sys.stdout:
            out_file.close()


def metric_field(row: dict[str, Any], name: str) -> float:
    metrics = (row.get("bssy") or {}).get("metrics") or {}
    return float(metrics.get(name, 0))


def write_sync_csv(payload: dict[str, Any], output: str) -> None:
    fieldnames = [
        "range_index",
        "action_index",
        "kernel",
        "bssy_pc",
        "barrier",
        "bssy_target",
        "bssy_sass",
        "bssy_file",
        "bssy_repo_file",
        "bssy_line",
        "bssy_source",
        "bssy_source_status",
        "bsync_pc",
        "bsync_sass",
        "bsync_file",
        "bsync_repo_file",
        "bsync_line",
        "bsync_source",
        "branch_point_id",
        "branch_mapping",
        *PC_METRICS,
    ]
    out_file = sys.stdout if output == "-" else open(output, "w", newline="", encoding="utf-8")
    try:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload["bssy_rows"]:
            bssy = row.get("bssy") or {}
            bsync = row.get("bsync") or {}
            csv_row = {
                "range_index": row["range_index"],
                "action_index": row["action_index"],
                "kernel": row["kernel"],
                "bssy_pc": bssy.get("pc"),
                "barrier": row.get("barrier"),
                "bssy_target": row.get("bssy_target"),
                "bssy_sass": bssy.get("sass"),
                "bssy_file": bssy.get("file"),
                "bssy_repo_file": bssy.get("repo_file"),
                "bssy_line": bssy.get("line"),
                "bssy_source": bssy.get("source"),
                "bssy_source_status": bssy.get("source_status"),
                "bsync_pc": bsync.get("pc"),
                "bsync_sass": bsync.get("sass"),
                "bsync_file": bsync.get("file"),
                "bsync_repo_file": bsync.get("repo_file"),
                "bsync_line": bsync.get("line"),
                "bsync_source": bsync.get("source"),
                "branch_point_id": row.get("branch_point_id"),
                "branch_mapping": row.get("branch_mapping"),
            }
            for metric_name in PC_METRICS:
                csv_row[metric_name] = metric_field(row, metric_name)
            writer.writerow(csv_row)
    finally:
        if out_file is not sys.stdout:
            out_file.close()


def print_summary(payload: dict[str, Any], print_branches: bool) -> None:
    summary = payload["summary"]
    print(f"Report: {payload['report']}")
    print(
        "Actions: "
        f"{summary['actions_with_imported_source']}/{summary['selected_actions']} "
        "with imported source"
    )
    print(
        f"Source control points: {summary['branch_point_count']}  "
        f"with BSSY: {summary['branch_points_with_bssy']}"
    )
    print(
        f"BSSY instructions: {summary['bssy_count']}  "
        f"repo-source BSSY: {summary['repo_source_bssy_count']}  "
        f"mapped to source control points: {summary['bssy_mapped_to_branch_points']}  "
        f"unmapped: {summary['bssy_unmapped_to_branch_points']}"
    )

    if not print_branches:
        return
    print()
    print("Source control points with mapped BSSY:")
    for point in payload["branch_points"]:
        if not point["mapped_bssy_count"]:
            continue
        source = point["source"].strip()
        if len(source) > 120:
            source = source[:117] + "..."
        print(
            f"{point['repo_file'] or point['file']}:{point['line']}-{point['end_line']} "
            f"{point['keyword']} bssy={point['mapped_bssy_count']}"
        )
        if source:
            print(f"    {source}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map emitted BSSY/BSYNC instructions onto actual source control points."
    )
    parser.add_argument("report", help="Path to .ncu-rep or .ncu-repz")
    parser.add_argument("--repo-root", help="Local repo root used to read matching source files")
    parser.add_argument("--kernel", help="Only include actions whose name contains this substring")
    parser.add_argument("--file", help="Only include source files whose path contains this substring")
    parser.add_argument(
        "--include-external",
        action="store_true",
        help="Also scan imported non-repo source files for control points",
    )
    parser.add_argument(
        "--context-radius",
        type=int,
        default=4,
        help="Reserved for compatibility; JSON control rows include their full source span",
    )
    parser.add_argument("--json", help="Write full JSON payload to this path, or '-' for stdout")
    parser.add_argument("--csv", help="Write source control point CSV to this path, or '-' for stdout")
    parser.add_argument("--sync-csv", help="Write raw BSSY instruction CSV to this path")
    parser.add_argument("--print-branches", action="store_true", help="Print mapped source control points")
    parser.add_argument("--quiet", action="store_true", help="Do not print the text summary")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        payload = analyze_report(args)
    except SystemExit as exc:
        message = str(exc)
        if message:
            print(message, file=sys.stderr)
        return 2
    if args.json:
        write_json(payload, args.json)
    if args.csv:
        write_branch_csv(payload, args.csv)
    if args.sync_csv:
        write_sync_csv(payload, args.sync_csv)
    if not args.quiet:
        print_summary(payload, args.print_branches)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
