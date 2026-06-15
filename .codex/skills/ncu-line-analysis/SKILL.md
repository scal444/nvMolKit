---
name: ncu-line-analysis
description: Extract and analyze source-line information from NVIDIA Nsight Compute `.ncu-rep` and `.ncu-repz` reports. Use when Codex needs to inspect NCU source correlation, imported CUDA source files, line-level SASS metrics, PC sampling stalls, instruction counts, BSSY/BSYNC branch synchronization, or to fail fast when a report lacks imported line/source information.
---

# NCU Line Analysis

## Overview

Use this skill to extract source-correlated line metrics from Nsight Compute reports and map SASS instructions back to CUDA/C++ source. Prefer the bundled scripts for repeatable extraction and use the CLI source page as a cross-check.

## Quick Start

Run the extractor from the skill directory or pass its absolute path:

```bash
python .codex/skills/ncu-line-analysis/scripts/extract_line_info.py path/to/report.ncu-rep
```

Write machine-readable output for deeper analysis:

```bash
python .codex/skills/ncu-line-analysis/scripts/extract_line_info.py report.ncu-rep --json /tmp/ncu-lines.json --csv /tmp/ncu-lines.csv
```

Map compiler-emitted branch synchronization:

```bash
python .codex/skills/ncu-line-analysis/scripts/analyze_branch_sync.py report.ncu-rep --repo-root /path/to/nvmolkit --json /tmp/ncu-branch-sync.json --csv /tmp/ncu-branch-points.csv --sync-csv /tmp/ncu-bssy.csv
```

The scripts fail if the selected report/actions do not include embedded imported source content or if no source-correlated metric instances can be mapped to file and line.

## Line Metric Workflow

1. Load the report with the Python Report Interface (`ncu_report`).
2. Select actions by kernel substring if requested.
3. Require at least one selected action with non-empty `IAction.source_files()` content. Empty source content means the report has line correlation but not imported source text.
4. Use source-correlated metric correlation IDs, then call `IAction.source_info(address)` to map each PC to file and line.
5. Aggregate metric values by `(range, action, file, line)`.
6. Sort the line metric summary by `smsp__pcsamp_sample_count` by default and report the source text, PC count, and leading stall reason.

## Branch Sync Workflow

Use `scripts/analyze_branch_sync.py` for BSSY/BSYNC analysis.

1. Scan the actual source files from `--repo-root` for control points (`if`, loops, `switch`, ternary, and control-transfer statements).
2. Enumerate every emitted `BSSY` instruction in the selected report actions.
3. Match each `BSSY` to the corresponding `BSYNC` when the barrier token and target address allow it.
4. Attach each `BSSY` to the source control point whose source span contains the NCU-correlated line. Report any `BSSY` that does not attach to a repo source control point.
5. For every source control point with attached `BSSY`, reason from the source predicate and surrounding data flow about whether the predicate can vary across lanes in the hardware warp.
6. Treat lane-owner, thread-rank, and sub-warp predicates as legitimate until the active group width and predicate are proven full-warp uniform.
7. If the source predicate is logically identical for all 32 lanes and a `BSSY` is still emitted, inspect whether nvMolKit's repo-specific `mark_warp_uniform` helper should be applied at the value definition that feeds the branch.

## Default Signals

The default extraction includes:

- PC sampling count: `smsp__pcsamp_sample_count`
- instruction counts: `inst_executed`, `thread_inst_executed`, `thread_inst_executed_true`
- PC sampling stall reason counters: `smsp__pcsamp_warps_issue_stalled_*`

Use `--metric` to override the metric set, `--all-correlated-metrics` to include every source-correlated metric, and `--list-metrics` to discover available source-correlated names.

## Verification

Use the Nsight Compute CLI source page to cross-check that a report has source output:

```bash
ncu --import report.ncu-rep --page source --print-source cuda,sass --csv
```

If this prints source/SASS rows but the Python script fails, inspect `references/ncu-line-extraction.md` and compare the installed `ncu` version with the Python Report Interface version being imported.

## Resources

- `scripts/extract_line_info.py`: Extract and summarize line-level metrics from `.ncu-rep` or `.ncu-repz`.
- `scripts/analyze_branch_sync.py`: Inventory `BSSY`/`BSYNC` instructions and map them to source branch context.
- `references/ncu-line-extraction.md`: Notes on the researched NCU APIs and CLI options.
