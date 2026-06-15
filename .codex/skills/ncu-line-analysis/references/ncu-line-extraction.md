# Nsight Compute Line Extraction Notes

Use the Python Report Interface as the primary extractor. It exposes report structure directly:

- `ncu_report.load_report(path)` returns an `IContext`.
- Iterate `IContext -> IRange -> IAction` for profiled actions.
- `IAction.source_files()` returns file names and embedded source content. If a file maps to an empty string, the source file was not imported into the report.
- `IMetric.has_correlation_ids()` and `IMetric.correlation_ids()` identify per-instance metric values that can be mapped to program counters.
- `IAction.source_info(address)` maps a source-correlated address to file and line.
- `IAction.sass_by_pc(address)` maps the same address to SASS text for context.
- In Nsight Compute 2026.2, `IMetric.value(idx)` can return valid source-page values even when `IMetric.has_value(idx)` is false for that instance. For line extraction, read `value(idx)` directly and skip nonnumeric or zero values.
- Not every correlated metric uses PC addresses as correlation IDs. Opcode/category metrics can use strings such as instruction names; skip those for source-line aggregation unless a later analysis explicitly handles them.

Use the CLI source page as a sanity check:

```bash
ncu --import report.ncu-rep --page source --print-source cuda,sass --csv
```

Important collection prerequisites:

- Source correlation requires line info from the CUDA build, usually `-lineinfo`.
- Imported source text requires profiling with `--import-source yes` when source files are available to the profiler. `--source-folders` can provide extra recursive search paths for missing source files at collection time.
- The CLI `--page source` output can show SASS/source correlation and metrics, but parsing it is less stable than using the Python API.

## BSSY/BSYNC review

Use `scripts/analyze_branch_sync.py` when reviewing compiler-emitted branch synchronization:

1. Scan the actual source files from `--repo-root` for control points.
2. Enumerate all emitted `BSSY` instructions.
3. Match each `BSSY` to a nearby `BSYNC` with the same barrier token when possible.
4. Attach each `BSSY` to the source control point whose source span contains the NCU-correlated line.
5. Review every source control point with attached `BSSY` for whether its predicate can differ within the hardware warp.
6. Treat lane-rank and sub-warp predicates as legitimate until proven otherwise.
7. If the predicate is logically identical for all 32 lanes, inspect whether nvMolKit's repo-specific `mark_warp_uniform` inline helper should be applied at the value definition that feeds that branch.

`mark_warp_uniform` is nvMolKit code, not an Nsight Compute or compiler directive. In this repo it is defined in `src/forcefields/kernel_utils.cuh` and implemented with a warp shuffle broadcast.

Official docs:

- Nsight Compute CLI command options: https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#command-line-options
- Python Report Interface: https://docs.nvidia.com/nsight-compute/PythonReportInterface/index.html
