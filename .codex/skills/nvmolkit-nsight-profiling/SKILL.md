---
name: nvmolkit-nsight-profiling
description: Profile local nvMolKit CUDA workloads with NVIDIA Nsight Systems (nsys), including user-run sudo commands, supported capture-range selection, NVTX stage analysis, and report inspection. Use for nsys/Nsight Systems profiling requests; use ncu-line-analysis instead for Nsight Compute source-line reports.
---

# nvMolKit Nsight Systems Profiling

Use this skill with `nvmolkit-gpu-execution` for local GPU execution rules.

## Preserve the execution workflow

Before launching a profiler, determine who is supposed to execute it. If the
user asks for a command to run themselves, asks for a sudo command, or has
already established that privileged GPU commands must be handed to them:

- Do not execute the profiler, benchmark, CUDA probe, or preliminary GPU
  command yourself.
- Provide the complete command and wait for the user to report completion.
- Do not request tool approval as a substitute for the requested handoff.
- After completion, inspect the generated report with read-only, non-GPU tools.

Do not assume that an externally executed command ran in the Codex app's
attached terminal. `read_thread_terminal` can only recover output from a
terminal actually attached to the current task. For commands run in another
terminal, output is unavailable unless the command persists it or the user
pastes it.

## Make unattended handoffs observable

When handing the user a long-running command, include durable artifacts from
the outset:

- Send stdout and stderr to a known log path while still displaying them, for
  example with `2>&1 | tee /tmp/<name>.log` under pipefail semantics.
- Use benchmark output options such as `--output /tmp/<name>.csv` when
  available.
- Preserve the command's exit status in a known status file; a generated
  `.nsys-rep` alone does not prove application success.
- Give reports, logs, CSV files, and status markers unique absolute paths so a
  stale prior run cannot be mistaken for the current one.

After sending off an unattended command, arrange a non-blocking check every
five minutes. Prefer the product's heartbeat/automation or wait mechanism;
do not occupy a shell with a long blocking `sleep`. Each check should inspect
the known process, log, status marker, and report paths. Stay quiet while the
run is unchanged, and resume analysis when it completes, fails, or needs user
action. If no monitoring mechanism is available, say so before the user starts
the run rather than implying that the task can observe it.

For a user-run nvMolKit Python profile, make path and environment selection
unambiguous:

- Start from a directory outside the repository, normally `cd /tmp`.
- Use `sudo -E` when the user requests a sudo-capable command.
- Use absolute paths for `nsys`, the Python executable, benchmark script, input
  data, and report output.
- Use `/home/kboyd/miniforge3/envs/rdcu_dev/bin/python` for the local
  `rdcu_dev` Python environment.
- Prefer a report prefix under `/tmp` and pass `--force-overwrite=true` when a
  repeatable overwrite is intended.

## Select a capture mechanism the target supports

Inspect the benchmark or driver before composing the command. Never add a
capture-range mode merely because Nsight Systems supports it.

1. If the target brackets the intended region with `cudaProfilerStart()` and
   `cudaProfilerStop()` (including calls through `torch.cuda.cudart()`), use:

   ```text
   -c cudaProfilerApi
   ```

2. If the target instead provides a deliberate NVTX start range, use:

   ```text
   -c nvtx --nvtx-capture=<range-name>[@<domain>]
   ```

   Supply the exact range and domain implemented by the target. An ordinary
   NVTX annotation is useful for attribution but does not automatically make a
   correct capture start range; confirm that the chosen range encloses all
   intended work.

3. If neither mechanism is implemented, omit `-c` and capture the process, or
   first add an appropriate bracket when changing the benchmark is in scope.

Using the target's supported capture range excludes unrelated setup such as
data loading and fingerprint generation and keeps the report focused. Using an
unsupported range can capture nothing or wait forever, so fail fast during
command review rather than guessing.

Match capture termination to the benchmark structure. For the usual single
timed BitBIRCH run, use `--runs 1 --warmups 0`, allowing one bracketed region
to define the capture. If multiple bracketed runs are intentionally required,
configure the appropriate `--capture-range-end` repeat behavior rather than
silently recording only the first region.

## BitBIRCH example

The BitBIRCH benchmark currently calls the CUDA Profiler API around the
nvMolKit timing region, so its command should include `-c cudaProfilerApi`:

```bash
cd /tmp && sudo -E /usr/local/bin/nsys profile \
  -c cudaProfilerApi \
  --trace=cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --cuda-memory-usage=true \
  --force-overwrite=true \
  --output=/tmp/bitbirch_profile \
  /home/kboyd/miniforge3/envs/rdcu_dev/bin/python \
  /home/kboyd/omg/repos/nvmolkit/benchmarks/bitbirch_clustering_bench.py \
  --smiles /absolute/path/to/input.cxsmiles \
  --num-mols 1000000 \
  --threshold 0.25 \
  --runs 1 \
  --warmups 0 \
  --no-bblean
```

Keep NVTX tracing enabled because nvMolKit uses named ranges to attribute time
to partial-tree construction, merge rounds, and forest finalization. Disable
CPU sampling and context-switch tracing when the question is GPU-stage
attribution; enable them only when investigating host-side gaps.

## Diagnose the target before the trace

Treat the profiled program's outcome as the first diagnostic. Preserve and
inspect its stdout, stderr, and exit status before interpreting report tables.
For a command the user runs, ask for that terminal output if it was not already
provided.

An application failure can yield a structurally readable `.nsys-rep` with
NVTX and CUDA API events but no kernel table. For example, a CUDA out-of-memory
error during workspace allocation can end the capture before kernels launch.
In that situation, debug the application failure; do not label the report
incomplete or prescribe profiler flushing merely because kernels are absent.

Warnings such as "Not all CUDA events might have been collected" are not, by
themselves, proof of a collection failure. Consider flushing or profiler
configuration changes only after confirming that the target completed the
intended workload successfully and the expected events are still missing.

## Report analysis

Once the user has produced the report, verify the exact `.nsys-rep` path. Use
`nsys stats` or export to SQLite for read-only analysis; neither operation
should launch CUDA work. Attribute elapsed time using both NVTX ranges and CUDA
kernel/API summaries, and distinguish:

- GPU kernel execution;
- CUDA allocation, copies, and synchronization;
- host gaps inside an NVTX stage;
- setup outside the selected capture range.

Report the dominant measured stage before proposing algorithmic changes. Do
not infer a kernel bottleneck solely from low whole-run GPU utilization when
uncaptured CPU setup or synchronization may dominate.
