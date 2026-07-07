#!/usr/bin/env python3
"""Offline fMCS block-size compilation and occupancy sweep.

The probe compiles the real ``fmcsKernel`` template with ptxas for requested
SM architectures and candidate block sizes.  It then feeds the emitted
register, shared-memory, and barrier counts into Nsight Compute's offline
occupancy calculator.  No GPU of the target architecture needs to be installed.

The default bisection search assumes resident blocks do not increase as block
size increases within one scratch-placement domain.  Use ``--search
exhaustive`` to audit that assumption after a change that might alter resource
scaling.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
CUDA_TARGETS_FILE = REPO_ROOT / "cmake" / "cuda_targets.cmake"
DEFAULT_NCU_ROOT = Path("/opt/nvidia/nsight-compute")
TIER_VALUES = (16, 32, 64, 128)
GLOBAL_SCRATCH_BLOCK_THRESHOLD = 512
SCRATCH_ENUM = {"shared": 0, "global": 1}
SCRATCH_FROM_ENUM = {value: key for key, value in SCRATCH_ENUM.items()}

KERNEL_TEMPLATE_RE = re.compile(
    r"ILi(?P<tier>\d+)ELi(?P<bonds>\d+)ELi(?P<block>\d+)ELb0ELb0ELNS0_19FmcsScratchLocationE(?P<scratch>[01])EEE"
)
STACK_RE = re.compile(
    r"(?P<stack>\d+) bytes stack frame, (?P<stores>\d+) bytes spill stores, (?P<loads>\d+) bytes spill loads"
)
USED_RE = re.compile(
    r"Used (?P<registers>\d+) registers, used (?P<barriers>\d+) barriers, (?P<shared>\d+) bytes smem"
)


@dataclasses.dataclass(frozen=True)
class Variant:
    tier: int
    scratch: str


@dataclasses.dataclass
class CompileResult:
    architecture: int
    block_size: int
    tier: int
    scratch: str
    compiled: bool = False
    registers_per_thread: int | None = None
    static_shared_bytes: int | None = None
    barriers: int | None = None
    stack_bytes: int | None = None
    spill_store_bytes: int | None = None
    spill_load_bytes: int | None = None
    compiler_error: str | None = None
    occupancy_model_architecture: int | None = None
    active_blocks_per_sm: int | None = None
    occupancy_percent: float | None = None
    limiters: list[str] = dataclasses.field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=["full"],
        help="SM numbers (for example 80 90 120) or 'full' for nvMolKit full-mode targets",
    )
    parser.add_argument("--tiers", nargs="+", type=int, default=list(TIER_VALUES), choices=TIER_VALUES)
    parser.add_argument(
        "--scratch",
        choices=("auto", "shared", "global", "both"),
        default="auto",
        help="Scratch specializations; auto extends the current tier-128-at-512 policy to 512+",
    )
    parser.add_argument("--min-block-size", type=int, default=32)
    parser.add_argument("--max-block-size", type=int, default=1024)
    parser.add_argument("--block-step", type=int, default=32)
    parser.add_argument(
        "--resident-blocks",
        nargs="+",
        type=int,
        default=[1, 2],
        help="Resident-block targets for which the largest qualifying block is reported",
    )
    parser.add_argument("--nvcc", default="/usr/local/cuda/bin/nvcc")
    parser.add_argument("--ncu-python-path", type=Path)
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--compile-timeout", type=float, default=180.0)
    parser.add_argument(
        "--search",
        choices=("bisect", "exhaustive"),
        default="bisect",
        help="Bisect residency thresholds (default) or compile every requested block size",
    )
    parser.add_argument("--reject-stack-spills", action="store_true")
    parser.add_argument("--show-all", action="store_true", help="Print every compiled configuration")
    parser.add_argument("--json", type=Path, help="Write all resources and occupancy results as JSON")
    parser.add_argument("--config-json", type=Path, help="Write concise generated launch recommendations")
    parser.add_argument("--config-header", type=Path, help="Write generated C++ launch constants")
    parser.add_argument("--production-architecture", type=int, default=120)
    parser.add_argument("--from-json", type=Path, help="Regenerate summaries/config from a previous --json result")
    args = parser.parse_args()

    if args.min_block_size < 32 or args.max_block_size > 1024:
        parser.error("block sizes must stay within CUDA's 32..1024 thread range")
    if args.min_block_size > args.max_block_size:
        parser.error("--min-block-size cannot exceed --max-block-size")
    bounds = ((args.min_block_size, "minimum"), (args.max_block_size, "maximum"), (args.block_step, "step"))
    for value, label in bounds:
        if value % 32 != 0:
            parser.error(f"the {label} block size must be a multiple of 32")
    if any(target < 1 for target in args.resident_blocks):
        parser.error("resident-block targets must be positive")
    return args


def nvcc_supported_architectures(nvcc: str) -> set[int]:
    result = subprocess.run([nvcc, "--list-gpu-code"], check=True, capture_output=True, text=True)
    return {int(value) for value in re.findall(r"sm_(\d+)", result.stdout)}


def repository_full_architectures() -> list[int]:
    text = CUDA_TARGETS_FILE.read_text()
    architectures = {
        int(value)
        for value in re.findall(r"(?<![\d.])(\d{2,3})(?:-real)?(?![\d.])", text)
    }
    return sorted(value for value in architectures if 70 <= value <= 999)


def resolve_architectures(values: list[str], nvcc: str) -> list[int]:
    requested: set[int] = set()
    for value in values:
        for item in value.split(","):
            if item == "full":
                requested.update(repository_full_architectures())
            else:
                requested.add(int(item.removeprefix("sm_")))
    supported = nvcc_supported_architectures(nvcc)
    unsupported = sorted(requested - supported)
    if unsupported:
        names = ", ".join(f"sm_{value}" for value in unsupported)
        print(f"warning: nvcc cannot emit {names}; skipping", file=sys.stderr)
    return sorted(requested & supported)


def variants_for(tiers: list[int], scratch_mode: str, block_size: int) -> list[Variant]:
    variants: list[Variant] = []
    for tier in sorted(set(tiers)):
        if scratch_mode == "both":
            scratch_values = ("shared", "global")
        elif scratch_mode == "auto":
            use_global = tier == 128 and block_size >= GLOBAL_SCRATCH_BLOCK_THRESHOLD
            scratch_values = ("global",) if use_global else ("shared",)
        else:
            scratch_values = (scratch_mode,)
        variants.extend(Variant(tier, scratch) for scratch in scratch_values)
    return variants


def logical_variants(tiers: list[int], scratch_mode: str) -> list[Variant]:
    if scratch_mode == "both":
        return [Variant(tier, scratch) for tier in sorted(set(tiers)) for scratch in ("shared", "global")]
    return [Variant(tier, scratch_mode) for tier in sorted(set(tiers))]


def result_for_variant(results: list[CompileResult], variant: Variant, scratch_mode: str) -> CompileResult:
    expected_scratch = expected_scratch_for(variant, scratch_mode, results[0].block_size)
    return next(
        result
        for result in results
        if result.tier == variant.tier and result.scratch == expected_scratch
    )


def expected_scratch_for(variant: Variant, scratch_mode: str, block_size: int) -> str:
    if scratch_mode == "auto":
        return variants_for([variant.tier], "auto", block_size)[0].scratch
    return variant.scratch


def generate_source(block_size: int, variants: list[Variant]) -> str:
    launches = []
    for variant in variants:
        launches.append(
            "  fmcsKernel<{tier}, {tier}, {block}, false, false, FmcsScratchLocation::{scratch}>\n"
            "    <<<1, {block}>>>(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, 0, 0);".format(
                tier=variant.tier,
                block=block_size,
                scratch=variant.scratch.capitalize(),
            )
        )
    return "\n".join(
        (
            "#define NVMOLKIT_FMCS_BLOCK_SIZE_PROBE 1",
            "#define NVMOLKIT_ENABLE_MCS_STATS 0",
            "#define NVMOLKIT_ENABLE_MCS_TIMINGS 0",
            '#include "fmcs_cuda/fmcs_kernel.cuh"',
            "",
            "using namespace mcs::fmcs;",
            "",
            'extern "C" void instantiate_fmcs_occupancy_probes() {',
            *launches,
            "}",
            "",
        )
    )


def concise_error(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if "error" in line.lower() or "ptxas fatal" in line.lower()]
    return " | ".join(lines[-3:]) or "nvcc did not emit this specialization"


def parse_ptxas_output(
    architecture: int,
    block_size: int,
    variants: list[Variant],
    output: str,
) -> list[CompileResult]:
    results = {
        (variant.tier, variant.scratch): CompileResult(
            architecture, block_size, variant.tier, variant.scratch
        )
        for variant in variants
    }
    current: CompileResult | None = None
    for line in output.splitlines():
        if "Compiling entry function" in line:
            match = KERNEL_TEMPLATE_RE.search(line)
            current = None
            if match and int(match.group("block")) == block_size:
                key = (int(match.group("tier")), SCRATCH_FROM_ENUM[int(match.group("scratch"))])
                current = results.get(key)
            continue
        if current is None:
            continue
        stack_match = STACK_RE.search(line)
        if stack_match:
            current.stack_bytes = int(stack_match.group("stack"))
            current.spill_store_bytes = int(stack_match.group("stores"))
            current.spill_load_bytes = int(stack_match.group("loads"))
        used_match = USED_RE.search(line)
        if used_match:
            current.registers_per_thread = int(used_match.group("registers"))
            current.barriers = int(used_match.group("barriers"))
            current.static_shared_bytes = int(used_match.group("shared"))
            current.compiled = True
    error = concise_error(output)
    for result in results.values():
        if not result.compiled:
            result.compiler_error = error
    return list(results.values())


def compile_block(
    nvcc: str,
    architecture: int,
    block_size: int,
    variants: list[Variant],
    work_dir: Path,
    timeout: float,
) -> list[CompileResult]:
    source = work_dir / f"probe_sm{architecture}_{block_size}.cu"
    cubin = work_dir / f"probe_sm{architecture}_{block_size}.cubin"
    source.write_text(generate_source(block_size, variants))
    command = [
        nvcc,
        "--cubin",
        str(source),
        "-o",
        str(cubin),
        f"-arch=sm_{architecture}",
        f"-I{REPO_ROOT / 'src' / 'mcs'}",
        f"-I{REPO_ROOT}",
        "-std=c++20",
        "-O3",
        "-DNDEBUG",
        "--use_fast_math",
        "--default-stream=per-thread",
        "--extended-lambda",
        "--diag-suppress=20012",
        "-Xptxas=-v",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        output = completed.stdout + "\n" + completed.stderr
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + "\n" + (exc.stderr or "") + f"\nerror: compile timed out after {timeout}s"
    return parse_ptxas_output(architecture, block_size, variants, output)


def find_ncu_python_path(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(explicit)
    env_path = os.environ.get("NCU_OCCUPANCY_PYTHON_PATH")
    if env_path:
        candidates.append(Path(env_path))
    if DEFAULT_NCU_ROOT.exists():
        candidates.extend(sorted(DEFAULT_NCU_ROOT.glob("*/extras/python"), reverse=True))
    for candidate in candidates:
        if (candidate / "ncu_occupancy.py").is_file():
            return candidate
    raise RuntimeError(
        "could not find Nsight Compute's ncu_occupancy.py; pass --ncu-python-path or set NCU_OCCUPANCY_PYTHON_PATH"
    )


def load_occupancy_module(path: Path) -> Any:
    sys.path.insert(0, str(path))
    try:
        import ncu_occupancy  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(f"failed to import ncu_occupancy from {path}: {exc}") from exc
    return ncu_occupancy


def make_calculator(module: Any, architecture: int) -> tuple[Any, dict[str, Any], int]:
    major, minor = divmod(architecture, 10)
    try:
        return module.OccupancyCalculator(major, minor), module.get_gpu_data(major, minor), architecture
    except ValueError:
        # CUDA and NVIDIA's specification tables group compute capabilities
        # 10.0 and 10.3 under the same 10.x occupancy limits.  Nsight Compute
        # 2025.2 predates the explicit 10.3 entry, so use its 10.0 model.
        if major == 10:
            return module.OccupancyCalculator(10, 0), module.get_gpu_data(10, 0), 100
        raise


def apply_occupancy(module: Any, results: list[CompileResult]) -> None:
    calculators: dict[int, tuple[Any, dict[str, Any], int]] = {}
    for result in results:
        if not result.compiled:
            continue
        if result.architecture not in calculators:
            calculators[result.architecture] = make_calculator(module, result.architecture)
        calculator, gpu_data, model_arch = calculators[result.architecture]
        result.occupancy_model_architecture = model_arch
        parameters = module.OccupancyParameters(
            shared_mem_size=max(gpu_data["shared_mem_size_configs"]),
            threads_per_block=result.block_size,
            registers_per_thread=result.registers_per_thread,
            shared_mem_per_block=result.static_shared_bytes,
            num_block_barriers=result.barriers,
        )
        try:
            utilization = calculator.get_resource_utilization(parameters)
            result.active_blocks_per_sm = int(utilization["allocated_blocks"])
            result.occupancy_percent = float(utilization["sm_occupancy"])
            result.limiters = [limiter.name.lower() for limiter in calculator.get_occupancy_limiters(parameters)]
        except ValueError as exc:
            result.compiler_error = f"occupancy calculator rejected resources: {exc}"


def result_qualifies(result: CompileResult, target: int, reject_stack_spills: bool) -> bool:
    if not result.compiled or result.active_blocks_per_sm is None or result.active_blocks_per_sm < target:
        return False
    if reject_stack_spills and any((result.stack_bytes, result.spill_store_bytes, result.spill_load_bytes)):
        return False
    return True


def print_summary(
    results: list[CompileResult],
    resident_targets: list[int],
    reject_stack_spills: bool,
    tiers: list[int],
    scratch_mode: str,
) -> None:
    print("\nLargest qualifying fMCS block sizes")
    for architecture in sorted({result.architecture for result in results}):
        arch_results = [result for result in results if result.architecture == architecture]
        model_arch = next(
            (
                result.occupancy_model_architecture
                for result in arch_results
                if result.occupancy_model_architecture
            ),
            architecture,
        )
        model_note = "" if model_arch == architecture else f" (occupancy model sm_{model_arch})"
        print(f"\nsm_{architecture}{model_note}")
        variants = logical_variants(tiers, scratch_mode)
        for target in sorted(set(resident_targets)):
            print(f"  target >= {target} resident block(s)/SM")
            selected_by_variant: dict[Variant, CompileResult | None] = {}
            for variant in variants:
                candidates = [
                    result
                    for result in arch_results
                    if result.tier == variant.tier
                    and result.scratch == expected_scratch_for(variant, scratch_mode, result.block_size)
                    and result_qualifies(result, target, reject_stack_spills)
                ]
                selected = max(candidates, key=lambda item: item.block_size) if candidates else None
                selected_by_variant[variant] = selected
                if selected:
                    label = selected.scratch if scratch_mode == "auto" else variant.scratch
                    print(
                        f"    tier {variant.tier:>3} {label:>6}: {selected.block_size:>4} threads, "
                        f"{selected.active_blocks_per_sm} blocks/SM, {selected.occupancy_percent:5.1f}% occupancy, "
                        f"{selected.registers_per_thread} regs, {selected.static_shared_bytes} B smem, "
                        f"limiter={','.join(selected.limiters) or 'none'}"
                    )
                else:
                    print(f"    tier {variant.tier:>3} {variant.scratch:>6}: no qualifying block")

            selections = list(selected_by_variant.values())
            common_text = str(min(item.block_size for item in selections if item)) if all(selections) else "none"
            print(f"    common across selected variants: {common_text}")


def compile_requests(
    requests: set[tuple[int, int]],
    args: argparse.Namespace,
    block_variants: dict[int, list[Variant]],
    work_dir: Path,
) -> dict[tuple[int, int], list[CompileResult]]:
    compiled: dict[tuple[int, int], list[CompileResult]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                compile_block,
                args.nvcc,
                architecture,
                block_size,
                block_variants[block_size],
                work_dir,
                args.compile_timeout,
            ): (architecture, block_size)
            for architecture, block_size in requests
        }
        for future in concurrent.futures.as_completed(futures):
            compiled[futures[future]] = future.result()
    return compiled


def bisect_configurations(
    architectures: list[int],
    block_sizes: list[int],
    args: argparse.Namespace,
    occupancy_module: Any,
    work_dir: Path,
) -> list[CompileResult]:
    variants = logical_variants(args.tiers, args.scratch)
    targets = sorted(set(args.resident_blocks))
    domains: dict[Variant, list[tuple[int, int]]] = {}
    for variant in variants:
        starts = [0]
        for index in range(1, len(block_sizes)):
            previous = expected_scratch_for(variant, args.scratch, block_sizes[index - 1])
            current = expected_scratch_for(variant, args.scratch, block_sizes[index])
            if current != previous:
                starts.append(index)
        ends = [start - 1 for start in starts[1:]] + [len(block_sizes) - 1]
        domains[variant] = list(zip(starts, ends, strict=True))
    bounds = {
        (architecture, variant, target, start, end): [start - 1, end + 1]
        for architecture in architectures
        for variant in variants
        for target in targets
        for start, end in domains[variant]
    }
    cache: dict[tuple[int, int], list[CompileResult]] = {}
    block_variants = {size: variants_for(args.tiers, args.scratch, size) for size in block_sizes}
    iteration = 0
    while True:
        pending: dict[tuple[int, Variant, int, int, int], int] = {}
        for key, (low, high) in bounds.items():
            if high - low > 1:
                pending[key] = (low + high) // 2
        if not pending:
            break
        requests = {
            (architecture, block_sizes[index])
            for (architecture, _variant, _target, _start, _end), index in pending.items()
            if (architecture, block_sizes[index]) not in cache
        }
        if requests:
            new_results = compile_requests(requests, args, block_variants, work_dir)
            flat_results = [result for group in new_results.values() for result in group]
            apply_occupancy(occupancy_module, flat_results)
            cache.update(new_results)
        for key, index in pending.items():
            architecture, variant, target, _start, _end = key
            result = result_for_variant(cache[(architecture, block_sizes[index])], variant, args.scratch)
            if result_qualifies(result, target, args.reject_stack_spills):
                bounds[key][0] = index
            else:
                bounds[key][1] = index
        iteration += 1
        print(
            f"\rbisection iteration {iteration}: {len(cache)} architecture/block combinations compiled",
            end="",
            flush=True,
        )
    print()
    return [result for group in cache.values() for result in group]


def exhaustive_configurations(
    architectures: list[int],
    block_sizes: list[int],
    args: argparse.Namespace,
    occupancy_module: Any,
    work_dir: Path,
) -> list[CompileResult]:
    requests = {(architecture, block_size) for architecture in architectures for block_size in block_sizes}
    block_variants = {size: variants_for(args.tiers, args.scratch, size) for size in block_sizes}
    cache = compile_requests(requests, args, block_variants, work_dir)
    results = [result for group in cache.values() for result in group]
    apply_occupancy(occupancy_module, results)
    print(f"compiled {len(requests)} architecture/block combinations")
    return results


def print_all(results: list[CompileResult]) -> None:
    print("\nAll configurations")
    for result in sorted(results, key=lambda item: (item.architecture, item.tier, item.scratch, item.block_size)):
        if result.compiled:
            occupancy = "unknown" if result.occupancy_percent is None else f"{result.occupancy_percent:.2f}%"
            print(
                f"sm_{result.architecture} tier={result.tier} scratch={result.scratch} block={result.block_size} "
                f"blocks/SM={result.active_blocks_per_sm} occupancy={occupancy} "
                f"regs={result.registers_per_thread} smem={result.static_shared_bytes} barriers={result.barriers} "
                f"stack={result.stack_bytes} spills={result.spill_store_bytes}/{result.spill_load_bytes}"
            )
        else:
            print(
                f"sm_{result.architecture} tier={result.tier} scratch={result.scratch} block={result.block_size} "
                f"COMPILE_FAILED: {result.compiler_error}"
            )


def recommendation_payload(
    results: list[CompileResult],
    architectures: list[int],
    tiers: list[int],
    scratch_mode: str,
    resident_targets: list[int],
    reject_stack_spills: bool,
) -> dict[str, Any]:
    configurations: dict[str, Any] = {}
    for architecture in architectures:
        arch_results = [result for result in results if result.architecture == architecture]
        model_arch = next(
            (
                result.occupancy_model_architecture
                for result in arch_results
                if result.occupancy_model_architecture
            ),
            architecture,
        )
        targets: dict[str, Any] = {}
        for target in sorted(set(resident_targets)):
            tier_results: dict[str, Any] = {}
            selected_sizes: list[int] = []
            for variant in logical_variants(tiers, scratch_mode):
                candidates = [
                    result
                    for result in arch_results
                    if result.tier == variant.tier
                    and result.scratch == expected_scratch_for(variant, scratch_mode, result.block_size)
                    and result_qualifies(result, target, reject_stack_spills)
                ]
                selected = max(candidates, key=lambda item: item.block_size) if candidates else None
                if selected is None:
                    tier_results[str(variant.tier)] = None
                    continue
                selected_sizes.append(selected.block_size)
                tier_results[str(variant.tier)] = {
                    "block_size": selected.block_size,
                    "scratch": selected.scratch,
                    "active_blocks_per_sm": selected.active_blocks_per_sm,
                    "occupancy_percent": selected.occupancy_percent,
                    "registers_per_thread": selected.registers_per_thread,
                    "static_shared_bytes": selected.static_shared_bytes,
                    "limiters": selected.limiters,
                }
            targets[str(target)] = {
                "common_block_size": min(selected_sizes) if len(selected_sizes) == len(tiers) else None,
                "tiers": tier_results,
            }
        configurations[f"sm_{architecture}"] = {
            "occupancy_model": f"sm_{model_arch}",
            "resident_block_targets": targets,
        }
    return {
        "schema_version": 1,
        "generator": "fmcs_occupancy_sweep.py",
        "regenerate_command": (
            "python src/mcs/fmcs_cuda/fmcs_occupancy_sweep.py --architectures full "
            "--tiers 16 32 64 128 --scratch auto --min-block-size 32 --max-block-size 1024 "
            "--block-step 32 --resident-blocks 1 2 --search bisect --reject-stack-spills "
            "--config-json src/mcs/fmcs_cuda/fmcs_occupancy_config.json "
            "--config-header src/mcs/fmcs_cuda/fmcs_occupancy_config.cuh --production-architecture 120"
        ),
        "method": (
            "Compile real fmcsKernel specializations with ptxas, then pass registers/thread, static shared memory, "
            "barriers, and threads/block to Nsight Compute's offline OccupancyCalculator. A target qualifies when "
            "get_resource_utilization()['allocated_blocks'] is at least the requested resident-block count."
        ),
        "requires_resident_gpu": False,
        "search_mode": "bisect",
        "search_assumption": "Resident blocks do not increase with block size inside one scratch-placement domain.",
        "configurations": configurations,
    }


def write_config_json(
    path: Path,
    results: list[CompileResult],
    architectures: list[int],
    tiers: list[int],
    scratch_mode: str,
    resident_targets: list[int],
    reject_stack_spills: bool,
) -> None:
    payload = recommendation_payload(
        results, architectures, tiers, scratch_mode, resident_targets, reject_stack_spills
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {path}")


def write_config_header(path: Path, payload: dict[str, Any], architecture: int) -> None:
    architecture_key = f"sm_{architecture}"
    try:
        targets = payload["configurations"][architecture_key]["resident_block_targets"]
        two_block_size = int(targets["2"]["common_block_size"])
        single_block_size = int(targets["1"]["common_block_size"])
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"config has no one-/two-block recommendations for {architecture_key}") from exc
    command = payload["regenerate_command"]
    content = f"""// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Generated by fmcs_occupancy_sweep.py for {architecture_key}. Do not edit.
// Regenerate from the repository root with:
//   {command}
//
// The generator compiles the real kernel with ptxas and passes its resource
// counts to Nsight Compute's offline OccupancyCalculator. No resident GPU of
// the queried architecture is required.

#ifndef FMCS_CUDA_FMCS_OCCUPANCY_CONFIG_CUH
#define FMCS_CUDA_FMCS_OCCUPANCY_CONFIG_CUH

namespace mcs {{
namespace fmcs {{

inline constexpr int kFmcsOccupancyTargetArchitecture = {architecture};
inline constexpr int kFmcsMaxBlockSizeTwoBlockOccupancy = {two_block_size};
inline constexpr int kFmcsMaxBlockSizeSingleOccupancy = {single_block_size};

}}  // namespace fmcs
}}  // namespace mcs

#endif  // FMCS_CUDA_FMCS_OCCUPANCY_CONFIG_CUH
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"\nwrote {path}")


def main() -> int:
    args = parse_args()
    if args.from_json:
        payload = json.loads(args.from_json.read_text())
        all_results = [CompileResult(**result) for result in payload["results"]]
        architectures = payload["architectures"]
        tiers = payload["tiers"]
        scratch_mode = payload["scratch_mode"]
        resident_targets = payload["resident_block_targets"]
        reject_stack_spills = payload["reject_stack_spills"]
        print_summary(all_results, resident_targets, reject_stack_spills, tiers, scratch_mode)
        config_payload = recommendation_payload(
            all_results, architectures, tiers, scratch_mode, resident_targets, reject_stack_spills
        )
        if args.config_json:
            args.config_json.parent.mkdir(parents=True, exist_ok=True)
            args.config_json.write_text(json.dumps(config_payload, indent=2, sort_keys=True) + "\n")
            print(f"\nwrote {args.config_json}")
        if args.config_header:
            write_config_header(args.config_header, config_payload, args.production_architecture)
        return 0

    architectures = resolve_architectures(args.architectures, args.nvcc)
    if not architectures:
        raise RuntimeError("no requested architectures are supported by nvcc")
    block_sizes = list(range(args.min_block_size, args.max_block_size + 1, args.block_step))
    ncu_path = find_ncu_python_path(args.ncu_python_path)
    occupancy_module = load_occupancy_module(ncu_path)

    print(f"nvcc: {args.nvcc}")
    print(f"Nsight Compute occupancy interface: {ncu_path}")
    print("architectures: " + ", ".join(f"sm_{architecture}" for architecture in architectures))
    print(f"block sizes: {block_sizes[0]}..{block_sizes[-1]} step {args.block_step}")
    print("tiers: " + ", ".join(map(str, sorted(set(args.tiers)))))
    print(f"scratch mode: {args.scratch}; search: {args.search}; compile jobs: {args.jobs}")

    with tempfile.TemporaryDirectory(prefix="nvmolkit-fmcs-occupancy-") as temp:
        work_dir = Path(temp)
        if args.search == "bisect":
            all_results = bisect_configurations(architectures, block_sizes, args, occupancy_module, work_dir)
        else:
            all_results = exhaustive_configurations(architectures, block_sizes, args, occupancy_module, work_dir)

    print_summary(all_results, args.resident_blocks, args.reject_stack_spills, args.tiers, args.scratch)
    if args.show_all:
        print_all(all_results)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "architectures": architectures,
            "block_sizes": block_sizes,
            "tiers": sorted(set(args.tiers)),
            "scratch_mode": args.scratch,
            "search_mode": args.search,
            "resident_block_targets": sorted(set(args.resident_blocks)),
            "reject_stack_spills": args.reject_stack_spills,
            "results": [dataclasses.asdict(result) for result in all_results],
        }
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.json}")
    if args.config_json:
        write_config_json(
            args.config_json,
            all_results,
            architectures,
            args.tiers,
            args.scratch,
            args.resident_blocks,
            args.reject_stack_spills,
        )
    if args.config_header:
        config_payload = recommendation_payload(
            all_results,
            architectures,
            args.tiers,
            args.scratch,
            args.resident_blocks,
            args.reject_stack_spills,
        )
        write_config_header(args.config_header, config_payload, args.production_architecture)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
