# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Maximum common substructure benchmark comparing nvmolkit against RDKit."""

import argparse
import csv
import random
import time
from multiprocessing import Pool
from pathlib import Path

import nvtx
from bench_utils import TimingResult, load_pickle, load_smiles, throughput_per_s, time_it_bounded
from rdkit import Chem
from rdkit.Chem import rdFMCS
from tqdm.auto import tqdm

from nvmolkit.mcs import findMCS

_worker_mols: list[Chem.Mol] | None = None
_worker_params: rdFMCS.MCSParameters | None = None

FMCS_STATS_COLUMNS = [
    "phase2_iters",
    "initial_seeds",
    "mismatched_initial_seeds",
    "popped",
    "seed_checks",
    "match_calls",
    "match_found",
    "bound_rejected",
    "expanded",
    "fill_zero",
    "stage0_attempts",
    "stage0_success",
    "stage1_attempts",
    "stage1_success",
    "stage2_attempts",
    "stage2_success",
    "individual_bond_excluded",
    "fast_attempts",
    "fast_success",
    "fallback_calls",
    "fallback_success",
    "fallback_fail",
    "fallback_overflow",
    "max_queue",
    "forced_exit",
    "total_clocks",
    "phase1_clocks",
    "phase2_clocks",
    "incremental_match_cycles_1024",
    "substructure_match_cycles_1024",
    "phase2_pop_sync_wait_cycles_1024",
    "phase2_sync_wait_cycles_1024",
    "phase2_idle_no_seed_wait_cycles_1024",
    "phase2_idle_no_match_wait_cycles_1024",
    "phase2_active_work_cycles_1024",
    "phase2_active_match_cycles_1024",
]


def _log(message: str) -> None:
    print(message, flush=True)


def _rdkit_params(args: argparse.Namespace) -> rdFMCS.MCSParameters:
    params = rdFMCS.MCSParameters()
    params.MaximizeBonds = True
    params.Timeout = int(args.timeout_seconds)
    params.AtomCompareParameters.MatchValences = bool(args.match_valences)
    params.AtomCompareParameters.MatchFormalCharge = bool(args.match_formal_charge)
    params.AtomCompareParameters.MatchIsotope = bool(args.match_isotope)
    params.AtomCompareParameters.RingMatchesRingOnly = bool(args.ring_matches_ring_only)
    params.BondCompareParameters.RingMatchesRingOnly = bool(args.ring_matches_ring_only)
    params.AtomCompareParameters.CompleteRingsOnly = bool(args.complete_rings_only)
    params.BondCompareParameters.CompleteRingsOnly = bool(args.complete_rings_only)

    atom_types = {
        "any": rdFMCS.AtomCompare.CompareAny,
        "elements": rdFMCS.AtomCompare.CompareElements,
        "isotopes": rdFMCS.AtomCompare.CompareIsotopes,
        "any_heavy_atom": rdFMCS.AtomCompare.CompareAnyHeavyAtom,
    }
    bond_types = {
        "any": rdFMCS.BondCompare.CompareAny,
        "order": rdFMCS.BondCompare.CompareOrder,
        "order_exact": rdFMCS.BondCompare.CompareOrderExact,
    }
    params.AtomTyper = atom_types[args.atom_compare]
    params.BondTyper = bond_types[args.bond_compare]
    return params


def _rdkit_worker_init(mol_binaries: list[bytes], params: rdFMCS.MCSParameters) -> None:
    global _worker_mols, _worker_params
    _worker_mols = [Chem.Mol(binary) for binary in mol_binaries]
    _worker_params = params


def _rdkit_worker_pair(pair: tuple[int, int]) -> tuple[int, int, float]:
    if _worker_mols is None or _worker_params is None:
        raise RuntimeError("RDKit MCS worker was not initialized")
    start = time.perf_counter()
    result = rdFMCS.FindMCS([_worker_mols[pair[0]], _worker_mols[pair[1]]], _worker_params)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return int(result.numAtoms), int(result.numBonds), elapsed_ms


def _filter_molecules(mols: list[Chem.Mol], max_atoms: int, max_bonds: int) -> list[Chem.Mol]:
    out = [
        mol
        for mol in mols
        if mol is not None
        and (max_atoms <= 0 or mol.GetNumAtoms() <= max_atoms)
        and (max_bonds <= 0 or mol.GetNumBonds() <= max_bonds)
    ]
    if not out:
        raise ValueError("No molecules remain after max atom/bond filtering")
    return out


def _sample_pairs(num_mols: int, num_pairs: int, seed: int) -> list[tuple[int, int]]:
    if num_pairs <= 0:
        raise ValueError("--pairs must be positive")
    rng = random.Random(seed)
    return [(rng.randrange(num_mols), rng.randrange(num_mols)) for _ in range(num_pairs)]




@nvtx.annotate("bench_nvmolkit_mcs", color="red")
def _bench_nvmolkit(mols: list[Chem.Mol], pairs: list[tuple[int, int]], args: argparse.Namespace):
    _log(
        "Starting nvmolkit MCS benchmark: "
        f"molecules={len(mols)} pairs={len(pairs)} runs={args.runs} warmups={args.warmups} "
        f"batch_size={args.batch_size} block_size={args.block_size} "
        f"workers={args.workers} prep_threads={args.prep_threads} "
        f"num_gpus={args.num_gpus} executors_per_runner={args.executors_per_runner} "
        f"collect_timings={args.collect_timings} collect_stats={args.collect_stats}"
    )
    last_result = None
    import torch

    def run_find(label: str):
        nonlocal last_result
        _log(f"nvmolkit {label}: starting findMCS")
        with nvtx.annotate(f"mcs_nvmolkit_findMCS_{label}", color="blue"):
            last_result = findMCS(
                mols,
                mode="pairs",
                pairs=pairs,
                atom_compare=args.atom_compare,
                bond_compare=args.bond_compare,
                match_valences=args.match_valences,
                match_formal_charge=args.match_formal_charge,
                ring_matches_ring_only=args.ring_matches_ring_only,
                complete_rings_only=args.complete_rings_only,
                match_isotope=args.match_isotope,
                timeout_seconds=args.timeout_seconds,
                batch_size=args.batch_size,
                block_size=args.block_size,
                worker_threads=args.workers,
                preprocessing_threads=args.prep_threads,
                executors_per_runner=args.executors_per_runner,
                gpu_ids=list(range(args.num_gpus)),
                collect_timings=args.collect_timings,
                collect_stats=args.collect_stats,
            )
        _log(f"nvmolkit {label}: findMCS returned; synchronizing GPU")
        torch.cuda.synchronize()
        _log(
            f"nvmolkit {label}: finished "
            f"gpu={int(last_result.used_gpu.sum())}/{len(last_result)} "
            f"fallback={int(last_result.used_fallback.sum())}/{len(last_result)} "
            f"overflow={int(last_result.overflowed.sum())}/{len(last_result)}"
        )

    for warmup_idx in range(args.warmups):
        with nvtx.annotate(f"mcs_nvmolkit_warmup_{warmup_idx + 1}", color="purple"):
            run_find(f"warmup {warmup_idx + 1}/{args.warmups}")

    times_ms: list[float] = []
    for run_idx in range(args.runs):
        label = f"run {run_idx + 1}/{args.runs}"
        with nvtx.annotate(f"mcs_nvmolkit_run_{run_idx + 1}", color="orange"):
            _log(f"nvmolkit {label}: pre-run GPU synchronize")
            torch.cuda.synchronize()
            start = time.perf_counter()
            run_find(label)
            times_ms.append((time.perf_counter() - start) * 1000.0)

    timing = TimingResult(times_ms=times_ms)
    _log("Finished nvmolkit MCS benchmark")
    return timing, last_result


@nvtx.annotate("bench_rdkit_mcs", color="green")
def _bench_rdkit(mols: list[Chem.Mol], pairs: list[tuple[int, int]], args: argparse.Namespace):
    _log(
        "Starting RDKit MCS benchmark: "
        f"molecules={len(mols)} pairs={len(pairs)} runs={args.runs} threads={args.rdkit_threads} "
        f"max_seconds={args.rdkit_max_seconds}"
    )
    params = _rdkit_params(args)
    sizes: list[tuple[int, int]] = []
    times_ms: list[float] = []
    pairs_done = 0
    rdkit_run_idx = 0

    if args.rdkit_threads > 1:
        _log("RDKit: serializing molecules for multiprocessing workers")
        mol_binaries = [mol.ToBinary() for mol in mols]
        chunksize = 1 if args.rdkit_max_seconds > 0 else max(1, len(pairs) // max(1, args.rdkit_threads * 8))
        _log(f"RDKit: launching multiprocessing pool with chunksize={chunksize}")

        def run(deadline):
            nonlocal sizes, times_ms, pairs_done, rdkit_run_idx
            rdkit_run_idx += 1
            with nvtx.annotate(f"mcs_rdkit_run_{rdkit_run_idx}", color="yellow"):
                sizes = []
                times_ms = []
                pairs_done = 0
                with Pool(args.rdkit_threads, initializer=_rdkit_worker_init, initargs=(mol_binaries, params)) as pool:
                    iterator = pool.imap(_rdkit_worker_pair, pairs, chunksize=chunksize)
                    with tqdm(total=len(pairs), desc="RDKit MCS pairs", unit="pair") as progress:
                        for atoms, bonds, elapsed_ms in iterator:
                            size = (atoms, bonds)
                            sizes.append(size)
                            times_ms.append(elapsed_ms)
                            pairs_done += 1
                            progress.update(1)
                            if deadline.expired():
                                _log(
                                    f"RDKit: stopping early after {pairs_done}/{len(pairs)} pairs due to max_seconds"
                                )
                                break

    else:

        def run(deadline):
            nonlocal sizes, times_ms, pairs_done, rdkit_run_idx
            rdkit_run_idx += 1
            with nvtx.annotate(f"mcs_rdkit_run_{rdkit_run_idx}", color="yellow"):
                sizes = []
                times_ms = []
                pairs_done = 0
                with tqdm(total=len(pairs), desc="RDKit MCS pairs", unit="pair") as progress:
                    for idx_a, idx_b in pairs:
                        if deadline.expired():
                            _log(f"RDKit: stopping early after {pairs_done}/{len(pairs)} pairs due to max_seconds")
                            break
                        start = time.perf_counter()
                        result = rdFMCS.FindMCS([mols[idx_a], mols[idx_b]], params)
                        elapsed_ms = (time.perf_counter() - start) * 1000.0
                        sizes.append((int(result.numAtoms), int(result.numBonds)))
                        times_ms.append(elapsed_ms)
                        pairs_done += 1
                        progress.update(1)

    avg_ms, std_ms, last_pairs = time_it_bounded(
        run, args.runs, args.rdkit_max_seconds, lambda: pairs_done, len(pairs)
    )
    _log(f"Finished RDKit MCS benchmark: pairs_completed={last_pairs}/{len(pairs)}")
    return avg_ms, std_ms, sizes, times_ms, last_pairs


def _format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def _mol_prop(mol: Chem.Mol, key: str) -> str:
    return mol.GetProp(key) if mol.HasProp(key) else ""


def _mol_input_smiles(mol: Chem.Mol) -> str:
    return _mol_prop(mol, "nvmolkit_input_smiles") or Chem.MolToSmiles(mol, canonical=True)


def _write_pair_timings(
    path: str,
    mols: list[Chem.Mol],
    pairs: list[tuple[int, int]],
    nv_result,
    rdkit_sizes: list[tuple[int, int]] | None,
    rdkit_times_ms: list[float] | None,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pair_index",
        "mol_a_idx",
        "mol_b_idx",
        "input_line_a",
        "input_line_b",
        "input_smiles_a",
        "input_smiles_b",
        "smiles_a",
        "smiles_b",
        "atoms_a",
        "bonds_a",
        "atoms_b",
        "bonds_b",
        "nvmolkit_time_ms",
        "rdkit_time_ms",
        "nvmolkit_atoms",
        "nvmolkit_bonds",
        "rdkit_atoms",
        "rdkit_bonds",
        "rdkit_completed",
        "nvmolkit_canceled",
        "nvmolkit_overflowed",
        "nvmolkit_used_gpu",
        "nvmolkit_used_fallback",
    ]
    stats_columns = []
    if nv_result is not None and getattr(nv_result, "fmcs_stats", None) is not None:
        stats_columns = [f"nvmolkit_{name}" for name in FMCS_STATS_COLUMNS]
        fieldnames.extend(stats_columns)

    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for pair_idx, (idx_a, idx_b) in enumerate(pairs):
            mol_a = mols[idx_a]
            mol_b = mols[idx_b]
            rdkit_completed = rdkit_sizes is not None and pair_idx < len(rdkit_sizes)
            row = {
                "pair_index": pair_idx,
                "mol_a_idx": idx_a,
                "mol_b_idx": idx_b,
                "input_line_a": _mol_prop(mol_a, "nvmolkit_input_line"),
                "input_line_b": _mol_prop(mol_b, "nvmolkit_input_line"),
                "input_smiles_a": _mol_input_smiles(mol_a),
                "input_smiles_b": _mol_input_smiles(mol_b),
                "smiles_a": Chem.MolToSmiles(mol_a, canonical=True),
                "smiles_b": Chem.MolToSmiles(mol_b, canonical=True),
                "atoms_a": mol_a.GetNumAtoms(),
                "bonds_a": mol_a.GetNumBonds(),
                "atoms_b": mol_b.GetNumAtoms(),
                "bonds_b": mol_b.GetNumBonds(),
                "nvmolkit_time_ms": "",
                "rdkit_time_ms": "",
                "nvmolkit_atoms": "",
                "nvmolkit_bonds": "",
                "rdkit_atoms": "",
                "rdkit_bonds": "",
                "rdkit_completed": int(rdkit_completed),
                "nvmolkit_canceled": "",
                "nvmolkit_overflowed": "",
                "nvmolkit_used_gpu": "",
                "nvmolkit_used_fallback": "",
            }
            for column in stats_columns:
                row[column] = ""
            if nv_result is not None:
                if nv_result.elapsed_ms is not None:
                    row["nvmolkit_time_ms"] = _format_float(float(nv_result.elapsed_ms[pair_idx]))
                row["nvmolkit_atoms"] = int(nv_result.num_atoms[pair_idx])
                row["nvmolkit_bonds"] = int(nv_result.num_bonds[pair_idx])
                row["nvmolkit_canceled"] = int(nv_result.canceled[pair_idx])
                row["nvmolkit_overflowed"] = int(nv_result.overflowed[pair_idx])
                row["nvmolkit_used_gpu"] = int(nv_result.used_gpu[pair_idx])
                row["nvmolkit_used_fallback"] = int(nv_result.used_fallback[pair_idx])
                if nv_result.fmcs_stats is not None:
                    for name in FMCS_STATS_COLUMNS:
                        row[f"nvmolkit_{name}"] = int(nv_result.fmcs_stats[name][pair_idx])
            if rdkit_completed:
                rd_atoms, rd_bonds = rdkit_sizes[pair_idx]
                row["rdkit_atoms"] = rd_atoms
                row["rdkit_bonds"] = rd_bonds
                if rdkit_times_ms is not None and pair_idx < len(rdkit_times_ms):
                    row["rdkit_time_ms"] = _format_float(rdkit_times_ms[pair_idx])
            writer.writerow(row)
    _log(f"Wrote per-pair timings to {output_path}")


def _validate(nv_result, rdkit_sizes: list[tuple[int, int]]) -> None:
    if nv_result is None:
        raise ValueError("Validation requires nvmolkit results")
    if len(rdkit_sizes) < len(nv_result):
        print(f"Validation skipped: RDKit completed only {len(rdkit_sizes)} / {len(nv_result)} pairs")
        return

    mismatches = []
    for i, (rd_atoms, rd_bonds) in enumerate(rdkit_sizes):
        if int(nv_result.num_atoms[i]) != rd_atoms or int(nv_result.num_bonds[i]) != rd_bonds:
            mismatches.append(
                (i, nv_result.pairs[i], int(nv_result.num_atoms[i]), int(nv_result.num_bonds[i]), rd_atoms, rd_bonds)
            )
            if len(mismatches) >= 10:
                break
    if mismatches:
        print("Validation failed; first mismatches:")
        for idx, pair, nv_atoms, nv_bonds, rd_atoms, rd_bonds in mismatches:
            print(f"  pair {idx} {pair}: nvmolkit={nv_atoms}/{nv_bonds} RDKit={rd_atoms}/{rd_bonds}")
        raise AssertionError(f"{len(mismatches)} MCS validation mismatches")
    print(f"Validation passed for {len(rdkit_sizes)} pairs")


def _load_molecules(args: argparse.Namespace) -> list[Chem.Mol]:
    source = args.pickle if args.pickle else args.smiles
    _log(
        "Loading molecules: "
        f"source={source} max_mols={args.max_mols} max_atoms={args.max_atoms} max_bonds={args.max_bonds}"
    )
    if args.pickle:
        mols = load_pickle(args.pickle, max_count=args.max_mols, seed=args.seed)
    else:
        mols = load_smiles(args.smiles, max_count=args.max_mols, seed=args.seed)
    filtered = _filter_molecules(mols, args.max_atoms, args.max_bonds)
    _log(f"Loaded {len(filtered)} molecules after filtering ({len(mols)} before filtering)")
    return filtered


def _add_option(parser: argparse.ArgumentParser, *flags: str, legacy_flags: tuple[str, ...] = (), **kwargs):
    action = parser.add_argument(*flags, **kwargs)
    for legacy_flag in legacy_flags:
        legacy_kwargs = {key: value for key, value in kwargs.items() if key not in {"default", "help"}}
        legacy_kwargs["default"] = argparse.SUPPRESS
        legacy_kwargs["dest"] = action.dest
        legacy_kwargs["help"] = argparse.SUPPRESS
        parser.add_argument(legacy_flag, **legacy_kwargs)
    return action


def _build_parser(default_smiles: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MCS benchmark: nvmolkit GPU fMCS vs RDKit FindMCS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--smiles",
        default=str(default_smiles),
        help="Path to a SMILES file. --max-mols reservoir-samples across the whole file.",
    )
    input_group.add_argument(
        "--pickle",
        help="Path to pickled RDKit binary molecules. --max-mols samples across the whole file.",
    )
    _add_option(
        parser,
        "--max-mols",
        legacy_flags=("--max_mols",),
        dest="max_mols",
        type=int,
        default=1000,
        help=(
            "Number of molecules to sample from the input before atom/bond filtering; "
            "0 loads all molecules."
        ),
    )
    _add_option(
        parser,
        "--max-atoms",
        legacy_flags=("--max_atoms",),
        dest="max_atoms",
        type=int,
        default=128,
        help="Maximum atoms per molecule; 0 disables the filter.",
    )
    _add_option(
        parser,
        "--max-bonds",
        legacy_flags=("--max_bonds",),
        dest="max_bonds",
        type=int,
        default=128,
        help="Maximum bonds per molecule; 0 disables the filter.",
    )
    parser.add_argument("--pairs", type=int, default=1000, help="Number of random explicit index pairs")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed for molecule and pair selection")
    parser.add_argument("--runs", type=int, default=3, help="Timed runs")
    parser.add_argument("--warmups", type=int, default=1, help="nvmolkit warmup runs")
    gpu_group = parser.add_argument_group("nvmolkit GPU options")
    _add_option(
        gpu_group,
        "--batch-size",
        "-b",
        legacy_flags=("--batch_size",),
        dest="batch_size",
        type=int,
        default=0,
        help="fMCS GPU tier chunk size in pairs; 0 uses a default of 512 pairs per tier chunk.",
    )
    _add_option(
        gpu_group,
        "--block-size",
        legacy_flags=("--block_size",),
        dest="block_size",
        type=int,
        choices=[128, 512],
        default=128,
        help="CUDA threads per fMCS pair block. 512 is experimental and only supports maxSize tiers up to 64.",
    )
    _add_option(
        gpu_group,
        "--workers",
        dest="workers",
        type=int,
        default=-1,
        help="nvmolkit GPU worker threads per GPU; -1 autoselects.",
    )
    _add_option(
        gpu_group,
        "--prep-threads",
        legacy_flags=("--prep_threads",),
        dest="prep_threads",
        type=int,
        default=-1,
        help="nvmolkit CPU preprocessing threads; -1 autoselects.",
    )
    _add_option(
        gpu_group,
        "--num-gpus",
        legacy_flags=("--num_gpus",),
        dest="num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use; maps to device IDs [0, num_gpus).",
    )
    _add_option(
        gpu_group,
        "--slots-per-runner",
        "--executors-per-runner",
        legacy_flags=("--slots_per_runner", "--executors_per_runner"),
        dest="executors_per_runner",
        type=int,
        default=-1,
        help="Asynchronous fMCS executor slots/streams per GPU worker; -1 autoselects, valid explicit range is 1-8.",
    )
    gpu_group.add_argument(
        "--collect-timings",
        action="store_true",
        help="Collect per-pair nvmolkit backend elapsed milliseconds in --timings-csv.",
    )
    gpu_group.add_argument(
        "--collect-stats",
        action="store_true",
        help="Collect diagnostic per-pair fMCS queue/search counters in --timings-csv.",
    )
    _add_option(
        parser,
        "--atom-compare",
        legacy_flags=("--atom_compare",),
        dest="atom_compare",
        choices=["any", "elements", "isotopes", "any_heavy_atom"],
        default="elements",
        help="Atom comparison rule.",
    )
    _add_option(
        parser,
        "--bond-compare",
        legacy_flags=("--bond_compare",),
        dest="bond_compare",
        choices=["any", "order", "order_exact"],
        default="order",
        help="Bond comparison rule.",
    )
    _add_option(
        parser,
        "--match-valences",
        legacy_flags=("--match_valences",),
        dest="match_valences",
        action="store_true",
        help="Require matching atom total valence.",
    )
    _add_option(
        parser,
        "--match-formal-charge",
        legacy_flags=("--match_formal_charge",),
        dest="match_formal_charge",
        action="store_true",
        help="Require matching atom formal charge.",
    )
    _add_option(
        parser,
        "--ring-matches-ring-only",
        legacy_flags=("--ring_matches_ring_only",),
        dest="ring_matches_ring_only",
        action="store_true",
        help="Require ring atoms/bonds to match only ring atoms/bonds.",
    )
    _add_option(
        parser,
        "--complete-rings-only",
        legacy_flags=("--complete_rings_only",),
        dest="complete_rings_only",
        action="store_true",
        help="Require complete ring matches; this may use RDKit fallback.",
    )
    _add_option(
        parser,
        "--match-isotope",
        legacy_flags=("--match_isotope",),
        dest="match_isotope",
        action="store_true",
        help="Require matching isotope labels.",
    )
    _add_option(
        parser,
        "--timeout-seconds",
        legacy_flags=("--timeout_seconds",),
        dest="timeout_seconds",
        type=int,
        default=0,
        help="Per-pair timeout in seconds for nvmolkit and RDKit fallback.",
    )
    _add_option(
        parser,
        "--no-nvmolkit",
        legacy_flags=("--no_nvmolkit",),
        dest="no_nvmolkit",
        action="store_true",
        help="Skip nvmolkit benchmark.",
    )
    _add_option(
        parser,
        "--no-rdkit",
        legacy_flags=("--no_rdkit",),
        dest="no_rdkit",
        action="store_true",
        help="Skip RDKit benchmark.",
    )
    parser.add_argument("--validate", action="store_true", help="Compare nvmolkit atom/bond counts against RDKit")
    parser.add_argument(
        "--timings-csv",
        help="Write per-pair nvmolkit/RDKit timings and pair metadata to this CSV file.",
    )
    _add_option(
        parser,
        "--rdkit-threads",
        legacy_flags=("--rdkit_threads",),
        dest="rdkit_threads",
        type=int,
        default=1,
        help="RDKit multiprocessing worker count.",
    )
    _add_option(
        parser,
        "--rdkit-max-seconds",
        legacy_flags=("--rdkit_max_seconds",),
        dest="rdkit_max_seconds",
        type=float,
        default=0.0,
        help=(
            "Stop the RDKit comparison after this many wall-clock seconds and report throughput on the work "
            "actually completed. 0 disables the cap. For single-threaded RDKit MCS, the cap is checked "
            "between pairs."
        ),
    )
    return parser


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    default_smiles = repo_root / "benchmarks" / "data" / "chembl_10k.smi"

    parser = _build_parser(default_smiles)
    args = parser.parse_args()
    if args.num_gpus <= 0:
        parser.error("--num-gpus must be positive")

    mols = _load_molecules(args)
    _log(f"Sampling {args.pairs} explicit MCS pairs with seed={args.seed}")
    pairs = _sample_pairs(len(mols), args.pairs, args.seed)
    print(f"Prepared {len(mols)} molecules and {len(pairs)} explicit MCS pairs")

    nv_result = None
    if not args.no_nvmolkit:
        timing, nv_result = _bench_nvmolkit(mols, pairs, args)
        print(
            "nvmolkit: "
            f"median={timing.median_ms:.3f} ms mean={timing.mean_ms:.3f} ms std={timing.std_ms:.3f} ms "
            f"throughput={throughput_per_s(len(pairs), timing.median_ms):.2f} pairs/s"
        )

    rdkit_sizes = None
    rdkit_times_ms = None
    if not args.no_rdkit or args.validate:
        avg_ms, std_ms, rdkit_sizes, rdkit_times_ms, rdkit_pairs = _bench_rdkit(mols, pairs, args)
        print(
            "RDKit: "
            f"mean={avg_ms:.3f} ms std={std_ms:.3f} ms pairs={rdkit_pairs}/{len(pairs)} "
            f"throughput={throughput_per_s(rdkit_pairs, avg_ms):.2f} pairs/s"
        )

    if args.timings_csv:
        _write_pair_timings(args.timings_csv, mols, pairs, nv_result, rdkit_sizes, rdkit_times_ms)

    if args.validate:
        if rdkit_sizes is None:
            raise ValueError("Validation requires RDKit results")
        _log("Starting validation against RDKit atom/bond counts")
        _validate(nv_result, rdkit_sizes)
        _log("Finished validation")


if __name__ == "__main__":
    main()
