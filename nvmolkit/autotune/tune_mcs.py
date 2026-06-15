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

"""Autotune wrapper for :func:`nvmolkit.mcs.findMCS`.

The tuner uses the explicit-pair route because that is the underlying batch
shape for MCS execution. Throughput is reported in molecule pairs per second.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

from rdkit.Chem import Mol

from nvmolkit.autotune._calibration import normalize_calibration_set
from nvmolkit.autotune._core import (
    CalibrationState,
    TuneResult,
    _require_optuna,
    collect_int_from_space,
    resolve_search_space,
    run_study,
    suggest_preprocessing_threads_with_cpu_budget,
    suggest_from_space,
)
from nvmolkit.autotune._ff_common import resolve_cpu_budget, resolve_num_gpus
from nvmolkit.mcs import MCSConfig, findMCS


def _default_mcs_search_space(num_gpus: int, cpus: int) -> dict:
    """Build the MCS execution search space scaled to the active hardware.

    MCS runs ``numGpus * workerThreads`` GPU coordinator threads and
    ``preprocessingThreads`` CPU preprocessing threads concurrently, so the two
    pools share the CPU budget. ``executorsPerRunner`` controls stream-level
    overlap within each runner and is not counted as a CPU pool.
    """
    per_gpu_worker_max = max(1, min(8, cpus // max(1, num_gpus)))
    return {
        "batchSize": [0, 64, 128, 256, 512],
        "blockSize": {"choices": [128, 512]},
        "workerThreads": (1, per_gpu_worker_max),
        "preprocessingThreads": (1, cpus),
        "executorsPerRunner": (1, 8),
    }


def _coerce_pairs(pairs: Sequence[Sequence[int]], num_mols: int) -> tuple[tuple[int, int], ...]:
    out: list[tuple[int, int]] = []
    for pair_idx, pair in enumerate(pairs):
        if len(pair) != 2:
            raise ValueError(f"pairs[{pair_idx}] must contain exactly two molecule indices")
        idx_a = int(pair[0])
        idx_b = int(pair[1])
        if not (0 <= idx_a < num_mols and 0 <= idx_b < num_mols):
            raise IndexError(f"pairs[{pair_idx}] contains a molecule index outside [0, {num_mols})")
        out.append((idx_a, idx_b))
    return tuple(out)


def tune_mcs(
    mols: Sequence[Mol],
    pairs: Sequence[Sequence[int]],
    *,
    atom_compare: str = "elements",
    bond_compare: str = "order",
    match_valences: bool = False,
    match_formal_charge: bool = False,
    ring_matches_ring_only: bool | None = None,
    atom_ring_matches_ring_only: bool | None = None,
    bond_ring_matches_ring_only: bool | None = None,
    complete_rings_only: bool | None = None,
    atom_complete_rings_only: bool | None = None,
    bond_complete_rings_only: bool | None = None,
    match_isotope: bool = False,
    maximize_bonds: bool = True,
    connected_only: bool = True,
    require_gpu: bool = False,
    timeout_seconds: int = 0,
    gpuIds: Optional[Iterable[int]] = None,
    calibration_set: Optional[Iterable[int]] = None,
    calibration_fraction: float = 0.1,
    calibration_max_size: int = 2000,
    target_seconds_per_trial: float = 10.0,
    n_trials: int = 30,
    search_space_overrides: Optional[dict[str, Any]] = None,
    cpu_budget: Optional[int] = None,
    sampler: Any = None,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> TuneResult:
    """Tune :class:`MCSConfig` for an explicit-pair MCS workflow.

    Args:
        mols: Molecule table referenced by ``pairs``.
        pairs: Explicit ``(left_index, right_index)`` pair specifications.
        atom_compare: ``"any"``, ``"elements"``, ``"isotopes"``, or
            ``"any_heavy_atom"``.
        bond_compare: ``"any"``, ``"order"``, or ``"order_exact"``.
        match_valences: Match atom total valence.
        match_formal_charge: Match atom formal charge.
        ring_matches_ring_only: Convenience value applied to both atom and bond
            ring matching unless the axis-specific arguments are supplied.
        atom_ring_matches_ring_only: Match ring atoms only to ring atoms.
        bond_ring_matches_ring_only: Match ring bonds only to ring bonds.
        complete_rings_only: Convenience value applied to both atom and bond
            complete-ring settings unless axis-specific values are supplied.
        atom_complete_rings_only: Delegate atom complete-ring matching to RDKit.
        bond_complete_rings_only: Delegate bond complete-ring matching to RDKit.
        match_isotope: Match isotope labels in addition to ``atom_compare``.
        maximize_bonds: Maximize bonds, matching RDKit's default fMCS objective.
        connected_only: Require connected MCS.
        require_gpu: Raise instead of using RDKit fallback for unsupported or
            overflowed pairs.
        timeout_seconds: Per-pair timeout in seconds.
        gpuIds: GPU device IDs to use. Fixed across the study.
        calibration_set: Optional explicit indices into ``pairs``.
        calibration_fraction: Fraction of pair workload to auto-sample.
        calibration_max_size: Cap on the auto-sampled pair workload.
        target_seconds_per_trial: Target wall-clock budget for one trial.
        n_trials: Number of Optuna trials to run after warm-up.
        search_space_overrides: Optional overrides for ``batchSize``,
            ``blockSize``, ``workerThreads``, ``preprocessingThreads``, or
            ``executorsPerRunner``.
        cpu_budget: Optional explicit cap on total CPU threads.
        sampler: Optional Optuna sampler.
        seed: Seed for the default sampler.
        verbose: Print warm-up and trial diagnostics.

    Returns:
        :class:`TuneResult` with ``best_config`` set to a fully-populated
        :class:`MCSConfig` instance.
    """
    optuna = _require_optuna()  # noqa: F841

    mol_list = list(mols)
    if not mol_list:
        raise ValueError("mols must be non-empty for autotuning")

    pair_list = _coerce_pairs(pairs, len(mol_list))
    if not pair_list:
        raise ValueError("pairs must be non-empty for autotuning")

    indices = normalize_calibration_set(
        calibration_set,
        len(pair_list),
        fraction=calibration_fraction,
        max_size=calibration_max_size,
    )
    fixed_gpu_ids = list(gpuIds) if gpuIds is not None else []
    num_gpus = resolve_num_gpus(fixed_gpu_ids)
    cpus = resolve_cpu_budget(cpu_budget)
    space = resolve_search_space(_default_mcs_search_space(num_gpus, cpus), search_space_overrides)

    def _make_config(values: dict[str, Any]) -> MCSConfig:
        return MCSConfig(
            batchSize=int(values.get("batchSize", 0)),
            blockSize=int(values.get("blockSize", 128)),
            workerThreads=int(values.get("workerThreads", -1)),
            preprocessingThreads=int(values.get("preprocessingThreads", -1)),
            executorsPerRunner=int(values.get("executorsPerRunner", -1)),
            gpuIds=fixed_gpu_ids if fixed_gpu_ids else None,
        )

    def _run_once(config: MCSConfig, state: CalibrationState) -> int:
        selected_pairs = [pair_list[i] for i in state.indices]
        findMCS(
            mol_list,
            mode="pairs",
            pairs=selected_pairs,
            atom_compare=atom_compare,
            bond_compare=bond_compare,
            match_valences=match_valences,
            match_formal_charge=match_formal_charge,
            ring_matches_ring_only=ring_matches_ring_only,
            atom_ring_matches_ring_only=atom_ring_matches_ring_only,
            bond_ring_matches_ring_only=bond_ring_matches_ring_only,
            complete_rings_only=complete_rings_only,
            atom_complete_rings_only=atom_complete_rings_only,
            bond_complete_rings_only=bond_complete_rings_only,
            match_isotope=match_isotope,
            maximize_bonds=maximize_bonds,
            connected_only=connected_only,
            require_gpu=require_gpu,
            timeout_seconds=timeout_seconds,
            config=config,
        )
        return len(selected_pairs)

    def default_runner(state: CalibrationState) -> int:
        return _run_once(
            MCSConfig(gpuIds=fixed_gpu_ids if fixed_gpu_ids else None),
            state,
        )

    def trial_runner(trial, state: CalibrationState) -> int:
        values: dict[str, Any] = {}
        for name, spec in space.items():
            if name == "preprocessingThreads":
                continue
            values[name] = suggest_from_space(trial, name, spec)
        worker_threads = int(values.get("workerThreads", 1))
        prep_spec = space.get("preprocessingThreads")
        if prep_spec is not None:
            values["preprocessingThreads"] = suggest_preprocessing_threads_with_cpu_budget(
                trial,
                prep_spec,
                worker_threads=worker_threads,
                num_gpus=num_gpus,
                cpus=cpus,
            )
        return _run_once(_make_config(values), state)

    def build_config(params_dict: dict[str, Any]) -> MCSConfig:
        merged = {name: params_dict.get(name, collect_int_from_space(spec)) for name, spec in space.items()}
        return _make_config(merged)

    initial_state = CalibrationState(indices=list(indices))
    return run_study(
        default_runner=default_runner,
        trial_runner=trial_runner,
        build_config=build_config,
        initial_state=initial_state,
        n_trials=n_trials,
        target_seconds_per_trial=target_seconds_per_trial,
        sampler=sampler,
        seed=seed,
        verbose=verbose,
    )
