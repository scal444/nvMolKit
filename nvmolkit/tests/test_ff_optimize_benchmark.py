# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only regression tests for force-field benchmark mode routing."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

from rdkit import Chem

from nvmolkit.types import PrecisionMode, PrecisionOptions

def _load_benchmark_module():
    benchmark_dir = Path(__file__).resolve().parents[2] / "benchmarks"
    sys.path.insert(0, str(benchmark_dir))
    try:
        return importlib.import_module("ff_optimize_bench")
    finally:
        sys.path.pop(0)


def test_compound_benchmark_modes_resolve_expected_profiles():
    benchmark = _load_benchmark_module()
    expected = {
        "bfgs_full": ("BFGS", PrecisionMode.LEGACY),
        "bfgs_fp32_hessian": ("BFGS", PrecisionMode.HESSIAN_F32),
        "bfgs_fp32_minimizer": ("BFGS", PrecisionMode.MINIMIZER_F32),
        "bfgs_fp32_forcefield": ("BFGS", PrecisionMode.FORCEFIELD_F32),
        "bfgs_single": ("BFGS", PrecisionMode.SINGLE),
        "fire_full": ("FIRE", PrecisionMode.LEGACY),
        "fire_single": ("FIRE", PrecisionMode.MINIMIZER_F32),
        "fire_single_forcefield": ("FIRE", PrecisionMode.FORCEFIELD_F32),
        "fire_single_both": ("FIRE", PrecisionMode.SINGLE),
    }
    assert benchmark.BENCHMARK_MODES == expected


def test_multiple_modes_clone_the_same_starting_conformers(monkeypatch):
    benchmark = _load_benchmark_module()
    mmff_module = importlib.import_module("nvmolkit.mmffOptimization")
    source = Chem.AddHs(Chem.MolFromSmiles("CC"))
    conformer = Chem.Conformer(source.GetNumAtoms())
    for atom_idx in range(source.GetNumAtoms()):
        conformer.SetAtomPosition(atom_idx, (float(atom_idx), 0.25 * atom_idx, -0.5 * atom_idx))
    source.AddConformer(conformer)
    expected_start = tuple(source.GetConformer().GetAtomPosition(0))
    calls = []

    def fake_optimize(mols, *, minimizerKind, precisionOptions, **kwargs):
        calls.append(
            (
                tuple(mols[0].GetConformer().GetAtomPosition(0)),
                minimizerKind,
                precisionOptions.mode,
            )
        )
        mols[0].GetConformer().SetAtomPosition(0, (999.0, 999.0, 999.0))
        return [[1.0]]

    def fake_time_it(function, **kwargs):
        function()
        return SimpleNamespace(mean_ms=1.0, std_ms=0.0)

    monkeypatch.setattr(mmff_module, "MMFFOptimizeMoleculesConfs", fake_optimize)
    monkeypatch.setattr(benchmark, "time_it", fake_time_it)

    for minimizer_kind, precision_mode in (
        ("BFGS", PrecisionMode.LEGACY),
        ("FIRE", PrecisionMode.MINIMIZER_F32),
    ):
        benchmark.bench_nvmolkit(
            [source],
            "mmff",
            minimizer_kind,
            PrecisionOptions(precision_mode),
            max_iters=3,
            hardware_options=None,
            runs=1,
            warmup=False,
        )

    assert calls == [
        (expected_start, "BFGS", PrecisionMode.LEGACY),
        (expected_start, "FIRE", PrecisionMode.MINIMIZER_F32),
    ]
    assert tuple(source.GetConformer().GetAtomPosition(0)) == expected_start
