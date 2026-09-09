# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for benchmark callers of deadline-aware ``time_it``."""

from bench_utils import TimingResult
from etkdg_bench import bench_rdkit as bench_rdkit_etkdg
from ff_optimize_bench import bench_rdkit as bench_rdkit_ff
from substruct_bench import bench_rdkit_substruct


class _Deadline:
    def __init__(self, expire_after: int | None = None) -> None:
        self._checks = 0
        self._expire_after = expire_after

    def expired(self) -> bool:
        self._checks += 1
        return self._expire_after is not None and self._checks >= self._expire_after


def _fake_time_it(deadline):
    def invoke(run, **kwargs):
        run(deadline)
        progress = kwargs["progress_getter"]()
        return TimingResult(
            times_ms=[7.0],
            progress=progress,
            progress_target=kwargs["progress_target"],
        )

    return invoke


def test_etkdg_reports_molecules_from_partial_timed_run(monkeypatch):
    embedded = []
    monkeypatch.setattr("etkdg_bench.Chem.RWMol", lambda mol: f"copy-{mol}")
    monkeypatch.setattr(
        "etkdg_bench.rdDistGeom.EmbedMultipleConfs",
        lambda mol, numConfs, params: embedded.append((mol, numConfs, params)),
    )
    monkeypatch.setattr("etkdg_bench.time_it", _fake_time_it(_Deadline(expire_after=2)))

    timing, measured_mols, measured_count = bench_rdkit_etkdg(
        ["a", "b", "c"],
        params="params",
        confs_per_mol=4,
        runs=3,
        warmup=False,
        max_seconds=1.0,
    )

    assert timing.truncated
    assert measured_count == 2
    assert measured_mols == ["copy-a", "copy-b"]
    assert embedded == [("copy-a", 4, "params"), ("copy-b", 4, "params")]


def test_force_field_reports_energies_from_partial_timed_run(monkeypatch):
    optimized = []
    monkeypatch.setattr("ff_optimize_bench.Chem.RWMol", lambda mol: f"copy-{mol}")

    def optimize(mol, numThreads, maxIters):
        optimized.append((mol, numThreads, maxIters))
        return [(0, float(len(optimized)))]

    monkeypatch.setattr("ff_optimize_bench.AllChem.MMFFOptimizeMoleculeConfs", optimize)
    monkeypatch.setattr("ff_optimize_bench.time_it", _fake_time_it(_Deadline(expire_after=2)))

    avg_ms, std_ms, energies, measured_count = bench_rdkit_ff(
        ["a", "b", "c"],
        ff="mmff",
        max_iters=20,
        runs=3,
        warmup=False,
        num_threads=1,
        max_seconds=1.0,
    )

    assert (avg_ms, std_ms, measured_count) == (7.0, 0.0, 2)
    assert energies == [1.0, 2.0]
    assert optimized == [("copy-a", 1, 20), ("copy-b", 1, 20)]


def test_substructure_reports_partial_results_and_pair_count(monkeypatch):
    class FakeMol:
        def __init__(self, name):
            self.name = name

        def HasSubstructMatch(self, query, _params):
            return (self.name, query)

    monkeypatch.setattr("substruct_bench._time_it", _fake_time_it(_Deadline(expire_after=3)))

    avg_ms, std_ms, results, measured_pairs = bench_rdkit_substruct(
        [FakeMol("a"), FakeMol("b"), FakeMol("c")],
        queries=["q1", "q2"],
        runs=3,
        mode="hasSubstructMatch",
        max_matches=0,
        threads=1,
        max_seconds=1.0,
    )

    assert (avg_ms, std_ms, measured_pairs) == (7.0, 0.0, 4)
    assert results == [[("a", "q1"), ("a", "q2")], [("b", "q1"), ("b", "q2")]]
