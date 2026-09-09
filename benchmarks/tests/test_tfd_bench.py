# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for deadline-aware RDKit TFD benchmark timing."""

from bench_utils import TimingResult
from tfd_bench import bench_rdkit_batch, run_benchmarks


class _Deadline:
    def __init__(self, expire_after: int | None = None) -> None:
        self._checks = 0
        self._expire_after = expire_after

    def expired(self) -> bool:
        self._checks += 1
        return self._expire_after is not None and self._checks >= self._expire_after


def _install_fake_timing(monkeypatch, deadline):
    captured = {}

    def fake_time_it(run, runs, warmups, max_seconds, progress_getter, progress_target):
        captured.update(
            runs=runs,
            warmups=warmups,
            max_seconds=max_seconds,
            progress_target=progress_target,
        )
        run(deadline)
        progress = progress_getter()
        return TimingResult(times_ms=[8.0], progress=progress, progress_target=progress_target)

    monkeypatch.setattr("tfd_bench.time_it", fake_time_it)
    return captured


def test_bench_rdkit_batch_tracks_full_progress_and_small_warmups(monkeypatch):
    measured = []
    monkeypatch.setattr(
        "tfd_bench.TorsionFingerprints.GetTFDMatrix",
        lambda mol, **kwargs: measured.append((mol, kwargs)),
    )
    captured = _install_fake_timing(monkeypatch, _Deadline())

    timing = bench_rdkit_batch(["a", "b", "c"], runs=4, warmups=2, max_seconds=3.0)

    assert timing.progress == 3
    assert not timing.truncated
    assert captured == {"runs": 4, "warmups": 0, "max_seconds": 3.0, "progress_target": 3}
    assert [mol for mol, _kwargs in measured] == ["a", "a", "a", "b", "c"]
    assert all(kwargs == {"useWeights": True, "maxDev": "equal"} for _mol, kwargs in measured)


def test_bench_rdkit_batch_stops_and_reports_partial_molecule_count(monkeypatch):
    measured = []
    monkeypatch.setattr(
        "tfd_bench.TorsionFingerprints.GetTFDMatrix",
        lambda mol, **_kwargs: measured.append(mol),
    )
    captured = _install_fake_timing(monkeypatch, _Deadline(expire_after=2))

    timing = bench_rdkit_batch(["a", "b", "c"], warmups=0, max_seconds=1.0)

    assert timing.progress == 2
    assert timing.truncated
    assert measured == ["a", "b"]
    assert captured["progress_target"] == 3


def test_bench_rdkit_batch_accepts_empty_workload(monkeypatch):
    monkeypatch.setattr(
        "tfd_bench.TorsionFingerprints.GetTFDMatrix",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected TFD call")),
    )
    _install_fake_timing(monkeypatch, _Deadline())

    timing = bench_rdkit_batch([], warmups=2, max_seconds=0.0)

    assert timing.progress == 0
    assert not timing.truncated


def test_run_benchmarks_uses_partial_rdkit_throughput_for_speedup(monkeypatch, tmp_path, capsys):
    class FakeMol:
        def __init__(self, conformers):
            self._conformers = conformers

        def GetNumConformers(self):
            return self._conformers

    def fake_rdkit(_mols, **_kwargs):
        return TimingResult(times_ms=[100.0], progress=1, progress_target=2)

    def fake_gpu_timing(_func, **_kwargs):
        return TimingResult(times_ms=[10.0])

    monkeypatch.setattr("tfd_bench.bench_rdkit_batch", fake_rdkit)
    monkeypatch.setattr("tfd_bench.time_it", fake_gpu_timing)

    dataframe = run_benchmarks(
        skip_rdkit=False,
        skip_nvmolkit=False,
        output_file=str(tmp_path / "results.csv"),
        mol_counts=[2],
        preloaded_mols=[FakeMol(3), FakeMol(5)],
        runs=1,
        warmups=0,
        rdkit_max_seconds=1.0,
    )

    row = dataframe.iloc[0]
    assert row["rdkit_molecules_processed"] == 1
    assert row["rdkit_pairs_processed"] == 3
    assert row["rdkit_truncated"] == 1
    assert row["rdkit_max_seconds"] == 1.0
    # RDKit: 3 pairs / 100 ms; GPU: 13 pairs / 10 ms => 43.3x.
    assert "43.3x" in capsys.readouterr().out
