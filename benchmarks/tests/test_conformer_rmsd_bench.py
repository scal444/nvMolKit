# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for deadline-aware RDKit conformer RMSD benchmark timing."""

from bench_utils import TimingResult
from conformer_rmsd_bench import bench_rdkit_batch


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
        captured.update(runs=runs, max_seconds=max_seconds, progress_target=progress_target)
        run(deadline)
        progress = progress_getter()
        return TimingResult(times_ms=[12.0], progress=progress, progress_target=progress_target)

    monkeypatch.setattr("conformer_rmsd_bench.time_it", fake_time_it)
    return captured


def test_bench_rdkit_batch_reports_full_progress_and_uses_one_item_warmup(monkeypatch):
    converted = []
    measured = []
    monkeypatch.setattr("conformer_rmsd_bench.Chem.Mol", lambda payload: converted.append(payload) or payload)
    monkeypatch.setattr(
        "conformer_rmsd_bench.AllChem.GetConformerRMSMatrix",
        lambda mol, prealigned: measured.append((mol, prealigned)),
    )
    captured = _install_fake_timing(monkeypatch, _Deadline())

    timing, processed = bench_rdkit_batch([b"a", b"b", b"c"], runs=4, warmup=True, max_seconds=2.5)

    assert timing.times_ms == [12.0]
    assert processed == 3
    assert captured == {"runs": 4, "max_seconds": 2.5, "progress_target": 3}
    assert converted == [b"a", b"a", b"b", b"c"]
    assert measured == [(b"a", False), (b"a", False), (b"b", False), (b"c", False)]


def test_bench_rdkit_batch_reports_progress_from_truncated_sample(monkeypatch):
    measured = []
    monkeypatch.setattr("conformer_rmsd_bench.Chem.Mol", lambda payload: payload)
    monkeypatch.setattr(
        "conformer_rmsd_bench.AllChem.GetConformerRMSMatrix",
        lambda mol, prealigned: measured.append(mol),
    )
    captured = _install_fake_timing(monkeypatch, _Deadline(expire_after=2))

    timing, processed = bench_rdkit_batch([b"a", b"b", b"c"], warmup=False, max_seconds=1.0)

    assert timing.median_ms == 12.0
    assert processed == 2
    assert measured == [b"a", b"b"]
    assert captured["progress_target"] == 3


def test_bench_rdkit_batch_accepts_empty_workload(monkeypatch):
    monkeypatch.setattr(
        "conformer_rmsd_bench.AllChem.GetConformerRMSMatrix",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected RMSD call")),
    )
    captured = _install_fake_timing(monkeypatch, _Deadline())

    timing, processed = bench_rdkit_batch([], warmup=True, max_seconds=0.0)

    assert timing.times_ms == [12.0]
    assert processed == 0
    assert captured["progress_target"] == 0
