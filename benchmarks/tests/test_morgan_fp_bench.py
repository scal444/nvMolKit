# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from morgan_fp_bench import _build_parser, _result_row, _validate_args, _validate_fingerprints
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


def _args(**overrides):
    values = {
        "num_mols": 0,
        "radius": 2,
        "rdkit_threads": 1,
        "prep_threads": 0,
        "gpu_id": 0,
        "runs": 3,
        "warmups": 1,
        "validation_mols": 100,
        "no_nvmolkit": False,
        "no_rdkit": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_parser_uses_scalar_thread_settings_and_common_input_options():
    """Thread controls accept exactly one value per invocation."""
    args = _build_parser().parse_args(
        ["--smiles", "input.smi", "--rdkit_threads", "8", "--prep_threads", "4", "--gpu_id", "2"]
    )

    assert args.rdkit_threads == 8
    assert args.prep_threads == 4
    assert args.gpu_id == 2
    assert args.smiles == "input.smi"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"radius": -1}, "radius"),
        ({"rdkit_threads": -1}, "rdkit_threads"),
        ({"prep_threads": -1}, "prep_threads"),
        ({"gpu_id": -1}, "gpu_id"),
        ({"no_rdkit": True, "no_nvmolkit": True}, "disable both"),
    ],
)
def test_argument_validation_rejects_invalid_settings(overrides, message):
    """Invalid compute settings fail before molecule loading."""
    with pytest.raises(ValueError, match=message):
        _validate_args(_args(**overrides))


def test_fingerprint_validation_accepts_matching_packed_bits_and_rejects_mismatch():
    """Validation compares nvMolKit's packed representation bit for bit."""
    from nvmolkit.fingerprints import pack_fingerprint

    mols = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("c1ccccc1")]
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=128)
    rdkit_fps = generator.GetFingerprints(mols, numThreads=1)
    bits = torch.tensor([fp.ToList() for fp in rdkit_fps], dtype=torch.bool)
    packed = pack_fingerprint(bits)

    assert _validate_fingerprints(rdkit_fps, packed, max_count=0) == 2

    bits[0, 0] = ~bits[0, 0]
    mismatched = pack_fingerprint(bits)
    with pytest.raises(AssertionError, match="fingerprint 0"):
        _validate_fingerprints(rdkit_fps, mismatched, max_count=0)


def test_result_rows_record_backend_specific_thread_and_gpu_settings():
    """CSV rows label settings only for the backend that consumes them."""
    from bench_utils import TimingResult

    args = _args()
    args.fp_size = 2048
    timing = TimingResult(times_ms=[10.0, 12.0, 11.0])

    rdkit = _result_row("rdkit", timing, input_file="mols.smi", input_type="smiles", num_mols=100, args=args)
    nvmolkit = _result_row(
        "nvmolkit",
        timing,
        input_file="mols.smi",
        input_type="smiles",
        num_mols=100,
        args=args,
        rdkit_mols_per_second=5000.0,
    )

    assert rdkit["rdkit_threads"] == 1
    assert rdkit["prep_threads"] == "N/A"
    assert nvmolkit["rdkit_threads"] == "N/A"
    assert nvmolkit["prep_threads"] == 0
    assert nvmolkit["gpu_id"] == 0
    assert nvmolkit["vs_rdkit_throughput_ratio"] == 1.8182
