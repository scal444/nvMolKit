"""Tiny ABI + kernel-launch smoke for an installed nvmolkit wheel.

Imports nvmolkit, generates a Morgan fingerprint for a single small molecule
on the GPU, materializes the result on the host and prints a one-line summary.
Exits 0 on success, non-zero on failure.

Catches:
- Missing or mismatched rdkit/boost/cuda symbols (import-time link errors).
- ``no kernel image is available for execution on the device`` (sm_120 etc.
  on RTX 50-series when the wheel was built without that arch).
- ``__cuda_array_interface__`` plumbing or torch tensor roundtrip regressions.
"""

from __future__ import annotations

import sys

from rdkit import Chem
import torch

import nvmolkit
from nvmolkit.fingerprints import MorganFingerprintGenerator


def main() -> int:
    if not torch.cuda.is_available():
        print("FAIL: torch.cuda.is_available() is False", file=sys.stderr)
        return 2

    device_name = torch.cuda.get_device_name(0)
    nvmolkit_version = getattr(nvmolkit, "__version__", "unknown")
    rdkit_version_str = Chem.rdBase.rdkitVersion

    mol = Chem.MolFromSmiles("CCO")
    if mol is None:
        print("FAIL: rdkit could not parse 'CCO'", file=sys.stderr)
        return 3

    gen = MorganFingerprintGenerator(radius=2, fpSize=2048)
    handle = gen.GetFingerprints([mol])
    torch.cuda.synchronize()
    tensor = handle.torch()

    expected_shape = (1, 2048 // 32)
    if tuple(tensor.shape) != expected_shape:
        print(
            f"FAIL: fingerprint shape {tuple(tensor.shape)} != expected {expected_shape}",
            file=sys.stderr,
        )
        return 4

    nonzero_bits = int((tensor != 0).sum().item())
    if nonzero_bits == 0:
        print("FAIL: all-zero fingerprint for ethanol", file=sys.stderr)
        return 5

    print(
        f"OK nvmolkit={nvmolkit_version} rdkit={rdkit_version_str} "
        f"device={device_name} fp_shape={tuple(tensor.shape)} "
        f"nonzero_ints={nonzero_bits}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
