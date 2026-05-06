# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Final-energy comparison plot: nvmolkit FIRE 2.0 vs RDKit MMFF94 across many conformers.

Loads conformers from an SDF, optimizes each conformer with both backends, and
writes a scatter plot (RDKit vs nvmolkit) and a histogram of the per-conformer
energy difference. Useful as a regression bench when changing FIRE internals.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from _fire_options_cli import add_fire_options_args, fire_options_from_args
from rdkit import Chem
from rdkit.Chem import AllChem

from nvmolkit.mmffOptimization import FireOptions, MMFFOptimizeMoleculesConfsFire


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_sdf", type=Path, help="Input SDF with conformers.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("nvmol_compare"),
        help="Directory for output figures (default: ./nvmol_compare).",
    )
    parser.add_argument("--max-iters", type=int, default=1000, help="Maximum iterations (default: 1000).")
    parser.add_argument(
        "--max-mols",
        type=int,
        default=None,
        help="Limit the number of molecules processed (default: all).",
    )
    add_fire_options_args(parser)
    return parser.parse_args()


def load_mols(sdf_path: Path, max_mols: int | None) -> list[Chem.Mol]:
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mols = []
    for mol in supplier:
        if mol is None:
            continue
        mols.append(mol)
        if max_mols is not None and len(mols) >= max_mols:
            break
    if not mols:
        raise ValueError(f"No molecules in {sdf_path}.")
    return mols


def rdkit_energies(mols: list[Chem.Mol], max_iters: int) -> np.ndarray:
    energies: list[float] = []
    for mol in mols:
        results = AllChem.MMFFOptimizeMoleculeConfs(Chem.Mol(mol), maxIters=max_iters, mmffVariant="MMFF94")
        for _, energy in results:
            energies.append(energy if energy is not None else float("nan"))
    return np.asarray(energies, dtype=float)


def nvmolkit_energies(mols: list[Chem.Mol], fire_opts: FireOptions, max_iters: int) -> np.ndarray:
    nv_mols = [Chem.Mol(mol) for mol in mols]
    energies_nested = MMFFOptimizeMoleculesConfsFire(nv_mols, maxIters=max_iters, fireOptions=fire_opts)
    flat: list[float] = []
    for energies in energies_nested:
        for energy in energies:
            flat.append(energy if energy is not None else float("nan"))
    return np.asarray(flat, dtype=float)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mols = load_mols(args.input_sdf, args.max_mols)
    total_confs = sum(mol.GetNumConformers() for mol in mols)
    print(f"Loaded {len(mols)} molecules with {total_confs} conformers from {args.input_sdf}.")

    fire_opts = fire_options_from_args(args)

    t0 = time.perf_counter()
    rdkit_arr = rdkit_energies(mols, args.max_iters)
    rdkit_elapsed = time.perf_counter() - t0

    t0 = time.perf_counter()
    nv_arr = nvmolkit_energies(mols, fire_opts, args.max_iters)
    nv_elapsed = time.perf_counter() - t0

    if rdkit_arr.size != nv_arr.size:
        raise RuntimeError(f"Energy size mismatch: rdkit={rdkit_arr.size}, nvmolkit={nv_arr.size}")

    diff = nv_arr - rdkit_arr
    finite = np.isfinite(diff)
    median = float(np.nanmedian(diff))
    mean = float(np.nanmean(diff))
    p95 = float(np.nanpercentile(np.abs(diff), 95)) if finite.any() else float("nan")

    print(f"RDKit:    {rdkit_elapsed:.2f}s   nvmolkit FIRE: {nv_elapsed:.2f}s")
    print(f"Energy diff (nvmolkit - rdkit) median={median:+.4f}, mean={mean:+.4f}, |.|p95={p95:.4f} kcal/mol")

    plt.figure(figsize=(6, 6))
    plt.scatter(rdkit_arr, nv_arr, s=8, alpha=0.6)
    lo = float(min(np.nanmin(rdkit_arr), np.nanmin(nv_arr)))
    hi = float(max(np.nanmax(rdkit_arr), np.nanmax(nv_arr)))
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    plt.xlabel("RDKit MMFF94 energy (kcal/mol)")
    plt.ylabel("nvmolkit FIRE energy (kcal/mol)")
    plt.title(f"{rdkit_arr.size} conformers")
    plt.tight_layout()
    plt.savefig(args.output_dir / "scatter.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(diff[finite], bins=80)
    plt.axvline(0.0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("nvmolkit - rdkit (kcal/mol)")
    plt.ylabel("count")
    plt.title(f"Energy difference (median {median:+.3f}, |.|p95 {p95:.3f})")
    plt.tight_layout()
    plt.savefig(args.output_dir / "diff_hist.png", dpi=150)
    plt.close()

    np.save(args.output_dir / "rdkit_energies.npy", rdkit_arr)
    np.save(args.output_dir / "nvmolkit_energies.npy", nv_arr)
    print(f"Wrote figures and arrays to {args.output_dir}/")


if __name__ == "__main__":
    main()
