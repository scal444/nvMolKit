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

"""Molecule preparation helpers shared across nvMolKit benchmarks."""

import os
import random
from functools import partial

from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom, rdForceFieldHelpers
from rdkit.Geometry import Point3D
from tqdm.contrib.concurrent import process_map

# Manually tuned so the per-conformer jitter recreates an ETKDGv3-like pairwise RMSD spread
JITTER_CENTER = 1.3
JITTER_SPREAD = 0.6


def _available_cpu_count() -> int:
    """Return the CPUs available to this process, respecting affinity when possible."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def prep_mols(
    mols: list[Chem.Mol],
    *,
    add_hs: bool = True,
    sanitize: bool = True,
    clear_conformers: bool = True,
) -> list[Chem.Mol]:
    """Return new molecule copies prepared for conformer generation.

    For each input:
        - skip ``None`` entries,
        - optionally add explicit hydrogens,
        - optionally sanitize (no-op when already sanitized),
        - optionally drop existing conformers so timed runs start clean.

    Molecules that fail any step are dropped and a count is printed.
    """
    prepped: list[Chem.Mol] = []
    drop_count = 0
    for mol in mols:
        if mol is None:
            drop_count += 1
            continue
        try:
            current = Chem.AddHs(mol) if add_hs else Chem.Mol(mol)
            if sanitize:
                Chem.SanitizeMol(current)
            if clear_conformers:
                current.RemoveAllConformers()
            prepped.append(current)
        except Exception:
            drop_count += 1
    if drop_count > 0:
        print(f"  Dropped {drop_count} molecules during prep (None or sanitize failure)")
    return prepped


def filter_force_field_mols(mols: list[Chem.Mol], force_field: str) -> list[Chem.Mol]:
    """Keep only molecules with complete RDKit parameters for ``force_field``."""
    force_field = force_field.lower()
    if force_field == "mmff":
        has_all_params = AllChem.MMFFHasAllMoleculeParams
    elif force_field == "uff":
        has_all_params = rdForceFieldHelpers.UFFHasAllMoleculeParams
    else:
        raise ValueError(f"Unknown force field: {force_field!r}")

    filtered = [mol for mol in mols if has_all_params(mol)]
    drop_count = len(mols) - len(filtered)
    if drop_count > 0:
        print(f"  Dropped {drop_count} molecules lacking {force_field.upper()} parameters")
    return filtered


def clone_mols_with_conformers(mols: list[Chem.Mol]) -> list[Chem.RWMol]:
    """Deep-copy molecules including their conformers.

    Useful for benches whose timed routines mutate conformer state in place
    (such as ETKDG embedding or FF optimization), so each iteration sees a
    pristine input.
    """
    return [Chem.RWMol(mol) for mol in mols]


def perturb_conformer(
    conf: Chem.Conformer,
    seed: int,
    center: float = JITTER_CENTER,
    spread: float = JITTER_SPREAD,
) -> None:
    """Apply per-atom uniform jitter to a conformer in place.

    A single half-width is drawn for the conformer as ``center * (1 + spread *
    U(-1, 1))`` and every x/y/z coordinate is then shifted by ``U(-half_width,
    half_width)``. Drawing a distinct half-width per conformer (each call uses
    a distinct ``seed``) gives a jittered ensemble a range of pairwise RMSDs
    rather than a single structure-independent value.
    """
    rng = random.Random(seed)
    half_width = max(0.0, center * (1.0 + spread * rng.uniform(-1.0, 1.0)))
    for atom_idx in range(conf.GetNumAtoms()):
        pos = conf.GetAtomPosition(atom_idx)
        conf.SetAtomPosition(
            atom_idx,
            Point3D(
                pos.x + rng.uniform(-half_width, half_width),
                pos.y + rng.uniform(-half_width, half_width),
                pos.z + rng.uniform(-half_width, half_width),
            ),
        )


def _embed_one(args_tuple: tuple[int, bytes], seed: int, add_hs: bool, min_atoms: int) -> bytes | None:
    """Embed a single ETKDGv3 conformer for one mol payload (multiprocessing worker)."""
    idx, mol_bytes = args_tuple
    mol = Chem.Mol(mol_bytes)
    if mol.GetNumAtoms() < min_atoms:
        return None
    if add_hs:
        mol = Chem.AddHs(mol)
    params = rdDistGeom.ETKDGv3()
    params.useRandomCoords = True
    params.randomSeed = seed + idx
    try:
        conf_id = rdDistGeom.EmbedMolecule(mol, params=params)
    except Exception:
        return None
    if conf_id < 0 or mol.GetNumConformers() == 0:
        return None
    if add_hs:
        mol = Chem.RemoveHs(mol)
    return mol.ToBinary()


def embed_and_jitter(
    mols: list[Chem.Mol],
    confs_per_mol: int,
    seed: int,
    num_workers: int = 0,
    add_hs: bool = False,
    min_atoms: int = 1,
    desc: str = "Embedding base conformers",
) -> list[Chem.Mol]:
    """Embed one ETKDGv3 base conformer per mol in parallel, then jitter to ``confs_per_mol``.

    The embed step runs across mols via ``process_map``; ``num_workers=0``
    uses every CPU available to the process. The jitter step is in-process
    and serial (cheap). Mols whose base embedding fails are dropped with a
    printed count. When ``add_hs`` is true, hydrogens are added before
    embedding and stripped from the returned mol.
    """
    if not mols:
        return []
    if confs_per_mol < 1:
        raise ValueError(f"confs_per_mol must be >= 1, got {confs_per_mol}")

    requested_workers = num_workers if num_workers > 0 else _available_cpu_count()
    workers = min(len(mols), requested_workers)
    binaries = [(i, mol.ToBinary()) for i, mol in enumerate(mols)]
    embedded_binaries = process_map(
        partial(_embed_one, seed=seed, add_hs=add_hs, min_atoms=min_atoms),
        binaries,
        max_workers=workers,
        # ETKDG cost varies substantially by molecule. Keep chunks bounded so
        # work is balanced and tqdm does not remain at 0% behind a huge chunk.
        chunksize=max(1, min(256, len(binaries) // (workers * 8) or 1)),
        desc=f"{desc} ({workers} workers)",
    )

    out: list[Chem.Mol] = []
    drop_count = 0
    for raw in embedded_binaries:
        if raw is None:
            drop_count += 1
            continue
        out.append(Chem.Mol(raw))
    if drop_count > 0:
        print(f"  Dropped {drop_count} molecules during embedding (no conformer generated)")

    if confs_per_mol > 1:
        for mol_idx, mol in enumerate(out):
            base_conf_id = mol.GetConformer().GetId()
            base_conf = mol.GetConformer(base_conf_id)
            for conf_idx in range(1, confs_per_mol):
                new_conf = Chem.Conformer(base_conf)
                perturb_conformer(new_conf, seed=seed + mol_idx * confs_per_mol + conf_idx)
                mol.AddConformer(new_conf, assignId=True)
            perturb_conformer(mol.GetConformer(base_conf_id), seed=seed + mol_idx * confs_per_mol)

    return out
