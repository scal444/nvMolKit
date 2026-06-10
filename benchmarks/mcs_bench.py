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
import random
from multiprocessing import Pool
from pathlib import Path

from bench_utils import add_rdkit_max_seconds_arg, load_pickle, load_smiles, throughput_per_s, time_it, time_it_bounded
from rdkit import Chem
from rdkit.Chem import rdFMCS

from nvmolkit.mcs import findMCS

_worker_mols: list[Chem.Mol] | None = None
_worker_params: rdFMCS.MCSParameters | None = None


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


def _rdkit_worker_pair(pair: tuple[int, int]) -> tuple[int, int]:
    if _worker_mols is None or _worker_params is None:
        raise RuntimeError("RDKit MCS worker was not initialized")
    result = rdFMCS.FindMCS([_worker_mols[pair[0]], _worker_mols[pair[1]]], _worker_params)
    return int(result.numAtoms), int(result.numBonds)


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


def _bench_nvmolkit(mols: list[Chem.Mol], pairs: list[tuple[int, int]], args: argparse.Namespace):
    last_result = None

    def run():
        nonlocal last_result
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
            require_gpu=args.require_gpu,
            timeout_seconds=args.timeout_seconds,
            batch_size=args.batch_size,
            executors_per_runner=args.executors_per_runner,
        )

    timing = time_it(run, runs=args.runs, warmups=args.warmups, gpu_sync=True)
    return timing, last_result


def _bench_rdkit(mols: list[Chem.Mol], pairs: list[tuple[int, int]], args: argparse.Namespace):
    params = _rdkit_params(args)
    sizes: list[tuple[int, int]] = []
    pairs_done = 0

    if args.rdkit_threads > 1:
        mol_binaries = [mol.ToBinary() for mol in mols]

        def run(_deadline):
            nonlocal sizes, pairs_done
            with Pool(args.rdkit_threads, initializer=_rdkit_worker_init, initargs=(mol_binaries, params)) as pool:
                sizes = pool.map(_rdkit_worker_pair, pairs)
            pairs_done = len(pairs)

    else:

        def run(deadline):
            nonlocal sizes, pairs_done
            sizes = []
            pairs_done = 0
            for idx_a, idx_b in pairs:
                if deadline.expired():
                    break
                result = rdFMCS.FindMCS([mols[idx_a], mols[idx_b]], params)
                sizes.append((int(result.numAtoms), int(result.numBonds)))
                pairs_done += 1

    avg_ms, std_ms, last_pairs = time_it_bounded(run, args.runs, args.rdkit_max_seconds, lambda: pairs_done, len(pairs))
    return avg_ms, std_ms, sizes, last_pairs


def _validate(nv_result, rdkit_sizes: list[tuple[int, int]]) -> None:
    if nv_result is None:
        raise ValueError("Validation requires nvmolkit results")
    if len(rdkit_sizes) < len(nv_result):
        print(f"Validation skipped: RDKit completed only {len(rdkit_sizes)} / {len(nv_result)} pairs")
        return

    mismatches = []
    for i, (rd_atoms, rd_bonds) in enumerate(rdkit_sizes):
        if int(nv_result.num_atoms[i]) != rd_atoms or int(nv_result.num_bonds[i]) != rd_bonds:
            mismatches.append((i, nv_result.pairs[i], int(nv_result.num_atoms[i]), int(nv_result.num_bonds[i]), rd_atoms, rd_bonds))
            if len(mismatches) >= 10:
                break
    if mismatches:
        print("Validation failed; first mismatches:")
        for idx, pair, nv_atoms, nv_bonds, rd_atoms, rd_bonds in mismatches:
            print(f"  pair {idx} {pair}: nvmolkit={nv_atoms}/{nv_bonds} RDKit={rd_atoms}/{rd_bonds}")
        raise AssertionError(f"{len(mismatches)} MCS validation mismatches")
    print(f"Validation passed for {len(rdkit_sizes)} pairs")


def _load_molecules(args: argparse.Namespace) -> list[Chem.Mol]:
    if args.pickle:
        mols = load_pickle(args.pickle, max_count=args.max_mols, seed=args.seed)
    else:
        mols = load_smiles(args.smiles, max_count=args.max_mols, seed=args.seed)
    return _filter_molecules(mols, args.max_atoms, args.max_bonds)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    default_smiles = repo_root / "benchmarks" / "data" / "chembl_10k.smi"

    parser = argparse.ArgumentParser(description="MCS benchmark: nvmolkit GPU fMCS vs RDKit FindMCS")
    parser.add_argument("--smiles", default=str(default_smiles), help="Path to a SMILES file")
    parser.add_argument("--pickle", help="Path to pickled RDKit binary molecules")
    parser.add_argument("--max_mols", type=int, default=1000, help="Maximum molecules to load before filtering")
    parser.add_argument("--max_atoms", type=int, default=128, help="Maximum atoms per molecule; 0 disables")
    parser.add_argument("--max_bonds", type=int, default=128, help="Maximum bonds per molecule; 0 disables")
    parser.add_argument("--pairs", type=int, default=1000, help="Number of random explicit index pairs")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument("--runs", type=int, default=3, help="Timed runs")
    parser.add_argument("--warmups", type=int, default=1, help="nvmolkit warmup runs")
    parser.add_argument("--batch_size", type=int, default=0, help="Native GPU batch chunk size; 0 auto-selects")
    parser.add_argument("--executors_per_runner", type=int, default=1, help="Asynchronous fMCS executor streams")
    parser.add_argument("--atom_compare", choices=["any", "elements", "isotopes", "any_heavy_atom"], default="elements")
    parser.add_argument("--bond_compare", choices=["any", "order", "order_exact"], default="order")
    parser.add_argument("--match_valences", action="store_true")
    parser.add_argument("--match_formal_charge", action="store_true")
    parser.add_argument("--ring_matches_ring_only", action="store_true")
    parser.add_argument("--complete_rings_only", action="store_true")
    parser.add_argument("--match_isotope", action="store_true")
    parser.add_argument("--require_gpu", action="store_true", help="Fail instead of using RDKit fallback inside nvmolkit")
    parser.add_argument("--timeout_seconds", type=int, default=0, help="RDKit fallback timeout in seconds")
    parser.add_argument("--no_nvmolkit", action="store_true", help="Skip nvmolkit benchmark")
    parser.add_argument("--no_rdkit", action="store_true", help="Skip RDKit benchmark")
    parser.add_argument("--validate", action="store_true", help="Compare nvmolkit atom/bond counts against RDKit")
    parser.add_argument("--rdkit_threads", type=int, default=1, help="RDKit multiprocessing worker count")
    add_rdkit_max_seconds_arg(parser, extra_help="For single-threaded RDKit MCS, the cap is checked between pairs.")
    args = parser.parse_args()

    mols = _load_molecules(args)
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
    if not args.no_rdkit or args.validate:
        avg_ms, std_ms, rdkit_sizes, rdkit_pairs = _bench_rdkit(mols, pairs, args)
        print(
            "RDKit: "
            f"mean={avg_ms:.3f} ms std={std_ms:.3f} ms pairs={rdkit_pairs}/{len(pairs)} "
            f"throughput={throughput_per_s(rdkit_pairs, avg_ms):.2f} pairs/s"
        )

    if args.validate:
        if rdkit_sizes is None:
            raise ValueError("Validation requires RDKit results")
        _validate(nv_result, rdkit_sizes)


if __name__ == "__main__":
    main()
