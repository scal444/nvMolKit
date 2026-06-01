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

#!/usr/bin/env python
"""Line-level profile of RDKit's QED computation.

RDKit's ``qed`` spends nearly all of its time inside ``QED.properties``, where
eight molecular properties are computed on separate lines (the structural-alert
and acceptor SMARTS loops dominate). cProfile cannot separate those steps
because several of them call the same C++ method (``HasSubstructMatch``). This
script instead reuses RDKit's own QED constants but defines line-for-line copies
of ``ads``, ``properties`` and ``qed`` so that line_profiler can attribute time
to each individual property computation.

Run it outside the Cursor sandbox so line_profiler's C tracer is unrestricted.
"""

import argparse
import math
import os
import sys
import time

from line_profiler import LineProfiler
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, MolSurf
from rdkit.Chem import rdMolDescriptors as rdmd
from rdkit.Chem import QED

from time_2d_descriptors import sample_smiles

Acceptors = QED.Acceptors
StructuralAlerts = QED.StructuralAlerts
AliphaticRings = QED.AliphaticRings
adsParameters = QED.adsParameters
QEDproperties = QED.QEDproperties
WEIGHT_MEAN = QED.WEIGHT_MEAN


def ads(x, adsParameter):
    """ADS desirability function (copied from rdkit.Chem.QED)."""
    p = adsParameter
    exp1 = 1 + math.exp(-1 * (x - p.C + p.D / 2) / p.E)
    exp2 = 1 + math.exp(-1 * (x - p.C - p.D / 2) / p.F)
    dx = p.A + p.B / exp1 * (1 - 1 / exp2)
    return dx / p.DMAX


def properties(mol):
    """Compute the eight QED properties (copied from rdkit.Chem.QED).

    Each property is assigned on its own line so line_profiler attributes the
    cost of every step separately.
    """
    if mol is None:
        raise ValueError("You need to provide a mol argument.")
    mol = Chem.RemoveHs(mol)
    molecular_weight = rdmd._CalcMolWt(mol)
    alogp = Crippen.MolLogP(mol)
    hba = sum(len(mol.GetSubstructMatches(pattern)) for pattern in Acceptors if mol.HasSubstructMatch(pattern))
    hbd = rdmd.CalcNumHBD(mol)
    psa = MolSurf.TPSA(mol)
    rotatable_bonds = rdmd.CalcNumRotatableBonds(mol, rdmd.NumRotatableBondsOptions.Strict)
    aromatic_rings = len(Chem.GetSSSR(Chem.DeleteSubstructs(Chem.Mol(mol), AliphaticRings)))
    alerts = sum(1 for alert in StructuralAlerts if mol.HasSubstructMatch(alert))
    return QEDproperties(
        MW=molecular_weight,
        ALOGP=alogp,
        HBA=hba,
        HBD=hbd,
        PSA=psa,
        ROTB=rotatable_bonds,
        AROM=aromatic_rings,
        ALERTS=alerts,
    )


def qed(mol, w=WEIGHT_MEAN, qedProperties=None):
    """Weighted ADS-mapped QED score (copied from rdkit.Chem.QED)."""
    if qedProperties is None:
        qedProperties = properties(mol)
    d = [ads(pi, adsParameters[name]) for name, pi in qedProperties._asdict().items()]
    t = sum(wi * math.log(di) for wi, di in zip(w, d))
    return math.exp(t / sum(w))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default=os.path.expanduser("~/data/enamine_real_10M.cxsmiles"),
        help="Path to the tab-separated cxsmiles dataset (with header row).",
    )
    parser.add_argument(
        "--num-molecules",
        type=int,
        default=20_000,
        help="Number of molecules to sample, evenly spaced across the dataset.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Wall-clock budget in seconds for the profiling loop.",
    )
    parser.add_argument(
        "--smiles-column",
        type=int,
        default=0,
        help="Zero-based index of the SMILES column in the dataset.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to also write the line_profiler report.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    RDLogger.DisableLog("rdApp.*")

    if not os.path.exists(args.dataset):
        sys.exit(f"Dataset not found: {args.dataset}")

    print(f"Sampling up to {args.num_molecules} molecules from {args.dataset} ...")
    smiles_list = sample_smiles(args.dataset, args.num_molecules, args.smiles_column)
    print(f"Sampled {len(smiles_list)} molecules.")

    profiler = LineProfiler()
    profiler.add_function(properties)
    profiler.add_function(qed)
    profiler.add_function(ads)

    processed = 0
    parse_failures = 0
    cancelled = False
    perf = time.perf_counter

    profiler.enable_by_count()
    start = perf()
    for smiles in smiles_list:
        if perf() - start >= args.timeout:
            cancelled = True
            break
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            parse_failures += 1
            continue
        qed(mol)
        processed += 1
    elapsed = perf() - start
    profiler.disable_by_count()

    status = "cancelled early (timeout)" if cancelled else "completed"
    print(f"Profiling {status} after {elapsed:.1f}s.")
    print(f"Molecules processed: {processed}")
    print(f"SMILES parse failures: {parse_failures}")
    print()

    profiler.print_stats()

    if args.output:
        with open(args.output, "w") as handle:
            profiler.print_stats(stream=handle)
        print(f"Wrote line_profiler report to {args.output}")


if __name__ == "__main__":
    main()
