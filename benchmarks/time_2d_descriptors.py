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
"""Time the individual runtime contribution of every RDKit 2D descriptor.

Samples molecules strided (evenly spaced) across an Enamine REAL cxsmiles file
-- never just from the top, because the file is size sorted -- and times each
descriptor in ``rdkit.Chem.Descriptors._descList`` separately. Runs until either
all sampled molecules are processed or a wall-clock budget is exhausted, then
writes a per-descriptor CSV of accumulated times.
"""

import argparse
import csv
import os
import sys
import time

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from tqdm import tqdm

PARSE_LABEL = "__MolFromSmiles__"
PARSE_MODULE = "parsing"


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
        default=100_000,
        help="Number of molecules to sample, evenly spaced across the dataset.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Wall-clock budget in seconds for the timing loop. Processing is cancelled early once exceeded.",
    )
    parser.add_argument(
        "--output",
        default="descriptor_times.csv",
        help="Output CSV path for per-descriptor timings.",
    )
    parser.add_argument(
        "--smiles-column",
        type=int,
        default=0,
        help="Zero-based index of the SMILES column in the dataset.",
    )
    return parser.parse_args()


def count_data_lines(path):
    """Count molecule lines (excluding the header) in the dataset."""
    with open(path, "r") as handle:
        total = sum(1 for _ in handle)
    return max(0, total - 1)


def sample_smiles(path, num_molecules, smiles_column):
    """Collect ``num_molecules`` SMILES evenly spaced across the dataset.

    The dataset is size sorted, so a strided sample is taken across the whole
    file rather than reading a contiguous block from the top.
    """
    data_lines = count_data_lines(path)
    if data_lines == 0:
        return []
    stride = max(1, data_lines // num_molecules)
    sampled = []
    with open(path, "r") as handle:
        next(handle, None)  # skip header
        for index, line in enumerate(handle):
            if index % stride:
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) <= smiles_column:
                continue
            sampled.append(fields[smiles_column])
            if len(sampled) >= num_molecules:
                break
    return sampled


def time_descriptors(smiles_list, descriptors, timeout):
    """Accumulate per-descriptor runtime over the sampled molecules.

    Returns a tuple of (totals, processed, parse_failures, descriptor_errors,
    elapsed, cancelled).
    """
    totals = {name: 0.0 for name, _ in descriptors}
    totals[PARSE_LABEL] = 0.0
    descriptor_errors = {name: 0 for name, _ in descriptors}
    processed = 0
    parse_failures = 0
    cancelled = False

    perf = time.perf_counter
    start = perf()
    for smiles in tqdm(smiles_list, desc="Timing descriptors", unit="mol"):
        if perf() - start >= timeout:
            cancelled = True
            break

        parse_start = perf()
        mol = Chem.MolFromSmiles(smiles)
        totals[PARSE_LABEL] += perf() - parse_start
        if mol is None:
            parse_failures += 1
            continue

        for name, function in descriptors:
            call_start = perf()
            try:
                function(mol)
            except Exception:
                descriptor_errors[name] += 1
            totals[name] += perf() - call_start
        processed += 1

    elapsed = perf() - start
    return totals, processed, parse_failures, descriptor_errors, elapsed, cancelled


def write_csv(path, descriptors, totals, descriptor_errors, processed):
    module_by_name = {name: getattr(fn, "__module__", "") for name, fn in descriptors}
    module_by_name[PARSE_LABEL] = PARSE_MODULE

    rows = []
    for name in [PARSE_LABEL] + [n for n, _ in descriptors]:
        total_seconds = totals[name]
        mean_us = (total_seconds / processed * 1e6) if processed else 0.0
        rows.append(
            {
                "descriptor": name,
                "module": module_by_name[name],
                "total_seconds": total_seconds,
                "num_molecules": processed,
                "mean_us_per_mol": mean_us,
                "errors": descriptor_errors.get(name, 0),
            }
        )

    rows.sort(key=lambda row: row["total_seconds"], reverse=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "descriptor",
                "module",
                "total_seconds",
                "num_molecules",
                "mean_us_per_mol",
                "errors",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    RDLogger.DisableLog("rdApp.*")

    if not os.path.exists(args.dataset):
        sys.exit(f"Dataset not found: {args.dataset}")

    descriptors = list(Descriptors._descList)
    print(f"Loaded {len(descriptors)} descriptors from rdkit.Chem.Descriptors._descList")

    print(f"Sampling up to {args.num_molecules} molecules from {args.dataset} ...")
    smiles_list = sample_smiles(args.dataset, args.num_molecules, args.smiles_column)
    print(f"Sampled {len(smiles_list)} molecules.")

    print(f"Timing descriptors with a {args.timeout:.0f}s budget ...")
    (
        totals,
        processed,
        parse_failures,
        descriptor_errors,
        elapsed,
        cancelled,
    ) = time_descriptors(smiles_list, descriptors, args.timeout)

    status = "cancelled early (timeout)" if cancelled else "completed"
    print(f"Timing {status} after {elapsed:.1f}s.")
    print(f"Molecules processed: {processed}")
    print(f"SMILES parse failures: {parse_failures}")

    write_csv(args.output, descriptors, totals, descriptor_errors, processed)
    print(f"Wrote per-descriptor timings to {args.output}")


if __name__ == "__main__":
    main()
