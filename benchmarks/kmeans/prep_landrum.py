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

"""Landrum BL sets (rdkit_blog, Oct 2019) -> Morgan r2 1024-bit packed fingerprints + target labels.

Input: https://raw.githubusercontent.com/greglandrum/rdkit_blog/master/data/BLSets_actives.txt
Filter pKi > 4 as in CLASSIX-T (arXiv 2601.08781). Writes the full set and a subset with each
SMILES kept only if it has a single target (clean ARI ceiling).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
parser = argparse.ArgumentParser()
parser.add_argument("data_dir", type=Path, help="directory containing BLSets_actives.txt; outputs are written here")
here = parser.parse_args().data_dir
frame = pd.read_csv(here / "BLSets_actives.txt", sep="\t")
frame = frame[frame["pKi"] > 4].reset_index(drop=True)
generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
fps, keep = [], []
for row, smiles in enumerate(frame["smiles"]):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        continue
    fps.append(generator.GetFingerprintAsNumPy(mol).astype(bool))
    keep.append(row)
frame = frame.iloc[keep].reset_index(drop=True)
packed = np.packbits(np.stack(fps), axis=1, bitorder="little")
labels = frame["target_chembl_id"].astype("category").cat.codes.to_numpy().astype(np.int32)
np.save(here / "morgan2_1024.npy", packed)
np.save(here / "labels.npy", labels)

targets_per_smiles = frame.groupby("smiles")["target_chembl_id"].nunique()
single = frame["smiles"].map(targets_per_smiles).eq(1) & ~frame["smiles"].duplicated()
np.save(here / "single_morgan2_1024.npy", packed[single.to_numpy()])
np.save(here / "single_labels.npy", labels[single.to_numpy()])
summary = {
    "rows_pki_gt_4": len(frame),
    "targets": int(labels.max() + 1),
    "single_target_unique_rows": int(single.sum()),
    "single_targets": len(np.unique(labels[single.to_numpy()])),
}
(here / "prep_summary.json").write_text(json.dumps(summary, indent=1))
print(summary)
