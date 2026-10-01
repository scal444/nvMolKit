# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Count pairs of distinct clusters with bit-identical majority centroids (ties set the bit).

Such pairs make Davies-Bouldin undefined. Usage:
  bitbirch_identical_centroids.py FINGERPRINTS.npy OUTPUT.csv ARCHIVE.npz [ARCHIVE.npz ...]
"""

import csv
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bitbirch_refinement import compact, identical_centroid_groups


def main():
    fingerprints = np.load(sys.argv[1], mmap_mode="r")
    output = Path(sys.argv[2])
    device = torch.device("cuda")
    rows = []
    for archive_path in sys.argv[3:]:
        threshold = float(re.search(r"labels_threshold_([0-9.]+)\.npz", archive_path).group(1))
        archive = np.load(archive_path)
        for name in sorted(archive.files):
            method, batch, seed = re.fullmatch(r"(\w+?)_b(\d+)_s(\d+)", name).groups()
            labels = compact(archive[name])
            group, pairs = identical_centroid_groups(np.ascontiguousarray(fingerprints[: len(labels)]), labels, device)
            rows.append(
                {
                    "threshold": threshold,
                    "method": method,
                    "batch_size": int(batch),
                    "seed": int(seed),
                    "clusters": len(group),
                    "identical_centroid_pairs": pairs,
                    "identical_pairs_per_1k_clusters": 1000 * pairs / len(group),
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
