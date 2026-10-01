# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Equal-granularity control for BitBIRCH refinement: nvMolKit at raised thresholds.

Refinement splits giant clusters and absorbs singletons, which by itself raises
size-weighted iSIM and scaffold purity. This clusters the same input orders as
``bitbirch_order_sensitivity.py`` (``default_rng(seed).permutation``) at higher
thresholds, so refined partitions can be compared with plain BitBIRCH runs of
similar granularity. Writes ``OUT/labels_threshold_<t>.npz`` with keys
``nvmolkit_b<batch>_s<seed>``; existing archives are skipped.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from nvmolkit.clustering import bitbirch


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fingerprints", help="packed uint8 .npy fingerprints, shape (N, 256)")
    parser.add_argument("--thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--branching-factor", type=int, default=254)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fingerprints = np.load(args.fingerprints, mmap_mode="r")
    for threshold in args.thresholds:
        target = args.out_dir / f"labels_threshold_{threshold}.npz"
        if target.exists():
            continue
        runs = {}
        for seed in range(args.seeds):
            permutation = np.random.default_rng(seed).permutation(len(fingerprints))
            ordered = np.ascontiguousarray(fingerprints[permutation]).view(np.int32)
            labels = np.empty(len(ordered), dtype=np.int32)
            device = torch.from_numpy(ordered).cuda()
            labels[permutation] = bitbirch(
                device, threshold, branching_factor=args.branching_factor, batch_size=args.batch_size
            ).numpy()
            del device
            torch.cuda.empty_cache()
            runs[f"nvmolkit_b{args.batch_size}_s{seed}"] = labels
            print(f"threshold {threshold} seed {seed}: {labels.max() + 1} clusters", flush=True)
        partial = target.with_suffix(".partial.npz")
        np.savez_compressed(partial, **runs)
        partial.rename(target)


if __name__ == "__main__":
    main()
