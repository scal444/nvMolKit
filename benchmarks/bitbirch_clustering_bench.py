# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reproducible end-to-end benchmark for native BitBIRCH execution modes."""

import argparse
import json
import statistics
import time

import numpy as np
import torch

from nvmolkit.clustering import bitbirch


def _make_fingerprints(args: argparse.Namespace) -> np.ndarray:
    rng = np.random.default_rng(args.seed)
    if args.workload == "random":
        return rng.integers(0, 2**32, size=(args.num_fingerprints, args.num_words), dtype=np.uint32)
    bases = rng.integers(0, 2**32, size=(args.num_bases, args.num_words), dtype=np.uint32)
    return np.ascontiguousarray(bases[np.arange(args.num_fingerprints) % args.num_bases])


def _time_configuration(x, args: argparse.Namespace, num_partitions: int) -> dict:
    for _ in range(args.warmup):
        bitbirch(
            x,
            args.threshold,
            branching_factor=args.branching_factor,
            num_partitions=num_partitions,
        ).torch()
    torch.cuda.synchronize()

    elapsed = []
    num_clusters = 0
    for _ in range(args.repeats):
        start = time.perf_counter()
        labels = bitbirch(
            x,
            args.threshold,
            branching_factor=args.branching_factor,
            num_partitions=num_partitions,
        ).torch()
        torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - start)
        num_clusters = int(labels.max().item()) + 1 if labels.numel() else 0
    return {
        "num_partitions": num_partitions,
        "num_clusters": num_clusters,
        "median_seconds": statistics.median(elapsed),
        "minimum_seconds": min(elapsed),
        "runs_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-fingerprints", type=int, default=4096)
    parser.add_argument("--num-words", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--branching-factor", type=int, default=254)
    parser.add_argument("--partitions", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--workload", choices=("duplicates", "random"), default="duplicates")
    parser.add_argument("--num-bases", type=int, default=32)
    parser.add_argument("--input-location", choices=("device", "host"), default="device")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    fingerprints = _make_fingerprints(args)
    x = (
        torch.from_numpy(fingerprints.astype(np.int32, copy=False)).cuda()
        if args.input_location == "device"
        else fingerprints
    )
    results = [_time_configuration(x, args, num_partitions) for num_partitions in args.partitions]
    output = {
        "device": torch.cuda.get_device_name(),
        "compute_capability": ".".join(str(value) for value in torch.cuda.get_device_capability()),
        "configuration": {
            "num_fingerprints": args.num_fingerprints,
            "num_words": args.num_words,
            "threshold": args.threshold,
            "branching_factor": args.branching_factor,
            "workload": args.workload,
            "num_bases": args.num_bases if args.workload == "duplicates" else None,
            "input_location": args.input_location,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "seed": args.seed,
        },
        "results": results,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
