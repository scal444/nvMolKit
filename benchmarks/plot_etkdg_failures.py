# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plot per-pipeline failure tallies from an etkdg_fire_accuracy_bench summary.json.

Two figures are produced:
    * ``failures_grouped.png``   - grouped bar chart of failure counts per category.
    * ``conformer_yield.png``    - stacked bar of full / partial / no conformers per pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FAILURE_KEYS = ("no_conformers", "partial_conformers", "mmff_param_failure", "nan_energies")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_json", type=Path, help="Path to summary.json from etkdg_fire_accuracy_bench.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the plots (default: alongside summary.json).",
    )
    return parser.parse_args()


def plot_grouped_failures(failures: dict[str, dict[str, int]], output_path: Path, num_molecules: int) -> None:
    pipelines = list(failures.keys())
    width = 0.8 / len(pipelines)
    indices = np.arange(len(FAILURE_KEYS))

    fig, ax = plt.subplots(figsize=(10, 6))
    for slot, pipeline in enumerate(pipelines):
        counts = [failures[pipeline].get(key, 0) for key in FAILURE_KEYS]
        offset = (slot - (len(pipelines) - 1) / 2) * width
        bars = ax.bar(indices + offset, counts, width=width, label=pipeline)
        for rect, count in zip(bars, counts):
            if count > 0:
                ax.annotate(
                    str(count),
                    xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
    ax.set_xticks(indices)
    ax.set_xticklabels(FAILURE_KEYS, rotation=15, ha="right")
    ax.set_ylabel(f"Molecule count (out of {num_molecules})")
    ax.set_title("Failure tallies per pipeline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_conformer_yield(failures: dict[str, dict[str, int]], output_path: Path, num_molecules: int) -> None:
    pipelines = list(failures.keys())
    no_conf = np.array([failures[name].get("no_conformers", 0) for name in pipelines])
    partial = np.array([failures[name].get("partial_conformers", 0) for name in pipelines])
    full = num_molecules - no_conf - partial

    fig, ax = plt.subplots(figsize=(8, 5))
    indices = np.arange(len(pipelines))
    ax.bar(indices, full, label="full conformer set", color="#4caf50")
    ax.bar(indices, partial, bottom=full, label="partial", color="#ff9800")
    ax.bar(indices, no_conf, bottom=full + partial, label="no conformers", color="#e53935")
    for slot, name in enumerate(pipelines):
        ax.annotate(
            f"full={full[slot]}\npartial={partial[slot]}\nzero={no_conf[slot]}",
            xy=(indices[slot], num_molecules),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(indices)
    ax.set_xticklabels(pipelines)
    ax.set_ylabel(f"Molecule count (total {num_molecules})")
    ax.set_ylim(0, num_molecules * 1.18)
    ax.set_title("Conformer-set yield per pipeline")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    with args.summary_json.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    failures = summary["failures"]
    num_molecules = summary["config"]["num_molecules_used"]

    output_dir = args.output_dir or args.summary_json.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped_path = output_dir / "failures_grouped.png"
    yield_path = output_dir / "conformer_yield.png"
    plot_grouped_failures(failures, grouped_path, num_molecules)
    plot_conformer_yield(failures, yield_path, num_molecules)
    print(f"Wrote {grouped_path}")
    print(f"Wrote {yield_path}")


if __name__ == "__main__":
    main()
