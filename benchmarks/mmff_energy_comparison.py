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

"""Analyze precomputed MMFF minimization results for RDKit reference and comparisons."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot histograms comparing a reference RDKit run against one or more comparison runs.",
    )
    parser.add_argument(
        "reference_prefix",
        type=Path,
        help="Path prefix for the RDKit reference outputs (e.g., /path/to/rdkit).",
    )
    parser.add_argument(
        "comparison_prefixes",
        type=Path,
        nargs="+",
        help="One or more path prefixes for comparison runs (e.g., /path/to/nvm).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory to save plots (default: display interactively).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins (default: 50).",
    )
    return parser.parse_args()


def load_array(path: Path) -> np.ndarray:
    array = np.load(path)
    if array.ndim != 1:
        array = array.reshape(-1)
    return array


@dataclass
class EnergySet:
    name: str
    prefix: Path
    final: np.ndarray
    initial: np.ndarray | None


INITIAL_SUFFIXES = [
    "_initial_energies.npy",
    "_initial_energies_rdkit.npy",
    "_initial_energies_nvm.npy",
]

FINAL_SUFFIXES = [
    "_final_energies.npy",
    "_final_energies_rdkit.npy",
    "_final_energies_nvm.npy",
]


def find_file(prefix: Path, suffixes: list[str]) -> Path | None:
    for suffix in suffixes:
        candidate = prefix.parent / f"{prefix.name}{suffix}"
        if candidate.exists():
            return candidate
    return None


def load_energies(prefix: Path, label: str | None = None) -> EnergySet:
    final_path = find_file(prefix, FINAL_SUFFIXES)
    if final_path is None:
        raise FileNotFoundError(
            f"Could not locate final energies file for prefix {prefix}. Expected one of: {FINAL_SUFFIXES}."
        )
    final = load_array(final_path)

    initial_path = find_file(prefix, INITIAL_SUFFIXES)
    initial = load_array(initial_path) if initial_path is not None else None

    name = label or prefix.name
    return EnergySet(name=name, prefix=prefix, final=final, initial=initial)


def safe_label(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")


def plot_histogram(
    energies_list: list[np.ndarray],
    labels: list[str],
    bins: int,
    title: str,
    output_path: Path | None,
    show: bool = False,
) -> None:
    plt.figure(figsize=(12, 6))
    for energies, label in zip(energies_list, labels):
        plt.hist(energies, bins=bins, alpha=0.6, label=label)
    plt.xlabel("Energy (kcal/mol)")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if not show and output_path is None:
        plt.close()
        return
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()


def plot_boxplot(
    data_list: list[np.ndarray],
    labels: list[str],
    title: str,
    output_path: Path | None,
    show: bool = False,
) -> None:
    cleaned = [data[np.isfinite(data)] for data in data_list]
    if all(len(arr) == 0 for arr in cleaned):
        return
    plt.figure(figsize=(10, 6))
    plt.boxplot(cleaned, labels=labels, vert=True, showfliers=False)
    plt.ylabel("Energy difference (kcal/mol)")
    plt.title(title)
    plt.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()


def plot_scatter(
    reference_energies: np.ndarray,
    comparison_energies: np.ndarray,
    comparison_label: str,
    output_path: Path | None,
    show: bool = False,
) -> None:
    if reference_energies.size == 0 or comparison_energies.size == 0:
        return
    min_len = min(reference_energies.size, comparison_energies.size)
    x_vals = reference_energies[:min_len]
    y_vals = comparison_energies[:min_len]
    mask = np.isfinite(x_vals) & np.isfinite(y_vals)
    if not np.any(mask):
        return
    x_vals = x_vals[mask]
    y_vals = y_vals[mask]
    min_val = min(np.min(x_vals), np.min(y_vals))
    max_val = max(np.max(x_vals), np.max(y_vals))
    plt.figure(figsize=(6, 6))
    plt.scatter(x_vals, y_vals, alpha=0.6, edgecolor="none")
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black", linewidth=1.0)
    plt.xlabel("RDKit final energy (kcal/mol)")
    plt.ylabel(f"{comparison_label} final energy (kcal/mol)")
    plt.title(f"RDKit vs {comparison_label} final energies")
    plt.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()


def main() -> None:
    args = parse_args()

    reference = load_energies(args.reference_prefix, label="RDKit")
    comparisons = [load_energies(prefix) for prefix in args.comparison_prefixes]

    output_dir = args.output_dir
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    if reference.initial is not None:
        plot_histogram(
            [reference.initial, reference.final],
            ["RDKit initial", "RDKit final"],
            args.bins,
            "RDKit MMFF energy distribution",
            output_dir / "rdkit_hist.png" if output_dir is not None else None,
            show=False,
        )

    # Combined final comparison plot
    combined_arrays = [reference.final] + [comp.final for comp in comparisons]
    combined_labels = ["RDKit final"] + [f"{comp.name} final" for comp in comparisons]
    plot_histogram(
        combined_arrays,
        combined_labels,
        args.bins,
        "Final energy comparison",
        output_dir / "final_comparison.png" if output_dir is not None else None,
        show=True,
    )

    combined_deltas: list[np.ndarray] = []
    combined_delta_labels: list[str] = []

    for comp in comparisons:
        labels = ["RDKit final", f"{comp.name} final"]
        arrays = [reference.final, comp.final]
        if comp.initial is not None:
            labels.append(f"{comp.name} initial")
            arrays.append(comp.initial)

        comp_hist_path = None
        delta_hist_path = None
        if output_dir is not None:
            base = safe_label(comp.name)
            comp_hist_path = output_dir / f"comparison_{base}.png"
            delta_hist_path = output_dir / f"delta_{base}.png"
            scatter_path = output_dir / f"scatter_{base}.png"
        else:
            scatter_path = None

        plot_histogram(
            arrays,
            labels,
            args.bins,
            f"RDKit vs {comp.name} energy distribution",
            comp_hist_path,
            show=False,
        )

        if reference.final.size and comp.final.size:
            min_len = min(reference.final.size, comp.final.size)
            delta = comp.final[:min_len] - reference.final[:min_len]
            combined_deltas.append(delta)
            combined_delta_labels.append(f"{comp.name} - RDKit")
            plot_histogram(
                [delta],
                [f"{comp.name} - RDKit"],
                args.bins,
                f"Energy difference: {comp.name} - RDKit",
                delta_hist_path,
                show=False,
            )
            plot_scatter(
                reference.final,
                comp.final,
                comp.name,
                scatter_path,
                show=True,
            )

    if combined_deltas:
        plot_histogram(
            combined_deltas,
            combined_delta_labels,
            args.bins,
            "Energy difference comparison",
            output_dir / "delta_combined.png" if output_dir is not None else None,
            show=True,
        )
        if output_dir is not None:
            plot_boxplot(
                combined_deltas,
                combined_delta_labels,
                "Energy difference comparison (boxplot)",
                output_dir / "delta_box_combined.png",
                show=True,
            )


if __name__ == "__main__":
    main()

