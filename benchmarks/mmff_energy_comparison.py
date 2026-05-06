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

"""Cross-run final-energy comparison for the minimize_*.py scripts.

`minimize_nvmolkit_conformers.py` writes `<prefix>_final_energies_nvm.npy`,
`minimize_rdkit_conformers.py` writes `<prefix>_final_energies.npy`. This
script loads N such runs, aligns them by index (which assumes the same input
SDF was processed in the same order), and produces summary stats and plots.

Pass each run as `--run <label>=<output-prefix>`, where `<output-prefix>` is
the value of the corresponding `--output-prefix` argument when the run was
produced. The first run is treated as the reference; all subsequent runs are
compared to it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PREFIX",
        help=(
            "Append a labeled run. PREFIX matches the --output-prefix of a previous "
            "minimize_*.py run; this script auto-detects whether the energies file is "
            "the nvmolkit (_final_energies_nvm.npy) or RDKit (_final_energies.npy) "
            "naming convention. Repeat for multiple runs; the first is the reference."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mmff_energy_comparison"),
        help="Directory for output figures (default: ./mmff_energy_comparison).",
    )
    return parser.parse_args()


@dataclass
class Run:
    label: str
    energies: np.ndarray


def discover_energies_npy(prefix: Path) -> Path:
    candidates = [
        prefix.parent / f"{prefix.name}_final_energies_nvm.npy",
        prefix.parent / f"{prefix.name}_final_energies.npy",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"No final-energies .npy file found for prefix {prefix}. Looked for: {[str(c) for c in candidates]}"
    )


def parse_run_arg(arg: str) -> tuple[str, Path]:
    if "=" not in arg:
        raise argparse.ArgumentTypeError(f"--run argument {arg!r} must be of the form LABEL=PREFIX")
    label, prefix = arg.split("=", 1)
    return label, Path(prefix)


def load_runs(run_args: list[str]) -> list[Run]:
    runs: list[Run] = []
    for arg in run_args:
        label, prefix = parse_run_arg(arg)
        path = discover_energies_npy(prefix)
        energies = np.load(path)
        runs.append(Run(label=label, energies=energies))
    return runs


def main() -> None:
    args = parse_args()
    runs = load_runs(args.run)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if len(runs) < 2:
        raise SystemExit("Need at least two --run entries to compare.")

    sizes = {len(run.energies) for run in runs}
    if len(sizes) != 1:
        raise SystemExit(
            f"Run sizes differ: {[(r.label, len(r.energies)) for r in runs]}. Did the runs use the same SDF?"
        )
    n_confs = sizes.pop()
    print(f"Loaded {len(runs)} runs of {n_confs} conformers each.")

    reference = runs[0]
    print(f"Reference: {reference.label}")
    rows = []
    for run in runs:
        diff = run.energies - reference.energies
        finite = np.isfinite(diff)
        rows.append(
            (
                run.label,
                float(np.nanmean(run.energies)),
                float(np.nanmedian(diff)),
                float(np.nanmean(diff)),
                float(np.nanpercentile(np.abs(diff[finite]), 95)) if finite.any() else float("nan"),
            )
        )
    print("\n   label                                     mean_E    median_diff    mean_diff    |diff|.p95")
    for label, mean_e, median_d, mean_d, p95_d in rows:
        print(f"   {label:<40s}  {mean_e:>10.4f}  {median_d:>+11.4f}  {mean_d:>+11.4f}  {p95_d:>10.4f}")

    plt.figure(figsize=(7, 4))
    bins = 80
    for run in runs[1:]:
        diff = run.energies - reference.energies
        plt.hist(diff[np.isfinite(diff)], bins=bins, alpha=0.6, label=f"{run.label} - {reference.label}")
    plt.axvline(0.0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("Energy diff vs reference (kcal/mol)")
    plt.ylabel("count")
    plt.title("Per-conformer final-energy difference vs reference")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(args.output_dir / "diff_hist.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 6))
    for run in runs[1:]:
        plt.scatter(reference.energies, run.energies, s=8, alpha=0.5, label=run.label)
    lo = float(min(np.nanmin(r.energies) for r in runs))
    hi = float(max(np.nanmax(r.energies) for r in runs))
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    plt.xlabel(f"{reference.label} energy (kcal/mol)")
    plt.ylabel("other run energies (kcal/mol)")
    plt.title("Per-conformer final-energy correlation")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(args.output_dir / "scatter.png", dpi=150)
    plt.close()

    print(f"\nWrote figures to {args.output_dir}/.")


if __name__ == "__main__":
    main()
