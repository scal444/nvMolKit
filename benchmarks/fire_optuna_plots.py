# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plot per-parameter slices, importances, and pairwise contours from a fire_optuna study.

Useful for spotting params that have independent effects on the objective vs. params that
are coupled (visible as non-flat contours but flat slices, or vice versa). Saves PNGs to
``--output-dir``; one slice/importance plot per study and one contour plot per pair of
top-importance params.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import optuna.visualization.matplotlib as ovm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("storage", type=Path, help="SQLite RDB file written by fire_optuna.py.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--study",
        action="append",
        help="Study name(s) to plot. Can be passed multiple times. Defaults to all studies in the storage.",
    )
    parser.add_argument(
        "--top-pairs",
        type=int,
        default=3,
        help="Number of top-importance param pairs to render contour plots for (default 3).",
    )
    parser.add_argument(
        "--clip-quantile",
        type=float,
        default=0.9,
        help=(
            "Drop trials whose objective value is above the given quantile of completed trials before "
            "plotting (default 0.9). Set to 1.0 to keep all trials. Outliers are not removed from the "
            "underlying study, only filtered out of the rendering path."
        ),
    )
    return parser.parse_args()


def list_studies(storage_url: str) -> list[str]:
    return [s.study_name for s in optuna.get_all_study_summaries(storage=storage_url)]


def _extract_figure(ax_or_axes) -> plt.Figure:
    if hasattr(ax_or_axes, "figure"):
        return ax_or_axes.figure
    flat = list(getattr(ax_or_axes, "flat", ax_or_axes))
    return flat[0].figure


def save(ax_or_axes, path: Path) -> None:
    fig = _extract_figure(ax_or_axes)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _filtered_study(study: optuna.Study, clip_quantile: float) -> optuna.Study:
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed or clip_quantile >= 1.0:
        return study
    values = np.array([t.value for t in completed if t.value is not None])
    if values.size == 0:
        return study
    threshold = float(np.quantile(values, clip_quantile))
    surviving = [t for t in completed if t.value is not None and t.value <= threshold]
    if len(surviving) == len(completed):
        return study

    direction = study.direction
    new_study = optuna.create_study(
        study_name=f"{study.study_name}_filtered",
        direction="minimize" if direction == optuna.study.StudyDirection.MINIMIZE else "maximize",
    )
    new_study.add_trials(surviving)
    print(
        f"[{study.study_name}] filtered to {len(surviving)}/{len(completed)} trials "
        f"(clip <= {threshold:.4g} at quantile {clip_quantile})."
    )
    return new_study


def plot_study(study: optuna.Study, output_dir: Path, top_pairs: int, clip_quantile: float) -> None:
    name = study.study_name
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print(f"[{name}] no completed trials, skipping.")
        return
    print(f"[{name}] {len(completed)} completed trials.")

    plot_study_obj = _filtered_study(study, clip_quantile)

    save(ovm.plot_optimization_history(plot_study_obj), output_dir / f"{name}_history.png")
    save(ovm.plot_slice(plot_study_obj), output_dir / f"{name}_slice.png")

    try:
        save(ovm.plot_param_importances(plot_study_obj), output_dir / f"{name}_importance.png")
    except Exception as exc:
        print(f"[{name}] importance plot skipped: {exc}")

    try:
        importances = optuna.importance.get_param_importances(plot_study_obj)
        ranked = list(importances.keys())
    except Exception as exc:
        print(f"[{name}] importance compute skipped: {exc}")
        ranked = []

    pair_count = 0
    for i, a in enumerate(ranked):
        for b in ranked[i + 1 :]:
            if pair_count >= top_pairs:
                break
            try:
                save(ovm.plot_contour(plot_study_obj, params=[a, b]), output_dir / f"{name}_contour_{a}_vs_{b}.png")
                pair_count += 1
            except Exception as exc:
                print(f"[{name}] contour {a} vs {b} skipped: {exc}")
        if pair_count >= top_pairs:
            break


def main() -> None:
    args = parse_args()
    storage_url = f"sqlite:///{args.storage.resolve()}"
    studies = args.study or list_studies(storage_url)
    if not studies:
        raise SystemExit("No studies found.")
    for name in studies:
        study = optuna.load_study(study_name=name, storage=storage_url)
        plot_study(study, args.output_dir, args.top_pairs, args.clip_quantile)
    print(f"\nPlots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
