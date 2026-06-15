#!/usr/bin/env python
"""Plot fMCS diagnostic counters relevant to parallelism and slow tails."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/home/kboyd/omg/repos/nvmolkit")
OUT_DIR = ROOT / "analysis" / "mcs_1k_timing_analysis"
STATS_CSV = ROOT / "analysis" / "enamine_mcs_1k_pair_timings_with_stats.csv"
OUTLIER_JSON = OUT_DIR / "outlier_pair_851_nvmolkit_stats_by_block.json"


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    print(path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(STATS_CSV)
    stat_cols = [
        "nvmolkit_match_calls",
        "nvmolkit_fallback_calls",
        "nvmolkit_popped",
        "nvmolkit_phase2_iters",
        "nvmolkit_match_found",
        "nvmolkit_fallback_fail",
        "nvmolkit_bound_rejected",
        "nvmolkit_stage2_attempts",
        "nvmolkit_max_queue",
    ]
    corr = (
        df[["nvmolkit_time_ms", *stat_cols]]
        .corr(method="spearman")["nvmolkit_time_ms"]
        .drop("nvmolkit_time_ms")
        .sort_values()
    )

    plt.figure(figsize=(7, 4))
    corr.plot.barh(color="#4f6f8f")
    plt.xlabel("Spearman correlation with nvmolkit_time_ms")
    plt.ylabel("")
    plt.title("fMCS runtime tracks search work counters")
    savefig(OUT_DIR / "fmcs_stats_time_correlations.png")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].scatter(df["nvmolkit_phase2_iters"], df["nvmolkit_time_ms"], s=12, alpha=0.55)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("phase2_iters")
    axes[0].set_ylabel("nvmolkit_time_ms")
    axes[0].set_title("Search iterations")

    axes[1].scatter(df["nvmolkit_max_queue"], df["nvmolkit_time_ms"], s=12, alpha=0.55, color="#9a6f3d")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("max_queue")
    axes[1].set_ylabel("nvmolkit_time_ms")
    axes[1].set_title("Queue depth")
    savefig(OUT_DIR / "fmcs_stats_iters_queue_vs_time.png")

    outlier = pd.DataFrame(json.loads(OUTLIER_JSON.read_text()))
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(outlier["block_size"], outlier["elapsed_ms"], marker="o", color="#345995")
    ax1.set_xlabel("block size")
    ax1.set_ylabel("elapsed_ms")
    ax1.set_title("Outlier pair: more warp groups helps but remains one block")
    ax2 = ax1.twinx()
    ax2.plot(
        outlier["block_size"],
        outlier["group_slot_utilization"],
        marker="s",
        color="#b03a2e",
    )
    ax2.set_ylabel("group slot utilization")
    ax2.set_ylim(0, 1.05)
    savefig(OUT_DIR / "fmcs_outlier_block_size_scaling.png")

    summary = {
        "spearman_correlations": corr.sort_values(ascending=False).to_dict(),
        "time_quantiles_ms": df["nvmolkit_time_ms"].quantile([0.5, 0.9, 0.95, 0.99, 0.999]).to_dict(),
        "top10_pairs": df.sort_values("nvmolkit_time_ms", ascending=False)
        .head(10)[
            [
                "pair_index",
                "nvmolkit_time_ms",
                "nvmolkit_atoms",
                "nvmolkit_bonds",
                "nvmolkit_phase2_iters",
                "nvmolkit_popped",
                "nvmolkit_max_queue",
                "nvmolkit_stage2_attempts",
                "nvmolkit_fallback_calls",
                "nvmolkit_fallback_fail",
                "input_smiles_a",
                "input_smiles_b",
            ]
        ]
        .to_dict(orient="records"),
    }
    summary_path = OUT_DIR / "fmcs_stats_parallelism_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(summary_path)


if __name__ == "__main__":
    main()
