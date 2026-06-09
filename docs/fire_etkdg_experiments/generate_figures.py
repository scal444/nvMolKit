#!/usr/bin/env python3
"""Generate figures for the ETKDG FIRE experiment summary."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def stat(summary: dict, key: str, field: str = "median") -> float:
    return float(summary[key][field])


def savefig(name: str) -> None:
    for suffix in ("png", "svg"):
        plt.savefig(OUT / f"{name}.{suffix}", bbox_inches="tight", dpi=180)
    plt.close()


plt.rcParams.update(
    {
        "figure.figsize": (10, 5.6),
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
    }
)


def figure_stage_sweep() -> None:
    summary = load_json("/tmp/etkdg_2x2_n400/summary.json")
    deltas = summary["delta_min_energy_summary"]
    labels = ["all BFGS", "DG FIRE\nETK BFGS", "DG BFGS\nETK FIRE", "all FIRE"]
    vs_bfgs = [
        0.0,
        stat(deltas, "nvm_mixedrev - nvm_bfgs"),
        stat(deltas, "nvm_mixed - nvm_bfgs"),
        stat(deltas, "nvm_fire - nvm_bfgs"),
    ]
    vs_rdkit = [
        stat(deltas, "nvm_bfgs - rdkit"),
        stat(deltas, "nvm_mixedrev - rdkit"),
        stat(deltas, "nvm_mixed - rdkit"),
        stat(deltas, "nvm_fire - rdkit"),
    ]
    x = range(len(labels))
    width = 0.36
    fig, ax = plt.subplots()
    ax.bar([i - width / 2 for i in x], vs_bfgs, width, label="vs all-BFGS", color="#2f6f9f")
    ax.bar([i + width / 2 for i in x], vs_rdkit, width, label="vs RDKit", color="#d07a2d")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("Final conformer MMFF energy by ETKDG stage minimizer")
    ax.set_ylabel("Median per-molecule min MMFF delta")
    ax.set_xticks(list(x), labels)
    ax.legend(frameon=False)
    savefig("stage_backend_mmff_delta")


def figure_same_start() -> None:
    summary = load_json("/tmp/etk_same_start_diag/summary.json")
    starts = ["embedded", "random"]
    labels = ["Embedded ETK start", "Rough/random ETK start"]
    med = [summary["results"][s]["summary"]["dE_ETK"]["median"] for s in starts]
    p90 = [summary["results"][s]["summary"]["dE_ETK"]["p90"] for s in starts]
    fig, ax = plt.subplots()
    x = range(len(labels))
    ax.bar(x, med, color=["#3d7f5f", "#a7473d"], label="median")
    ax.scatter(x, p90, color="#202020", marker="D", zorder=3, label="p90")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_title("Same-start ETK: FIRE minus BFGS")
    ax.set_ylabel("ETK energy delta on the same potential")
    ax.set_xticks(list(x), labels)
    ax.legend(frameon=False)
    savefig("same_start_etk_delta")


def figure_rough_dynamics() -> None:
    summary = load_json("/tmp/etk_fire_dynamics/summary.json")
    rows = summary["results"]["random"]["budget_rows"]
    budgets = [row["budget"] for row in rows]
    bfgs = [row["bfgs_energy"]["median"] for row in rows]
    fire = [row["fire_energy"]["median"] for row in rows]
    fig, ax = plt.subplots()
    ax.plot(budgets, bfgs, marker="o", label="BFGS", color="#2f6f9f", linewidth=2.5)
    ax.plot(budgets, fire, marker="o", label="FIRE", color="#a7473d", linewidth=2.5)
    ax.set_title("Rough ETK starts: useful descent happens early")
    ax.set_xlabel("Iteration budget")
    ax.set_ylabel("Median ETK energy")
    ax.legend(frameon=False)
    savefig("rough_start_energy_budget")


def figure_exit_mechanism() -> None:
    summary = load_json("/tmp/etk_fire_exit_mechanism/summary.json")
    arms = ["fire_stuck300", "fire_open300"]
    labels = ["stuck enabled\n300 iters", "stuck disabled\n300 iters"]
    reasons = ["budget", "fmax", "stuck"]
    colors = {"budget": "#8e8e8e", "fmax": "#3d7f5f", "stuck": "#a7473d"}
    bottoms = [0, 0]
    fig, ax = plt.subplots()
    for reason in reasons:
        counts = [
            summary["results"]["random"]["arm_summaries"][arm]["fire_exit_reasons"][reason]["count"] for arm in arms
        ]
        ax.bar(labels, counts, bottom=bottoms, color=colors[reason], label=reason)
        bottoms = [b + c for b, c in zip(bottoms, counts)]
    med_deltas = [
        summary["results"]["random"]["arm_summaries"][arm]["dE_ETK_vs_bfgs"]["median"] for arm in arms
    ]
    for i, delta in enumerate(med_deltas):
        ax.text(i, 415, f"median dE {delta:.0f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 455)
    ax.set_title("Stuck exits were not the main failure mechanism")
    ax.set_ylabel("Molecules")
    ax.legend(frameon=False, ncols=3, loc="upper left")
    savefig("exit_reasons_random")


def figure_bfgs_burnin() -> None:
    summary = load_json("/tmp/etk_burnin_diag/summary.json")
    comps = summary["results"]["random"]["comparisons"]
    labels = ["0", "5", "10", "20", "50", "100*", "150*", "200*"]
    med = [
        comps["fire"]["summary"]["dE_ETK"]["median"],
        comps["bfgs5_fire"]["summary"]["dE_ETK"]["median"],
        comps["bfgs10_fire"]["summary"]["dE_ETK"]["median"],
        comps["bfgs20_fire"]["summary"]["dE_ETK"]["median"],
        comps["bfgs50_fire"]["summary"]["dE_ETK"]["median"],
        67.4,
        4.78,
        -0.19,
    ]
    fig, ax = plt.subplots()
    x = range(len(labels))
    ax.plot(x, med, marker="o", color="#2f6f9f", linewidth=2.5)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("BFGS burn-in fixes rough ETK only when it is long")
    ax.set_xlabel("BFGS burn-in iterations before FIRE")
    ax.set_ylabel("Median dE_ETK vs BFGS300")
    ax.set_xticks(list(x), labels)
    ax.text(5.6, max(med) * 0.72, "* follow-up spot runs", fontsize=10)
    savefig("bfgs_burnin_random")


def figure_mitigation_summary() -> None:
    burnin = load_json("/tmp/etk_burnin_diag/summary.json")["results"]["random"]["comparisons"]
    abc_rows = load_json("/tmp/etk_fire_abc_segment_sweep/summary.json")["rows"]
    force_rows = load_json("/tmp/etk_fire_force_abc_refine/summary.json")["rows"]
    best_abc = min((row for row in abc_rows if row["arm"] != "tuned_fire300"), key=lambda row: row["dE_ETK"]["median"])
    best_force = min((row for row in force_rows if row["arm"] != "tuned_fire300"), key=lambda row: row["dE_ETK"]["median"])
    items = [
        ("tuned FIRE", burnin["fire"]["summary"]["dE_ETK"]),
        ("best ABC\nsegmented", best_abc["dE_ETK"]),
        ("best force+ABC", best_force["dE_ETK"]),
        ("BFGS50+FIRE", burnin["bfgs50_fire"]["summary"]["dE_ETK"]),
        ("BFGS150+FIRE*", {"median": 4.78, "p90": 0.0}),
    ]
    labels = [item[0] for item in items]
    med = [float(item[1]["median"]) for item in items]
    p90 = [float(item[1].get("p90", item[1]["median"])) for item in items]
    fig, ax = plt.subplots()
    x = range(len(labels))
    ax.bar(x, med, color=["#a7473d", "#6c7f3d", "#9b6b9e", "#2f6f9f", "#2f6f9f"])
    ax.scatter(x[:-1], p90[:-1], color="#202020", marker="D", label="p90", zorder=3)
    ax.set_title("Mitigations improved rough-start FIRE but did not replace BFGS")
    ax.set_ylabel("Median dE_ETK vs BFGS300")
    ax.set_xticks(list(x), labels)
    ax.legend(frameon=False)
    savefig("mitigation_comparison")


def figure_absolute_scale() -> None:
    rows = load_json("/tmp/etk_fire_dynamics/summary.json")["results"]["random"]["budget_rows"]
    by_budget = {row["budget"]: row for row in rows}
    labels = ["initial", "BFGS300", "FIRE300", "best force+ABC"]
    values = [
        by_budget[0]["bfgs_energy"]["median"],
        by_budget[300]["bfgs_energy"]["median"],
        by_budget[300]["fire_energy"]["median"],
        by_budget[300]["bfgs_energy"]["median"] + 448.9,
    ]
    fig, ax = plt.subplots()
    ax.bar(labels, values, color=["#777777", "#2f6f9f", "#a7473d", "#9b6b9e"])
    ax.set_title("Absolute rough-start ETK magnitudes")
    ax.set_ylabel("Median ETK energy")
    ax.text(2, values[2] + 1800, "+~950 vs BFGS", ha="center", fontsize=10)
    ax.text(3, values[3] + 1800, "+~449 vs BFGS", ha="center", fontsize=10)
    savefig("absolute_rough_magnitudes")


def main() -> None:
    figure_stage_sweep()
    figure_same_start()
    figure_rough_dynamics()
    figure_exit_mechanism()
    figure_bfgs_burnin()
    figure_mitigation_summary()
    figure_absolute_scale()


if __name__ == "__main__":
    main()
