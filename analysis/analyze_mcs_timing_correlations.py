#!/usr/bin/env python3
"""Analyze per-pair MCS timing correlations and structural descriptors."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, rdMolDescriptors


def _mol_from_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    return mol


def _ring_descriptors(mol: Chem.Mol) -> dict[str, float]:
    ring_info = mol.GetRingInfo()
    atom_rings = list(ring_info.AtomRings())
    ring_atom_counts = Counter(atom_idx for ring in atom_rings for atom_idx in ring)
    ring_atoms = set(ring_atom_counts)
    ring_bonds = {bond.GetIdx() for bond in mol.GetBonds() if bond.IsInRing()}

    aromatic_rings = 0
    for ring in atom_rings:
        if all(mol.GetAtomWithIdx(atom_idx).GetIsAromatic() for atom_idx in ring):
            aromatic_rings += 1

    return {
        "rings": float(len(atom_rings)),
        "aromatic_rings": float(aromatic_rings),
        "aliphatic_rings": float(len(atom_rings) - aromatic_rings),
        "ring_atoms": float(len(ring_atoms)),
        "ring_bonds": float(len(ring_bonds)),
        "largest_ring": float(max((len(ring) for ring in atom_rings), default=0)),
        "fused_ring_atoms": float(sum(1 for count in ring_atom_counts.values() if count > 1)),
    }


def _component_diameter(adj: dict[int, set[int]], start: int) -> int:
    seen: set[int] = set()
    component: list[int] = []
    stack = [start]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        component.append(node)
        stack.extend(adj[node] - seen)

    def farthest(src: int) -> tuple[int, int]:
        visited = {src}
        queue: deque[tuple[int, int]] = deque([(src, 1)])
        far_node = src
        far_dist = 1
        while queue:
            node, dist = queue.popleft()
            if dist > far_dist:
                far_node = node
                far_dist = dist
            for nxt in adj[node]:
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, dist + 1))
        return far_node, far_dist

    # The selected acyclic carbon graph is normally a forest. If branching or
    # unusual chemistry leaves a cycle, two sweeps are still a useful path proxy.
    far_node, _ = farthest(component[0])
    _, diameter = farthest(far_node)
    return diameter


def _longest_acyclic_carbon_chain(mol: Chem.Mol) -> float:
    carbon_nodes = {
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 6 and not atom.GetIsAromatic() and not atom.IsInRing()
    }
    if not carbon_nodes:
        return 0.0

    adj = {atom_idx: set() for atom_idx in carbon_nodes}
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE or bond.IsInRing():
            continue
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin in carbon_nodes and end in carbon_nodes:
            adj[begin].add(end)
            adj[end].add(begin)

    seen: set[int] = set()
    best = 1
    for node in carbon_nodes:
        if node in seen:
            continue
        stack = [node]
        component: set[int] = set()
        while stack:
            cur = stack.pop()
            if cur in component:
                continue
            component.add(cur)
            stack.extend(adj[cur] - component)
        seen.update(component)
        best = max(best, _component_diameter(adj, node))
    return float(best)


def _mol_descriptors(mol: Chem.Mol, prefix: str) -> dict[str, float]:
    atoms = float(mol.GetNumAtoms())
    bonds = float(mol.GetNumBonds())
    carbons = float(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6))
    hetero_atoms = float(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in (1, 6)))
    ring_desc = _ring_descriptors(mol)

    desc = {
        "atoms": atoms,
        "bonds": bonds,
        "rotatable_bonds": float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "hetero_atoms": hetero_atoms,
        "carbon_atoms": carbons,
        "carbon_fraction": carbons / atoms if atoms else 0.0,
        "hetero_fraction": hetero_atoms / atoms if atoms else 0.0,
        "longest_acyclic_carbon_chain": _longest_acyclic_carbon_chain(mol),
        "logp": float(Crippen.MolLogP(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
    }
    desc.update(ring_desc)
    return {f"{prefix}_{key}": value for key, value in desc.items()}


def _pair_features(row: pd.Series) -> dict[str, float]:
    mol_a = _mol_from_smiles(row["smiles_a"])
    mol_b = _mol_from_smiles(row["smiles_b"])
    desc = {}
    desc.update(_mol_descriptors(mol_a, "a"))
    desc.update(_mol_descriptors(mol_b, "b"))

    base_keys = [key[2:] for key in desc if key.startswith("a_")]
    for key in base_keys:
        a_value = desc[f"a_{key}"]
        b_value = desc[f"b_{key}"]
        desc[f"pair_min_{key}"] = min(a_value, b_value)
        desc[f"pair_max_{key}"] = max(a_value, b_value)
        desc[f"pair_avg_{key}"] = (a_value + b_value) / 2.0
        desc[f"pair_absdiff_{key}"] = abs(a_value - b_value)

    return desc


def _add_runtime_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["nvmolkit_time_ms"] = pd.to_numeric(out["nvmolkit_time_ms"], errors="coerce")
    out["rdkit_time_ms"] = pd.to_numeric(out["rdkit_time_ms"], errors="coerce")
    out["nvmolkit_atoms"] = pd.to_numeric(out["nvmolkit_atoms"], errors="coerce")
    out["nvmolkit_bonds"] = pd.to_numeric(out["nvmolkit_bonds"], errors="coerce")
    out["rdkit_atoms"] = pd.to_numeric(out["rdkit_atoms"], errors="coerce")
    out["rdkit_bonds"] = pd.to_numeric(out["rdkit_bonds"], errors="coerce")

    out["log_nvmolkit_time_ms"] = np.log10(out["nvmolkit_time_ms"].clip(lower=1e-6))
    out["log_rdkit_time_ms"] = np.log10(out["rdkit_time_ms"].clip(lower=1e-6))
    out["time_ratio_nv_over_rdkit"] = out["nvmolkit_time_ms"] / out["rdkit_time_ms"]
    out["log_time_ratio_nv_over_rdkit"] = np.log10(out["time_ratio_nv_over_rdkit"].clip(lower=1e-6))
    out["mcs_atoms"] = out["nvmolkit_atoms"]
    out["mcs_bonds"] = out["nvmolkit_bonds"]
    out["mcs_atom_fraction_small"] = out["mcs_atoms"] / out["pair_min_atoms"].replace(0, np.nan)
    out["mcs_bond_fraction_small"] = out["mcs_bonds"] / out["pair_min_bonds"].replace(0, np.nan)
    return out


def _corr_table(df: pd.DataFrame, columns: list[str], targets: list[str]) -> pd.DataFrame:
    rows = []
    for feature in columns:
        row = {"feature": feature}
        for target in targets:
            valid = df[[feature, target]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid) < 3 or valid[feature].nunique() < 2 or valid[target].nunique() < 2:
                pearson = math.nan
                spearman = math.nan
            else:
                pearson = valid[feature].corr(valid[target], method="pearson")
                spearman = valid[feature].corr(valid[target], method="spearman")
            row[f"{target}_pearson"] = pearson
            row[f"{target}_spearman"] = spearman
        rows.append(row)
    return pd.DataFrame(rows)


def _save_scatter(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=160)
    scatter = ax.scatter(
        df["rdkit_time_ms"],
        df["nvmolkit_time_ms"],
        c=df["pair_max_rings"],
        s=18,
        alpha=0.72,
        cmap="viridis",
        linewidths=0,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("RDKit time per pair (ms, log)")
    ax.set_ylabel("nvMolKit time per pair (ms, log)")
    ax.set_title("Per-pair fMCS timing: nvMolKit vs RDKit")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("max ring count in pair")
    ax.grid(True, which="both", linewidth=0.35, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_top_feature_bars(corr: pd.DataFrame, target_col: str, out_path: Path, title: str) -> None:
    plot_df = corr[["feature", target_col]].dropna().copy()
    plot_df["abs_corr"] = plot_df[target_col].abs()
    plot_df = plot_df.sort_values("abs_corr", ascending=False).head(18).sort_values(target_col)
    fig, ax = plt.subplots(figsize=(8.2, 6.0), dpi=160)
    colors = np.where(plot_df[target_col] >= 0, "#2868a8", "#b25b34")
    ax.barh(plot_df["feature"], plot_df[target_col], color=colors)
    ax.axvline(0, color="#222222", linewidth=0.8)
    ax.set_xlabel("Spearman correlation")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_size_plot(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), dpi=160, sharey=True)
    for ax, feature, title in zip(
        axes,
        ["pair_min_atoms", "pair_avg_atoms", "pair_max_atoms"],
        ["Smaller molecule", "Average of pair", "Larger molecule"],
        strict=True,
    ):
        ax.scatter(df[feature], df["nvmolkit_time_ms"], s=16, alpha=0.65, linewidths=0)
        ax.set_yscale("log")
        ax.set_xlabel("atoms")
        ax.set_title(title)
        ax.grid(True, which="both", linewidth=0.35, alpha=0.35)
    axes[0].set_ylabel("nvMolKit time per pair (ms, log)")
    fig.suptitle("Molecule size versus nvMolKit timing")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_ring_hydrocarbon_plot(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.0), dpi=160)
    specs = [
        ("pair_max_rings", "max ring count"),
        ("pair_max_ring_atoms", "max ring atoms"),
        ("pair_max_aromatic_rings", "max aromatic rings"),
        ("pair_max_longest_acyclic_carbon_chain", "max acyclic carbon chain"),
    ]
    for ax, (feature, label) in zip(axes.ravel(), specs, strict=True):
        grouped = (
            df.groupby(feature, dropna=True)["nvmolkit_time_ms"]
            .agg(["median", "count", "max"])
            .reset_index()
            .sort_values(feature)
        )
        ax.plot(grouped[feature], grouped["median"], marker="o", label="median")
        ax.plot(grouped[feature], grouped["max"], marker=".", linestyle="--", label="max")
        ax.set_yscale("log")
        ax.set_xlabel(label)
        ax.set_ylabel("nvMolKit time (ms, log)")
        ax.grid(True, which="both", linewidth=0.35, alpha=0.35)
    axes[0, 0].legend()
    fig.suptitle("Ring and hydrocarbon descriptors versus nvMolKit timing")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _slow_tail_lift_table(df: pd.DataFrame) -> pd.DataFrame:
    threshold = df["nvmolkit_time_ms"].quantile(0.95)
    slow = df[df["nvmolkit_time_ms"] >= threshold]
    rest = df[df["nvmolkit_time_ms"] < threshold]
    features = [
        "mcs_atoms",
        "mcs_bonds",
        "mcs_atom_fraction_small",
        "pair_min_atoms",
        "pair_avg_atoms",
        "pair_max_atoms",
        "pair_min_bonds",
        "pair_avg_bonds",
        "pair_max_bonds",
        "pair_avg_rings",
        "pair_max_rings",
        "pair_avg_ring_atoms",
        "pair_max_ring_atoms",
        "pair_avg_aromatic_rings",
        "pair_max_aromatic_rings",
        "pair_avg_aliphatic_rings",
        "pair_max_aliphatic_rings",
        "pair_avg_fused_ring_atoms",
        "pair_max_fused_ring_atoms",
        "pair_avg_rotatable_bonds",
        "pair_max_rotatable_bonds",
        "pair_avg_longest_acyclic_carbon_chain",
        "pair_max_longest_acyclic_carbon_chain",
    ]
    rows = []
    for feature in features:
        slow_mean = slow[feature].mean()
        rest_mean = rest[feature].mean()
        rows.append(
            {
                "feature": feature,
                "slow_top5_mean": slow_mean,
                "rest_mean": rest_mean,
                "difference": slow_mean - rest_mean,
                "ratio": slow_mean / rest_mean if rest_mean else math.nan,
            }
        )
    return pd.DataFrame(rows)


def _save_slow_tail_lift_plot(lift: pd.DataFrame, out_path: Path) -> None:
    plot_df = lift.copy()
    plot_df["abs_difference"] = plot_df["difference"].abs()
    plot_df = plot_df.sort_values("abs_difference", ascending=False).head(18).sort_values("difference")
    fig, ax = plt.subplots(figsize=(8.4, 6.0), dpi=160)
    colors = np.where(plot_df["difference"] >= 0, "#2868a8", "#b25b34")
    ax.barh(plot_df["feature"], plot_df["difference"], color=colors)
    ax.axvline(0, color="#222222", linewidth=0.8)
    ax.set_xlabel("top 5% slow mean - remaining 95% mean")
    ax.set_title("Descriptor lift in the nvMolKit slow tail")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_slowest_table(df: pd.DataFrame, out_path: Path) -> None:
    cols = [
        "pair_index",
        "nvmolkit_time_ms",
        "rdkit_time_ms",
        "time_ratio_nv_over_rdkit",
        "input_line_a",
        "input_line_b",
        "input_smiles_a",
        "input_smiles_b",
        "pair_min_atoms",
        "pair_avg_atoms",
        "pair_max_atoms",
        "pair_max_rings",
        "pair_max_ring_atoms",
        "pair_max_aromatic_rings",
        "pair_max_aliphatic_rings",
        "pair_max_longest_acyclic_carbon_chain",
        "mcs_atoms",
        "mcs_bonds",
    ]
    df.sort_values("nvmolkit_time_ms", ascending=False)[cols].head(30).to_csv(out_path, index=False)


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    columns = list(df.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        cells = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                cells.append(f"{value:.6g}")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _descriptor_columns(df: pd.DataFrame) -> list[str]:
    prefixes = ("pair_min_", "pair_max_", "pair_avg_", "pair_absdiff_")
    return [
        col
        for col in df.columns
        if col.startswith(prefixes)
        or col
        in {
            "mcs_atoms",
            "mcs_bonds",
            "mcs_atom_fraction_small",
            "mcs_bond_fraction_small",
        }
    ]


def _write_summary(df: pd.DataFrame, corr: pd.DataFrame, lift: pd.DataFrame, out_path: Path) -> None:
    def stat_line(series: pd.Series) -> str:
        return (
            f"min={series.min():.3f}, median={series.median():.3f}, "
            f"p95={series.quantile(0.95):.3f}, p99={series.quantile(0.99):.3f}, "
            f"max={series.max():.3f}, mean={series.mean():.3f}"
        )

    runtime_corr = df[["nvmolkit_time_ms", "rdkit_time_ms"]].corr(method="spearman").iloc[0, 1]
    log_runtime_corr = df[["log_nvmolkit_time_ms", "log_rdkit_time_ms"]].corr(method="spearman").iloc[0, 1]
    top_nv = (
        corr[["feature", "log_nvmolkit_time_ms_spearman"]]
        .dropna()
        .assign(abs_corr=lambda x: x["log_nvmolkit_time_ms_spearman"].abs())
        .sort_values("abs_corr", ascending=False)
        .head(12)
    )
    top_ratio = (
        corr[["feature", "log_time_ratio_nv_over_rdkit_spearman"]]
        .dropna()
        .assign(abs_corr=lambda x: x["log_time_ratio_nv_over_rdkit_spearman"].abs())
        .sort_values("abs_corr", ascending=False)
        .head(12)
    )
    top_rows = df.sort_values("nvmolkit_time_ms", ascending=False).head(10)
    top_lift = lift.assign(abs_difference=lambda x: x["difference"].abs()).sort_values(
        "abs_difference", ascending=False
    ).head(12)

    lines = [
        "# fMCS 1k Pair Timing Analysis",
        "",
        "## Dataset",
        "",
        f"- pairs: {len(df)}",
        f"- nvMolKit timing (ms): {stat_line(df['nvmolkit_time_ms'])}",
        f"- RDKit timing (ms): {stat_line(df['rdkit_time_ms'])}",
        f"- Spearman nvMolKit vs RDKit time: {runtime_corr:.3f}",
        f"- Spearman log10(nvMolKit) vs log10(RDKit): {log_runtime_corr:.3f}",
        "",
        "## Strongest Descriptor Correlations With log10(nvMolKit time)",
        "",
        _markdown_table(top_nv[["feature", "log_nvmolkit_time_ms_spearman"]]),
        "",
        "## Strongest Descriptor Correlations With log10(nvMolKit/RDKit ratio)",
        "",
        _markdown_table(top_ratio[["feature", "log_time_ratio_nv_over_rdkit_spearman"]]),
        "",
        "## Largest Descriptor Differences In The Slowest 5% nvMolKit Pairs",
        "",
        _markdown_table(top_lift[["feature", "slow_top5_mean", "rest_mean", "difference", "ratio"]]),
        "",
        "## Ten Slowest nvMolKit Pairs",
        "",
        _markdown_table(
            top_rows[
                [
                    "pair_index",
                    "nvmolkit_time_ms",
                    "rdkit_time_ms",
                    "input_line_a",
                    "input_line_b",
                    "pair_min_atoms",
                    "pair_avg_atoms",
                    "pair_max_atoms",
                    "pair_max_rings",
                    "pair_max_ring_atoms",
                    "pair_max_longest_acyclic_carbon_chain",
                    "mcs_atoms",
                    "mcs_bonds",
                ]
            ]
        ),
        "",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="analysis/enamine_mcs_1k_pair_timings.csv",
        help="Per-pair timing CSV from benchmarks/mcs_bench.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/mcs_1k_timing_analysis",
        help="Directory for descriptor tables, plots, and summary.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    descriptor_rows = [_pair_features(row) for _, row in df.iterrows()]
    descriptor_df = pd.DataFrame(descriptor_rows)
    full = pd.concat([df.reset_index(drop=True), descriptor_df], axis=1)
    full = _add_runtime_features(full)

    descriptor_cols = _descriptor_columns(full)
    targets = [
        "log_nvmolkit_time_ms",
        "log_rdkit_time_ms",
        "log_time_ratio_nv_over_rdkit",
    ]
    corr = _corr_table(full, descriptor_cols, targets)

    full.to_csv(out_dir / "pair_descriptor_timings.csv", index=False)
    corr.to_csv(out_dir / "descriptor_correlations.csv", index=False)
    lift = _slow_tail_lift_table(full)
    lift.to_csv(out_dir / "slow_tail_top5_lift.csv", index=False)
    _save_slowest_table(full, out_dir / "slowest_nvmolkit_pairs.csv")

    _save_scatter(full, out_dir / "timing_scatter_nv_vs_rdkit.png")
    _save_top_feature_bars(
        corr,
        "log_nvmolkit_time_ms_spearman",
        out_dir / "top_descriptor_correlations_nvmolkit.png",
        "Top descriptor correlations with log10(nvMolKit time)",
    )
    _save_top_feature_bars(
        corr,
        "log_time_ratio_nv_over_rdkit_spearman",
        out_dir / "top_descriptor_correlations_nv_over_rdkit_ratio.png",
        "Top descriptor correlations with log10(nvMolKit/RDKit ratio)",
    )
    _save_size_plot(full, out_dir / "size_vs_nvmolkit_time.png")
    _save_ring_hydrocarbon_plot(full, out_dir / "rings_hydrocarbons_vs_nvmolkit_time.png")
    _save_slow_tail_lift_plot(lift, out_dir / "slow_tail_top5_descriptor_lift.png")
    _write_summary(full, corr, lift, out_dir / "summary.md")

    manifest = {
        "input": str(Path(args.input).resolve()),
        "output_dir": str(out_dir.resolve()),
        "rows": int(len(full)),
        "plots": sorted(str(path.name) for path in out_dir.glob("*.png")),
        "tables": sorted(str(path.name) for path in out_dir.glob("*.csv")),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
