# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Candidate-pruning study for BitBIRCH nearest-centroid refinement.

For a sample of molecules, the exhaustive best other centroid (Triton kernel in
``bitbirch_refinement.py``) is the ground truth. Reports, per scheme:

* ``centroid_knn_M``: candidates are the M centroids most similar to the
  molecule's own centroid (a centroid kNN graph, computed once per iteration).
  Recall = share of would-be movers whose exhaustive best other centroid is in
  the candidate set; ``found_move`` = share of would-be movers for which the
  best candidate still beats the own centroid (the move still happens, maybe
  to a slightly worse cluster).
* ``ivf_C_pP``: centroids are grouped into C coarse cells (majority-centroid
  k-means on the centroids, Tanimoto assignment); each molecule scans only the
  centroids in its P most similar cells.
* ``popcount_bound``: exact pruning by Tanimoto <= min(|a|,|b|)/max(|a|,|b|):
  a centroid can only beat the own-centroid similarity s if its popcount lies in
  [s|a|, |a|/s]. Reports the share of (molecule, centroid) pairs that survive.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bitbirch_refinement import (
    best_other_centroid,
    build_coarse,
    cluster_stats,
    compact,
    popcounts,
    tanimoto_matrix,
)


def ivf_eval(cent, cpop, qb, own, own_s, best_s, best_i, movers, n_cells, probes, device):
    k = cent.shape[0]
    coarse, cell_of = build_coarse(cent, cpop, n_cells, device)
    cell_sizes = torch.bincount(cell_of, minlength=n_cells).cpu().numpy()
    qh, qp = qb.half(), qb.float().sum(1)
    qcell = tanimoto_matrix(qh, qp, coarse.half(), coarse.float().sum(1))
    rank = torch.empty_like(qcell, dtype=torch.int32)
    rank.scatter_(
        1,
        qcell.argsort(1, descending=True),
        torch.arange(n_cells, device=device, dtype=torch.int32).expand_as(rank).contiguous(),
    )
    own_t = torch.from_numpy(own.astype(np.int64)).to(device)
    found_s = torch.full((len(probes), len(own)), -1.0, device=device)
    for c in range(0, k, 32_768):
        cells = cell_of[c : c + 32_768]
        for r in range(0, len(own), 1024):
            sim = tanimoto_matrix(
                qh[r : r + 1024], qp[r : r + 1024], cent[c : c + 32_768].half(), cpop[c : c + 32_768].float()
            )
            sim[own_t[r : r + 1024, None] == torch.arange(c, c + len(cells), device=device)[None]] = -1.0
            crank = rank[r : r + 1024][:, cells]
            for j, p in enumerate(probes):
                masked = torch.where(crank < p, sim, torch.full_like(sim, -1.0))
                found_s[j, r : r + 1024] = torch.maximum(found_s[j, r : r + 1024], masked.max(1).values)
    best_cell_rank = rank.gather(1, cell_of[torch.from_numpy(best_i).to(device)][:, None]).squeeze(1).cpu().numpy()
    rank_np = rank.cpu().numpy()
    out = []
    for j, p in enumerate(probes):
        fs = found_s[j].cpu().numpy()
        found = fs > own_s
        scanned = np.array([cell_sizes[rank_np[i] < p].sum() for i in range(len(own))])
        gain = (fs[movers & found] - own_s[movers & found]).sum() / (best_s[movers] - own_s[movers]).sum()
        out.append(
            {
                "scheme": f"ivf_{n_cells}_p{p}",
                "candidates_per_molecule": float(scanned.mean()),
                "movers_fraction": float(movers.mean()),
                "recall_exact_best": float((best_cell_rank < p)[movers].mean()),
                "found_move": float(found[movers].mean()),
                "false_moves": float((found & ~movers).mean()),
                "gain_captured": float(gain),
                "pair_fraction": float(scanned.mean() / k),
            }
        )
        print(out[-1], flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fingerprints")
    parser.add_argument("labels", help="npz archive")
    parser.add_argument("run", help="archive key")
    parser.add_argument("--sample", type=int, default=20_000)
    parser.add_argument("--knn", type=int, nargs="+", default=[8, 32, 128, 512, 2048])
    parser.add_argument("--ivf-cells", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--ivf-probes", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    fps = np.load(args.fingerprints)
    labels = compact(np.load(args.labels)[args.run])
    fps = fps[: len(labels)]
    _, _, cent, msim = cluster_stats(fps, labels, device)
    torch.cuda.empty_cache()
    cpop = popcounts(cent)
    rng = np.random.default_rng(0)
    q = np.sort(rng.choice(len(labels), size=min(args.sample, len(labels)), replace=False))
    best_s, best_i = best_other_centroid(fps, labels, cent, cpop, device, rows=q)
    own_s = msim[q]
    movers = best_s > own_s
    own = labels[q]

    # Top-M neighbours (excluding self) of each sampled molecule's own centroid.
    max_m = max(args.knn)
    uniq, inv = np.unique(own, return_inverse=True)
    topk_i = torch.empty((len(uniq), max_m), dtype=torch.int64)
    ub = cent[torch.from_numpy(uniq).to(device)].half()
    up = ub.float().sum(1)
    for r in range(0, len(uniq), 1024):
        rows = ub[r : r + 1024]
        rp = up[r : r + 1024]
        run_v = torch.full((len(rows), max_m), -2.0, device=device)
        run_i = torch.zeros((len(rows), max_m), dtype=torch.int64, device=device)
        for c in range(0, cent.shape[0], 32_768):
            cb = cent[c : c + 32_768].half()
            inter = (rows @ cb.T).float()
            union = rp[:, None] + cpop[c : c + 32_768][None].float() - inter
            sim = torch.where(union > 0, inter / union.clamp_min(1), torch.ones_like(inter))
            self_mask = (
                torch.from_numpy(uniq[r : r + 1024]).to(device)[:, None]
                == torch.arange(c, c + cb.shape[0], device=device)[None]
            )
            sim[self_mask] = -1.0
            v, i = sim.topk(min(max_m, sim.shape[1]), dim=1)
            allv = torch.cat([run_v, v], 1)
            alli = torch.cat([run_i, i + c], 1)
            run_v, sel = allv.topk(max_m, dim=1)
            run_i = alli.gather(1, sel)
        topk_i[r : r + 1024] = run_i.cpu()
    topk_i = topk_i.numpy()[inv]

    # Similarity of each sampled molecule to all of its top-M candidates (for found_move).
    shifts = torch.arange(8, device=device, dtype=torch.uint8)
    qb = ((torch.from_numpy(fps[q]).to(device).unsqueeze(-1) >> shifts) & 1).reshape(len(q), -1).to(torch.int8)
    rows_out = []
    cand_sims = np.empty((len(q), max_m), dtype=np.float32)
    for r in range(0, len(q), 16):
        cb = cent[torch.from_numpy(topk_i[r : r + 16]).to(device)]  # (b, M, 2048)
        a = qb[r : r + 16, None, :]
        inter = (a & cb).sum(-1, dtype=torch.int32).float()
        union = (a | cb).sum(-1, dtype=torch.int32).float()
        cand_sims[r : r + 16] = (
            torch.where(union > 0, inter / union.clamp_min(1), torch.ones_like(inter)).cpu().numpy()
        )
    for m in args.knn:
        hit = (topk_i[:, :m] == best_i[:, None]).any(1)
        cand_best = cand_sims[:, :m].max(1)
        found = cand_best > own_s
        gain_ratio = (cand_best[movers & found] - own_s[movers & found]).sum() / (best_s[movers] - own_s[movers]).sum()
        rows_out.append(
            {
                "scheme": f"centroid_knn_{m}",
                "candidates_per_molecule": m,
                "movers_fraction": float(movers.mean()),
                "recall_exact_best": float(hit[movers].mean()),
                "found_move": float(found[movers].mean()),
                "false_moves": float((found & ~movers).mean()),
                "gain_captured": float(gain_ratio),
                "pair_fraction": m / cent.shape[0],
            }
        )

    for n_cells in args.ivf_cells:
        rows_out += ivf_eval(cent, cpop, qb, own, own_s, best_s, best_i, movers, n_cells, args.ivf_probes, device)

    # Exact popcount bound.
    cp = np.sort(cpop.cpu().numpy())
    qp = qb.sum(1).cpu().numpy().astype(np.float64)
    s = np.clip(own_s.astype(np.float64), 1e-6, 1.0)
    lo = np.searchsorted(cp, np.ceil(s * qp - 1e-9), side="left")
    hi = np.searchsorted(cp, np.floor(qp / s + 1e-9), side="right")
    rows_out.append(
        {
            "scheme": "popcount_bound",
            "candidates_per_molecule": float((hi - lo).mean()),
            "movers_fraction": float(movers.mean()),
            "recall_exact_best": 1.0,
            "found_move": 1.0,
            "false_moves": 0.0,
            "gain_captured": 1.0,
            "pair_fraction": float((hi - lo).mean() / len(cp)),
        }
    )
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)
    for row in rows_out:
        print(row)


if __name__ == "__main__":
    main()
