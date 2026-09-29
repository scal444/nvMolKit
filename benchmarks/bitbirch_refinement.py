# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prototype post-hoc refinement of BitBIRCH partitions by nearest-centroid reassignment.

Starting from a BitBIRCH partition (label archives written by
``bitbirch_order_sensitivity.py --save-labels`` or equivalent), each iteration:

1. Computes every cluster's majority-bit centroid (ties set the bit, as in
   ``bitbirch_cluster_quality.py``) and each member's Tanimoto to it.
2. Finds, for every molecule, the most similar centroid of a *different*
   cluster with a fused int8 tensor-core Triton kernel (exhaustive N x K).
3. Proposes moving every molecule whose best other centroid beats its own.
4. Variants:

   * ``lloyd``: apply every proposal (a k-means/Lloyd step, no threshold guard).
   * ``guarded``: apply proposals, then repeatedly reject every move into or out
     of any multi-member cluster whose iSIM fell below the threshold. Rejection
     only removes moves, so it terminates, and the result keeps BitBIRCH's
     diameter guarantee (every multi-member cluster has iSIM >= threshold).
   * ``topn``: the published BitBIRCH "cluster reassignment" refinement
     (López Pérez et al., JCIM 2025): single pass, only members of the 20 most
     populated clusters move, only among those 20 centroids.
   * ``guarded-fine``: like ``guarded``, but a violating destination first sheds
     only its weakest incoming moves (lowest similarity to the destination
     centroid) in halves before falling back to rejecting everything.

Quality metrics after each iteration come from ``bitbirch_cluster_quality.quality``.
"""

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bitbirch_cluster_quality import chunk_bounds, isim_from_counts, quality

N_BITS = 2048


@triton.jit
def _best_other_kernel(
    Q,
    C,
    QP,
    CP,
    EXCL,
    OUT_S,
    OUT_I,
    M,
    K,
    n_per_split,
    W: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    split = pid // num_m
    mb = pid % num_m
    rm = mb * BLOCK_M + tl.arange(0, BLOCK_M)
    qmask = rm < M
    qp = tl.load(QP + rm, mask=qmask, other=0)
    excl = tl.load(EXCL + rm, mask=qmask, other=-1)
    best_s = tl.full([BLOCK_M], -1.0, tl.float32)
    best_i = tl.full([BLOCK_M], -1, tl.int32)
    n_start = split * n_per_split
    n_end = tl.minimum(n_start + n_per_split, K)
    for n0 in range(n_start, n_end, BLOCK_N):
        rn = n0 + tl.arange(0, BLOCK_N)
        nmask = rn < n_end
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)
        for k0 in range(0, W, BLOCK_K):
            rk = k0 + tl.arange(0, BLOCK_K)
            a = tl.load(Q + rm[:, None] * W + rk[None, :], mask=qmask[:, None], other=0)
            b = tl.load(C + rn[None, :] * W + rk[:, None], mask=nmask[None, :], other=0)
            acc = tl.dot(a, b, acc)
        cp = tl.load(CP + rn, mask=nmask, other=0)
        union = qp[:, None] + cp[None, :] - acc
        sim = tl.where(union > 0, acc.to(tl.float32) / tl.maximum(union, 1).to(tl.float32), 1.0)
        valid = nmask[None, :] & (rn[None, :] != excl[:, None])
        sim = tl.where(valid, sim, -1.0)
        tile_best = tl.max(sim, 1)
        cand = tl.where(sim == tile_best[:, None], rn[None, :], 2147483647)
        tile_idx = tl.min(cand, 1)
        upd = tile_best > best_s
        best_s = tl.where(upd, tile_best, best_s)
        best_i = tl.where(upd, tile_idx, best_i)
    tl.store(OUT_S + split * M + rm, best_s, mask=qmask)
    tl.store(OUT_I + split * M + rm, best_i, mask=qmask)


@triton.jit
def _best_other_ivf_kernel(
    Q,
    C,
    QP,
    CP,
    CID,
    EXCL,
    QIDX,
    ITEM_Q0,
    ITEM_QN,
    ITEM_C0,
    ITEM_C1,
    OUT_S,
    OUT_I,
    W: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One work item = up to BLOCK_M (query, cell) pairs probing the same cell.

    Centroids are sorted by cell; the item scans centroid rows [c0, c1). Output is
    per pair position (index into QIDX).
    """
    item = tl.program_id(0)
    q0 = tl.load(ITEM_Q0 + item)
    qn = tl.load(ITEM_QN + item)
    c0 = tl.load(ITEM_C0 + item)
    c1 = tl.load(ITEM_C1 + item)
    pos = q0 + tl.arange(0, BLOCK_M)
    pmask = tl.arange(0, BLOCK_M) < qn
    rm = tl.load(QIDX + pos, mask=pmask, other=0)
    qp = tl.load(QP + rm, mask=pmask, other=0)
    excl = tl.load(EXCL + rm, mask=pmask, other=-1)
    best_s = tl.full([BLOCK_M], -1.0, tl.float32)
    best_i = tl.full([BLOCK_M], -1, tl.int32)
    for n0 in range(c0, c1, BLOCK_N):
        rn = n0 + tl.arange(0, BLOCK_N)
        nmask = rn < c1
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)
        for k0 in range(0, W, BLOCK_K):
            rk = k0 + tl.arange(0, BLOCK_K)
            a = tl.load(Q + rm[:, None] * W + rk[None, :], mask=pmask[:, None], other=0)
            b = tl.load(C + rn[None, :] * W + rk[:, None], mask=nmask[None, :], other=0)
            acc = tl.dot(a, b, acc)
        cp = tl.load(CP + rn, mask=nmask, other=0)
        cid = tl.load(CID + rn, mask=nmask, other=-1)
        union = qp[:, None] + cp[None, :] - acc
        sim = tl.where(union > 0, acc.to(tl.float32) / tl.maximum(union, 1).to(tl.float32), 1.0)
        sim = tl.where(nmask[None, :] & (cid[None, :] != excl[:, None]), sim, -1.0)
        tile_best = tl.max(sim, 1)
        cand = tl.where(sim == tile_best[:, None], cid[None, :], 2147483647)
        tile_idx = tl.min(cand, 1)
        upd = tile_best > best_s
        best_s = tl.where(upd, tile_best, best_s)
        best_i = tl.where(upd, tile_idx, best_i)
    tl.store(OUT_S + pos, best_s, mask=pmask)
    tl.store(OUT_I + pos, best_i, mask=pmask)


def unpack_bits(packed_u8, shifts):
    """Packed little-order uint8 (n, 256) on device -> int8 bits (n, 2048)."""
    return ((packed_u8.unsqueeze(-1) >> shifts) & 1).reshape(packed_u8.shape[0], -1).to(torch.int8)


def best_other_centroid(
    fingerprints,
    labels,
    centroid_bits,
    centroid_pop,
    device,
    rows=None,
    query_chunk=65_536,
    split=8_192,
    block=(128, 128, 64),
):
    """Most similar centroid other than each molecule's own, for ``rows`` (default all).

    ``centroid_bits``: int8 (K, 2048) on device. Returns (similarity float32, index int64) numpy arrays.
    """
    rows = np.arange(len(labels)) if rows is None else rows
    shifts = torch.arange(8, device=device, dtype=torch.uint8)
    n_clusters = centroid_bits.shape[0]
    n_splits = triton.cdiv(n_clusters, split)
    out_s = np.empty(len(rows), dtype=np.float32)
    out_i = np.empty(len(rows), dtype=np.int64)
    bm, bn, bk = block
    for begin in range(0, len(rows), query_chunk):
        chunk = rows[begin : begin + query_chunk]
        q = unpack_bits(torch.from_numpy(fingerprints[chunk]).to(device), shifts)
        qp = q.sum(1, dtype=torch.int32)
        excl = torch.from_numpy(labels[chunk].astype(np.int32)).to(device)
        m = len(chunk)
        part_s = torch.empty((n_splits, m), dtype=torch.float32, device=device)
        part_i = torch.empty((n_splits, m), dtype=torch.int32, device=device)
        grid = (triton.cdiv(m, bm) * n_splits,)
        _best_other_kernel[grid](
            q,
            centroid_bits,
            qp,
            centroid_pop,
            excl,
            part_s,
            part_i,
            m,
            n_clusters,
            split,
            W=N_BITS,
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=bk,
            num_warps=8,
            num_stages=3,
        )
        best_split = part_s.argmax(0)
        out_s[begin : begin + m] = part_s.gather(0, best_split[None]).squeeze(0).cpu().numpy()
        out_i[begin : begin + m] = part_i.gather(0, best_split[None]).squeeze(0).long().cpu().numpy()
    return out_s, out_i


def tanimoto_matrix(a_half, a_pop, b_half, b_pop):
    inter = (a_half @ b_half.T).float()
    union = a_pop[:, None] + b_pop[None, :] - inter
    return torch.where(union > 0, inter / union.clamp_min(1), torch.ones_like(inter))


def build_coarse(centroid_bits, centroid_pop, n_cells, device, iterations=5, seed=0):
    """Majority-centroid k-means over the cluster centroids (Tanimoto assignment).

    Returns (coarse int8 (n_cells, 2048), cell of every centroid int64).
    """
    k = centroid_bits.shape[0]
    n_cells = min(n_cells, k)
    rng = np.random.default_rng(seed)
    coarse = centroid_bits[torch.from_numpy(rng.choice(k, n_cells, replace=False)).to(device)].clone()
    cell_of = torch.empty(k, dtype=torch.int64, device=device)
    for step in range(iterations + 1):
        ch, cp = coarse.half(), coarse.float().sum(1)
        for c in range(0, k, 32_768):
            sim = tanimoto_matrix(centroid_bits[c : c + 32_768].half(), centroid_pop[c : c + 32_768].float(), ch, cp)
            cell_of[c : c + 32_768] = sim.argmax(1)
        if step == iterations:
            break
        n = torch.bincount(cell_of, minlength=n_cells)
        counts = torch.zeros((n_cells, N_BITS), dtype=torch.int32, device=device)
        for c in range(0, k, 32_768):
            counts.index_add_(0, cell_of[c : c + 32_768], centroid_bits[c : c + 32_768].to(torch.int32))
        updated = (counts >= (n // 2 + n % 2).unsqueeze(1)).to(torch.int8)
        coarse = torch.where((n > 0)[:, None], updated, coarse)
    return coarse, cell_of


def best_other_centroid_ivf(
    fingerprints,
    labels,
    centroid_bits,
    centroid_pop,
    device,
    n_cells=4096,
    probes=16,
    query_chunk=131_072,
    block=(64, 64, 64),
):
    """Like ``best_other_centroid`` but each molecule scans only its ``probes`` most similar coarse cells."""
    shifts = torch.arange(8, device=device, dtype=torch.uint8)
    coarse, cell_of = build_coarse(centroid_bits, centroid_pop, n_cells, device)
    n_cells = coarse.shape[0]
    order = torch.argsort(cell_of)
    sorted_bits = centroid_bits[order]
    sorted_pop = centroid_pop[order]
    sorted_id = order.to(torch.int32)
    cell_start = torch.zeros(n_cells + 1, dtype=torch.int64, device=device)
    cell_start[1:] = torch.cumsum(torch.bincount(cell_of, minlength=n_cells), 0)
    coarse_h, coarse_p = coarse.half(), coarse.float().sum(1)
    bm, bn, bk = block
    out_s = np.empty(len(labels), dtype=np.float32)
    out_i = np.empty(len(labels), dtype=np.int64)
    for begin in range(0, len(labels), query_chunk):
        m = min(query_chunk, len(labels) - begin)
        q = unpack_bits(torch.from_numpy(fingerprints[begin : begin + m]).to(device), shifts)
        qp = q.sum(1, dtype=torch.int32)
        excl = torch.from_numpy(labels[begin : begin + m].astype(np.int32)).to(device)
        probe = torch.empty((m, probes), dtype=torch.int64, device=device)
        for r in range(0, m, 8192):
            sim = tanimoto_matrix(q[r : r + 8192].half(), qp[r : r + 8192].float(), coarse_h, coarse_p)
            probe[r : r + 8192] = sim.topk(probes, dim=1).indices
        pair_cell = probe.reshape(-1)
        pair_query = torch.arange(m, device=device).repeat_interleave(probes)
        by_cell = torch.argsort(pair_cell, stable=True)
        pair_cell, pair_query = pair_cell[by_cell], pair_query[by_cell].to(torch.int32)
        per_cell = torch.bincount(pair_cell, minlength=n_cells)
        pair_start = torch.zeros(n_cells + 1, dtype=torch.int64, device=device)
        pair_start[1:] = torch.cumsum(per_cell, 0)
        blocks = (per_cell + bm - 1) // bm
        item_cell = torch.repeat_interleave(torch.arange(n_cells, device=device), blocks)
        block_in_cell = torch.arange(len(item_cell), device=device) - torch.repeat_interleave(
            torch.cumsum(blocks, 0) - blocks, blocks
        )
        item_q0 = pair_start[item_cell] + block_in_cell * bm
        item_qn = torch.minimum(per_cell[item_cell] - block_in_cell * bm, torch.tensor(bm, device=device))
        pair_s = torch.empty(len(pair_query), dtype=torch.float32, device=device)
        pair_i = torch.empty(len(pair_query), dtype=torch.int32, device=device)
        _best_other_ivf_kernel[(len(item_cell),)](
            q,
            sorted_bits,
            qp,
            sorted_pop,
            sorted_id,
            excl,
            pair_query,
            item_q0,
            item_qn.to(torch.int32),
            cell_start[item_cell],
            cell_start[item_cell + 1],
            pair_s,
            pair_i,
            W=N_BITS,
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=bk,
            num_warps=4,
            num_stages=3,
        )
        query_long = pair_query.long()
        best = torch.full((m,), -1.0, device=device).scatter_reduce(0, query_long, pair_s, "amax")
        winner = torch.where(pair_s == best[query_long], pair_i, torch.full_like(pair_i, 2147483647))
        index = torch.full((m,), 2147483647, dtype=torch.int32, device=device).scatter_reduce(
            0, query_long, winner, "amin"
        )
        out_s[begin : begin + m] = best.cpu().numpy()
        out_i[begin : begin + m] = index.long().cpu().numpy()
        del q, probe, pair_cell, pair_query, pair_s, pair_i
    return out_s, out_i


def cluster_stats(fingerprints, labels, device, want_centroids=True, chunk_molecules=50_000):
    """Sizes, iSIM, and optionally int8 centroids + member-to-own-centroid similarity."""
    shifts = torch.arange(8, device=device, dtype=torch.uint8)
    sizes = np.bincount(labels)
    n_clusters = len(sizes)
    order = np.argsort(labels, kind="stable")
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    isim = np.full(n_clusters, np.nan)
    member_sim = np.ones(len(labels), dtype=np.float32) if want_centroids else None
    centroids = torch.zeros((n_clusters, N_BITS), dtype=torch.int8, device=device) if want_centroids else None
    spans = chunk_bounds(sizes, chunk_molecules)
    for first, last in spans:
        rows = order[offsets[first] : offsets[last]]
        chunk_sizes = torch.from_numpy(sizes[first:last]).to(device)
        local = torch.repeat_interleave(torch.arange(last - first, device=device), chunk_sizes)
        bits = unpack_bits(torch.from_numpy(fingerprints[rows]).to(device), shifts)
        counts = torch.zeros((last - first, N_BITS), dtype=torch.int32, device=device)
        counts.index_add_(0, local, bits.to(torch.int32))
        isim[first:last] = isim_from_counts(counts, chunk_sizes).cpu().numpy()
        if want_centroids:
            majority = (counts >= (chunk_sizes // 2 + chunk_sizes % 2).unsqueeze(1)).to(torch.int8)
            centroids[first:last] = majority
            cbits = majority[local]
            inter = (bits & cbits).sum(1, dtype=torch.int32)
            union = (bits | cbits).sum(1, dtype=torch.int32)
            sim = torch.where(union > 0, inter / union.clamp_min(1), torch.ones_like(inter, dtype=torch.float32))
            member_sim[rows] = sim.cpu().numpy()
        del bits, counts
    return sizes, isim, centroids, member_sim


def popcounts(bits, chunk=65_536):
    """Row popcounts of an int8 bit matrix without materializing an int32 copy."""
    return torch.cat([bits[i : i + chunk].sum(1, dtype=torch.int32) for i in range(0, bits.shape[0], chunk)])


def compact(labels):
    _, inverse = np.unique(labels, return_inverse=True)
    return inverse.astype(np.int32)


def violating(fingerprints, labels, threshold, device, touched):
    """Boolean mask over clusters: touched multi-member clusters with iSIM < threshold."""
    sizes, isim, _, _ = cluster_stats(fingerprints, labels, device, want_centroids=False)
    return touched[: len(sizes)] & (sizes >= 2) & (isim < threshold - 1e-12)


def topn_step(fingerprints, labels, device, top_n=20):
    """Published BitBIRCH cluster reassignment (López Pérez et al., JCIM 2025).

    Members of the ``top_n`` most populated clusters move to the closest of those clusters' centroids.
    """
    t0 = time.perf_counter()
    shifts = torch.arange(8, device=device, dtype=torch.uint8)
    sizes = np.bincount(labels)
    top = np.argsort(-sizes, kind="stable")[:top_n]
    rows = np.flatnonzero(np.isin(labels, top))
    bits = unpack_bits(torch.from_numpy(fingerprints[rows]).to(device), shifts)
    lookup = np.full(len(sizes), -1)
    lookup[top] = np.arange(len(top))
    local = torch.from_numpy(lookup[labels[rows]]).to(device)
    counts = torch.zeros((len(top), N_BITS), dtype=torch.int32, device=device)
    counts.index_add_(0, local, bits.to(torch.int32))
    n = torch.from_numpy(sizes[top]).to(device)
    cent = (counts >= (n // 2 + n % 2).unsqueeze(1)).to(torch.float32)
    inter = bits.to(torch.float32) @ cent.T
    union = bits.sum(1, dtype=torch.int32)[:, None].float() + cent.sum(1)[None] - inter
    sim = torch.where(union > 0, inter / union.clamp_min(1), torch.ones_like(inter))
    # Keep the current cluster on ties.
    sim[torch.arange(len(rows), device=device), local] += 1e-6
    new = labels.copy()
    new[rows] = top[sim.argmax(1).cpu().numpy()]
    moved = int((new != labels).sum())
    stats = {
        "proposed_moves": moved,
        "guard_rounds": 0,
        "violations_unguarded": -1,
        "accepted_moves": moved,
        "seconds_centroids": 0.0,
        "seconds_search": time.perf_counter() - t0,
        "seconds_guard": 0.0,
    }
    return compact(new), stats


def refine_step(fingerprints, labels, threshold, variant, device, margin=0.0, search="exhaustive", ivf=(4096, 16)):
    """One reassignment iteration. Returns new (compacted) labels and a stats dict."""
    if variant == "topn":
        return topn_step(fingerprints, labels, device)
    t0 = time.perf_counter()
    sizes, _, centroids, member_sim = cluster_stats(fingerprints, labels, device)
    torch.cuda.empty_cache()
    centroid_pop = popcounts(centroids)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    if search == "ivf":
        best_s, best_i = best_other_centroid_ivf(
            fingerprints, labels, centroids, centroid_pop, device, n_cells=ivf[0], probes=ivf[1]
        )
    else:
        best_s, best_i = best_other_centroid(fingerprints, labels, centroids, centroid_pop, device)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    del centroids, centroid_pop
    torch.cuda.empty_cache()

    propose = best_s > member_sim + margin
    movers = np.flatnonzero(propose)
    dest = best_i[movers]
    src = labels[movers]
    stats = {"proposed_moves": len(movers), "guard_rounds": 0, "violations_unguarded": -1}
    accept = np.ones(len(movers), dtype=bool)
    if variant in ("guarded", "guarded-fine"):
        n_clusters = len(sizes)
        # Sub-fraction of incoming moves each violating destination keeps in the fine variant.
        keep_frac = np.ones(n_clusters)
        while True:
            new = labels.copy()
            new[movers[accept]] = dest[accept]
            touched = np.zeros(n_clusters, dtype=bool)
            touched[dest[accept]] = True
            touched[src[accept]] = True
            bad = violating(fingerprints, new, threshold, device, touched)
            if stats["guard_rounds"] == 0:
                stats["violations_unguarded"] = int(bad.sum())
            stats["guard_rounds"] += 1
            if not bad.any():
                break
            if variant == "guarded":
                accept &= ~(bad[dest] | bad[src])
                continue
            # Fine: halve incoming moves of violating destinations (keep the most similar);
            # once a destination is down to zero incoming, reject moves out of it too.
            for cluster in np.flatnonzero(bad):
                keep_frac[cluster] = keep_frac[cluster] / 2 if keep_frac[cluster] > 1 / 16 else 0.0
            incoming_bad = accept & bad[dest]
            if incoming_bad.any():
                idx = np.flatnonzero(incoming_bad)
                d = dest[idx]
                s = best_s[movers[idx]]
                order = np.lexsort((-s, d))
                idx, d = idx[order], d[order]
                starts = np.flatnonzero(np.r_[True, d[1:] != d[:-1]])
                counts = np.diff(np.r_[starts, len(d)])
                rank = np.arange(len(d)) - np.repeat(starts, counts)
                keep = rank < np.floor(np.repeat(counts, counts) * keep_frac[d])
                accept[idx[~keep]] = False
            zero = keep_frac == 0.0
            accept &= ~(zero[src] | zero[dest])
            # A violating source that received no incoming moves: reject its outgoing moves.
            no_incoming = bad & ~np.isin(np.arange(n_clusters), dest[accept])
            accept &= ~no_incoming[src]
    new = labels.copy()
    new[movers[accept]] = dest[accept]
    t3 = time.perf_counter()
    stats.update(
        accepted_moves=int(accept.sum()),
        seconds_centroids=t1 - t0,
        seconds_search=t2 - t1,
        seconds_guard=t3 - t2,
    )
    return compact(new), stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fingerprints", help="packed uint8 .npy fingerprints, shape (N, 256)")
    parser.add_argument("labels", nargs="+", type=Path, help="labels_threshold_<t>.npz archives")
    parser.add_argument("--runs", nargs="+", default=None, help="archive keys to refine (default all)")
    parser.add_argument("--variants", nargs="+", default=["lloyd", "guarded"])
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--search", choices=["exhaustive", "ivf"], default="exhaustive")
    parser.add_argument("--ivf-cells", type=int, default=4096)
    parser.add_argument("--ivf-probes", type=int, default=16)
    parser.add_argument("--scaffolds")
    parser.add_argument("--misassignment-sample", type=int, default=20_000)
    parser.add_argument("--output", type=Path, required=True, help="CSV of per-iteration metrics")
    parser.add_argument("--save-labels", type=Path, help="directory for refined label npz archives")
    args = parser.parse_args()

    device = torch.device("cuda")
    fingerprints = np.load(args.fingerprints)
    scaffolds = np.load(args.scaffolds) if args.scaffolds else None
    # Completed runs are kept as part files next to the output and skipped on rerun.
    parts = args.output.with_suffix(".parts")
    parts.mkdir(parents=True, exist_ok=True)
    for archive_path in args.labels:
        threshold = float(re.search(r"labels_threshold_([0-9.]+)\.npz", archive_path.name).group(1))
        archive = np.load(archive_path)
        names = [n for n in sorted(archive.files) if args.runs is None or n in args.runs]
        for name in names:
            part = parts / f"{threshold}_{name}.csv"
            if part.exists():
                continue
            rows = []
            method, batch, seed = re.fullmatch(r"(\w+?)_b(\d+)_s(\d+)", name).groups()
            base = compact(archive[name])
            fps = fingerprints[: len(base)]
            scaf = None if scaffolds is None else scaffolds[: len(base)]

            search_name = "exhaustive" if args.search == "exhaustive" else f"ivf{args.ivf_cells}p{args.ivf_probes}"

            def record(variant, iteration, labels, step):
                metrics = quality(fps, labels, threshold, scaf, args.misassignment_sample, device, 50_000)
                rows.append(
                    {
                        "threshold": threshold,
                        "method": method,
                        "batch_size": int(batch),
                        "seed": int(seed),
                        "variant": variant if variant in ("none", "topn") else f"{variant}-{search_name}",
                        "iteration": iteration,
                        **step,
                        **metrics,
                    }
                )
                print(
                    f"t={threshold} {name} {variant} it{iteration}: clusters={metrics['clusters']} "
                    f"misassigned={metrics['misassigned_fraction']:.4f} isim_w={metrics['isim_size_weighted']:.4f} "
                    f"below={metrics['isim_below_threshold']} {step}",
                    flush=True,
                )

            empty = dict.fromkeys(
                [
                    "proposed_moves",
                    "accepted_moves",
                    "guard_rounds",
                    "violations_unguarded",
                    "seconds_centroids",
                    "seconds_search",
                    "seconds_guard",
                ],
                0,
            )
            record("none", 0, base, empty)
            for variant in args.variants:
                labels = base
                saved = {}
                for iteration in range(1, args.iterations + 1):
                    labels, step = refine_step(
                        fps,
                        labels,
                        threshold,
                        variant,
                        device,
                        args.margin,
                        args.search,
                        (args.ivf_cells, args.ivf_probes),
                    )
                    record(variant, iteration, labels, step)
                    saved[f"{method}_b{batch}_s{seed}_it{iteration}"] = labels
                    if step["accepted_moves"] == 0 or variant == "topn":
                        break
                if args.save_labels:
                    args.save_labels.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        args.save_labels
                        / f"refined_{variant}-{search_name}_{threshold}_{method}_b{batch}_s{seed}.npz",
                        **saved,
                    )
            write(part, rows)
            combined = [r for path in sorted(parts.glob("*.csv")) for r in csv.DictReader(open(path))]
            write(args.output, combined)


def write(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
