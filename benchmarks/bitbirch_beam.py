# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Linear-cost candidate search for BitBIRCH refinement: beam search over a CF-tree of the clusters.

Tree: a bblean ``BitBirch`` (branching factor 254) into which every cluster of the
partition is inserted as one BitFeature (linear sum + size) with the ``never`` merge
criterion, via ``BitBirch._fit_buffers(..., reinsert_index_seqs=[[cluster_id], ...])``.
Every insertion therefore routes like a BitBIRCH insertion (closest majority centroid
per level, ties set the bit) and appends a new leaf entry, splitting nodes exactly as
bblean does, so the leaf entries are the clusters themselves. Clusters are inserted
largest first (bblean's own refine paths reinsert size-sorted BitFeatures), in chunks
of one buffer dtype (uint8 for n < 256, uint16 otherwise) because ``_fit_buffers``
uses the first row's dtype for the whole call. The topology is kept fixed across
refinement iterations; node summaries are recomputed on the GPU from the current
memberships each iteration (internal entry = sum of its descendants' molecules).

Search: for each molecule, keep the best ``b`` entries per level (Tanimoto to the
entry's majority centroid), expand their children, and at the leaf level return the
best cluster other than the molecule's own. ``b = 1`` is plain greedy routing. Cost per
molecule is about depth x b x branching, independent of the number of clusters.

Commands (outputs are skipped when they exist):
  tree    FPS ARCHIVE RUN --out TREE.npz [--prefix N]
  refine  FPS ARCHIVE RUN --tree TREE.npz --beams B... --out-dir DIR
  recall  FPS ARCHIVE RUN --tree TREE.npz --beams B... --output CSV --cache NPZ
  cost    FPS ARCHIVE RUN --sizes N... --beams B... --tree-dir DIR --output CSV
"""

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bitbirch_refinement import (
    N_BITS,
    best_other_centroid,
    best_other_centroid_ivf,
    cluster_stats,
    compact,
    dedup_step,
    popcounts,
    refine_step,
    unpack_bits,
)

WORDS = N_BITS // 32


@triton.jit
def _gather_tanimoto_kernel(Q, C, QP, CP, IDX, OUT, n_cand, W: tl.constexpr, BLOCK_C: tl.constexpr):
    """OUT[m, j] = Tanimoto(Q[m], C[IDX[m, j]]) on packed int32 words; -1 where IDX < 0."""
    m = tl.program_id(0)
    rc = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = rc < n_cand
    idx = tl.load(IDX + m.to(tl.int64) * n_cand + rc, mask=cmask, other=-1)
    valid = idx >= 0
    rw = tl.arange(0, W)
    q = tl.load(Q + m.to(tl.int64) * W + rw)
    c = tl.load(C + idx.to(tl.int64)[:, None] * W + rw[None, :], mask=valid[:, None], other=0)
    inter = tl.sum(libdevice.popc(q[None, :] & c), axis=1)
    qp = tl.load(QP + m)
    cp = tl.load(CP + idx, mask=valid, other=0)
    union = qp + cp - inter
    sim = tl.where(union > 0, inter.to(tl.float32) / tl.maximum(union, 1).to(tl.float32), 1.0)
    tl.store(OUT + m.to(tl.int64) * n_cand + rc, tl.where(valid, sim, -1.0), mask=cmask)


def gather_tanimoto(q_words, q_pop, c_words, c_pop, idx, block=128):
    m, n_cand = idx.shape
    out = torch.empty((m, n_cand), dtype=torch.float32, device=idx.device)
    _gather_tanimoto_kernel[(m, triton.cdiv(n_cand, block))](
        q_words, c_words, q_pop, c_pop, idx, out, n_cand, W=WORDS, BLOCK_C=block, num_warps=4
    )
    return out


def pack_words(bits_int8):
    """int8 bits (n, 2048) -> int32 words (n, 64), same byte layout as packed little-order fingerprints."""
    weights = (1 << torch.arange(8, device=bits_int8.device, dtype=torch.int32)).view(1, 1, 8)
    out = torch.empty((bits_int8.shape[0], WORDS), dtype=torch.int32, device=bits_int8.device)
    for c in range(0, bits_int8.shape[0], 65_536):
        block = (bits_int8[c : c + 65_536].reshape(-1, N_BITS // 8, 8).to(torch.int32) * weights).sum(-1)
        out[c : c + 65_536] = block.to(torch.uint8).view(torch.int32)
    return out


# ---------------------------------------------------------------------------------------------
# Tree build (CPU, bblean)


def cluster_sums(fingerprints, labels, device, chunk=50_000):
    """Per-cluster linear sums (K, 2048) as host uint16 and sizes."""
    sizes = np.bincount(labels)
    order = np.argsort(labels, kind="stable")
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    sums = np.empty((len(sizes), N_BITS), dtype=np.uint16)
    shifts = torch.arange(8, device=device, dtype=torch.uint8)
    begin = 0
    while begin < len(sizes):
        end = max(int(np.searchsorted(offsets, offsets[begin] + chunk, side="right")) - 1, begin + 1)
        rows = order[offsets[begin] : offsets[end]]
        local = torch.from_numpy(np.repeat(np.arange(end - begin), sizes[begin:end])).to(device)
        counts = torch.zeros((end - begin, N_BITS), dtype=torch.int32, device=device)
        counts.index_add_(
            0, local, unpack_bits(torch.from_numpy(fingerprints[rows]).to(device), shifts).to(torch.int32)
        )
        sums[begin:end] = counts.cpu().numpy().astype(np.uint16)
        begin = end
    return sums, sizes


def build_tree(fingerprints, labels, device, threshold, branching_factor=254, chunk=50_000):
    """Bblean CF-tree whose leaf entries are the clusters of ``labels``. Returns topology arrays + seconds."""
    import bblean

    t0 = time.perf_counter()
    sums, sizes = cluster_sums(fingerprints, labels, device)
    t_sums = time.perf_counter() - t0
    t0 = time.perf_counter()
    tree = bblean.BitBirch(threshold=threshold, branching_factor=branching_factor, merge_criterion="never")
    order = np.argsort(-sizes, kind="stable")
    for dtype, select in ((np.uint16, sizes[order] >= 256), (np.uint8, sizes[order] < 256)):
        part = order[select]
        for c in range(0, len(part), chunk):
            ids = part[c : c + chunk]
            buffers = np.empty((len(ids), N_BITS + 1), dtype=dtype)
            buffers[:, :-1] = sums[ids]
            buffers[:, -1] = sizes[ids]
            tree._fit_buffers(buffers, reinsert_index_seqs=[[int(i)] for i in ids], check_indices=False)
    t_build = time.perf_counter() - t0

    # Breadth-first topology. Level 0 = root entries; children of level l entries are contiguous at l+1.
    levels, nodes = [], [tree._root]
    while True:
        entries = [s for node in nodes for s in node._subclusters]
        counts = np.array([len(node._subclusters) for node in nodes])
        is_leaf = [s.child is None for s in entries]
        if all(is_leaf):
            leaf_cluster = np.array([s.mol_indices[0] for s in entries], dtype=np.int64)
            assert all(len(s.mol_indices) == 1 for s in entries)
            levels.append({"node_sizes": counts})
            break
        assert not any(is_leaf), "CF-tree is not height-balanced"
        levels.append({"node_sizes": counts})
        nodes = [s.child for s in entries]
    assert np.array_equal(np.sort(leaf_cluster), np.arange(len(sizes)))
    out = {"leaf_cluster": leaf_cluster, "seconds_sums": t_sums, "seconds_build": t_build, "depth": len(levels)}
    # node_sizes[l][i] = number of entries in the i-th node of level l; nodes of level l+1 are the children
    # of level-l entries in order, so child ranges of level-l entries are consecutive slices of level l+1.
    for level, info in enumerate(levels):
        out[f"node_sizes_{level}"] = info["node_sizes"]
    return out


class Tree:
    """Fixed topology loaded from ``build_tree`` output."""

    def __init__(self, arrays, device):  # noqa: D107
        self.depth = int(arrays["depth"])
        self.device = device
        self.leaf_cluster = torch.from_numpy(np.asarray(arrays["leaf_cluster"])).to(device)
        self.n_clusters = len(self.leaf_cluster)
        node_sizes = [np.asarray(arrays[f"node_sizes_{level}"]) for level in range(self.depth)]
        # Entries at level l+1 are grouped in nodes; node j of level l+1 is the child of entry j of level l.
        self.child_start, self.child_count, self.parent = [], [], []
        for level in range(self.depth - 1):
            sizes = node_sizes[level + 1]
            start = np.concatenate([[0], np.cumsum(sizes)[:-1]])
            self.child_start.append(torch.from_numpy(start).to(device))
            self.child_count.append(torch.from_numpy(sizes).to(device))
            self.parent.append(np.repeat(np.arange(len(sizes)), sizes))  # level l+1 entry -> level l entry
        self.n_entries = [int(node_sizes[level].sum()) for level in range(self.depth)]
        self.max_children = int(max(s.max() for s in node_sizes))
        # ancestor[l][cluster] = level-l entry above that cluster (internal levels only)
        leaf_pos = np.empty(self.n_clusters, dtype=np.int64)
        leaf_pos[np.asarray(arrays["leaf_cluster"])] = np.arange(self.n_clusters)
        self.ancestor = [None] * (self.depth - 1)
        current = leaf_pos
        for level in range(self.depth - 2, -1, -1):
            current = self.parent[level][current]
            self.ancestor[level] = current


class BeamSearch:
    """Callable for ``refine_step``: best other cluster per molecule by beam search over ``tree``."""

    def __init__(self, tree, beam, device, chunk=None):  # noqa: D107
        # Keep chunk x beam x branching (candidate matrix) near 33M entries.
        self.tree, self.beam, self.device = tree, beam, device
        self.chunk = chunk or max(64, min(8192, 131_072 // beam))
        self.n_clusters = tree.n_clusters

    def summaries(self, fingerprints, labels):
        """Packed majority centroids, popcounts and alive masks of every internal level."""
        tree, device = self.tree, self.device
        shifts = torch.arange(8, device=device, dtype=torch.uint8)
        counts = [torch.zeros((n, N_BITS), dtype=torch.int32, device=device) for n in tree.n_entries[:-1]]
        sizes = [torch.zeros(n, dtype=torch.int64, device=device) for n in tree.n_entries[:-1]]
        for begin in range(0, len(labels), 32_768):
            lab = labels[begin : begin + 32_768]
            bits = unpack_bits(torch.from_numpy(fingerprints[begin : begin + 32_768]).to(device), shifts).to(
                torch.int32
            )
            for level in range(tree.depth - 1):
                entry = torch.from_numpy(tree.ancestor[level][lab]).to(device)
                counts[level].index_add_(0, entry, bits)
                sizes[level] += torch.bincount(entry, minlength=tree.n_entries[level])
        out = []
        for level in range(tree.depth - 1):
            n = sizes[level]
            majority = ((counts[level] >= (n // 2 + n % 2).unsqueeze(1)) & (n > 0).unsqueeze(1)).to(torch.int8)
            out.append((pack_words(majority), popcounts(majority), n > 0))
        return out

    def __call__(self, fingerprints, labels, centroids, centroid_pop, sizes):
        """Best other cluster (similarity, id) per molecule; ``centroids`` are int8 cluster centroids."""
        tree, device, beam = self.tree, self.device, self.beam
        levels = self.summaries(fingerprints, labels)
        leaf_words = pack_words(centroids)
        leaf_alive = torch.from_numpy(sizes > 0).to(device)
        fp_words = torch.from_numpy(np.ascontiguousarray(fingerprints).view(np.int32))
        span = torch.arange(tree.max_children, device=device)
        out_s = np.empty(len(labels), dtype=np.float32)
        out_i = np.empty(len(labels), dtype=np.int64)
        for begin in range(0, len(labels), self.chunk):
            q = fp_words[begin : begin + self.chunk].to(device)
            m = q.shape[0]
            qp = torch.from_numpy(np.unpackbits(fingerprints[begin : begin + m], axis=1).sum(1).astype(np.int32)).to(
                device
            )
            own = torch.from_numpy(labels[begin : begin + m].astype(np.int64)).to(device)
            words, pop, alive = levels[0]
            kept = None
            idx = torch.where(alive, torch.arange(len(alive), device=device), -1).expand(m, -1).contiguous()
            for level in range(tree.depth):
                if level > 0:
                    start = tree.child_start[level - 1][kept.clamp_min(0)]
                    count = torch.where(kept >= 0, tree.child_count[level - 1][kept.clamp_min(0)], 0)
                    child = start[..., None] + span
                    child = torch.where(span < count[..., None], child, -1).reshape(m, -1)
                    if level < tree.depth - 1:
                        words, pop, alive = levels[level]
                        idx = torch.where(child >= 0, torch.where(alive[child.clamp_min(0)], child, -1), -1)
                    else:
                        cluster = tree.leaf_cluster[child.clamp_min(0)]
                        ok = (child >= 0) & leaf_alive[cluster] & (cluster != own[:, None])
                        idx = torch.where(ok, cluster, -1)
                if level < tree.depth - 1:
                    sim = gather_tanimoto(q, qp, words, pop, idx.to(torch.int32).contiguous())
                    top = sim.topk(min(beam, sim.shape[1]), dim=1)
                    kept = torch.where(top.values >= 0, idx.gather(1, top.indices), -1)
                else:
                    sim = gather_tanimoto(q, qp, leaf_words, centroid_pop, idx.to(torch.int32).contiguous())
                    best, pos = sim.max(1)
                    out_s[begin : begin + m] = best.cpu().numpy()
                    out_i[begin : begin + m] = idx.gather(1, pos[:, None]).squeeze(1).cpu().numpy()
        return out_s, out_i


# ---------------------------------------------------------------------------------------------


def load_run(args):
    threshold = float(re.search(r"labels_threshold_([0-9.]+)\.npz", Path(args.labels).name).group(1))
    labels = compact(np.load(args.labels)[args.run])
    if getattr(args, "prefix", None):
        labels = compact(labels[: args.prefix])
    fingerprints = np.ascontiguousarray(np.load(args.fingerprints, mmap_mode="r")[: len(labels)])
    return threshold, labels, fingerprints


def get_tree(path, fingerprints, labels, threshold, device):
    path = Path(path)
    if not path.exists():
        arrays = build_tree(fingerprints, labels, device, threshold)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path.with_suffix(".partial.npz"), **arrays)
        path.with_suffix(".partial.npz").rename(path)
        print(f"tree {path}: depth {arrays['depth']}, build {arrays['seconds_build']:.1f}s", flush=True)
    return Tree(np.load(path), device)


def cmd_refine(args, device):
    threshold, base, fps = load_run(args)
    method, batch, seed = re.fullmatch(r"(\w+?)_b(\d+)_s(\d+)", args.run).groups()
    tree = get_tree(args.tree, fps, base, threshold, device)
    for beam in args.beams:
        target = args.out_dir / f"{args.run}_beam{beam}" / f"labels_threshold_{threshold}.npz"
        if target.exists():
            continue
        search = BeamSearch(tree, beam, device)
        labels, steps, out = base, [], {}
        for iteration in range(1, args.iterations + 1):
            labels, step = refine_step(fps, labels, threshold, device, search)
            steps.append(step)
            print(f"beam {beam} iteration {iteration}: {step}", flush=True)
            if iteration == 1:
                out["r1"] = compact(labels)
        out["r3"] = compact(labels)
        out["r3dedup"], dedup = dedup_step(fps, out["r3"], threshold, device)
        target.parent.mkdir(parents=True, exist_ok=True)
        (target.parent / f"steps_{threshold}.json").write_text(
            json.dumps({"beam": beam, "r": steps, "r3dedup": dedup}, indent=1)
        )
        partial = target.with_suffix(".partial.npz")
        np.savez_compressed(
            partial, **{f"{method}_r{v[1:]}beam{beam}_b{batch}_s{seed}": lab for v, lab in out.items()}
        )
        partial.rename(target)


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def pass_inputs(fps, labels, device):
    sizes, _, centroids, member_sim = cluster_stats(fps, labels, device)
    torch.cuda.empty_cache()
    return sizes, centroids, popcounts(centroids), member_sim


def cmd_recall(args, device):
    threshold, labels, fps = load_run(args)
    tree = get_tree(args.tree, fps, labels, threshold, device)
    sizes, centroids, pop, own_s = pass_inputs(fps, labels, device)
    cache = Path(args.cache)
    if cache.exists():
        data = np.load(cache)
        exh_s, exh_i, t_exh = data["s"], data["i"], float(data["seconds"])
    else:
        (exh_s, exh_i), t_exh = timed(lambda: best_other_centroid(fps, labels, centroids, pop, device))
        np.savez(cache, s=exh_s, i=exh_i, seconds=t_exh)
    movers = exh_s > own_s
    total_gain = (exh_s[movers] - own_s[movers]).sum()
    rows = []

    def record(name, beam, s, i, seconds):
        found = s > own_s
        rows.append(
            {
                "run": args.run,
                "search": name,
                "beam": beam,
                "seconds": seconds,
                "exhaustive_movers": float(movers.mean()),
                "recall_exact_best": float((i == exh_i)[movers].mean()),
                "found_move": float(found[movers].mean()),
                "gain_captured": float((s[movers & found] - own_s[movers & found]).sum() / total_gain),
                "movers_found_fraction": float(found.mean()),
            }
        )
        print(rows[-1], flush=True)

    record("exhaustive", 0, exh_s, exh_i, t_exh)
    best_other_centroid_ivf(fps[:20_000], labels[:20_000], centroids, pop, device)
    (s, i), t = timed(lambda: best_other_centroid_ivf(fps, labels, centroids, pop, device))
    record("ivf4096p16", 0, s, i, t)
    for beam in args.beams:
        search = BeamSearch(tree, beam, device)
        (s, i), t = timed(lambda: search(fps, labels, centroids, pop, sizes))
        record("beam", beam, s, i, t)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cmd_cost(args, device):
    rows = []
    for n in args.sizes:
        args.prefix = n
        threshold, labels, fps = load_run(args)
        tree_path = args.tree_dir / f"tree_{args.run}_{threshold}_{n}.npz"
        tree = get_tree(tree_path, fps, labels, threshold, device)
        built = np.load(tree_path)
        sizes, centroids, pop, _ = pass_inputs(fps, labels, device)
        for beam in args.beams:
            search = BeamSearch(tree, beam, device)
            search(fps[:20_000], labels[:20_000], centroids, pop, sizes)
            torch.cuda.synchronize()
            start = time.perf_counter()
            search(fps, labels, centroids, pop, sizes)
            torch.cuda.synchronize()
            t = time.perf_counter() - start
            rows.append(
                {
                    "molecules": n,
                    "clusters": len(sizes),
                    "depth": tree.depth,
                    "beam": beam,
                    "beam_seconds": t,
                    "tree_sums_seconds": float(built["seconds_sums"]),
                    "tree_build_cpu_seconds": float(built["seconds_build"]),
                }
            )
            print(rows[-1], flush=True)
        del centroids, pop
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["tree", "refine", "recall", "cost"])
    parser.add_argument("fingerprints")
    parser.add_argument("labels")
    parser.add_argument("run")
    parser.add_argument("--tree")
    parser.add_argument("--out", help="tree output (tree command)")
    parser.add_argument("--prefix", type=int)
    parser.add_argument("--beams", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache")
    parser.add_argument("--sizes", type=int, nargs="+", default=[250_000, 500_000, 1_000_000, 2_409_133])
    parser.add_argument("--tree-dir", type=Path)
    args = parser.parse_args()
    device = torch.device("cuda")
    if args.command == "tree":
        threshold, labels, fps = load_run(args)
        get_tree(args.out, fps, labels, threshold, device)
    elif args.command == "refine":
        cmd_refine(args, device)
    elif args.command == "recall":
        cmd_recall(args, device)
    else:
        cmd_cost(args, device)


if __name__ == "__main__":
    main()
