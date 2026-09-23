# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Contains GPU-accelerated Butina clustering implementations.

The standard ``butina()`` path accepts a full N x N distance matrix and
materializes its neighbor relationships. This is the right choice when you
already have a distance matrix or plan to reuse it (e.g. at multiple cutoffs),
and when N is small enough that the O(N^2) data fits comfortably in GPU memory.

``fused_butina()`` avoids materializing the distance matrix entirely.  Each
clustering round recomputes only the similarities it needs on the fly with
CUDA kernels that fuse popcount-based fingerprint similarity with the
neighbor-count and cluster-extraction steps. This trades extra compute for
drastically lower memory: usage is O(N) rather than O(N^2), making it the
better choice for large N where the full matrix would be prohibitively large.

``bitbirch()`` also uses linear storage, but incrementally builds ordered trees
from binary-fingerprint summaries. It can construct independent partial trees
concurrently and merge their Bit Features without an all-pairs matrix.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Literal, overload

import numpy as np
import torch

from nvmolkit import _clustering
from nvmolkit._fingerprint_inputs import _prepare_packed_fingerprints
from nvmolkit.types import ArrayInput, AsyncGpuResult, _as_cuda_tensor, _resolve_cuda_stream

_VALID_NEIGHBORLIST_SIZES = (8, 16, 24, 32, 64, 128)

_RDKitClusters = tuple[tuple[int, ...], ...]


class ButinaOutputMode(Enum):
    """Output format for :func:`butina` and :func:`fused_butina`.

    ``RDKIT`` returns the same tuple of clusters as RDKit's
    ``Butina.ClusterData``. ``DEVICE`` returns a :class:`ButinaDeviceResult`.
    """

    RDKIT = "rdkit"
    DEVICE = "device"


@dataclass(frozen=True)
class ButinaDeviceResult:
    """GPU-resident Butina clustering result.

    Attributes:
        cluster_ids: One int32 cluster ID per input item, shape ``(N,)``.
        centroids: Centroid indices by cluster ID, int32 and shape ``(num_clusters,)``.
        cluster_sizes: Member counts by cluster ID, int64 and shape ``(num_clusters,)``.
    """

    cluster_ids: AsyncGpuResult
    centroids: AsyncGpuResult
    cluster_sizes: AsyncGpuResult


def _wrap_cluster_arrays(result) -> tuple[AsyncGpuResult, AsyncGpuResult]:
    cluster_ids_obj, centroids_obj = result
    return AsyncGpuResult(cluster_ids_obj), AsyncGpuResult(centroids_obj)


def _wrap_device_result(result) -> ButinaDeviceResult:
    cluster_ids, centroids = _wrap_cluster_arrays(result)
    cluster_ids_int64 = cluster_ids.torch().to(torch.int64)
    cluster_sizes = torch.zeros_like(centroids.torch(), dtype=torch.int64)
    cluster_sizes.index_add_(0, cluster_ids_int64, torch.ones_like(cluster_ids_int64))
    return ButinaDeviceResult(cluster_ids, centroids, AsyncGpuResult(cluster_sizes))


def _wrap_bitbirch_result(result, return_centroids: bool, host_output: bool = False):
    if return_centroids:
        cluster_ids, centroids = result
        labels = np.asarray(cluster_ids) if host_output else AsyncGpuResult(cluster_ids)
        return labels, AsyncGpuResult(centroids)
    return np.asarray(result) if host_output else AsyncGpuResult(result)


def _to_rdkit_clusters(cluster_ids: AsyncGpuResult, centroids: AsyncGpuResult) -> _RDKitClusters:
    cluster_ids_array = cluster_ids.numpy()
    centroids_array = centroids.numpy()
    member_order = np.argsort(cluster_ids_array, kind="stable")
    sorted_cluster_ids = cluster_ids_array[member_order]
    cluster_offsets = np.searchsorted(sorted_cluster_ids, np.arange(centroids_array.size + 1))

    clusters = []
    for cluster_id, centroid_value in enumerate(centroids_array):
        centroid = int(centroid_value)
        members = member_order[cluster_offsets[cluster_id] : cluster_offsets[cluster_id + 1]]
        clusters.append(tuple([centroid] + [int(member) for member in members if member != centroid]))
    return tuple(clusters)


def _resolve_output(result, output: ButinaOutputMode) -> _RDKitClusters | ButinaDeviceResult:
    if output is ButinaOutputMode.DEVICE:
        return _wrap_device_result(result)
    return _to_rdkit_clusters(*_wrap_cluster_arrays(result))


def _validate_output(output: ButinaOutputMode) -> None:
    if not isinstance(output, ButinaOutputMode):
        raise TypeError(f"output must be a ButinaOutputMode, got {type(output).__name__}")


def _check_distance_matrix(name: str, x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError(f"{name} must be a square 2D matrix, got shape={tuple(x.shape)}")
    if x.dtype != torch.float64:
        raise ValueError(f"{name} must have dtype float64")
    return x.contiguous()


@overload
def butina(
    distance_matrix: ArrayInput,
    cutoff: float,
    neighborlist_max_size: int = 64,
    reordering: bool = True,
    stream: torch.cuda.Stream | None = None,
    *,
    output: Literal[ButinaOutputMode.DEVICE] = ButinaOutputMode.DEVICE,
) -> ButinaDeviceResult: ...


@overload
def butina(
    distance_matrix: ArrayInput,
    cutoff: float,
    neighborlist_max_size: int = 64,
    reordering: bool = True,
    stream: torch.cuda.Stream | None = None,
    *,
    output: Literal[ButinaOutputMode.RDKIT],
) -> _RDKitClusters: ...


def butina(
    distance_matrix: ArrayInput,
    cutoff: float,
    neighborlist_max_size: int = 64,
    reordering: bool = True,
    stream: torch.cuda.Stream | None = None,
    *,
    output: ButinaOutputMode = ButinaOutputMode.DEVICE,
) -> _RDKitClusters | ButinaDeviceResult:
    """Perform Butina clustering on a distance matrix.

    The Butina algorithm is a deterministic clustering method that groups items based
    on distance thresholds. It iteratively:
    1. Finds the item with the most neighbors within the cutoff distance
    2. Forms a cluster with that item and all its neighbors
    3. Removes clustered items from consideration
    4. Repeats until all items are clustered

    Args:
        distance_matrix: Square distance matrix of shape (N, N) where N is the number
                        of items. Can be an AsyncGpuResult, torch.Tensor, or numpy.ndarray.
                        CPU tensors and NumPy arrays are copied to CUDA. Inputs
                        must have dtype float64.
        cutoff: Distance threshold for clustering. Items are neighbors if their
                distance is less than or equal to this cutoff.
        neighborlist_max_size: Maximum size of the neighborlist used for small cluster
                              optimization. Must be 8, 16, 24, 32, 64, or 128. Larger values
                              allow parallel processing of larger clusters but use more
                              shared memory. Ignored when reordering is False.
        reordering: Whether to update neighbor counts among unassigned items
                    after each cluster is formed. Defaults to True, while
                    RDKit's ``Butina.ClusterData`` defaults to False.
        stream: CUDA stream to use. If None, uses the current stream.
        output: Output representation. Defaults to ``ButinaOutputMode.DEVICE``.

    Returns:
        The representation selected by ``output``.

        ``ButinaOutputMode.RDKIT`` returns a tuple containing one tuple per
        cluster. Each cluster tuple contains input indices, with the centroid
        first. Constructing this representation synchronizes the CUDA work and
        copies the clustering result to the host.

        ``ButinaOutputMode.DEVICE`` returns a :class:`ButinaDeviceResult`
        containing three :class:`AsyncGpuResult` objects on the active CUDA
        device. ``cluster_ids`` is int32 with shape ``(N,)`` and maps each input
        index to a cluster ID. Cluster IDs are contiguous from zero through
        ``num_clusters - 1``. ``centroids`` is int32 with shape
        ``(num_clusters,)``; element ``k`` is an input index whose cluster ID is
        ``k``. ``cluster_sizes`` is int64 with shape ``(num_clusters,)``;
        element ``k`` equals the number of entries in ``cluster_ids`` that are
        equal to ``k``, and the sizes sum to ``N``. The return is
        asynchronous: each field's ``.torch()`` method exposes its CUDA tensor
        without a host copy, while ``.numpy()`` synchronizes and copies that
        field to the host.

    Note:
        The distance matrix should be symmetric and have zeros on the diagonal.
    """
    _validate_output(output)
    if neighborlist_max_size not in _VALID_NEIGHBORLIST_SIZES:
        raise ValueError(
            f"neighborlist_max_size must be one of {_VALID_NEIGHBORLIST_SIZES}, got {neighborlist_max_size}"
        )
    active_stream = _resolve_cuda_stream(stream, distance_matrix)
    with torch.cuda.stream(active_stream):
        distance_matrix_tensor = _as_cuda_tensor("distance_matrix", distance_matrix, stream=active_stream)
        distance_matrix_tensor = _check_distance_matrix("distance_matrix", distance_matrix_tensor)
        result = _clustering.butina(
            distance_matrix_tensor.__cuda_array_interface__,
            cutoff,
            neighborlist_max_size,
            True,
            reordering,
            active_stream.cuda_stream,
        )
        return _resolve_output(result, output)


@overload
def fused_butina(
    x: ArrayInput,
    cutoff: float,
    metric: str = "tanimoto",
    stream: torch.cuda.Stream | None = None,
    *,
    output: Literal[ButinaOutputMode.DEVICE] = ButinaOutputMode.DEVICE,
) -> ButinaDeviceResult: ...


@overload
def fused_butina(
    x: ArrayInput,
    cutoff: float,
    metric: str = "tanimoto",
    stream: torch.cuda.Stream | None = None,
    *,
    output: Literal[ButinaOutputMode.RDKIT],
) -> _RDKitClusters: ...


def fused_butina(
    x: ArrayInput,
    cutoff: float,
    metric: str = "tanimoto",
    stream: torch.cuda.Stream | None = None,
    *,
    output: ButinaOutputMode = ButinaOutputMode.DEVICE,
) -> _RDKitClusters | ButinaDeviceResult:
    """Perform fused Butina clustering on a set of fingerprints.

    This function uses a fused implementation of Butina clustering that computes
    similarities and neighbors on-the-fly, avoiding the need to compute and store
    the full distance matrix. This makes it suitable for large datasets.

    Args:
        x: Tensor-like object of shape (N, D) containing packed int32 or uint32 fingerprints
           to cluster. Can be an AsyncGpuResult, torch.Tensor, or numpy.ndarray.
           CPU tensors and NumPy arrays are copied to CUDA.
        cutoff: Distance threshold for clustering. Items are neighbors if their
                distance is at most this cutoff (i.e. similarity >= 1 - cutoff).
        metric: Metric to use for similarity computation. Currently only "tanimoto"
                and "cosine" are supported.
        stream: CUDA stream to use. If None, uses the current stream.
        output: Output representation. Defaults to ``ButinaOutputMode.DEVICE``.

    Returns:
        The representation selected by ``output``.

        ``ButinaOutputMode.RDKIT`` returns a tuple containing one tuple per
        cluster. Each cluster tuple contains input indices, with the centroid
        first. Constructing this representation synchronizes the CUDA work and
        copies the clustering result to the host.

        ``ButinaOutputMode.DEVICE`` returns a :class:`ButinaDeviceResult`
        containing three :class:`AsyncGpuResult` objects on the active CUDA
        device. ``cluster_ids`` is int32 with shape ``(N,)`` and maps each input
        index to a cluster ID. Cluster IDs are contiguous from zero through
        ``num_clusters - 1``. ``centroids`` is int32 with shape
        ``(num_clusters,)``; element ``k`` is an input index whose cluster ID is
        ``k``. ``cluster_sizes`` is int64 with shape ``(num_clusters,)``;
        element ``k`` equals the number of entries in ``cluster_ids`` that are
        equal to ``k``, and the sizes sum to ``N``. The return is
        asynchronous: each field's ``.torch()`` method exposes its CUDA tensor
        without a host copy, while ``.numpy()`` synchronizes and copies that
        field to the host.

    """
    _validate_output(output)
    if metric not in ("tanimoto", "cosine"):
        raise ValueError(f"metric must be one of ['tanimoto', 'cosine'], got {metric}")

    if not 0 <= cutoff <= 1:
        raise ValueError(f"cutoff must be in [0, 1], got {cutoff}")

    (x,), active_stream = _prepare_packed_fingerprints(("x", x), stream=stream)
    with torch.cuda.stream(active_stream):
        result = _clustering.fused_butina(x.__cuda_array_interface__, cutoff, True, metric, active_stream.cuda_stream)
        return _resolve_output(result, output)


def _automatic_bitbirch_partitions(num_fingerprints: int) -> int:
    if num_fingerprints < 512:
        return 1
    return (num_fingerprints + 254) // 255


def bitbirch_shared(
    x: ArrayInput,
    threshold: float,
    *,
    branching_factor: int = 254,
    insertion_batch_size: int = 1024,
    insertion_policy: str = "ordered-leaf",
    ordered_prefix_size: int = 0,
    summary_cache_bytes: int = 0,
    fingerprint_cache_bytes: int = 0,
    host_input: bool = False,
    host_output: bool = False,
    return_centroids: bool = False,
    stream: torch.cuda.Stream | None = None,
) -> AsyncGpuResult | np.ndarray | tuple[AsyncGpuResult | np.ndarray, AsyncGpuResult]:
    """Experimental single-tree insertion, with frozen parent routing per batch.

    With ``ordered-leaf``, each leaf has one ordered writer. ``filtered-group``
    first tests snapshot proposals individually and commits jointly admissible
    groups to disjoint entries, then handles residuals with ordered leaf writers.
    With ``filtered-group``, ``ordered_prefix_size`` optionally builds the first
    input rows using ordered leaf writers before enabling grouped insertion.
    Splits occur at barriers and unprocessed inputs are rerouted. This is not the
    independent-trees-plus-merge algorithm. Batch size one with single-path
    routing preserves native serial-tree semantics. Larger batches can change
    clustering; grouped insertion can do so even before the first split.
    Only the diameter criterion is supported. A positive ``summary_cache_bytes``
    puts materialized BF sums in pinned CPU memory with a capped GPU page cache;
    cold pages use mapped host memory and cache rotation occurs between batches.
    Zero keeps BF sums entirely on the GPU. ``host_input=True`` accepts a packed
    NumPy matrix (including a memory map), copies only the current insertion
    batch to the GPU, and retains packed singleton fingerprints with the tree.
    ``host_output=True`` keeps labels in mapped pinned CPU memory during
    insertion and returns them as a NumPy array, removing the remaining N-wide
    GPU allocation. A positive ``fingerprint_cache_bytes`` similarly puts
    retained singleton fingerprints in pinned CPU memory with a capped GPU page
    cache; it requires ``host_input=True``. Centroids, topology, and scratch
    remain GPU resident. This is not yet a fully bounded-memory interface or a
    persistent append API.
    """
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and in [0, 1]")
    if branching_factor < 3 or insertion_batch_size < 1:
        raise ValueError("branching_factor must be at least 3 and insertion_batch_size positive")
    if insertion_policy not in ("ordered-leaf", "filtered-group"):
        raise ValueError("insertion_policy must be ordered-leaf or filtered-group")
    if ordered_prefix_size < 0:
        raise ValueError("ordered_prefix_size must be nonnegative")
    if summary_cache_bytes < 0:
        raise ValueError("summary_cache_bytes must be nonnegative")
    if fingerprint_cache_bytes < 0:
        raise ValueError("fingerprint_cache_bytes must be nonnegative")
    if fingerprint_cache_bytes and not host_input:
        raise ValueError("fingerprint_cache_bytes requires host_input=True")
    if host_input:
        if not isinstance(x, np.ndarray) or x.ndim != 2 or x.dtype not in (np.int32, np.uint32):
            raise ValueError("host_input requires a packed 2D NumPy int32 or uint32 array")
        if x.shape[1] == 0:
            raise ValueError("x must contain at least one fingerprint word")
        x = np.ascontiguousarray(x)
        interface = x.__array_interface__
        active_stream = _resolve_cuda_stream(stream)
    else:
        (x,), active_stream = _prepare_packed_fingerprints(("x", x), stream=stream)
        interface = x.__cuda_array_interface__
    with torch.cuda.stream(active_stream):
        result = _clustering.bitbirch_shared(
            interface,
            threshold,
            branching_factor,
            insertion_batch_size,
            insertion_policy,
            ordered_prefix_size,
            summary_cache_bytes,
            fingerprint_cache_bytes,
            host_input,
            host_output,
            return_centroids,
            active_stream.cuda_stream,
        )
        return _wrap_bitbirch_result(result, return_centroids, host_output)


def bitbirch(
    x: ArrayInput,
    threshold: float,
    *,
    branching_factor: int = 254,
    merge_criterion: str = "diameter",
    tolerance: float = 0.05,
    num_partitions: int | None = None,
    return_centroids: bool = False,
    stream: torch.cuda.Stream | None = None,
) -> AsyncGpuResult | tuple[AsyncGpuResult, AsyncGpuResult]:
    """Cluster packed binary fingerprints with an ordered BitBIRCH tree.

    The operation uses linear device storage and avoids pairwise similarity
    matrices. With multiple partitions, independent ordered trees are built
    concurrently and their leaf Bit Features are inserted into a final tree.

    Args:
        x: Shape ``(N, W)`` packed int32 or uint32 fingerprints. Host inputs
           are copied to CUDA through the standard nvMolKit input path.
        threshold: Minimum combined-cluster iSIM Jaccard--Tanimoto similarity.
        branching_factor: Maximum entries per tree node. Must be at least 3.
        merge_criterion: ``"diameter"`` or ``"tolerance-diameter"``.
        tolerance: Maximum permitted degradation for tolerance-diameter merge.
        num_partitions: Number of contiguous partial trees. ``None`` selects
                        one tree below 512 inputs and roughly one tree per 256
                        inputs otherwise, keeping partial-tree counts in an
                        8-bit representation. Tolerance-diameter mode selects
                        one tree. Set to 1 for exact deterministic serial-tree
                        semantics.
        return_centroids: Return packed majority centroids with shape
                          ``(num_clusters, W)`` in addition to labels.
        stream: CUDA stream to use. If None, uses the current stream.

    Returns:
        One cluster ID per input fingerprint. Cluster IDs are ordered by the
        earliest input member. If requested, also returns packed uint32
        majority centroids in cluster-ID order.

    Notes:
        Fingerprints must be word-aligned; a row represents exactly
        ``32 * W`` logical bits. Results are deterministic for fixed input,
        options, and GPU architecture, but changing ``num_partitions`` can
        change the partition. The call currently waits for tree-status and
        cluster-count metadata on the host before returning; labels and
        centroids remain device-resident.
    """
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    if branching_factor < 3:
        raise ValueError(f"branching_factor must be at least 3, got {branching_factor}")
    if merge_criterion not in ("diameter", "tolerance-diameter"):
        raise ValueError(f"merge_criterion must be one of ['diameter', 'tolerance-diameter'], got {merge_criterion}")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError(f"tolerance must be nonnegative, got {tolerance}")
    if num_partitions is not None and num_partitions < 1:
        raise ValueError(f"num_partitions must be positive, got {num_partitions}")

    (x,), active_stream = _prepare_packed_fingerprints(("x", x), stream=stream)
    num_fingerprints = x.shape[0]
    if num_partitions is None:
        if merge_criterion == "tolerance-diameter":
            num_partitions = 1
        else:
            num_partitions = _automatic_bitbirch_partitions(num_fingerprints)
    if num_fingerprints > 0 and num_partitions > num_fingerprints:
        raise ValueError(
            f"num_partitions must not exceed the number of fingerprints ({num_fingerprints}), got {num_partitions}"
        )
    if num_fingerprints == 0:
        num_partitions = 1
    if merge_criterion == "tolerance-diameter" and num_partitions > 1:
        raise ValueError("tolerance-diameter merging currently requires num_partitions=1")
    with torch.cuda.stream(active_stream):
        result = _clustering.bitbirch(
            x.__cuda_array_interface__,
            threshold,
            branching_factor,
            merge_criterion,
            tolerance,
            num_partitions,
            return_centroids,
            active_stream.cuda_stream,
        )
        return _wrap_bitbirch_result(result, return_centroids)
