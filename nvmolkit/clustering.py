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

"""GPU-accelerated clustering from distance matrices, fingerprints, or ordered RDKit molecules."""

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Sequence, overload

import numpy as np
import torch

from nvmolkit import _clustering
from nvmolkit._fingerprint_inputs import _prepare_packed_fingerprints
from nvmolkit.similarity import (
    AAPSimilarity,
    FusedSimilarityMetric,
    PackedSimilarityMetric,
    _packed_metric_name,
    aap_similarity,  # noqa: F401 - compatibility re-export
)
from nvmolkit.types import ArrayInput, AsyncGpuResult, _as_cuda_tensor, _resolve_cuda_stream

_VALID_NEIGHBORLIST_SIZES = (8, 16, 24, 32, 64, 128)

_RDKitClusters = tuple[tuple[int, ...], ...]


class OutputMode(Enum):
    """Select a host-compatible or device-resident algorithm result.

    ``RDKIT`` materializes tuples of input indices on the host. ``DEVICE``
    returns result objects containing :class:`~nvmolkit.types.AsyncGpuResult`
    buffers on the active CUDA device.
    """

    RDKIT = "rdkit"
    DEVICE = "device"


@dataclass(frozen=True)
class ClusterDeviceResult:
    """GPU-resident clustering result shared by clustering algorithms.

    Attributes:
        cluster_ids: One zero-based int32 cluster ID per input molecule.
        centroids: Centroid indices by cluster ID, int32 and shape ``(num_clusters,)``.
        cluster_sizes: Member counts by cluster ID, int64 and shape ``(num_clusters,)``.
    """

    cluster_ids: AsyncGpuResult
    centroids: AsyncGpuResult
    cluster_sizes: AsyncGpuResult


@dataclass(frozen=True)
class SelectionDeviceResult:
    """GPU-resident ordered selection result.

    Attributes:
        indices: Ordered selected indices as an int32 GPU buffer.
        last_distance: Separation of the last MaxMin addition, or ``None`` for
            Leader. MaxMin uses ``-1`` when no candidate was added after the
            initial selection.
    """

    indices: AsyncGpuResult
    last_distance: float | None = None


# Compatibility names for the output types introduced alongside the original
# Butina and AAP APIs. They intentionally refer to the common types rather than
# defining algorithm-specific wrappers.
ButinaOutputMode = OutputMode
DISEOutputMode = OutputMode
ButinaDeviceResult = ClusterDeviceResult
DISEDeviceResult = ClusterDeviceResult


def _validate_output(output: OutputMode) -> None:
    if not isinstance(output, OutputMode):
        raise TypeError(f"output must be an OutputMode, got {type(output).__name__}")


def _validate_maxmin_threshold(threshold: float | None, *, maximum: float | None = None) -> float:
    if threshold is None:
        return -1.0
    if not np.isfinite(threshold) or threshold < 0 or (maximum is not None and threshold > maximum):
        distance_range = "non-negative" if maximum is None else f"in [0, {maximum:g}]"
        raise ValueError(f"threshold must be finite and {distance_range}, got {threshold}")
    return float(threshold)


def _cluster_arrays_to_rdkit(cluster_ids_array, centroids_array) -> _RDKitClusters:
    cluster_ids_array = np.asarray(cluster_ids_array)
    centroids_array = np.asarray(centroids_array)
    member_order = np.argsort(cluster_ids_array, kind="stable")
    sorted_cluster_ids = cluster_ids_array[member_order]
    cluster_offsets = np.searchsorted(sorted_cluster_ids, np.arange(centroids_array.size + 1))

    clusters = []
    for cluster_id, centroid_value in enumerate(centroids_array):
        centroid = int(centroid_value)
        members = member_order[cluster_offsets[cluster_id] : cluster_offsets[cluster_id + 1]]
        clusters.append(tuple([centroid] + [int(member) for member in members if member != centroid]))
    return tuple(clusters)


def _resolve_cluster_output(result, output: OutputMode) -> _RDKitClusters | ClusterDeviceResult:
    cluster_ids_obj, centroids_obj, cluster_sizes_obj = result
    if output is OutputMode.DEVICE:
        return ClusterDeviceResult(
            AsyncGpuResult(cluster_ids_obj),
            AsyncGpuResult(centroids_obj),
            AsyncGpuResult(cluster_sizes_obj),
        )
    return _cluster_arrays_to_rdkit(cluster_ids_obj, centroids_obj)


def _wrap_cluster_arrays(result) -> tuple[AsyncGpuResult, AsyncGpuResult]:
    cluster_ids_obj, centroids_obj = result
    return AsyncGpuResult(cluster_ids_obj), AsyncGpuResult(centroids_obj)


def _wrap_butina_device_result(result) -> ClusterDeviceResult:
    cluster_ids, centroids = _wrap_cluster_arrays(result)
    cluster_ids_int64 = cluster_ids.torch().to(torch.int64)
    cluster_sizes = torch.zeros_like(centroids.torch(), dtype=torch.int64)
    cluster_sizes.index_add_(0, cluster_ids_int64, torch.ones_like(cluster_ids_int64))
    return ClusterDeviceResult(cluster_ids, centroids, AsyncGpuResult(cluster_sizes))


def _to_rdkit_clusters(cluster_ids: AsyncGpuResult, centroids: AsyncGpuResult) -> _RDKitClusters:
    return _cluster_arrays_to_rdkit(cluster_ids.numpy(), centroids.numpy())


def _resolve_butina_output(result, output: OutputMode) -> _RDKitClusters | ClusterDeviceResult:
    if output is OutputMode.DEVICE:
        return _wrap_butina_device_result(result)
    return _to_rdkit_clusters(*_wrap_cluster_arrays(result))


def _resolve_selection_output(result, output: OutputMode, *, maxmin: bool):
    indices_obj, last_distance = result
    indices = AsyncGpuResult(indices_obj)
    if output is OutputMode.DEVICE:
        return SelectionDeviceResult(indices, float(last_distance) if maxmin else None)
    host_indices = tuple(int(index) for index in indices.numpy())
    if maxmin:
        return host_indices, float(last_distance)
    return host_indices


def _check_distance_matrix(name: str, x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError(f"{name} must be a square 2D matrix, got shape={tuple(x.shape)}")
    if x.dtype != torch.float64:
        raise ValueError(f"{name} must have dtype float64")
    return x.contiguous()


def _prepare_distance_matrix(distance_matrix: ArrayInput, stream: torch.cuda.Stream | None):
    active_stream = _resolve_cuda_stream(stream, distance_matrix)
    with torch.cuda.stream(active_stream):
        tensor = _as_cuda_tensor("distance_matrix", distance_matrix, stream=active_stream)
        tensor = _check_distance_matrix("distance_matrix", tensor)
    return tensor, active_stream


def leader(
    distance_matrix: ArrayInput,
    cutoff: float,
    pick_size: int = 0,
    first_picks: Sequence[int] = (),
    stream: torch.cuda.Stream | None = None,
    *,
    output: OutputMode = OutputMode.DEVICE,
) -> SelectionDeviceResult | tuple[int, ...]:
    """Select ordered sphere-exclusion leaders from a distance matrix.

    Matrix row ``i`` is interpreted as distances from selected leader ``i``
    to the remaining candidates, so directed matrices are supported. Items at
    distance ``<= cutoff`` are excluded. ``pick_size=0`` selects until no
    candidates remain.

    Args:
        distance_matrix: Full square float64 distance matrix. CPU inputs are
            copied to CUDA.
        cutoff: Inclusive exclusion distance. Must be finite and non-negative.
        pick_size: Maximum number of leaders, or zero for no explicit limit.
        first_picks: Unique in-range leader indices to process first, in the
            supplied order.
        stream: CUDA stream to use. If omitted, uses the current stream.
        output: Device-resident or RDKit-compatible host output.

    Returns:
        A :class:`SelectionDeviceResult` in device mode, or an ordered tuple of
        selected indices in RDKit mode.

    Note:
        The control loop synchronizes ``stream`` between selection passes.
    """
    _validate_output(output)
    matrix, active_stream = _prepare_distance_matrix(distance_matrix, stream)
    with torch.cuda.stream(active_stream):
        result = _clustering.leader(
            matrix.__cuda_array_interface__,
            cutoff,
            pick_size,
            tuple(first_picks),
            active_stream.cuda_stream,
        )
    return _resolve_selection_output(result, output, maxmin=False)


def fused_leader(
    x,
    cutoff: float,
    metric: FusedSimilarityMetric = "tanimoto",
    pick_size: int = 0,
    first_picks: Sequence[int] = (),
    stream: torch.cuda.Stream | None = None,
    *,
    output: OutputMode = OutputMode.DEVICE,
) -> SelectionDeviceResult | tuple[int, ...]:
    """Select ordered leaders while computing similarities on demand.

    Packed fingerprints support Tanimoto and cosine similarity. RDKit
    molecules use :class:`~nvmolkit.similarity.AAPSimilarity`, whose directed
    score is evaluated from each selected leader to each candidate.

    Args:
        x: Packed fingerprints for Tanimoto/cosine, or RDKit molecules for AAP.
        cutoff: Inclusive distance cutoff in ``[0, 1]``.
        metric: Similarity provider configuration or packed-provider string.
        pick_size: Maximum number of leaders, or zero for no explicit limit.
        first_picks: Unique in-range leader indices to process first, in the
            supplied order.
        stream: CUDA stream to use. If omitted, uses the current stream.
        output: Device-resident or RDKit-compatible host output.

    Returns:
        A :class:`SelectionDeviceResult` in device mode, or an ordered tuple of
        selected indices in RDKit mode.

    Note:
        This function stores ``O(N)`` algorithm state and does not materialize
        an ``N x N`` distance matrix. The control loop synchronizes ``stream``
        between selection passes.
    """
    _validate_output(output)
    if not 0 <= cutoff <= 1:
        raise ValueError(f"cutoff must be in [0, 1], got {cutoff}")

    if isinstance(metric, AAPSimilarity):
        active_stream = _resolve_cuda_stream(stream)
        result = _clustering.aap_leader(
            list(x),
            1.0 - cutoff,
            pick_size,
            tuple(first_picks),
            metric.max_path_length,
            metric.histogram_bins,
            metric.sinkhorn_iterations,
            metric.sinkhorn_temperature,
            active_stream.cuda_stream,
        )
        return _resolve_selection_output(result, output, maxmin=False)

    metric_name = _packed_metric_name(metric)
    (fingerprints,), active_stream = _prepare_packed_fingerprints(("x", x), stream=stream)
    with torch.cuda.stream(active_stream):
        result = _clustering.fused_leader(
            fingerprints.__cuda_array_interface__,
            cutoff,
            metric_name,
            pick_size,
            tuple(first_picks),
            active_stream.cuda_stream,
        )
    return _resolve_selection_output(result, output, maxmin=False)


def maxmin(
    distance_matrix: ArrayInput,
    pick_size: int,
    first_picks: Sequence[int] = (),
    seed: int = -1,
    threshold: float | None = None,
    stream: torch.cuda.Stream | None = None,
    *,
    output: OutputMode = OutputMode.DEVICE,
) -> SelectionDeviceResult | tuple[tuple[int, ...], float]:
    """Select a diverse subset with RDKit-compatible greedy MaxMin picking.

    The first pick uses RDKit's Boost MT19937 behavior when ``first_picks`` is
    empty. If ``threshold`` is provided, selection stops when the next item's
    distance to its nearest pick is at most that value.

    Args:
        distance_matrix: Full square float64 distance matrix. CPU inputs are
            copied to CUDA.
        pick_size: Target number of picks. Must be positive and no larger than
            the input size.
        first_picks: Unique in-range initial picks in the supplied order.
        seed: RDKit-compatible random seed used when ``first_picks`` is empty.
            A negative value seeds from system entropy.
        threshold: Optional finite, non-negative early-stop distance. The next
            candidate is not added when its nearest-pick distance is at most
            this value.
        stream: CUDA stream to use. If omitted, uses the current stream.
        output: Device-resident or RDKit-compatible host output.

    Returns:
        A :class:`SelectionDeviceResult` in device mode. RDKit mode returns
        ``(indices, last_distance)``.

    Note:
        MaxMin normally assumes symmetric distances. The control loop
        synchronizes ``stream`` between selection passes.
    """
    _validate_output(output)
    matrix, active_stream = _prepare_distance_matrix(distance_matrix, stream)
    native_threshold = _validate_maxmin_threshold(threshold)
    with torch.cuda.stream(active_stream):
        result = _clustering.maxmin(
            matrix.__cuda_array_interface__,
            pick_size,
            tuple(first_picks),
            seed,
            native_threshold,
            active_stream.cuda_stream,
        )
    return _resolve_selection_output(result, output, maxmin=True)


def fused_maxmin(
    x: ArrayInput,
    pick_size: int,
    metric: PackedSimilarityMetric = "tanimoto",
    first_picks: Sequence[int] = (),
    seed: int = -1,
    threshold: float | None = None,
    stream: torch.cuda.Stream | None = None,
    *,
    output: OutputMode = OutputMode.DEVICE,
) -> SelectionDeviceResult | tuple[tuple[int, ...], float]:
    """Run MaxMin directly on packed fingerprints without an ``N x N`` matrix.

    Args:
        x: Packed int32 or uint32 fingerprints with shape ``(N, num_words)``.
        pick_size: Target number of picks. Must be positive and no larger than
            the input size.
        metric: Tanimoto or cosine provider configuration/string. Directed AAP
            is not supported by MaxMin.
        first_picks: Unique in-range initial picks in the supplied order.
        seed: RDKit-compatible random seed used when ``first_picks`` is empty.
        threshold: Optional early-stop distance in ``[0, 1]``.
        stream: CUDA stream to use. If omitted, uses the current stream.
        output: Device-resident or RDKit-compatible host output.

    Returns:
        A :class:`SelectionDeviceResult` in device mode. RDKit mode returns
        ``(indices, last_distance)``.

    Note:
        This function stores ``O(N)`` algorithm state. The control loop
        synchronizes ``stream`` between selection passes.
    """
    _validate_output(output)
    metric_name = _packed_metric_name(metric)
    native_threshold = _validate_maxmin_threshold(threshold, maximum=1.0)
    (fingerprints,), active_stream = _prepare_packed_fingerprints(("x", x), stream=stream)
    with torch.cuda.stream(active_stream):
        result = _clustering.fused_maxmin(
            fingerprints.__cuda_array_interface__,
            pick_size,
            metric_name,
            tuple(first_picks),
            seed,
            native_threshold,
            active_stream.cuda_stream,
        )
    return _resolve_selection_output(result, output, maxmin=True)


def dise(
    distance_matrix: ArrayInput,
    cutoff: float,
    assignment: Literal["first", "nearest"] = "nearest",
    stream: torch.cuda.Stream | None = None,
    *,
    output: OutputMode = OutputMode.DEVICE,
) -> ClusterDeviceResult | _RDKitClusters:
    """Cluster an ordered distance matrix with directed sphere exclusion.

    Rows are interpreted as distances from a selected centroid to candidates,
    so the matrix may be directed. Centroids are selected in input order using
    inclusive Leader exclusion.

    Args:
        distance_matrix: Full square float64 distance matrix. CPU inputs are
            copied to CUDA.
        cutoff: Inclusive sphere-exclusion distance. Must be finite and
            non-negative.
        assignment: ``"first"`` keeps the first qualifying centroid;
            ``"nearest"`` assigns each non-centroid to its nearest centroid.
        stream: CUDA stream to use. If omitted, uses the current stream.
        output: Device-resident or RDKit-compatible host output.

    Returns:
        A :class:`ClusterDeviceResult` in device mode, or centroid-first
        cluster tuples ordered by descending size in RDKit mode.

    Note:
        The implementation synchronizes ``stream`` during centroid selection
        and while constructing the result.
    """
    _validate_output(output)
    if assignment not in ("first", "nearest"):
        raise ValueError(f"assignment must be one of ['first', 'nearest'], got {assignment!r}")
    matrix, active_stream = _prepare_distance_matrix(distance_matrix, stream)
    result = _clustering.dise(
        matrix.__cuda_array_interface__,
        cutoff,
        assignment == "nearest",
        output is OutputMode.DEVICE,
        active_stream.cuda_stream,
    )
    return _resolve_cluster_output(result, output)


def fused_dise(
    x,
    cutoff: float,
    metric: FusedSimilarityMetric = "tanimoto",
    assignment: Literal["first", "nearest"] = "nearest",
    stream: torch.cuda.Stream | None = None,
    *,
    output: OutputMode = OutputMode.DEVICE,
) -> ClusterDeviceResult | _RDKitClusters:
    """Cluster ordered inputs with on-demand directed sphere exclusion.

    Input order defines centroid priority. AAP is directed; packed Tanimoto and
    cosine providers are symmetric.

    Args:
        x: Packed fingerprints for Tanimoto/cosine, or RDKit molecules for AAP.
        cutoff: Inclusive distance cutoff in ``[0, 1]``.
        metric: Similarity provider configuration or packed-provider string.
        assignment: ``"first"`` keeps the first qualifying centroid;
            ``"nearest"`` assigns each non-centroid to its nearest centroid.
        stream: CUDA stream to use. If omitted, uses the current stream.
        output: Device-resident or RDKit-compatible host output.

    Returns:
        A :class:`ClusterDeviceResult` in device mode, or centroid-first
        cluster tuples ordered by descending size in RDKit mode.

    Note:
        This function avoids an ``N x N`` matrix and stores ``O(N)`` algorithm
        state. The implementation currently synchronizes ``stream`` during its
        host-controlled selection and result construction.
    """
    _validate_output(output)
    if not 0 <= cutoff <= 1:
        raise ValueError(f"cutoff must be in [0, 1], got {cutoff}")
    if assignment not in ("first", "nearest"):
        raise ValueError(f"assignment must be one of ['first', 'nearest'], got {assignment!r}")

    if isinstance(metric, AAPSimilarity):
        active_stream = _resolve_cuda_stream(stream)
        function = _clustering.aap_similarity_clustering if assignment == "first" else _clustering.aap_dise_clustering
        result = function(
            list(x),
            1.0 - cutoff,
            metric.max_path_length,
            metric.histogram_bins,
            metric.sinkhorn_iterations,
            metric.sinkhorn_temperature,
            output is OutputMode.DEVICE,
            active_stream.cuda_stream,
        )
        return _resolve_cluster_output(result, output)

    metric_name = _packed_metric_name(metric)
    (fingerprints,), active_stream = _prepare_packed_fingerprints(("x", x), stream=stream)
    result = _clustering.fused_dise(
        fingerprints.__cuda_array_interface__,
        cutoff,
        metric_name,
        assignment == "nearest",
        output is OutputMode.DEVICE,
        active_stream.cuda_stream,
    )
    return _resolve_cluster_output(result, output)


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
        return _resolve_butina_output(result, output)


@overload
def fused_butina(
    x: ArrayInput,
    cutoff: float,
    metric: PackedSimilarityMetric = "tanimoto",
    stream: torch.cuda.Stream | None = None,
    *,
    output: Literal[ButinaOutputMode.DEVICE] = ButinaOutputMode.DEVICE,
) -> ButinaDeviceResult: ...


@overload
def fused_butina(
    x: ArrayInput,
    cutoff: float,
    metric: PackedSimilarityMetric = "tanimoto",
    stream: torch.cuda.Stream | None = None,
    *,
    output: Literal[ButinaOutputMode.RDKIT],
) -> _RDKitClusters: ...


def fused_butina(
    x: ArrayInput,
    cutoff: float,
    metric: PackedSimilarityMetric = "tanimoto",
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
        metric: Tanimoto or cosine provider configuration/string. Directed AAP
                is intentionally unsupported by Butina.
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
    metric_name = _packed_metric_name(metric)

    if not 0 <= cutoff <= 1:
        raise ValueError(f"cutoff must be in [0, 1], got {cutoff}")

    (x,), active_stream = _prepare_packed_fingerprints(("x", x), stream=stream)
    with torch.cuda.stream(active_stream):
        result = _clustering.fused_butina(
            x.__cuda_array_interface__, cutoff, True, metric_name, active_stream.cuda_stream
        )
        return _resolve_butina_output(result, output)
