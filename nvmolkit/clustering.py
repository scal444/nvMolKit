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

"""GPU-accelerated clustering.

Each algorithm has a matrix form that takes a precomputed distance matrix and a
``fused_`` form that computes distances from fingerprints or molecules as needed.
"""

import importlib.util
from dataclasses import dataclass
from enum import Enum
from typing import Literal, overload

import numpy as np
import torch

from nvmolkit import _clustering
from nvmolkit._distance_inputs import _aap_args, _packed_metric_name, _prepare_distance_matrix, _prepare_fused_input
from nvmolkit._fingerprint_inputs import _prepare_packed_fingerprints
from nvmolkit.fingerprints import unpack_fingerprint
from nvmolkit.similarity import AAPMetric, Metric, _resolve_metric
from nvmolkit.types import ArrayInput, AsyncGpuResult, _as_cuda_tensor, _resolve_cuda_stream

_VALID_NEIGHBORLIST_SIZES = (8, 16, 24, 32, 64, 128)

_RDKitClusters = tuple[tuple[int, ...], ...]

CUML_INSTALL_HINT = (
    "nvmolkit.clustering.kmeans requires cuML, which is an optional dependency. "
    "Install it with 'pip install cuml-cu12' (or 'pip install nvMolKit[kmeans]' if you installed nvMolKit via pip). "
    "conda users: 'conda install -c rapidsai -c conda-forge cuml'."
)


class OutputMode(Enum):
    """Result representation for clustering and selection functions.

    ``DEVICE`` returns :class:`~nvmolkit.types.AsyncGpuResult` buffers on the
    GPU. ``RDKIT`` returns host tuples of input indices in RDKit's format.
    """

    RDKIT = "rdkit"
    DEVICE = "device"


@dataclass(frozen=True)
class ClusterDeviceResult:
    """Device-resident clustering result.

    Attributes:
        cluster_ids: int32 cluster ID of each input, shape ``(N,)``. IDs are
            contiguous from zero.
        centroids: int32 input index of each cluster's centroid, shape
            ``(num_clusters,)``.
        cluster_sizes: int64 member count of each cluster, shape
            ``(num_clusters,)``.
    """

    cluster_ids: AsyncGpuResult
    centroids: AsyncGpuResult
    cluster_sizes: AsyncGpuResult


def _validate_output(output: OutputMode) -> None:
    if not isinstance(output, OutputMode):
        raise TypeError(f"output must be an OutputMode, got {type(output).__name__}")


def _validate_assignment(assignment: str) -> None:
    if assignment not in ("first", "nearest"):
        raise ValueError(f"assignment must be one of ['first', 'nearest'], got {assignment!r}")


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


def _resolve_butina_output(result, output: OutputMode) -> _RDKitClusters | ClusterDeviceResult:
    if output is OutputMode.DEVICE:
        return _wrap_butina_device_result(result)
    cluster_ids, centroids = _wrap_cluster_arrays(result)
    return _cluster_arrays_to_rdkit(cluster_ids.numpy(), centroids.numpy())


def _check_distance_matrix(name: str, x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError(f"{name} must be a square 2D matrix, got shape={tuple(x.shape)}")
    if x.dtype != torch.float64:
        raise ValueError(f"{name} must have dtype float64")
    return x.contiguous()


def dise(
    distance_matrix: ArrayInput,
    cutoff: float,
    *,
    assignment: Literal["first", "nearest"] = "nearest",
    stream: torch.cuda.Stream | None = None,
    output: OutputMode = OutputMode.DEVICE,
) -> ClusterDeviceResult | _RDKitClusters:
    """Cluster a distance matrix by directed sphere exclusion (DISE).

    Centroids are the leaders selected by :func:`~nvmolkit.pickers.leader`. With
    ``assignment="first"``, each item joins the first centroid that excluded
    it; with ``"nearest"``, each non-centroid joins its nearest centroid, with
    ties going to the earlier centroid. Clusters are ordered by descending
    size, then by centroid selection order.

    Distances are compared in float32, so a distance within float32 rounding
    of the cutoff can be classified differently than in double precision, and
    distances within float32 rounding of each other tie in ``"nearest"``
    assignment.

    Args:
        distance_matrix: Square float32 or float64 matrix of shape ``(N, N)``.
            Element ``[i, j]`` is the distance from item ``i`` to item ``j``.
            Values are converted to float32 for comparisons. Values that
            overflow float32 during conversion become infinity.
        cutoff: Inclusive exclusion distance, rounded to float32. Must be
            between zero and the largest finite float32 value.
        assignment: ``"first"`` or ``"nearest"``.
        stream: CUDA stream to use. If None, uses the current stream.
        output: Result representation.

    Returns:
        A :class:`ClusterDeviceResult` for ``OutputMode.DEVICE``, or
        centroid-first tuples of input indices for ``OutputMode.RDKIT``.
    """
    _validate_output(output)
    _validate_assignment(assignment)
    matrix, active_stream = _prepare_distance_matrix(distance_matrix, stream)
    with torch.cuda.stream(active_stream):
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
    *,
    metric: Metric = "tanimoto",
    assignment: Literal["first", "nearest"] = "nearest",
    stream: torch.cuda.Stream | None = None,
    output: OutputMode = OutputMode.DEVICE,
) -> ClusterDeviceResult | _RDKitClusters:
    """Cluster by directed sphere exclusion (DISE), computing distances as needed.

    Equivalent to :func:`dise` on the matrix of ``1 - similarity`` values,
    with memory that scales as ``O(N)``. Distances are computed in float32, so
    results can differ slightly from RDKit's double-precision results.

    Args:
        x: Packed int32 or uint32 fingerprints of shape ``(N, num_words)`` for
            Tanimoto and cosine, or a sequence of RDKit molecules for AAP.
        cutoff: Inclusive exclusion distance in ``[0, 1]``, rounded to float32.
        metric: Similarity metric.
        assignment: ``"first"`` or ``"nearest"``.
        stream: CUDA stream to use. If None, uses the current stream.
        output: Result representation.

    Returns:
        A :class:`ClusterDeviceResult` for ``OutputMode.DEVICE``, or
        centroid-first tuples of input indices for ``OutputMode.RDKIT``.

    Note:
        For the method, see `Gobbi et al. (2015)
        <https://doi.org/10.1186/s13321-015-0056-8>`_.
    """
    _validate_output(output)
    _validate_assignment(assignment)
    resolved, inputs, active_stream = _prepare_fused_input(x, metric, stream)
    with torch.cuda.stream(active_stream):
        if isinstance(resolved, AAPMetric):
            result = _clustering.aap_dise(
                inputs,
                cutoff,
                assignment == "nearest",
                *_aap_args(resolved),
                output is OutputMode.DEVICE,
                active_stream.cuda_stream,
            )
        else:
            result = _clustering.fused_dise(
                inputs.__cuda_array_interface__,
                cutoff,
                _packed_metric_name(resolved),
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
    output: Literal[OutputMode.DEVICE] = OutputMode.DEVICE,
) -> ClusterDeviceResult: ...


@overload
def butina(
    distance_matrix: ArrayInput,
    cutoff: float,
    neighborlist_max_size: int = 64,
    reordering: bool = True,
    stream: torch.cuda.Stream | None = None,
    *,
    output: Literal[OutputMode.RDKIT],
) -> _RDKitClusters: ...


def butina(
    distance_matrix: ArrayInput,
    cutoff: float,
    neighborlist_max_size: int = 64,
    reordering: bool = True,
    stream: torch.cuda.Stream | None = None,
    *,
    output: OutputMode = OutputMode.DEVICE,
) -> _RDKitClusters | ClusterDeviceResult:
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
        output: Result representation.

    Returns:
        A :class:`ClusterDeviceResult` for ``OutputMode.DEVICE``, or the
        centroid-first cluster tuples returned by RDKit's
        ``Butina.ClusterData`` for ``OutputMode.RDKIT``.

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
    metric: Metric = "tanimoto",
    stream: torch.cuda.Stream | None = None,
    *,
    output: Literal[OutputMode.DEVICE] = OutputMode.DEVICE,
) -> ClusterDeviceResult: ...


@overload
def fused_butina(
    x: ArrayInput,
    cutoff: float,
    metric: Metric = "tanimoto",
    stream: torch.cuda.Stream | None = None,
    *,
    output: Literal[OutputMode.RDKIT],
) -> _RDKitClusters: ...


def fused_butina(
    x: ArrayInput,
    cutoff: float,
    metric: Metric = "tanimoto",
    stream: torch.cuda.Stream | None = None,
    *,
    output: OutputMode = OutputMode.DEVICE,
) -> _RDKitClusters | ClusterDeviceResult:
    """Perform Butina clustering on fingerprints, computing distances as needed.

    Equivalent to :func:`butina` on the matrix of ``1 - similarity`` values,
    without forming the ``N x N`` matrix.

    Args:
        x: Packed int32 or uint32 fingerprints of shape ``(N, num_words)``. Can
           be an AsyncGpuResult, torch.Tensor, or numpy.ndarray. CPU tensors
           and NumPy arrays are copied to CUDA.
        cutoff: Inclusive neighbor distance in ``[0, 1]``.
        metric: Similarity metric. :class:`~nvmolkit.similarity.AAPMetric` is
            not yet supported.
        stream: CUDA stream to use. If None, uses the current stream.
        output: Result representation.

    Returns:
        A :class:`ClusterDeviceResult` for ``OutputMode.DEVICE``, or the
        centroid-first cluster tuples returned by RDKit's
        ``Butina.ClusterData`` for ``OutputMode.RDKIT``.
    """
    _validate_output(output)
    resolved = _resolve_metric(metric)
    if isinstance(resolved, AAPMetric):
        raise NotImplementedError("fused_butina does not yet support AAPMetric")

    if not 0 <= cutoff <= 1:
        raise ValueError(f"cutoff must be in [0, 1], got {cutoff}")

    (x,), active_stream = _prepare_packed_fingerprints(("x", x), stream=stream)
    with torch.cuda.stream(active_stream):
        result = _clustering.fused_butina(
            x.__cuda_array_interface__, cutoff, True, _packed_metric_name(resolved), active_stream.cuda_stream
        )
        return _resolve_butina_output(result, output)


@dataclass(frozen=True)
class KMeansResult:
    """Device-resident k-means result.

    Attributes:
        labels: int32 cluster ID of each input, shape ``(N,)``.
        centers: float32 cluster centers in unpacked bit space, shape
            ``(n_clusters, num_words * 32)``. Column ``j`` is bit ``j % 32`` of
            word ``j // 32``, as in :func:`~nvmolkit.fingerprints.unpack_fingerprint`.
            For ``metric="cosine"``, centers are means of L2-normalized rows.
        cluster_sizes: int64 member count of each cluster, shape
            ``(n_clusters,)``. Sizes can be zero, because cuML keeps the
            previous center of a cluster that loses all its members.
        inertia: Sum of squared Euclidean distances from the fitted rows to
            their nearest center, in the space selected by ``metric``. Covers
            only the sampled rows when ``fit_size`` is smaller than ``N``.
        n_iter: Number of Lloyd iterations run by cuML.
    """

    labels: AsyncGpuResult
    centers: AsyncGpuResult
    cluster_sizes: AsyncGpuResult
    inertia: float
    n_iter: int


def is_cuml_available() -> bool:
    """Return ``True`` if cuML can be imported, without importing it."""
    return importlib.util.find_spec("cuml") is not None


def _require_cuml_kmeans():
    if not is_cuml_available():
        raise ImportError(CUML_INSTALL_HINT)
    from cuml.cluster import KMeans

    return KMeans


def _unpack_features(x: torch.Tensor, rows: torch.Tensor | None, metric: str, batch_size: int) -> torch.Tensor:
    """Unpack packed fingerprints to float32 rows, ``batch_size`` rows at a time to bound temporaries."""
    num_rows = x.shape[0] if rows is None else rows.numel()
    features = torch.empty((num_rows, x.shape[1] * 32), dtype=torch.float32, device=x.device)
    for start in range(0, num_rows, batch_size):
        stop = min(start + batch_size, num_rows)
        # CUDA indexing does not support uint32, so gather through an int32 view.
        packed = x[start:stop] if rows is None else x.view(torch.int32)[rows[start:stop]]
        chunk = features[start:stop]
        chunk.copy_(unpack_fingerprint(packed))
        if metric == "cosine":
            chunk.div_(chunk.sum(dim=1, keepdim=True).sqrt_().clamp_min_(1.0))
    return features


def kmeans(
    x: ArrayInput,
    n_clusters: int,
    *,
    metric: Literal["euclidean", "cosine"] = "euclidean",
    init: Literal["k-means||", "k-means++", "random"] | ArrayInput = "k-means||",
    n_init: int = 1,
    max_iter: int = 300,
    tol: float = 1e-4,
    fit_size: int | None = None,
    batch_size: int = 65536,
    random_state: int | None = None,
    stream: torch.cuda.Stream | None = None,
) -> KMeansResult:
    """Cluster packed fingerprints with cuML's k-means.

    Fingerprints are unpacked to float32 bit vectors and clustered by
    ``cuml.cluster.KMeans`` (Lloyd's algorithm on squared Euclidean distance,
    with mean centers). With ``metric="cosine"``, rows are L2-normalized
    first, so the Euclidean distance between rows is a monotone function of
    their cosine similarity. cuML does not support Tanimoto distance.

    Unpacking costs 4 bytes per fingerprint bit, for example 8 KiB per
    2048-bit fingerprint. To bound memory on large inputs, set ``fit_size`` to
    fit the centers on a random subsample; all rows are then labeled against
    those centers ``batch_size`` rows at a time.

    Requires the optional ``cuml`` package. cuML runs on its own per-thread
    CUDA stream; this function synchronizes ``stream`` before handing data to
    cuML and returns once cuML has finished.

    Args:
        x: Packed int32 or uint32 fingerprints of shape ``(N, num_words)``. Can
            be an AsyncGpuResult, torch.Tensor, or numpy.ndarray. CPU tensors
            and NumPy arrays are copied to CUDA.
        n_clusters: Number of clusters, between 1 and the number of fitted rows.
        metric: ``"euclidean"`` on raw bits, or ``"cosine"`` on L2-normalized bits.
        init: ``"k-means||"`` (cuML's scalable k-means++), ``"k-means++"``,
            ``"random"``, or initial centers of shape ``(n_clusters, num_words * 32)``
            in unpacked bit space. With ``metric="cosine"``, initial centers
            are L2-normalized.
        n_init: Number of initializations; the run with the lowest inertia is
            kept. Ignored when ``init`` is an array of centers.
        max_iter: Maximum Lloyd iterations per initialization.
        tol: cuML convergence tolerance: stop when the relative inertia
            improvement or the summed squared center shift falls below it.
        fit_size: Number of randomly sampled rows to fit the centers on, or
            ``None`` to fit on all rows.
        batch_size: Rows processed at a time: unpacked when building fit
            features and labeling rows outside the fit, and scored against all
            centers by cuML, whose distance workspace takes
            ``4 * batch_size * n_clusters`` bytes.
        random_state: Seed for the fit subsample and cuML's initialization.
        stream: CUDA stream to use. If None, uses the current stream.

    Returns:
        A :class:`KMeansResult`.
    """
    if metric not in ("euclidean", "cosine"):
        raise ValueError(f"metric must be one of ['euclidean', 'cosine'], got {metric!r}")
    if n_clusters < 1:
        raise ValueError(f"n_clusters must be positive, got {n_clusters}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if fit_size is not None and fit_size < n_clusters:
        raise ValueError(f"fit_size must be at least n_clusters ({n_clusters}), got {fit_size}")
    KMeans = _require_cuml_kmeans()

    (x,), active_stream = _prepare_packed_fingerprints(("x", x), stream=stream)
    num_rows = x.shape[0]
    num_bits = x.shape[1] * 32
    if n_clusters > num_rows:
        raise ValueError(f"n_clusters ({n_clusters}) must not exceed the number of fingerprints ({num_rows})")

    with torch.cuda.stream(active_stream):
        if isinstance(init, str):
            if init not in ("k-means||", "k-means++", "random"):
                raise ValueError(f"init must be 'k-means||', 'k-means++', 'random', or an array, got {init!r}")
            cuml_init = "scalable-k-means++" if init == "k-means||" else init
        else:
            cuml_init = _as_cuda_tensor("init", init, stream=active_stream).to(torch.float32).contiguous()
            if cuml_init.shape != (n_clusters, num_bits):
                raise ValueError(f"init must have shape {(n_clusters, num_bits)}, got {tuple(cuml_init.shape)}")
            if metric == "cosine":
                cuml_init = cuml_init / cuml_init.norm(dim=1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)

        fit_rows = None
        if fit_size is not None and fit_size < num_rows:
            generator = torch.Generator(device=x.device)
            if random_state is None:
                generator.seed()
            else:
                generator.manual_seed(random_state)
            fit_rows = torch.randperm(num_rows, generator=generator, device=x.device)[:fit_size]
        fit_features = _unpack_features(x, fit_rows, metric, batch_size)

        model = KMeans(
            n_clusters=n_clusters,
            init=cuml_init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            max_samples_per_batch=batch_size,
            output_type="cupy",
        )
        active_stream.synchronize()
        model.fit(fit_features)
        centers = torch.from_dlpack(model.cluster_centers_)

        if fit_rows is None:
            labels = torch.from_dlpack(model.labels_).to(torch.int32)
        else:
            del fit_features
            labels = torch.empty(num_rows, dtype=torch.int32, device=x.device)
            for start in range(0, num_rows, batch_size):
                stop = min(start + batch_size, num_rows)
                features = _unpack_features(x[start:stop], None, metric, batch_size)
                active_stream.synchronize()
                labels[start:stop] = torch.from_dlpack(model.predict(features))

        cluster_sizes = torch.bincount(labels.to(torch.int64), minlength=n_clusters)
        return KMeansResult(
            AsyncGpuResult(labels),
            AsyncGpuResult(centers),
            AsyncGpuResult(cluster_sizes),
            float(model.inertia_),
            int(model.n_iter_),
        )
