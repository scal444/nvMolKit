# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded multiprocessing helpers for benchmark preparation/reference work."""

import multiprocessing
import multiprocessing.pool
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from multiprocessing import TimeoutError as MultiprocessingTimeoutError
from typing import Any, Generic, TypeVar

from bench_utils.timing import Deadline

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@dataclass
class BoundedProcessMapResult(Generic[OutputT]):
    """Completed results from a bounded process map."""

    results: list[OutputT]
    timed_out: bool


def _report_ready(_: int) -> int:
    time.sleep(0.05)
    return os.getpid()


def ready_spawn_pool(
    max_workers: int,
    initializer: Callable[..., Any] | None = None,
    initargs: tuple[Any, ...] = (),
) -> multiprocessing.pool.Pool:
    """Start a "spawn" pool and wait until every worker has finished its initializer.

    Spawned workers re-import the benchmark module, which takes seconds per worker. Creating the pool
    with this helper before timing keeps that start-up out of the measured region. Forking is avoided
    because the parent already runs threads (CUDA, torch, OpenMP), and a forked child can inherit a
    held lock and deadlock.
    """
    pool = multiprocessing.get_context("spawn").Pool(max_workers, initializer=initializer, initargs=initargs)
    ready: set[int] = set()
    while len(ready) < max_workers:
        ready.update(pool.map(_report_ready, range(max_workers * 2), chunksize=1))
    return pool


def process_map_bounded(
    function: Callable[[InputT], OutputT],
    items: Sequence[InputT],
    *,
    max_workers: int,
    deadline: Deadline,
    initializer: Callable[..., Any] | None = None,
    initargs: tuple[Any, ...] = (),
    pool: multiprocessing.pool.Pool | None = None,
) -> BoundedProcessMapResult[OutputT]:
    """Map process-safe work and retain completed items when a deadline expires.

    Unlike :func:`tqdm.contrib.concurrent.process_map`, this helper uses the
    deadline as a total wall-clock budget, returns completed results, and
    terminates workers still processing items when the budget expires.
    Callers should submit coarse batches when individual tasks are tiny.
    Result order is completion order; include an index in each result when
    input order must be reconstructed.

    Pass ``pool`` (from :func:`ready_spawn_pool`) to reuse workers started outside the timed region; it
    is left open unless the deadline expires, in which case it is terminated and must not be reused.
    """
    if not items:
        return BoundedProcessMapResult([], False)

    owns_pool = pool is None
    if owns_pool:
        pool = ready_spawn_pool(max_workers, initializer, initargs)
    iterator = pool.imap_unordered(function, items, chunksize=1)
    completed: list[OutputT] = []
    timed_out = False
    try:
        for _ in items:
            remaining = deadline.remaining_seconds()
            if remaining is not None:
                if remaining <= 0:
                    timed_out = True
                    break
                try:
                    completed.append(iterator.next(timeout=remaining))
                except MultiprocessingTimeoutError:
                    timed_out = True
                    break
            else:
                completed.append(next(iterator))
    finally:
        if timed_out:
            pool.terminate()
            pool.join()
        elif owns_pool:
            pool.close()
            pool.join()
    return BoundedProcessMapResult(completed, timed_out)
