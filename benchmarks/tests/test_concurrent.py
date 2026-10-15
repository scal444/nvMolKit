# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

from bench_utils.concurrent import process_map_bounded, ready_spawn_pool
from bench_utils.timing import Deadline


def _sleep_and_return(item):
    delay, value = item
    time.sleep(delay)
    return value


def test_process_map_bounded_returns_all_results_without_deadline():
    result = process_map_bounded(
        _sleep_and_return,
        [(0.001, 1), (0.001, 2)],
        max_workers=2,
        deadline=Deadline(0.0),
    )
    assert sorted(result.results) == [1, 2]
    assert not result.timed_out


def test_process_map_bounded_retains_completed_results_on_timeout():
    # Start workers before the deadline, as benchmarks do, so spawn start-up is not part of the budget.
    pool = ready_spawn_pool(1)
    result = process_map_bounded(
        _sleep_and_return,
        [(0.001, 1), (2.0, 2)],
        max_workers=1,
        deadline=Deadline(0.25),
        pool=pool,
    )
    assert result.results == [1]
    assert result.timed_out


def test_process_map_bounded_reuses_a_ready_pool_across_calls():
    pool = ready_spawn_pool(2)
    try:
        first = process_map_bounded(_sleep_and_return, [(0.001, 1)], max_workers=2, deadline=Deadline(0.0), pool=pool)
        second = process_map_bounded(_sleep_and_return, [(0.001, 2)], max_workers=2, deadline=Deadline(0.0), pool=pool)
    finally:
        pool.terminate()
        pool.join()
    assert first.results == [1]
    assert second.results == [2]
