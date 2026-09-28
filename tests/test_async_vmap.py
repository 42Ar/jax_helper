import asyncio
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime

import numpy as np
import pytest

import jax_helper
from jax_helper.async_vmap import _Pool, async_vmap_pool


def _arity_key(request):
    return len(request)


def _pool(execute, key=_arity_key, max_batch_size=8, **kwargs):
    """A pool bound to the running loop, for testing the async core directly."""
    return _Pool(
        execute, key, max_batch_size, asyncio.get_running_loop(), **kwargs
    )


# --- pooling behaviour -------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_calls_coalesce_into_one_batch():
    batches = []

    def execute(requests):
        batches.append([request[0] for request in requests])
        return [request[0] * 2 for request in requests]

    pool = _pool(execute)
    results = await asyncio.gather(*[pool.submit(i) for i in range(8)])

    assert results == [i * 2 for i in range(8)]
    assert len(batches) == 1
    assert batches[0] == list(range(8))


async def _staged_submit(pool, index):
    """Submit ``index`` after ``index`` event-loop turns, one call per turn.

    The callers reach the pool on successive turns (one arrival per turn, no
    quiet gap).
    """
    for _ in range(index):
        await asyncio.sleep(0)
    return await pool.submit(index)


@pytest.mark.asyncio
async def test_parked_coalesces_calls_across_turns():
    """The worker keeps collecting until every caller has parked.

    Callers that reach the pool on successive turns are all captured by the
    one batch: nobody is considered settled until the last caller has
    submitted and parked on a pool future.
    """
    sizes = []

    def execute(requests):
        sizes.append(len(requests))
        return [request[0] for request in requests]

    pool = _pool(execute, max_batch_size=8)
    results = await asyncio.gather(
        *[_staged_submit(pool, i) for i in range(8)]
    )

    assert results == list(range(8))
    assert sizes == [8]


@pytest.mark.asyncio
async def test_parked_collecting_bounds_batch_size():
    """Parked collecting never grows a batch past ``max_batch_size``."""
    sizes = []

    def execute(requests):
        sizes.append(len(requests))
        return [request[0] for request in requests]

    pool = _pool(execute, max_batch_size=4)
    results = await asyncio.gather(
        *[_staged_submit(pool, i) for i in range(8)]
    )

    assert results == list(range(8))
    assert max(sizes) <= 4
    assert sum(sizes) == 8


@pytest.mark.asyncio
async def test_parked_waits_for_a_caller_still_computing():
    """A caller doing synchronous work before submitting is waited for.

    A caller that is still computing is *not* parked on the pool, so the
    batch is not dispatched until the last of them has submitted and parked.
    This is the case where a settle-after-one-turn rule would have split the
    burst into several small executions.
    """
    sizes = []

    def execute(requests):
        sizes.append(len(requests))
        return [request[0] for request in requests]

    pool = _pool(execute, max_batch_size=8)

    async def slow(i):
        for _ in range(4_000_000):
            pass
        return await pool.submit(i)

    results = await asyncio.gather(*[slow(i) for i in range(8)])

    assert results == list(range(8))
    assert sizes == [8]


@pytest.mark.asyncio
async def test_a_gather_waiter_is_parked_and_does_not_hold_the_batch():
    """A caller suspended on a gather is parked and does not hold the batch.

    ``_is_parked`` counts a ``gather`` parent as parked: it cannot enqueue
    until the gather resumes it, and any submitter it spawned is tracked as a
    task in its own right. So an unrelated orchestrator still suspended on its
    gather does not keep the open batch waiting -- its later submit joins the
    next batch.
    """
    sizes = []

    def execute(requests):
        sizes.append(len(requests))
        return [request[0] for request in requests]

    pool = _pool(execute, max_batch_size=8)

    gate = asyncio.Event()

    async def caller():
        await asyncio.gather(gate.wait())
        return await pool.submit("late")

    late = asyncio.create_task(caller())
    seed = asyncio.create_task(pool.submit("seed"))

    # ``caller`` is parked mid-gather here: it cannot have submitted yet, so
    # ``seed`` is the whole batch and closes on its own.
    assert await seed == "seed"
    gate.set()

    assert await late == "late"
    assert sizes == [1, 1]


@pytest.mark.asyncio
async def test_a_gather_of_pool_submits_is_counted_parked():
    """A gather burst still coalesces into one batch.

    The submitters are all queued before the batch closes (the drain merges
    whatever is queued), so counting the ``gather`` parent as parked never
    splits a gather-based burst.
    """
    sizes = []

    def execute(requests):
        sizes.append(len(requests))
        return [request[0] for request in requests]

    pool = _pool(execute, max_batch_size=8)

    results = await asyncio.gather(*[pool.submit(i) for i in range(4)])

    assert results == list(range(4))
    assert sizes == [4]


@pytest.mark.asyncio
async def test_a_task_awaiting_a_task_awaiting_the_pool_dispatches_parked(capsys):
    """An ``await`` chain ending in the pool counts every hop as parked."""
    sizes = []

    def execute(requests):
        sizes.append(len(requests))
        return [request[0] for request in requests]

    pool = _pool(execute, max_batch_size=8, debug=True, label="chain")

    async def leaf(value):
        return await pool.submit(value)

    async def middle(value):
        return await asyncio.create_task(leaf(value))

    results = await asyncio.gather(*[middle(i) for i in range(3)])

    assert results == list(range(3))
    assert sizes == [3]
    assert "dispatching after" not in capsys.readouterr().err


@pytest.mark.asyncio
async def test_deep_chain_awaits_pool_dispatches_parked(capsys):
    """Several hops of task-awaiting-task still count as parked."""
    sizes = []

    def execute(requests):
        sizes.append(len(requests))
        return [request[0] for request in requests]

    pool = _pool(execute, max_batch_size=8, debug=True, label="deep")

    async def bottom(value):
        return await pool.submit(value)

    async def mid(value):
        return await asyncio.create_task(bottom(value))

    async def top(value):
        return await asyncio.create_task(mid(value))

    results = await asyncio.gather(*[top(i) for i in range(4)])

    assert results == list(range(4))
    assert sizes == [4]
    assert "dispatching after" not in capsys.readouterr().err


@pytest.mark.asyncio
async def test_gather_inside_a_task_still_counts_as_parked(capsys):
    """A gather nested in a task, reached through another task, stays parked."""
    sizes = []

    def execute(requests):
        sizes.append(len(requests))
        return [request[0] for request in requests]

    pool = _pool(execute, max_batch_size=8, debug=True, label="nested")

    async def inner(value):
        return await asyncio.gather(*(pool.submit(v) for v in range(value)))

    async def outer(value):
        return await asyncio.create_task(inner(value))

    results = await asyncio.gather(*[outer(2), outer(2)])

    assert sorted(sum(r) for r in results) == [1, 1]
    assert sizes == [4]
    assert "dispatching after" not in capsys.readouterr().err


@pytest.mark.asyncio
async def test_background_task_does_not_starve_the_batch():
    """A task parked on non-pool work cannot stall a batch forever.

    The parked rule falls back to a few silent turns, so a background
    coroutine idling on its own await never blocks a lone caller.
    """
    sizes = []

    def execute(requests):
        sizes.append(len(requests))
        return [request[0] for request in requests]

    pool = _pool(execute, max_batch_size=8)

    async def background():
        await asyncio.Event().wait()  # parked on a non-pool future, forever

    bg = asyncio.create_task(background())
    try:
        assert await asyncio.wait_for(pool.submit("x"), timeout=2) == "x"
    finally:
        bg.cancel()

    assert sizes == [1]


@pytest.mark.asyncio
async def test_batch_never_exceeds_max_batch_size():
    sizes = []

    def execute(requests):
        sizes.append(len(requests))
        return [request[0] for request in requests]

    pool = _pool(execute, max_batch_size=8)
    results = await asyncio.gather(*[pool.submit(i) for i in range(20)])

    assert results == list(range(20))
    assert max(sizes) <= 8
    assert sum(sizes) == 20


@pytest.mark.asyncio
async def test_a_full_batch_dispatches_immediately(capsys):
    """Reaching ``max_batch_size`` dispatches at once, unparked task or not."""
    sizes = []

    def execute(requests):
        sizes.append(len(requests))
        return [request[0] for request in requests]

    pool = _pool(execute, max_batch_size=8, debug=True, label="full")

    async def background():
        await asyncio.Event().wait()  # parked on a non-pool future, forever

    bg = asyncio.create_task(background())
    try:
        results = await asyncio.gather(*[pool.submit(i) for i in range(8)])
    finally:
        bg.cancel()

    assert results == list(range(8))
    assert sizes == [8]
    assert "dispatching after" not in capsys.readouterr().err


@pytest.mark.asyncio
async def test_a_wave_exceeding_max_splits_into_capped_batches(capsys):
    """An overshooting wave runs a full batch now; the overflow is the split."""
    sizes = []

    def execute(requests):
        sizes.append(len(requests))
        return [request[0] for request in requests]

    pool = _pool(execute, max_batch_size=4, debug=True, label="split")
    results = await asyncio.gather(*[pool.submit(i) for i in range(10)])

    assert results == list(range(10))
    assert sizes == [4, 4, 2]
    assert all(s <= 4 for s in sizes)
    assert "dispatching after" not in capsys.readouterr().err


def _is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0


@pytest.mark.asyncio
async def test_padding_down_runs_power_of_two_prefixes():
    """``padding="down"`` runs the floor power of two and defers the rest."""
    sizes = []

    def execute(requests):
        sizes.append(len(requests))
        return [request[0] for request in requests]

    pool = _pool(execute, max_batch_size=8, padding="down")
    results = await asyncio.gather(*[pool.submit(i) for i in range(5)])

    assert results == list(range(5))
    assert sizes == [4, 1]
    assert all(_is_power_of_two(n) for n in sizes)


@pytest.mark.asyncio
async def test_padding_down_splits_a_seven_into_four_two_one():
    sizes = []

    def execute(requests):
        sizes.append(len(requests))
        return [request[0] for request in requests]

    pool = _pool(execute, max_batch_size=8, padding="down")
    results = await asyncio.gather(*[pool.submit(i) for i in range(7)])

    assert results == list(range(7))
    assert sizes == [4, 2, 1]
    assert all(_is_power_of_two(n) for n in sizes)


@pytest.mark.asyncio
async def test_padding_up_keeps_the_whole_batch():
    """The default ``padding="up"`` still serves the whole batch at once."""
    sizes = []

    def execute(requests):
        sizes.append(len(requests))
        return [request[0] for request in requests]

    pool = _pool(execute, max_batch_size=8)
    results = await asyncio.gather(*[pool.submit(i) for i in range(5)])

    assert results == list(range(5))
    assert sizes == [5]


@pytest.mark.asyncio
async def test_padding_down_split_respects_min_batch_size():
    """A split needs its prefix to reach ``min_batch_size``."""
    sizes = []

    def execute(requests):
        sizes.append(len(requests))
        return [request[0] for request in requests]

    pool = _pool(execute, max_batch_size=8, padding="down", min_batch_size=2)
    results = await asyncio.gather(*[pool.submit(i) for i in range(6)])

    assert results == list(range(6))
    assert sizes == [4, 2]


@pytest.mark.asyncio
async def test_sub_min_batch_is_dispatched_whole_and_padded():
    """A batch below ``min_batch_size`` never splits; it runs padded."""
    sizes = []

    def execute(requests):
        sizes.append(len(requests))
        return [request[0] for request in requests]

    pool = _pool(execute, max_batch_size=8, padding="down", min_batch_size=4)
    results = await asyncio.gather(*[pool.submit(i) for i in range(3)])

    # floor(3) = 2 < min(4), so no split: one whole (padded) batch.
    assert results == list(range(3))
    assert sizes == [3]


@pytest.mark.asyncio
async def test_padding_down_never_leaves_a_sub_min_tail():
    """A split must leave the remainder at ``min_batch_size`` at least."""
    sizes = []

    def execute(requests):
        sizes.append(len(requests))
        return [request[0] for request in requests]

    pool = _pool(execute, max_batch_size=8, padding="down", min_batch_size=4)
    results = await asyncio.gather(*[pool.submit(i) for i in range(6)])

    # floor(6) = 4 meets the minimum, but the tail (2) does not: splitting
    # would run 4 and then recompile a lone 2, so the whole group runs padded.
    assert results == list(range(6))
    assert sizes == [6]


@pytest.mark.asyncio
async def test_below_min_batch_dispatches_as_one_batch():
    """A sub-minimum batch of parked callers dispatches as a single batch."""
    sizes = []

    def execute(requests):
        sizes.append(len(requests))
        return [request[0] for request in requests]

    pool = _pool(execute, max_batch_size=8, min_batch_size=3)
    results = await asyncio.gather(pool.submit(1), pool.submit(2))

    assert results == [1, 2]
    assert sizes == [2]


@pytest.mark.asyncio
async def test_below_min_batch_dispatches_when_all_parked(capsys):
    """All tasks parked dispatches below ``min_batch_size``, no valve wait."""
    sizes = []

    def execute(requests):
        sizes.append(len(requests))
        return [request[0] for request in requests]

    pool = _pool(execute, max_batch_size=8, min_batch_size=4, debug=True)
    results = await asyncio.gather(pool.submit(1), pool.submit(2))

    assert results == [1, 2]
    assert sizes == [2]
    assert "dispatching after" not in capsys.readouterr().err


@pytest.mark.asyncio
async def test_min_batch_size_still_coalesces_a_full_burst():
    """A burst still coalesces with ``min_batch_size`` above 1."""
    sizes = []

    def execute(requests):
        sizes.append(len(requests))
        return [request[0] for request in requests]

    pool = _pool(execute, max_batch_size=8, min_batch_size=4)
    results = await asyncio.gather(*[pool.submit(i) for i in range(4)])

    assert results == list(range(4))
    assert sizes == [4]


@pytest.mark.asyncio
async def test_execution_runs_on_the_loop_thread_when_not_threaded():
    """With ``threaded=False`` batches execute inline in the worker task."""
    thread_ids = []

    def execute(requests):
        thread_ids.append(threading.get_ident())
        return [request[0] for request in requests]

    pool = _pool(execute, threaded=False)
    await pool.submit(1)
    await pool.submit(2)

    assert thread_ids == [threading.get_ident()] * 2


@pytest.mark.asyncio
async def test_execution_runs_on_a_worker_thread():
    """Batches execute off the event-loop thread by default."""
    thread_ids = []

    def execute(requests):
        thread_ids.append(threading.get_ident())
        return [request[0] for request in requests]

    pool = _pool(execute)
    await pool.submit(1)
    await pool.submit(2)

    # One worker thread, reused, distinct from the loop thread.
    assert len(set(thread_ids)) == 1
    assert thread_ids[0] != threading.get_ident()


@pytest.mark.asyncio
async def test_loop_stays_responsive_during_execution():
    """The loop keeps scheduling tasks while a batch executes on a thread."""
    def execute(requests):
        time.sleep(0.05)  # simulate a long JAX run
        return [request[0] for request in requests]

    pool = _pool(execute, threaded=True)
    start = time.perf_counter()
    task = asyncio.create_task(pool.submit(1))
    # A batch executing on the worker thread must not freeze the loop.
    await asyncio.sleep(0.02)
    elapsed = time.perf_counter() - start

    assert await task == 1
    assert elapsed < 0.045, elapsed


@pytest.mark.asyncio
async def test_loop_is_busy_during_inline_execution():
    """With the loop inline, a long batch holds the loop until it returns."""
    def execute(requests):
        time.sleep(0.05)
        return [request[0] for request in requests]

    pool = _pool(execute, threaded=False)
    start = time.perf_counter()
    task = asyncio.create_task(pool.submit(1))
    # An independent timer cannot run while the batch holds the loop.
    await asyncio.sleep(0.02)
    elapsed = time.perf_counter() - start

    assert await task == 1
    assert elapsed >= 0.045, elapsed


@pytest.mark.asyncio
async def test_aclose_waits_for_the_in_flight_batch():
    """Closing during an executing batch still delivers that batch's result."""
    def execute(requests):
        time.sleep(0.02)
        return [request[0] for request in requests]

    pool = _pool(execute, threaded=True)
    task = asyncio.create_task(pool.submit(7))
    await asyncio.sleep(0)  # let the batch open (and start executing)
    await pool.aclose()     # must wait for the in-flight run, not drop it

    assert await task == 7


@pytest.mark.asyncio
async def test_single_call_is_executed_immediately():
    calls = []

    def execute(requests):
        calls.append([request[0] for request in requests])
        return [request[0] for request in requests]

    pool = _pool(execute)
    assert await pool.submit(3) == 3
    assert calls == [[3]]


@pytest.mark.asyncio
async def test_max_batch_size_one_never_coalesces():
    sizes = []

    def execute(requests):
        sizes.append(len(requests))
        return [request[0] for request in requests]

    pool = _pool(execute, max_batch_size=1)
    results = await asyncio.gather(*[pool.submit(i) for i in range(4)])

    assert results == [0, 1, 2, 3]
    assert sizes == [1, 1, 1, 1]


@pytest.mark.asyncio
async def test_multi_argument_requests_coalesce():
    batches = []

    def execute(requests):
        batches.append([(r[0], r[1]) for r in requests])
        return [r[0] + r[1] for r in requests]

    pool = _pool(execute)
    results = await asyncio.gather(*[pool.submit(i, i * 10) for i in range(4)])

    assert results == [i + i * 10 for i in range(4)]
    assert len(batches) == 1
    assert batches[0] == [(i, i * 10) for i in range(4)]


@pytest.mark.asyncio
async def test_executor_exception_reaches_every_caller():
    def execute(requests):
        raise RuntimeError("boom")

    pool = _pool(execute)
    # wait_for turns a regression back into the original hang into a timeout.
    results = await asyncio.wait_for(
        asyncio.gather(*[pool.submit(i) for i in range(4)], return_exceptions=True),
        timeout=5,
    )

    assert len(results) == 4
    assert all(isinstance(r, RuntimeError) and str(r) == "boom" for r in results)


@pytest.mark.asyncio
async def test_wrong_result_count_fails_the_batch():
    def execute(requests):
        return [r[0] for r in requests][:-1]

    pool = _pool(execute)
    results = await asyncio.wait_for(
        asyncio.gather(*[pool.submit(i) for i in range(4)], return_exceptions=True),
        timeout=5,
    )

    assert all(isinstance(r, ValueError) for r in results)
    assert "batch of 4" in str(results[0])


@pytest.mark.asyncio
async def test_failed_batch_does_not_kill_the_worker():
    calls = []

    def execute(requests):
        calls.append([r[0] for r in requests])
        if len(calls) == 1:
            raise RuntimeError("boom")
        return [r[0] for r in requests]

    pool = _pool(execute)
    first = await asyncio.gather(pool.submit(1), return_exceptions=True)
    second = await asyncio.gather(*[pool.submit(i) for i in range(2)])

    assert isinstance(first[0], RuntimeError)
    assert second == [0, 1]


@pytest.mark.asyncio
async def test_cancelled_caller_does_not_break_the_batch():
    def execute(requests):
        return [r[0] * 2 for r in requests]

    pool = _pool(execute)
    tasks = [asyncio.ensure_future(pool.submit(i)) for i in range(4)]
    tasks[0].cancel()
    results = await asyncio.gather(*tasks, return_exceptions=True)

    assert isinstance(results[0], asyncio.CancelledError)
    assert results[1:] == [2, 4, 6]


@pytest.mark.asyncio
async def test_dead_worker_fails_its_waiters():
    def execute(requests):
        raise asyncio.CancelledError()

    pool = _pool(execute)
    results = await asyncio.wait_for(
        asyncio.gather(*[pool.submit(i) for i in range(3)], return_exceptions=True),
        timeout=5,
    )

    assert all(isinstance(r, asyncio.CancelledError) for r in results)


# --- grouping ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_incompatible_requests_split_into_groups():
    seen = []

    def key(request):
        return request[0]

    def execute(requests):
        seen.append([r[0] for r in requests])
        return [r[1] for r in requests]

    pool = _pool(execute, key=key, max_batch_size=8)
    results = await asyncio.gather(
        *[
            pool.submit(group, value)
            for group, value in [("a", 1), ("a", 2), ("b", 3), ("a", 4), ("b", 5)]
        ]
    )

    assert results == [1, 2, 3, 4, 5]
    assert seen == [["a", "a", "a"], ["b", "b"]]


@pytest.mark.asyncio
async def test_a_raising_key_fails_the_batch_not_the_worker():
    def key(request):
        if request[0] == "bad":
            raise ValueError("bad key")
        return request[0]

    def execute(requests):
        return [r[1] for r in requests]

    pool = _pool(execute, key=key)
    first = await asyncio.gather(pool.submit("bad", 1), return_exceptions=True)
    # The worker must survive a key it could not compute.
    assert await pool.submit("good", 2) == 2
    assert isinstance(first[0], ValueError)


@pytest.mark.asyncio
async def test_a_failing_group_does_not_affect_a_sibling_group():
    def key(request):
        return request[0]

    def execute(requests):
        if requests[0][0] == "bad":
            raise RuntimeError("boom")
        return [r[1] for r in requests]

    pool = _pool(execute, key=key)
    results = await asyncio.gather(
        *[
            pool.submit(group, value)
            for group, value in [("bad", 1), ("good", 2), ("good", 3)]
        ],
        return_exceptions=True,
    )

    assert isinstance(results[0], RuntimeError)
    assert results[1:] == [2, 3]


# --- validation --------------------------------------------------------------


def test_max_batch_size_must_be_an_int():
    with pytest.raises(TypeError, match="max_batch_size must be an int"):
        async_vmap_pool(2.5)  # type: ignore[arg-type]


def test_max_batch_size_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        async_vmap_pool(0)


def test_coalescing_must_be_parked():
    with pytest.raises(ValueError, match="coalescing must be 'parked'"):
        async_vmap_pool(8, coalescing="nope")
    with pytest.raises(ValueError, match="coalescing must be 'parked'"):
        async_vmap_pool(8, coalescing="quiescent")
    with pytest.raises(ValueError, match="coalescing must be 'parked'"):
        async_vmap_pool(8, coalescing="opportunistic")
    async_vmap_pool(8, coalescing="parked")
    async_vmap_pool(8)
    async_vmap_pool(8, run_in_thread=True)


def test_run_in_thread_must_be_a_bool():
    with pytest.raises(TypeError, match="run_in_thread must be a bool"):
        async_vmap_pool(8, run_in_thread="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="run_in_thread must be a bool"):
        async_vmap_pool(8, run_in_thread=1)  # type: ignore[arg-type]
    async_vmap_pool(8, run_in_thread=False)
    async_vmap_pool(8, run_in_thread=True)


def test_padding_must_be_up_or_down():
    with pytest.raises(ValueError, match="padding must be 'up' or 'down'"):
        async_vmap_pool(8, padding="sideways")
    with pytest.raises(ValueError, match="padding must be 'up' or 'down'"):
        async_vmap_pool(8, padding="round")
    async_vmap_pool(8, padding="up")
    async_vmap_pool(8, padding="down")


def test_min_batch_size_validation():
    with pytest.raises(TypeError, match="min_batch_size must be an int"):
        async_vmap_pool(8, min_batch_size="2")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="min_batch_size must be an int"):
        async_vmap_pool(8, min_batch_size=2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least 1"):
        async_vmap_pool(8, min_batch_size=0)
    with pytest.raises(ValueError, match="cannot exceed max_batch_size"):
        async_vmap_pool(8, min_batch_size=9)
    async_vmap_pool(8, min_batch_size=1)
    async_vmap_pool(8, min_batch_size=8)


@pytest.mark.asyncio
async def test_a_request_needs_at_least_one_argument():
    @async_vmap_pool(4)
    def f(x):
        return x

    with pytest.raises(ValueError, match="at least one argument is required"):
        await f()


# --- the real decorator ------------------------------------------------------


def test_decorated_function_works_in_two_event_loops():
    @async_vmap_pool(4)
    def double(x):
        return x * 2

    async def main():
        return await asyncio.gather(*[double(i) for i in range(4)])

    assert [float(v) for v in asyncio.run(main())] == [0, 2, 4, 6]
    assert [float(v) for v in asyncio.run(main())] == [0, 2, 4, 6]


@pytest.mark.asyncio
async def test_varying_values_still_share_a_single_compiled_execution():
    traces = []

    @async_vmap_pool(8)
    def f(x, args):
        traces.append(1)
        return args["scale"] * x + args["bias"]

    items = [(float(i), {"scale": float(i), "bias": -float(i)}) for i in range(8)]
    results = await asyncio.gather(*[f(x, args) for x, args in items])

    assert [float(v) for v in results] == [i * i - i for i in range(8)]
    assert len(traces) == 1


def test_values_are_not_part_of_the_request_key():
    """Grouping must ignore values, or free-varying arguments never batch.

    This cannot be observed from the compiled function: padded singleton
    batches share one shape and hit the jit cache, so the trace count looks
    identical whether the requests coalesced or were fragmented. The key
    itself is the only place the property is visible.
    """
    from jax_helper.async_vmap import _request_key

    # Same arity, structure and shapes; every value differs.
    assert _request_key((1.0, {"scale": 2.0, "bias": 3.0})) == _request_key(
        (9.0, {"scale": -7.0, "bias": 0.5})
    )
    # Array leaves are compared by shape, not contents.
    assert _request_key((1.0, np.ones(2))) == _request_key((1.0, np.full(2, 99.0)))

    # Structure, shape and arity still discriminate.
    assert _request_key((1.0, {"a": 1.0})) != _request_key((1.0, {"b": 1.0}))
    assert _request_key((1.0, np.ones(2))) != _request_key((1.0, np.ones(3)))
    assert _request_key((1.0,)) != _request_key((1.0, 2.0))


@pytest.mark.asyncio
async def test_single_argument_still_works():
    @async_vmap_pool(4)
    def double(x):
        return x * 2

    results = await asyncio.gather(*[double(float(i)) for i in range(4)])

    assert [float(v) for v in results] == [0, 2, 4, 6]


@pytest.mark.asyncio
async def test_arities_one_two_and_three():
    traces = []

    @async_vmap_pool(8)
    def f(*args):
        traces.append(len(args))
        return args[0] * 2

    results = await asyncio.gather(
        f(1.0), f(1.0, 99.0), f(1.0, 99.0, 99.0)
    )

    assert [float(v) for v in results] == [2, 2, 2]
    # in_axes length must match the arity, so each arity compiles separately.
    assert sorted(traces) == [1, 2, 3]


@pytest.mark.asyncio
async def test_dict_argument():
    @async_vmap_pool(8)
    def f(x, args):
        return x * args["a"] + args["b"].sum()

    results = await asyncio.gather(
        f(1.0, {"a": 2.0, "b": np.ones(2)}),
        f(2.0, {"a": 3.0, "b": np.ones(2)}),
    )

    assert [float(v) for v in results] == [4.0, 8.0]


@pytest.mark.asyncio
async def test_nested_pytree_argument():
    @async_vmap_pool(8)
    def f(x, args):
        return x * 2.0 + args["outer"]["k"] + args["pair"][0].sum() + args["pair"][1].sum()

    results = await asyncio.gather(
        f(1.0, {"outer": {"k": 10.0}, "pair": (np.ones(2), np.ones(3))}),
        f(2.0, {"outer": {"k": 20.0}, "pair": (np.ones(2), np.ones(3))}),
    )

    assert [float(v) for v in results] == [17.0, 29.0]


@pytest.mark.asyncio
async def test_mixed_leaf_shapes_split_into_groups():
    traces = []

    @async_vmap_pool(8)
    def f(x, args):
        traces.append(1)
        return args["w"].shape[0] * x

    results = await asyncio.gather(
        f(1.0, {"w": np.ones(2)}),
        f(1.0, {"w": np.ones(2)}),
        f(1.0, {"w": np.ones(3)}),
    )

    assert [float(v) for v in results] == [2, 2, 3]
    # Two leaf shapes cannot share one vmap, so they compile separately -- and
    # the two shape-2 requests still batched together.
    assert len(traces) == 2


@pytest.mark.asyncio
async def test_mismatched_structures_split_into_groups():
    traces = []

    @async_vmap_pool(8)
    def f(x, args):
        traces.append(1)
        return args.get("a", args.get("b", 0.0)) * x

    results = await asyncio.gather(
        f(1.0, {"a": 2.0}),
        f(1.0, {"b": 3.0}),
    )

    assert [float(v) for v in results] == [2, 3]
    assert len(traces) == 2


@pytest.mark.asyncio
async def test_padding_covers_every_argument_and_leaf():
    @async_vmap_pool(4)
    def f(x, args):
        return x * 10 + args["k"]

    # Three requests round up to a padded batch of 4 on both arguments.
    results = await asyncio.gather(
        f(1.0, {"k": 0.5}),
        f(2.0, {"k": 0.5}),
        f(3.0, {"k": 0.5}),
    )

    # Trimming back to the real batch size is what proves padding was applied
    # to both arguments and then removed.
    assert [float(v) for v in results] == [10.5, 20.5, 30.5]


@pytest.mark.asyncio
async def test_traces_once_per_rounded_power_of_two():
    traces = []

    @async_vmap_pool(8)
    def f(x, args):
        traces.append(1)
        return x * args["k"]

    first = await asyncio.gather(f(1.0, {"k": 2.0}), f(2.0, {"k": 2.0}))
    second = await asyncio.gather(*[f(float(i), {"k": 2.0}) for i in range(5)])

    assert [float(v) for v in first] == [2, 4]
    assert [float(v) for v in second] == [0, 2, 4, 6, 8]
    # Batches of 2 and 5 round to the powers of two 2 and 8: one trace each.
    assert len(traces) == 2


@pytest.mark.asyncio
async def test_each_leaf_shape_reuses_its_compiled_entry():
    traces = []

    @async_vmap_pool(8)
    def f(x, args):
        traces.append(args["w"].shape[0])
        return x

    await asyncio.gather(f(1.0, {"w": np.ones(2)}))
    await asyncio.gather(f(1.0, {"w": np.ones(3)}))
    await asyncio.gather(f(1.0, {"w": np.ones(2)}))

    # Two distinct shapes compile twice; the repeat of the first reuses it.
    assert traces == [2, 3]


@pytest.mark.asyncio
async def test_batches_recompile_only_per_rounded_power_of_two():
    """Batches must compile once per distinct rounded batch size, never more.

    Every batch of n requests is padded up to the next power of two (capped at
    ``max_batch_size``), and each distinct padded size is a single compiled
    entry. The stack/pad/trim glue runs on NumPy, so those sizes never add
    XLA compilations of their own.
    """
    import logging

    import jax

    compiles = []

    class _Capture(logging.Handler):
        def emit(self, record):
            compiles.append(record.getMessage())

    old, old_level = getattr(jax.config, "jax_log_compiles"), logging.getLogger().level
    jax.config.update("jax_log_compiles", True)
    logging.getLogger().setLevel(logging.WARNING)
    handler = _Capture()
    logging.getLogger().addHandler(handler)
    try:
        @async_vmap_pool(8)
        def f(x):
            return x * 2

        await asyncio.gather(f(1.0), f(2.0))
        await asyncio.gather(*[f(float(i)) for i in range(5)])
    finally:
        logging.getLogger().removeHandler(handler)
        logging.getLogger().setLevel(old_level)
        jax.config.update("jax_log_compiles", old)

    jits = [m for m in compiles if "Compiling jit(" in m]
    # Batches of 2 and 5 round to the powers of two 2 and 8: one compile each.
    assert len(jits) == 2
    assert all("jit(f)" in j for j in jits)


@pytest.mark.asyncio
async def test_same_rounded_size_shares_one_compile():
    """Batches of different sizes that round to the same power of two share it.

    A batch of 3 and a batch of 4 both round to 4, so the second must reuse
    the first batch's compiled entry.
    """
    import logging

    import jax

    compiles = []

    class _Capture(logging.Handler):
        def emit(self, record):
            compiles.append(record.getMessage())

    old, old_level = getattr(jax.config, "jax_log_compiles"), logging.getLogger().level
    jax.config.update("jax_log_compiles", True)
    logging.getLogger().setLevel(logging.WARNING)
    handler = _Capture()
    logging.getLogger().addHandler(handler)
    try:
        @async_vmap_pool(8)
        def f(x):
            return x * 2

        three = await asyncio.gather(f(1.0), f(2.0), f(3.0))
        four = await asyncio.gather(*[f(float(i)) for i in range(4)])
    finally:
        logging.getLogger().removeHandler(handler)
        logging.getLogger().setLevel(old_level)
        jax.config.update("jax_log_compiles", old)

    assert [float(v) for v in three] == [2.0, 4.0, 6.0]
    assert [float(v) for v in four] == [0.0, 2.0, 4.0, 6.0]
    jits = [m for m in compiles if "Compiling jit(" in m]
    assert len(jits) == 1
    assert "jit(f)" in jits[0]


@pytest.mark.asyncio
async def test_padded_results_are_plain_numpy_arrays():
    """The executor hands back host NumPy arrays, not jax arrays."""
    import jax

    @async_vmap_pool(8)
    def f(x):
        return x * 2

    # Three requests round up to a padded batch of 4, exercising the padded
    # path on a batch that cannot fill its rounded size.
    results = await asyncio.gather(f(1.0), f(2.0), f(3.0))

    assert [r.shape for r in results] == [(), (), ()]
    assert all(
        isinstance(r, (np.ndarray, np.generic)) and not isinstance(r, jax.Array)
        for r in results
    )
    assert [float(r) for r in results] == [2.0, 4.0, 6.0]


@pytest.mark.asyncio
async def test_non_numeric_leaf_fails_every_caller():
    @async_vmap_pool(4)
    def f(x, args):
        return x

    results = await asyncio.wait_for(
        asyncio.gather(
            f(1.0, {"name": "abc"}),
            f(2.0, {"name": "abc"}),
            return_exceptions=True,
        ),
        timeout=30,
    )

    assert all(isinstance(r, Exception) for r in results)
    assert not any(isinstance(r, str) for r in results)


# --- lazy export -------------------------------------------------------------


def test_importing_the_package_does_not_import_jax():
    # Checked in a clean interpreter, and for both layers of laziness: the
    # package must not import the submodule eagerly (the __getattr__ shim), and
    # the submodule must not import jax at module scope (it imports jax inside
    # _build_executor). Asserting only on jax would pass even with an eager
    # import in __init__, because the submodule is lazy regardless.
    src = os.path.dirname(os.path.dirname(jax_helper.__file__))
    env = {**os.environ, "PYTHONPATH": src}
    code = (
        "import sys, jax_helper;"
        "print('jax' in sys.modules, 'jax_helper.async_vmap' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )

    assert out.stdout.strip() == "False False"


def test_async_vmap_pool_is_exported_lazily():
    from jax_helper import async_vmap_pool as exported

    assert callable(exported)
    assert "async_vmap_pool" in jax_helper.__all__


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        jax_helper.nope


# --- debug mode ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_debug_off_prints_nothing(capsys):
    def execute(requests):
        return [request[0] for request in requests]

    pool = _pool(execute)
    await pool.submit(1)

    assert capsys.readouterr().err == ""


@pytest.mark.asyncio
async def test_debug_reports_when_the_settle_valve_fires(capsys):
    """The settle-turns safety valve says so, so a stuck caller is found."""

    def execute(requests):
        return [request[0] for request in requests]

    pool = _pool(execute, debug=True, label="pinned")

    async def background():
        await asyncio.Event().wait()  # parked on a non-pool future, forever

    bg = asyncio.create_task(background())
    try:
        assert await pool.submit("x") == "x"
    finally:
        bg.cancel()

    output = capsys.readouterr().err
    assert "pinned: dispatching after" in output
    assert "1 task(s) were not parked on the pool" in output
    assert "background parked on Future" in output


@pytest.mark.asyncio
async def test_settle_message_names_each_unparked_task(capsys):
    """Two stuck callers are both reported, each with its hold-up."""
    def execute(requests):
        return [request[0] for request in requests]

    pool = _pool(execute, debug=True, label="pinned")

    async def sleeper():
        await asyncio.sleep(30)

    s = asyncio.create_task(sleeper())

    async def chaser():
        await s  # parks on the sleeper task, not on the pool

    async def blocker():
        await asyncio.Event().wait()  # parks on a plain future

    c = asyncio.create_task(chaser())
    b = asyncio.create_task(blocker())
    try:
        assert await pool.submit("x") == "x"
    finally:
        for t in (s, c, b):
            t.cancel()

    output = capsys.readouterr().err
    assert "3 task(s) were not parked on the pool" in output
    assert "blocker parked on Future" in output
    assert "sleeper parked on Future" in output
    assert "chaser awaiting " in output
    assert "sleeper" in output


@pytest.mark.asyncio
async def test_settle_message_condenses_a_large_unparked_set(capsys):
    """A huge unparked set is reported as per-kind tallies, not dumped whole."""
    def execute(requests):
        return [request[0] for request in requests]

    pool = _pool(execute, debug=True, label="pinned")

    event = asyncio.Event()
    pinned = [asyncio.create_task(event.wait()) for _ in range(30)]
    try:
        assert await pool.submit("x") == "x"
    finally:
        for t in pinned:
            t.cancel()

    output = capsys.readouterr().err
    assert "30 task(s) were not parked on the pool" in output
    assert "30 x Event.wait parked on Future" in output
    assert output.count("Event.wait parked on Future") == 1


@pytest.mark.asyncio
async def test_valve_line_prints_up_to_the_limit_verbatim(capsys):
    """Exactly ``_UNPARKED_LIST_LIMIT`` tasks are listed individually."""
    def execute(requests):
        return [request[0] for request in requests]

    pool = _pool(execute, debug=True, label="pinned")

    event = asyncio.Event()
    pinned = [asyncio.create_task(event.wait()) for _ in range(20)]
    try:
        assert await pool.submit("x") == "x"
    finally:
        for t in pinned:
            t.cancel()

    output = capsys.readouterr().err
    assert "20 task(s) were not parked on the pool" in output
    assert output.count("Event.wait parked on Future") == 20
    assert "20 x Event.wait parked on Future" not in output


@pytest.mark.asyncio
async def test_valve_line_condenses_beyond_the_limit(capsys):
    """One past the limit, the list collapses to a per-kind tally."""
    def execute(requests):
        return [request[0] for request in requests]

    pool = _pool(execute, debug=True, label="pinned")

    event = asyncio.Event()
    pinned = [asyncio.create_task(event.wait()) for _ in range(21)]
    try:
        assert await pool.submit("x") == "x"
    finally:
        for t in pinned:
            t.cancel()

    output = capsys.readouterr().err
    assert "21 task(s) were not parked on the pool" in output
    assert "21 x Event.wait parked on Future" in output
    assert output.count("Event.wait parked on Future") == 1


@pytest.mark.asyncio
async def test_settle_budget_measures_idle_time_not_total_hold(capsys):
    """Arrivals reset the settle budgets, so only *idle* time trips the valve.

    A trickle of submissions spaced ~5 ms apart for well past the 100 ms
    budget coalesces into one open batch and never fires the valve: each
    arrival resets both the idle-turn and wall-clock budgets. When the trickle
    stops and nothing else is parked, the batch dispatches through the parked
    condition instead -- no ``dispatching after`` line, no valve.
    """
    sizes = []

    def execute(requests):
        sizes.append(len(requests))
        return [request[0] for request in requests]

    pool = _pool(execute, max_batch_size=1024, debug=True, label="burst")
    k = 40
    stopped = False
    tasks = []

    async def clog():
        # Makes each loop turn last ~5 ms, so the turn budget never races the
        # trickle (one worker check per turn) while the wall-clock hold grows
        # past the 100 ms settle timeout.
        nonlocal stopped
        while not stopped:
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < 0.005:
                pass
            await asyncio.sleep(0)

    async def producer():
        nonlocal stopped
        for i in range(k):
            tasks.append(asyncio.create_task(pool.submit(i)))
            await asyncio.sleep(0)
        stopped = True  # the last submitter has run; let clog go before the tail

    clog_task = asyncio.create_task(clog())
    started = time.perf_counter()
    await asyncio.gather(producer(), clog_task)
    results = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - started

    assert results == list(range(k))
    assert sizes == [k]
    assert elapsed > 0.1  # one open batch held far beyond the time budget
    assert "dispatching after" not in capsys.readouterr().err


@pytest.mark.asyncio
async def test_settle_message_names_the_idle_turn_budget(capsys):
    """The settle line names the idle-turn budget when it fires that way."""

    def execute(requests):
        return [request[0] for request in requests]

    pool = _pool(execute, debug=True, label="pinned")

    async def background():
        await asyncio.Event().wait()  # unparked forever, so the valve trips

    bg = asyncio.create_task(background())
    try:
        assert await pool.submit("x") == "x"
    finally:
        bg.cancel()

    output = capsys.readouterr().err
    assert "pinned: dispatching after" in output
    assert "safety valve: idle-turn budget" in output


@pytest.mark.asyncio
async def test_settle_message_names_the_time_budget(capsys):
    """The settle line names the wall-clock budget when it fires first."""
    def execute(requests):
        return [request[0] for request in requests]

    pool = _pool(execute, debug=True, label="pinned")

    async def background():
        await asyncio.sleep(0)       # let the worker start collecting
        time.sleep(0.15)             # block the loop past the 100 ms budget
        await asyncio.Event().wait()  # then stay unparked, pinning the valve

    bg = asyncio.create_task(background())
    try:
        assert await pool.submit("x") == "x"
    finally:
        bg.cancel()

    output = capsys.readouterr().err
    assert "pinned: dispatching after" in output
    assert "safety valve: time budget (100.0 ms)" in output


@pytest.mark.asyncio
async def test_debug_reports_each_execution_and_its_size(capsys):
    def execute(requests):
        return [request[0] for request in requests]

    pool = _pool(execute, max_batch_size=4, debug=True, label="solver")
    # 10 requests at max_batch_size 4 -> 4, 4, 2
    await asyncio.gather(*[pool.submit(i) for i in range(10)])

    pattern = re.compile(r"executing (\d+) request")
    sizes = [int(m.group(1)) for m in
             (pattern.search(line) for line in capsys.readouterr().err.splitlines())
             if m is not None]
    assert sizes == [4, 4, 2]


@pytest.mark.asyncio
async def test_debug_reports_the_function_name(capsys):
    def execute(requests):
        return [request[0] for request in requests]

    pool = _pool(execute, debug=True, label="roots_scan")
    await pool.submit(1)

    assert "roots_scan" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_debug_reports_a_datetime_timestamp(capsys):
    def execute(requests):
        return [request[0] for request in requests]

    pool = _pool(execute, debug=True, label="f")
    await pool.submit(1)

    line = capsys.readouterr().err.splitlines()[0]
    stamp, rest = line.split("]", 1)
    datetime.fromisoformat(stamp[1:])  # raises if not a valid datetime
    assert "executing 1 request(s)" in rest


@pytest.mark.asyncio
async def test_debug_reports_completion_with_timestamp(capsys):
    def execute(requests):
        return [request[0] for request in requests]

    pool = _pool(execute, debug=True, label="f")
    await pool.submit(1)

    lines = capsys.readouterr().err.splitlines()
    start, done = lines[0], lines[1]
    assert "executing 1 request(s)" in start.split("]", 1)[1]
    assert "executed 1 request(s)" in done.split("]", 1)[1]
    before = datetime.fromisoformat(start.split("]", 1)[0][1:])
    after = datetime.fromisoformat(done.split("]", 1)[0][1:])
    assert after >= before


@pytest.mark.asyncio
async def test_debug_goes_to_stderr_not_stdout(capsys):
    def execute(requests):
        return [request[0] for request in requests]

    pool = _pool(execute, debug=True, label="f")
    await pool.submit(1)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "f" in captured.err


@pytest.mark.asyncio
async def test_debug_reports_a_batch_that_raises(capsys):
    def execute(requests):
        raise RuntimeError("boom")

    pool = _pool(execute, debug=True, label="boom")
    await asyncio.gather(
        *[pool.submit(i) for i in range(3)], return_exceptions=True
    )

    assert "boom: executing 3 request(s)" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_debug_reports_each_group_of_a_split_batch(capsys):
    """Incompatible shapes run as separate executions, so each is reported.

    This also pins that the size is the *group's*, not the drained batch's:
    four requests drained, but each group holds only two.
    """
    def execute(requests):
        return [request[0] for request in requests]

    pool = _pool(execute, key=lambda r: r[1], max_batch_size=8, debug=True,
                 label="split")
    await asyncio.gather(
        pool.submit(1, "a"), pool.submit(2, "a"),
        pool.submit(3, "b"), pool.submit(4, "b"),
    )

    lines = capsys.readouterr().err.splitlines()
    start = [l for l in lines if "executing 2 request(s)" in l]
    done = [l for l in lines if "executed 2 request(s)" in l]
    assert len(start) == 2
    assert len(done) == 2


@pytest.mark.asyncio
async def test_debug_reports_group_size_not_drained_batch_size(capsys):
    """Unequal groups expose the difference: 3 + 1 drained, so not 4 + 4."""
    def execute(requests):
        return [request[0] for request in requests]

    pool = _pool(execute, key=lambda r: r[1], max_batch_size=8, debug=True,
                 label="skew")
    await asyncio.gather(
        pool.submit(1, "a"), pool.submit(2, "a"), pool.submit(3, "a"),
        pool.submit(4, "b"),
    )

    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 4
    assert "executing 3 request(s)" in lines[0]
    assert "executed 3 request(s)" in lines[1]
    assert "executing 1 request(s)" in lines[2]
    assert "executed 1 request(s)" in lines[3]


@pytest.mark.asyncio
async def test_debug_reports_the_real_count_not_the_padded_one(capsys):
    """Padding changes the compiled shape, not the number of callers."""
    def execute(requests):
        return [request[0] for request in requests]

    # The real executor pads, but the pool only ever sees real requests.
    pool = _pool(execute, max_batch_size=8, debug=True, label="padded")
    await pool.submit(1)

    assert "executing 1 request(s)" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_debug_reports_when_a_new_batch_size_compiles(capsys):
    """A debug line marks the first time a rounded batch size is compiled."""
    @async_vmap_pool(max_batch_size=8, debug=True)
    def f(x):
        return x * 2

    one = await f(1.0)                       # batch of 1  -> size 1, compiles
    three = await asyncio.gather(f(1.0), f(2.0), f(3.0))  # size 4, compiles
    again = await asyncio.gather(f(1.0), f(2.0), f(3.0))  # size 4, cached

    lines = capsys.readouterr().err.splitlines()
    compiles = [l for l in lines if "compiling batch size" in l]
    assert len(compiles) == 2
    assert "f: compiling batch size 1 (for 1 requests)" in compiles[0]
    assert "f: compiling batch size 4 (for 3 requests)" in compiles[1]
    assert float(one) == 2.0
    assert [float(v) for v in three] == [2.0, 4.0, 6.0]
    assert [float(v) for v in again] == [2.0, 4.0, 6.0]


@pytest.mark.asyncio
async def test_below_min_batch_compiles_at_min_batch_size(capsys):
    """A below-minimum batch pads up to ``min_batch_size``, not its own size."""
    @async_vmap_pool(max_batch_size=8, min_batch_size=4, debug=True)
    def f(x):
        return x * 2

    three = await asyncio.gather(f(1.0), f(2.0), f(3.0))

    compiles = [
        line
        for line in capsys.readouterr().err.splitlines()
        if "compiling batch size" in line
    ]
    assert len(compiles) == 1
    assert "f: compiling batch size 4 (for 3 requests)" in compiles[0]
    assert [float(v) for v in three] == [2.0, 4.0, 6.0]


@pytest.mark.asyncio
async def test_below_min_batches_reuse_one_rounded_entry(capsys):
    """Sub-minimum batches of different sizes share one rounded entry."""
    @async_vmap_pool(max_batch_size=8, min_batch_size=4, debug=True)
    def f(x):
        return x * 2

    three = await asyncio.gather(f(1.0), f(2.0), f(3.0))  # size 4
    two = await asyncio.gather(f(7.0), f(8.0))            # size 4, cached

    compiles = [
        line
        for line in capsys.readouterr().err.splitlines()
        if "compiling batch size" in line
    ]
    assert len(compiles) == 1
    assert "f: compiling batch size 4 (for 3 requests)" in compiles[0]
    assert [float(v) for v in three] == [2.0, 4.0, 6.0]
    assert [float(v) for v in two] == [14.0, 16.0]


@pytest.mark.asyncio
async def test_decorator_debug_flag_reports_every_dispatch(capsys):
    @async_vmap_pool(max_batch_size=4, debug=True)
    def f(x):
        return x * 2

    results = await asyncio.gather(*[f(i) for i in range(6)])

    assert [float(v) for v in results] == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    err = capsys.readouterr().err
    assert "f: executing 4 request(s)" in err
    assert "f: executing 2 request(s)" in err


@pytest.mark.asyncio
async def test_decorator_debug_defaults_to_off(capsys):
    @async_vmap_pool(max_batch_size=4)
    def f(x):
        return x * 2

    await f(1)

    assert capsys.readouterr().err == ""
