import asyncio
import os
import re
import subprocess
import sys
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

    results = await asyncio.gather(f(1.0, {"k": 0.5}), f(2.0, {"k": 0.5}))

    # Trimming back to the real batch size is what proves padding was applied
    # to both arguments and then removed.
    assert [float(v) for v in results] == [10.5, 20.5]


@pytest.mark.asyncio
async def test_padded_pair_form_traces_once_across_batch_sizes():
    traces = []

    @async_vmap_pool(8, pad_to_max=True)
    def f(x, args):
        traces.append(1)
        return x * args["k"]

    first = await asyncio.gather(f(1.0, {"k": 2.0}), f(2.0, {"k": 2.0}))
    second = await asyncio.gather(*[f(float(i), {"k": 2.0}) for i in range(5)])

    assert [float(v) for v in first] == [2, 4]
    assert [float(v) for v in second] == [0, 2, 4, 6, 8]
    # Both batch sizes were padded to 8, so the second reused the first trace.
    assert len(traces) == 1


@pytest.mark.asyncio
async def test_each_leaf_shape_reuses_its_compiled_entry():
    traces = []

    @async_vmap_pool(8, pad_to_max=False)
    def f(x, args):
        traces.append(args["w"].shape[0])
        return x

    await asyncio.gather(f(1.0, {"w": np.ones(2)}))
    await asyncio.gather(f(1.0, {"w": np.ones(3)}))
    await asyncio.gather(f(1.0, {"w": np.ones(2)}))

    # Two distinct shapes compile twice; the repeat of the first reuses it.
    assert traces == [2, 3]


@pytest.mark.asyncio
async def test_pad_to_max_never_recompiles_across_batch_sizes():
    """Varying batch sizes must not recompile the batch glue.

    The stack/pad/trim helpers run on NumPy, so a second -- differently sized
    -- batch must not spawn any new XLA compilations beyond the single compile
    of the vectorised function itself.
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
        @async_vmap_pool(8, pad_to_max=True)
        def f(x):
            return x * 2

        await asyncio.gather(f(1.0), f(2.0))
        await asyncio.gather(*[f(float(i)) for i in range(5)])
    finally:
        logging.getLogger().removeHandler(handler)
        logging.getLogger().setLevel(old_level)
        jax.config.update("jax_log_compiles", old)

    jits = [m for m in compiles if "Compiling jit(" in m]
    assert len(jits) == 1
    assert "jit(f)" in jits[0]


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
    """pad_to_max changes the compiled shape, not the number of callers."""
    def execute(requests):
        return [request[0] for request in requests]

    # The real executor pads, but the pool only ever sees real requests.
    pool = _pool(execute, max_batch_size=8, debug=True, label="padded")
    await pool.submit(1)

    assert "executing 1 request(s)" in capsys.readouterr().err


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
