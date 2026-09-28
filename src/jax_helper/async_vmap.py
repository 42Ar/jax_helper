"""Asynchronous pooled execution of a vectorised multi-argument function.

A call submits one tuple of per-item arguments and awaits its result. Requests
that are already pending when the worker wakes are coalesced into a single
batched computation, so a burst of concurrent calls costs one vectorised
execution instead of N.

The decorated function may take any number of arguments, and each one may be
an arbitrary pytree. Every value in every pytree is free to change between
calls: only the *structure* and *leaf shapes* decide which requests may share
a batch, never the values themselves. Two requests batch together exactly when
stacking them is possible, so free-varying arguments cost nothing.

Batching is *coalescing*: the worker keeps the current batch open while
concurrent callers keep arriving and dispatches it once the event loop has
settled. ``coalescing="quiescent"`` (the default) waits  until every
currently-runnable coroutine has had a turn and none enqueued more, so the
whole burst runs as one batch; ``coalescing="opportunistic"`` yields once and
dispatches whatever is queued. Either way a lone caller runs with no added
latency, and a batch never grows past ``max_batch_size``.

The pooling machinery is plain asyncio and knows nothing about JAX: it takes
injected ``key`` and ``execute`` callables, which is what makes it testable on
its own. The JAX adapter at the bottom of this module supplies both.
"""

from __future__ import annotations

import asyncio
import datetime
import functools
import sys
import weakref
from collections import defaultdict
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    List,
    Optional,
    Sequence,
    Tuple,
    cast,
)

import numpy as np

#: One queued request: a tuple of per-item arguments, and the future awaiting
#: that request's result.
_Request = Tuple[Tuple[Any, ...], "asyncio.Future[Any]"]

#: Maps a request to a hashable key. Requests sharing a key are compatible and
#: may be executed in one batch. A key must be value-insensitive, or varying
#: arguments would fork their own batches and defeat the pooling.
Key = Callable[[Tuple[Any, ...]], Hashable]

#: Executes one compatible group of requests, returning one result per request
#: in the same order.
Execute = Callable[[List[Tuple[Any, ...]]], Sequence[Any]]


def _log(label: Optional[str], message: str) -> None:
    """Print one debug line for the pool, timestamped, to stderr."""
    stamp = datetime.datetime.now().isoformat(sep=" ", timespec="milliseconds")
    print(
        f"[{stamp}] [async_vmap_pool] {label or 'function'}: {message}",
        file=sys.stderr,
    )


class _Pool:
    """Coalesces concurrent submissions into calls to ``execute``.

    A pool owns one queue and one worker task, and is bound to the event loop
    it was created in. Use one pool per loop: sharing one across loops breaks,
    because the futures in a queue are bound to the loop that first used them.
    """

    def __init__(
        self,
        execute: Execute,
        key: Key,
        max_batch_size: int,
        loop: asyncio.AbstractEventLoop,
        debug: bool = False,
        label: str = "",
        coalescing: str = "quiescent",
    ) -> None:
        self._execute = execute
        self._key = key
        self._max_batch_size = max_batch_size
        self._loop = loop
        self._debug = debug
        self._label = label
        self._coalescing = coalescing
        self._queue: "asyncio.Queue[_Request]" = asyncio.Queue()
        self._task: Optional["asyncio.Task[None]"] = None
        self._pending: "set[asyncio.Future[Any]]" = set()

    def _ensure_worker(self) -> None:
        if self._task is None or self._task.done():
            self._task = self._loop.create_task(self._run())
            self._task.add_done_callback(self._on_worker_done)

    async def submit(self, *items: Any) -> Any:
        """Queue one request and await the result its batch produces."""
        self._ensure_worker()
        future: "asyncio.Future[Any]" = self._loop.create_future()
        self._pending.add(future)
        try:
            await self._queue.put((tuple(items), future))
            return await future
        finally:
            self._pending.discard(future)

    async def _run(self) -> None:
        while True:
            request, future = await self._queue.get()
            batch: List[_Request] = [(request, future)]
            if self._coalescing == "quiescent":
                await self._collect_until_settled(batch)
            else:
                await self._collect_immediately(batch)
            self._dispatch_grouped(batch)

    async def _collect_until_settled(self, batch: List[_Request]) -> None:
        """Grow the batch until no other coroutine is ready to enqueue.

        Each turn runs every currently-runnable coroutine, then the batch
        takes whatever arrived. The batch is only dispatched once a full turn
        adds nothing -- the event loop has settled -- or it is full. A lone
        caller is alone in the loop, so the first turn adds nothing and it
        runs with no added latency; a burst is all captured in one batch.
        """
        while len(batch) < self._max_batch_size:
            await asyncio.sleep(0)
            grew = False
            while len(batch) < self._max_batch_size and not self._queue.empty():
                batch.append(self._queue.get_nowait())
                grew = True
            if not grew:
                return

    async def _collect_immediately(self, batch: List[_Request]) -> None:
        """Yield once for ready callers, drain once, and execute right away.

        The opportunistic mode: whatever is queued after a single turn is the
        batch, and anything that enqueues a turn later starts its own batch.
        """
        # Yield once so other ready coroutines can enqueue, then take
        # whatever is already queued. Deliberately no linger: waiting for
        # more would add latency to every caller.
        await asyncio.sleep(0)
        while len(batch) < self._max_batch_size and not self._queue.empty():
            batch.append(self._queue.get_nowait())

    def _dispatch_grouped(self, batch: List[_Request]) -> None:
        """Split the batch into compatible groups and execute each.

        Grouping is what keeps batching working when arguments vary: requests
        that cannot legally share a ``vmap`` are separated, so one odd shape
        costs a single extra dispatch instead of failing everything queued
        alongside it.
        """
        try:
            groups: Dict[Hashable, List[_Request]] = defaultdict(list)
            for request, future in batch:
                groups[self._key(request)].append((request, future))
        except Exception as exc:
            # A key that raises (a leaf that is not array-like) must fail the
            # batch, never the worker.
            self._settle(batch, exc)
            return

        for group in groups.values():
            self._dispatch(group)

    def _report(self, group: List[_Request]) -> None:
        """Log the start of one execution, ahead of running the executor.

        Emitted before the executor runs, so a batch that raises is still
        reported. Goes to stderr to keep it out of a program's own output.
        """
        self._log(group, "executing")

    def _report_done(self, group: List[_Request]) -> None:
        """Log completion of one execution, after the executor has run."""
        self._log(group, "executed")

    def _log(self, group: List[_Request], verb: str) -> None:
        if not self._debug:
            return
        _log(self._label, f"{verb} {len(group)} request(s)")

    def _dispatch(self, group: List[_Request]) -> None:
        self._report(group)
        try:
            results = self._execute([request for request, _ in group])
            if len(results) != len(group):
                raise ValueError(
                    f"executor returned {len(results)} results "
                    f"for a batch of {len(group)}"
                )
        except Exception as exc:
            # Every caller in the group gets the failure. Not re-raised: the
            # group has reported itself, and the worker stays alive to serve
            # the next batch.
            self._report_done(group)
            self._settle(group, exc)
            return

        for (_, future), value in zip(group, results):
            if not future.done():
                future.set_result(value)
        self._report_done(group)

    @staticmethod
    def _settle(group: List[_Request], exc: BaseException) -> None:
        for _, future in group:
            if not future.done():
                future.set_exception(exc)

    def _on_worker_done(self, task: "asyncio.Task[None]") -> None:
        """Settle stranded waiters if the worker died without dispatching.

        The dispatch path only catches ``Exception``. A worker killed by
        something that is not an ``Exception`` (``CancelledError`` propagating
        out of the executor, say) would otherwise leave its waiters hanging
        forever, which is the worst failure mode an async API has.
        """
        failure: BaseException
        if task.cancelled():
            failure = asyncio.CancelledError()
        else:
            exc = task.exception()
            if exc is None:
                return
            failure = exc
        for future in list(self._pending):
            if not future.done():
                future.set_exception(failure)

    async def aclose(self) -> None:
        """Cancel the worker and strand no one."""
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        for future in list(self._pending):
            if not future.done():
                future.cancel()


def _request_key(request: Tuple[Any, ...]) -> Hashable:
    """Compatibility key for a request: its tree structure and its leaf shapes.

    Values are excluded on purpose. A request whose leaves hold different
    numbers batches perfectly well with any other request of the same shape,
    so including values here would fragment the batch and make pooling useless
    for exactly the case it exists to serve.

    The argument count needs no separate term: the request is a tuple, and a
    tuple's ``treedef`` already distinguishes arities, so ``(x,)`` and
    ``(x, y)`` never collide.
    """
    import jax
    import jax.numpy as jnp

    leaves, treedef = jax.tree_util.tree_flatten(request)
    return (treedef, tuple(tuple(jnp.shape(leaf)) for leaf in leaves))


def _build_executor(
    scalar_fn: Callable[..., Any],
    max_batch_size: int,
    debug: bool = False,
    label: Optional[str] = None,
) -> Execute:
    """Compile ``scalar_fn`` under ``jit(vmap(...))`` and return an executor.

    The import is local so that importing this module -- and therefore the
    package, which re-exports :func:`async_vmap_pool` lazily -- does not pull
    in JAX until the decorator is actually used.

    All of a request's arguments travel as one tuple, so ``vmap`` gets a single
    argument and a plain ``in_axes=0``. That keeps the compiled function
    independent of how many arguments a call happens to use: one ``vmap``
    object serves every arity, and it recompiles per distinct structure by
    itself.

    Short batches are zero-padded along axis 0 of every leaf up to the next
    power of two (no more than ``max_batch_size``), so padding never exceeds a
    factor of two and the vectorised function only ever sees a small, bounded
    set of leading dimensions: one per power of two from 2 up to
    ``max_batch_size``. JAX's ``jit`` cache memoises the result, so each
    distinct rounded size is compiled once and reused forever after -- a
    workload with wildly varying batch sizes pays at most
    ``log2(max_batch_size) + 1`` compilations. The glue -- stacking, padding,
    and trimming back to the real batch size -- runs on NumPy arrays, where
    varying sizes are free and never touch the compiler.
    """
    import jax

    @functools.wraps(scalar_fn)
    def unpack(request: Tuple[Any, ...]) -> Any:
        return scalar_fn(*request)

    vmapped = jax.jit(jax.vmap(unpack, in_axes=0))

    def _padded(arr: Any, size: int) -> Any:
        width = [(0, size - arr.shape[0])] + [(0, 0)] * (arr.ndim - 1)
        return np.pad(arr, width)

    compiled_sizes: set = set()

    def execute(requests: List[Tuple[Any, ...]]) -> Sequence[Any]:
        n = len(requests)
        # Round the real batch size up to the next power of two (capped at
        # max_batch_size), so a batch of 100 runs on 128 and one of 1000 on
        # 1000: padding overhead ≤ 2x, and each distinct rounded size is a
        # single compiled entry reused by every batch that rounds to it.
        size = min(1 << (n - 1).bit_length(), max_batch_size)
        # The stack, pad and trim are the batch *glue*, and it deliberately
        # runs on NumPy rather than jnp. The vectorised function below is the
        # only thing that should ever reach the compiler; sizing it with every
        # call's real batch size would make it recompile per distinct `n`.
        #
        # Originally the glue used jnp.stack / jnp.pad / result[:n]. With
        # padded leading dimensions the fused function's signature is stable
        # per rounded size, but the glue still ran on the real `n`, so its ops
        # kept changing shape: _pad compiled once per distinct n, jnp.stack
        # compiled per split of the drain, and slicing a device array with a
        # runtime bound n created a fresh jit(dynamic_slice) on every dispatch
        # - even with an identical signature - flooding the compile logs
        # whenever batch sizes varied. That was tens of XLA compilations per
        # burst of distinct sizes, each around half a second, instead of the
        # handful of cached fused-function entries that matter.
        #
        # NumPy escapes all of that: np.stack / np.pad see plain host arrays,
        # and np.asarray(...)[:n] trims after the XLA result was pulled to the
        # host, so variable sizes never enter JAX dispatch.
        if debug and size not in compiled_sizes:
            compiled_sizes.add(size)
            _log(label, f"compiling batch size {size} (for {n} requests)")
        stacked = jax.tree_util.tree_map(lambda *xs: np.stack(xs), *requests)
        # Leafwise, so pytree arguments, bare scalars, and mixtures of the two
        # all stack correctly: each leaf is stacked against its counterparts.
        if size > n:
            padded = jax.tree_util.tree_map(
                lambda arr: _padded(arr, size), stacked
            )
            # np.asarray blocks on the XLA result and transfers it to the
            # host; the [:n] then slices a plain NumPy array. Results are the
            # real batch size, as plain NumPy arrays.
            return cast(Sequence[Any], np.asarray(vmapped(padded))[:n])
        return cast(Sequence[Any], np.asarray(vmapped(stacked)))

    return execute


def async_vmap_pool(
    max_batch_size: int,
    debug: bool = False,
    coalescing: str = "quiescent",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Turn a JAX scalar function into an asynchronous pooled executor.

    The decorated function may take any number of arguments, each an arbitrary
    pytree, and returns a future resolving to that request's result. Concurrent
    calls are batched into one ``vmap`` execution.

    Values in any pytree may differ freely between calls; batching groups on
    structure and leaf shape only. Every batch is zero-padded along axis 0 of
    every leaf up to the next power of two (never more than
    ``max_batch_size``), so padding adds at most a factor of two of work and
    the vectorised function compiles once per distinct rounded size -- never
    per batch size. Stacking, padding and trimming all happen on NumPy arrays,
    so variable batch sizes never reach the compiler. Each request resolves to
    a NumPy array; results are pulled off the device host-side every batch,
    which is negligible on CPU.

    Args:
        max_batch_size: The maximum number of requests drained into one
            batch: each compatible group within a batch is at most this size,
            and it caps the padded-up leading dimension.
        debug: If True, print a line to stderr for every execution, naming the
            function and the number of requests in that batch, plus a line
            whenever a new batch size is compiled. Useful for confirming that
            concurrent calls really are coalescing and that sizes are shared.
        coalescing: How long the worker keeps a batch open while collecting.
            ``"quiescent"`` (the default) waits until every currently-runnable
            coroutine has had a turn and none enqueued more, so all batched
            callers are captured by one execution; ``"opportunistic"`` yields
            once and dispatches whatever is queued, so callers that arrive a
            turn later run as their own batch. A lone caller runs with no
            added latency under both.

    Returns:
        A decorator producing an async function that awaits to its result.

    The pooled state is per event loop: each loop that calls the decorated
    function gets its own queue and worker, so the function is usable from
    several loops (or from several tests) without them interfering.

    Raises:
        TypeError: If ``max_batch_size`` is not an ``int``.
        ValueError: If ``max_batch_size`` is less than 1, or ``coalescing``
            is not ``"quiescent"`` or ``"opportunistic"``.
    """
    if not isinstance(max_batch_size, int):
        raise TypeError(
            f"max_batch_size must be an int, got {type(max_batch_size).__name__}"
        )
    if max_batch_size < 1:
        raise ValueError(
            f"max_batch_size must be at least 1, got {max_batch_size}"
        )
    if coalescing not in ("quiescent", "opportunistic"):
        raise ValueError(
            "coalescing must be 'quiescent' or 'opportunistic', "
            f"got {coalescing!r}"
        )

    def decorator(scalar_fn: Callable[..., Any]) -> Callable[..., Any]:
        execute = _build_executor(
            scalar_fn,
            max_batch_size,
            debug=debug,
            label=getattr(scalar_fn, "__name__", ""),
        )

        # Keyed by the running loop, so each loop gets a private pool. Weak, so
        # a finished loop's pool is collected with it.
        pools: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _Pool]"
        pools = weakref.WeakKeyDictionary()

        @functools.wraps(scalar_fn)
        async def wrapper(*items: Any) -> Any:
            if not items:
                raise ValueError(
                    "at least one argument is required: an empty request has "
                    "no array to map over"
                )
            loop = asyncio.get_running_loop()
            pool = pools.get(loop)
            if pool is None:
                pool = _Pool(
                    execute,
                    _request_key,
                    max_batch_size,
                    loop,
                    debug=debug,
                    label=getattr(scalar_fn, "__name__", ""),
                    coalescing=coalescing,
                )
                pools[loop] = pool
            return await pool.submit(*items)

        return wrapper

    return decorator
