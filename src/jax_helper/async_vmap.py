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

Batching is *coalescing*: the worker keeps the current batch open and
dispatches it once every live task is parked awaiting a result from this pool
and nothing new is queued -- the whole program is standing on the pool, so no
more requests can arrive until one of the batches executes -- and, if set, the
batch holds at least ``min_batch_size`` requests. A caller that is still
computing (or still on an earlier await) is waited for, so a burst of
concurrent calls -- even ones that reach the ``submit`` at different moments
-- runs as one execution. ``coalescing="parked"`` is the only mode. A batch
never grows past ``max_batch_size``, and a lone caller runs with no added
latency: with no one else active, the first quiet turn dispatches it.

With ``padding="up"`` (the default) a batch is zero-padded along axis 0 of
every leaf to the next power of two, so one execution serves the whole batch
at the cost of at most a factor of two of compute. With ``padding="down"`` a
batch instead runs at the largest power-of-two prefix and the remainder is
shifted to the next batch, so no request is ever padded, at the cost of extra
executions -- and only when both the prefix and the shift leave at least
``min_batch_size`` requests each, so a split never leaves a sub-minimum batch
that would recompile a small shape. In either mode a batch dispatched below
``min_batch_size`` is padded up to it, so the compiled leading dimension
never dips below the minimum: a lone call or a wave leftover shares one
compiled entry with every other sub-minimum batch that rounds to the same
size instead of recompiling a small shape. Stacking, padding and trimming
happen on NumPy arrays off the compiler either way.

By default each batch runs on a dedicated worker thread, so the event loop is
never frozen while NumPy or JAX compute runs; the GIL-releasing C work there
lets the loop keep servicing callers -- the ones becoming ready mid-batch
submit straight into the next batch's queue -- at the cost of one worker
thread per pool. Pass ``run_in_thread=False`` to run the batch inline in the
worker task instead: the loop is then busy for the duration of the run, but no
thread is created and nothing crosses a thread boundary.

The pooling machinery is plain asyncio and knows nothing about JAX: it takes
injected ``key`` and ``execute`` callables, which is what makes it testable on
its own. The JAX adapter at the bottom of this module supplies both.
"""

from __future__ import annotations

import asyncio
import datetime
import functools
import importlib
import sys
import weakref
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
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


class _Close:
    """Sentinel type slipped onto the queue by :meth:`_Pool.aclose` to stop
    the worker after any in-flight batch has finished. Never a real request:
    every submit wraps its arguments in a tuple, so a sentinel is
    unambiguous."""

    __slots__ = ()


#: The pool's close sentinel, also used to type the queue items.
_CLOSE = _Close()

#: Item on the pool queue: either a request, or the close sentinel.
_QueueItem = Union[_Request, _Close]

#: Number of consecutive turns with nothing queued and something still not
#: parked on the pool before the worker dispatches anyway. A safety valve so a
#: task parked on non-pool work (a listener, an I/O loop, an unrelated sleep)
#: can never starve a batch waiting for the "everyone is parked" condition.
_SETTLE_TURNS = 10

#: The pool never runs more than one batch at a time, so a single worker
#: thread is all it can ever use.
_MAX_WORKERS = 1


def _gathering_future_type() -> Optional[type]:
    """The asyncio type a ``gather`` parent parks on.

    The parent of a batch of concurrent calls usually waits on an internal
    gathering future rather than on the pool's futures, yet it cannot enqueue
    anything itself: its children do the submitting and park on pool futures
    directly. The type is used by :meth:`_Pool._all_parked`, which insists the
    gathering actually reach this pool before counting the parent as parked.
    The class lives in ``asyncio.tasks`` on 3.14+ and ``asyncio.futures``
    before that; both are internal, like ``Task._fut_waiter`` itself.
    """
    for modname in ("asyncio.tasks", "asyncio.futures"):
        mod = importlib.import_module(modname)
        cls = getattr(mod, "_GatheringFuture", None)
        if cls is not None:
            return cls
    return None


#: Cached orchestrator type for :meth:`_Pool._all_parked`.
_GATHERING_FUTURE = _gathering_future_type()


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
        coalescing: str = "parked",
        threaded: bool = True,
        padding: str = "up",
        min_batch_size: int = 1,
    ) -> None:
        self._execute = execute
        self._key = key
        self._max_batch_size = max_batch_size
        self._loop = loop
        self._debug = debug
        self._label = label
        self._coalescing = coalescing
        self._threaded = threaded
        self._padding = padding
        self._min_batch_size = min_batch_size
        self._queue: "asyncio.Queue[_QueueItem]" = asyncio.Queue()
        self._task: Optional["asyncio.Task[None]"] = None
        self._threadpool: Optional[ThreadPoolExecutor] = None
        self._pending: "set[asyncio.Future[Any]]" = set()
        self._closing = False

    def _ensure_worker(self) -> None:
        if self._task is None or self._task.done():
            self._closing = False
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
            item = await self._queue.get()
            if isinstance(item, _Close):
                break
            request, future = item
            batch: List[_Request] = [(request, future)]
            if not self._closing:
                await self._collect_until_all_parked(batch)
            await self._dispatch_grouped(batch)

    async def _collect_until_all_parked(self, batch: List[_Request]) -> None:
        """Grow the batch while any caller could still submit.

        Waits until every live task is parked awaiting one of this pool's
        futures and nothing new is queued: at that point no request can reach
        the pool until a batch executes, so the batch is complete. A caller
        still computing is *not* parked, so it is waited for rather than
        missed -- no turn-counting, no fixed linger. A batch is not dispatched
        while it is below ``min_batch_size``, so a preference for full batches
        holds until the safety valve gives up.

        Two turn-based safety valves bound the wait: the batch is full, or the
        queue has been empty for a few turns while the batch is still below
        ``min_batch_size`` or something is not parked on the pool. The latter
        keeps a background task that parks on unrelated work (a listener, an
        I/O loop) from starving the batch.
        """
        idle_turns = 0
        while len(batch) < self._max_batch_size and not self._closing:
            # One turn lets every currently-runnable task reach its submit.
            await asyncio.sleep(0)
            grew = False
            while len(batch) < self._max_batch_size and not self._queue.empty():
                item = self._queue.get_nowait()
                if isinstance(item, _Close):
                    # aclose slipped in mid-drain: leave it for the worker's
                    # own get() so it still stops the loop, and dispatch what
                    # we have.
                    self._queue.put_nowait(item)
                    break
                batch.append(item)
                grew = True
            if grew:
                idle_turns = 0
                continue
            if len(batch) >= self._min_batch_size and self._all_parked():
                return
            idle_turns += 1
            if idle_turns >= _SETTLE_TURNS:
                # Safety valve: either something stayed unparked (a listener,
                # an I/O loop, a caller still computing) past the settle
                # budget, or the batch is simply below ``min_batch_size``. In
                # debug, say who and what they are parked on -- it is how you
                # spot the task the batch could have waited for.
                details = self._describe_unparked()
                bits = (
                    [f"{len(details)} task(s) were not parked on the pool: "
                     + "; ".join(details)]
                    if details
                    else []
                )
                if self._min_batch_size > 1 and len(batch) < self._min_batch_size:
                    bits.append(
                        f"{len(batch)} request(s) below min_batch_size "
                        f"{self._min_batch_size}"
                    )
                detail = ("; " + "; ".join(bits)) if bits else ""
                _log(
                    self._label,
                    "dispatching after "
                    f"{_SETTLE_TURNS} idle turns with {len(batch)} "
                    f"request(s){detail}",
                )
                return

    def _gather_depends_on_pool(self, gather_future: Any) -> bool:
        """Whether a ``gather``, transitively, is waiting on this pool.

        A gathering future's children (the per-item tasks ``gather`` created)
        wait on pool futures directly when the gather includes calls into this
        pool. Only then may the orchestrating parent count as parked: it can
        enqueue nothing itself, and its next submit happens after this pool
        resolves, so dispatching is safe. A gather over unrelated work is a
        caller still computing, not a parked one.
        """
        for child in getattr(gather_future, "_children", ()):
            waiter = getattr(child, "_fut_waiter", None)
            if waiter in self._pending:
                return True
            if _GATHERING_FUTURE is not None and isinstance(
                waiter, _GATHERING_FUTURE
            ):
                if self._gather_depends_on_pool(waiter):
                    return True
        return False

    def _all_parked(self) -> bool:
        """True when no live task can enqueue without a batch executing first."""
        return not self._unparked_tasks()

    def _unparked_tasks(self) -> List["asyncio.Task[Any]"]:
        """The live tasks that are not parked on this pool.

        A task is parked on this pool when the future it awaits is one of the
        futures handed out by :meth:`submit`. ``asyncio.Task._fut_waiter`` is
        a private field but stable across CPython versions. A task awaiting a
        ``gather`` counts as parked only if that gather transitively waits on
        this pool (:meth:`_gather_depends_on_pool`): the orchestrating parent
        parks on an internal gathering future and cannot submit, and its
        children -- which park on pool futures directly -- already count.
        """
        unparked: List["asyncio.Task[Any]"] = []
        for task in asyncio.all_tasks(self._loop):
            if task is self._task or task.done():
                continue
            waiter = getattr(task, "_fut_waiter", None)
            if waiter in self._pending:
                continue
            if _GATHERING_FUTURE is not None and isinstance(
                waiter, _GATHERING_FUTURE
            ):
                if self._gather_depends_on_pool(waiter):
                    continue
            unparked.append(task)
        return unparked

    def _describe_unparked(self) -> List[str]:
        """Name each unparked task and what it is currently parked on.

        Future types are mostly opaque, so the practical clue is which
        coroutine is involved and what its waiter is: a not-yet-started task,
        another task being awaited, a ``gather`` over unrelated work, or a
        future class. The names feed the settle-valve debug line.
        """
        descriptions: List[str] = []
        for task in self._unparked_tasks():
            coro = task.get_coro()
            name = getattr(coro, "__qualname__", None) or f"task {task.get_name()}"
            waiter = getattr(task, "_fut_waiter", None)
            if waiter is None:
                descriptions.append(f"{name} not started yet")
            elif isinstance(waiter, asyncio.Task):
                awaited = getattr(
                    waiter.get_coro(), "__qualname__", "another task"
                )
                descriptions.append(f"{name} awaiting {awaited}")
            elif _GATHERING_FUTURE is not None and isinstance(
                waiter, _GATHERING_FUTURE
            ):
                descriptions.append(f"{name} awaiting an unrelated gather")
            else:
                descriptions.append(
                    f"{name} parked on {type(waiter).__name__}"
                )
        return descriptions

    async def _dispatch_grouped(self, batch: List[_Request]) -> None:
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
            n = len(group)
            if (
                self._padding == "down"
                and n > 1
                and (n & (n - 1)) != 0
            ):
                # ``padding="down"``: run the largest power-of-two prefix of
                # the group and shift the remainder to the next batch, so no
                # request is ever zero-padded. Only split when both the prefix
                # and the remainder reach ``min_batch_size``; otherwise the
                # whole group runs and is padded as usual. Requiring the
                # remainder to reach the minimum too stops the split leaving a
                # sub-min batch that would run alone and recompile a small
                # shape (e.g. 68 with min 64 must not become 64 + 4).
                floor = 1 << (n.bit_length() - 1)
                if (
                    floor >= self._min_batch_size
                    and (n - floor) >= self._min_batch_size
                ):
                    head, tail = group[:floor], group[floor:]
                    for item in tail:
                        self._queue.put_nowait(item)
                    await self._dispatch(head)
                    continue
            await self._dispatch(group)

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

    async def _dispatch(self, group: List[_Request]) -> None:
        self._report(group)
        requests = [request for request, _ in group]
        try:
            if self._threaded:
                # Run the batch on a dedicated worker thread so the event loop
                # is not frozen by NumPy/JAX compute; the GIL-releasing C work
                # there lets the loop keep servicing callers in parallel.
                if self._threadpool is None:
                    self._threadpool = ThreadPoolExecutor(
                        max_workers=_MAX_WORKERS
                    )
                results = await self._loop.run_in_executor(
                    self._threadpool, self._execute, requests
                )
            else:
                # Inline in the worker task: the loop is busy for the run, but
                # no thread is created and no values move across a thread
                # boundary.
                results = self._execute(requests)
            if len(results) != len(group):
                raise ValueError(
                    f"executor returned {len(results)} results "
                    f"for a batch of {len(group)}"
                )
        except Exception as exc:
            # Every caller in the group gets the failure. Not re-raised: the
            # group has reported itself, and the worker stays alive to serve
            # the next batch. ``CancelledError`` is a ``BaseException`` and
            # propagates to ``_on_worker_done``, which streams it to waiters.
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
        """Stop the worker after any in-flight batch, stranding no one.

        The sentinel ensures a batch already executing on the worker thread is
        finished (and its results delivered) before the worker exits, so a
        close never drops a result.
        """
        if self._task is None:
            return
        self._closing = True
        await self._queue.put(_CLOSE)
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        for future in list(self._pending):
            if not future.done():
                future.cancel()
        if self._threadpool is not None:
            self._threadpool.shutdown(wait=True)


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
    min_batch_size: int = 1,
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
    power of two (no more than ``max_batch_size``) -- and never below the next
    power of two of ``min_batch_size``, so a below-minimum batch that is
    dispatched is padded up to the minimum rather than compiling a small
    shape. Padding therefore never exceeds a factor of two and the vectorised
    function only ever sees a small, bounded set of leading dimensions: one
    per power of two from ``min_batch_size`` up to ``max_batch_size`` (or from
    1 when the minimum is left at its default). JAX's ``jit`` cache memoises
    the result, so each distinct rounded size is compiled once and reused
    forever after -- a workload with wildly varying batch sizes pays at most
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
        # single compiled entry reused by every batch that rounds to it. The
        # floor is never below ``min_batch_size``: a below-minimum batch that
        # gets dispatched (a wave leftover, a lone caller) is padded up to the
        # minimum instead of compiling a small shape, and every sub-minimum
        # batch that rounds there shares that one compiled entry.
        size = min(
            1 << (max(n, min_batch_size) - 1).bit_length(), max_batch_size
        )
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
    coalescing: str = "parked",
    run_in_thread: bool = True,
    padding: str = "up",
    min_batch_size: int = 1,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Turn a JAX scalar function into an asynchronous pooled executor.

    The decorated function may take any number of arguments, each an arbitrary
    pytree, and returns a future resolving to that request's result. Concurrent
    calls are batched into one ``vmap`` execution.

    Values in any pytree may differ freely between calls; batching groups on
    structure and leaf shape only. With ``padding="up"`` (the default) every
    batch is zero-padded along axis 0 of every leaf up to the next power of
    two (never more than ``max_batch_size``), so padding adds at most a factor
    of two of work and the vectorised function compiles once per distinct
    rounded size -- never per batch size. With ``padding="down"`` a batch is
    instead run at the largest power-of-two prefix and the rest is shifted to
    the next batch, so no request is ever padded. Stacking, padding and
    trimming all happen on NumPy arrays, so variable batch sizes never reach
    the compiler. Each request resolves to a NumPy array; results are pulled
    off the device host-side every batch, which is negligible on CPU.

    Args:
        max_batch_size: The maximum number of requests drained into one
            batch: each compatible group within a batch is at most this size,
            and it caps the padded-up leading dimension.
        debug: If True, print a line to stderr for every execution, naming the
            function and the number of requests in that batch, plus a line
            whenever a new batch size is compiled. Useful for confirming that
            concurrent calls really are coalescing and that sizes are shared.
        coalescing: How the worker decides a batch is complete. ``"parked"``
            (the default, and the only mode) waits until every live task is
            parked awaiting a result from this pool and nothing new is queued,
            so a burst of concurrent calls -- even ones reaching the decorator
            at slightly different moments -- runs as one ``vmap`` execution
            rather than one per arrival. A lone caller runs with no added
            latency.
        run_in_thread: If True (the default), execute each batch on a dedicated
            worker thread (one thread per pool), so the event loop is never
            blocked by the vectorised run; NumPy and JAX release the GIL during
            their C compute, letting the loop keep servicing callers. If False,
            the batch runs inline in the worker task and the loop is busy for
            the duration of the run, but no thread is created and nothing
            crosses a thread boundary. The batch glue only passes values in and
            out, so threading is safe either way.
        padding: How a batch whose size is not a power of two is handled.
            ``"up"`` (the default) zero-pads the batch up to the next power of
            two, so one execution serves the whole batch at the cost of at most
            a factor of two of compute. ``"down"`` runs the largest power-of-two
            prefix of the batch and shifts the remaining requests to the next
            batch, so nothing is ever padded, at the cost of extra executions.
            A batch is split only when both the prefix and the shifted
            remainder reach ``min_batch_size``, so a split never leaves a
            sub-minimum batch behind.
        min_batch_size: The minimum number of requests a batch may hold before
            the worker dispatches it, when the parked condition is met.
            Defaults to 1, which waits only for every caller to be parked; a
            larger value makes the pool hold small batches until more requests
            arrive or the settle valve gives up, and gates the ``"down"``
            padding split. It also floors the compiled leading dimension: a
            dispatched batch of fewer than ``min_batch_size`` requests is
            padded up to the next power of two of the minimum, so a wave
            leftover or a lone call neither recompiles a small shape nor shares
            nothing -- it reuses the rounded entry like any other batch.

    Returns:
        A decorator producing an async function that awaits to its result.

    The pooled state is per event loop: each loop that calls the decorated
    function gets its own queue and worker, so the function is usable from
    several loops (or from several tests) without them interfering.

    Raises:
        TypeError: If ``max_batch_size`` or ``min_batch_size`` is not an
            ``int``, or ``run_in_thread`` is not a ``bool``.
        ValueError: If ``max_batch_size`` is less than 1, ``min_batch_size``
            is less than 1 or greater than ``max_batch_size``, ``coalescing``
            is not ``"parked"``, or ``padding`` is not ``"up"`` or ``"down"``.
    """
    if not isinstance(max_batch_size, int):
        raise TypeError(
            f"max_batch_size must be an int, got {type(max_batch_size).__name__}"
        )
    if max_batch_size < 1:
        raise ValueError(
            f"max_batch_size must be at least 1, got {max_batch_size}"
        )
    if coalescing != "parked":
        raise ValueError(
            f"coalescing must be 'parked', got {coalescing!r}"
        )
    if not isinstance(run_in_thread, bool):
        raise TypeError(
            f"run_in_thread must be a bool, got {type(run_in_thread).__name__}"
        )
    if padding not in ("up", "down"):
        raise ValueError(
            f"padding must be 'up' or 'down', got {padding!r}"
        )
    if not isinstance(min_batch_size, int):
        raise TypeError(
            f"min_batch_size must be an int, got {type(min_batch_size).__name__}"
        )
    if min_batch_size < 1:
        raise ValueError(
            f"min_batch_size must be at least 1, got {min_batch_size}"
        )
    if min_batch_size > max_batch_size:
        raise ValueError(
            f"min_batch_size ({min_batch_size}) cannot exceed "
            f"max_batch_size ({max_batch_size})"
        )

    def decorator(scalar_fn: Callable[..., Any]) -> Callable[..., Any]:
        execute = _build_executor(
            scalar_fn,
            max_batch_size,
            debug=debug,
            label=getattr(scalar_fn, "__name__", ""),
            min_batch_size=min_batch_size,
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
                    threaded=run_in_thread,
                    padding=padding,
                    min_batch_size=min_batch_size,
                )
                pools[loop] = pool
            return await pool.submit(*items)

        return wrapper

    return decorator
