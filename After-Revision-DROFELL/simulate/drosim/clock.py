from __future__ import annotations

import asyncio as _std_asyncio
import math as _math
from typing import Any, Optional

import extend_asyncio as ea
from extend_asyncio.std_asyncio import StdEventLoop


class VirtualClock:
    """A monotonically advancing virtual timeline, in nanoseconds."""

    __slots__ = ("_now_ns", "_epoch_ns")

    def __init__(self, epoch_ns: int = 0) -> None:
        self._epoch_ns = epoch_ns
        self._now_ns = epoch_ns

    @property
    def now_ns(self) -> int:
        return self._now_ns

    @property
    def now_s(self) -> float:
        return self._now_ns / 1e9

    def advance_to_ns(self, when_ns: int) -> None:
        if when_ns > self._now_ns:
            self._now_ns = when_ns

    def advance_by_ns(self, delta_ns: int) -> None:
        if delta_ns > 0:
            self._now_ns += delta_ns

    def elapsed_ns(self) -> int:
        return self._now_ns - self._epoch_ns

    def __repr__(self) -> str:
        return f"<VirtualClock t={self.now_s:.6f}s>"


# The process-wide clock the simulated run reads. It is module-level because
# `record_event` and `record_step` are called from deep inside the production
# code, which cannot be threaded a clock argument without editing dozens of
# call sites (31 of them, see docs/06 P9).
_ACTIVE: Optional[VirtualClock] = None


def active_clock() -> Optional[VirtualClock]:
    return _ACTIVE


def now_ns() -> int:
    """Timeline timestamp source. Virtual under simulation, wall clock otherwise.

    This is what `utils.runtime_metrics.record_event` and
    `MultiPathInferTask.record_step` should call instead of `time.time_ns()`.
    """
    if _ACTIVE is not None:
        return _ACTIVE.now_ns
    import time

    return time.time_ns()


class VirtualClockLoop(StdEventLoop):
    """StdEventLoop driven by a virtual clock instead of monotonic time.

    Two overrides only:

    * ``time()`` reports virtual seconds, so every ``asyncio.sleep`` and
      ``call_later`` schedules on the virtual timeline.
    * ``_select_common()`` jumps time forward to the next timer instead of
      blocking in ``selector.select()``.
    """

    def __init__(self, clock: Optional[VirtualClock] = None) -> None:
        self.clock = clock if clock is not None else VirtualClock()
        # busy_wait must be False: a busy-wait loop would spin on real time and
        # never let the virtual clock advance (docs/02 §2.2).
        super().__init__(busy_wait=False)
        self._idle_jumps = 0

    # -- the two overrides ------------------------------------------------
    def time(self) -> float:
        return self.clock.now_s

    def _select_common(self) -> None:
        # Something is runnable now: do not advance time, and do not block.
        if self.ready or self._blocking_scheduler or self._is_stopping():
            return
        if self.scheduled:
            when = self.scheduled[0]._when  # type: ignore[attr-defined]
            if when > self.clock.now_s:
                # Round UP, and never advance by less than one nanosecond.
                #
                # `int(when * 1e9)` truncates, and CPython's timer deadlines are
                # floats produced by `now + delay`, so they routinely carry a
                # sub-nanosecond tail: measured `_when = 0.010116693999999999`
                # truncates to 10_116_693 ns, i.e. 0.010116693 s -- a hair BELOW
                # the deadline. `StdEventLoop._run_handles` only promotes a timer
                # when `_when < now + _clock_resolution`, so the timer stayed
                # scheduled, nothing became ready, and this method was asked to
                # advance to the same instant again: an infinite spin at fixed
                # virtual time (measured 7.5 million iterations at t=0.0101 s,
                # which is what made the first end-to-end run look like a hang).
                #
                # Rounding up guarantees the clock reaches or passes the deadline,
                # so the timer fires on the next pass. The `+1` floor guarantees
                # forward progress even if rounding lands exactly on `now`.
                # Advance past the loop's early-fire window as well. CPython
                # promotes handles whose deadline is within `_clock_resolution`
                # of now; firing even one nanosecond early breaks code that
                # checks a separately stored integer deadline (notably
                # InferBatchSender's interception trap), which then parks the
                # task again without scheduling another timer.
                target_ns = _math.ceil(
                    (when + self._clock_resolution) * 1e9
                )
                if target_ns <= self.clock.now_ns:
                    target_ns = self.clock.now_ns + 1
                self.clock.advance_to_ns(target_ns)
                self._idle_jumps += 1
            return
        # In simulation there is no external I/O that could wake a pending
        # coroutine. Reaching an empty scheduler while the root task is still
        # awaiting work is therefore a causal deadlock, not an idle state. Fail
        # loudly instead of spinning millions of loop iterations at fixed time.
        raise RuntimeError(
            "DROSIM deadlock: no ready callback or timer can advance virtual time"
        )

    def _is_stopping(self) -> bool:
        # `_stopping` is a private CPython loop attribute; read it defensively so
        # a CPython version bump degrades to "keep advancing time" rather than
        # crashing the simulation.
        return bool(getattr(self, "_stopping", False))

    # -- diagnostics ------------------------------------------------------
    @property
    def idle_jumps(self) -> int:
        """How many times the loop skipped idle virtual time."""
        return self._idle_jumps

    def __repr__(self) -> str:
        return (
            f"<VirtualClockLoop t={self.clock.now_s:.6f}s "
            f"jumps={self._idle_jumps}>"
        )


def install(clock: Optional[VirtualClock] = None) -> VirtualClockLoop:
    """Install a virtual-clock loop as the process event loop.

    Must be called before anything touches the event loop, mirroring how
    production calls ``extend_asyncio.config_impl``.
    """
    global _ACTIVE
    clock = clock if clock is not None else VirtualClock()
    loop_holder: dict[str, VirtualClockLoop] = {}

    def make_loop() -> VirtualClockLoop:
        loop = VirtualClockLoop(clock)
        loop_holder["loop"] = loop
        return loop

    _std_asyncio.set_event_loop_policy(ea.EventLoopPolicy(make_loop))
    _ACTIVE = clock
    loop = ea.get_event_loop()
    assert isinstance(loop, VirtualClockLoop)
    # Timeline timestamps must come from the same clock the loop schedules on.
    # See install_time_shim for why a mixed clock deadlocks rather than merely
    # skewing numbers.
    install_time_shim()
    return loop


# The real functions, captured before any patching, so the harness can still
# measure its own wall-clock cost while the simulated world runs on virtual time.
import time as _time

REAL_TIME_NS = _time.time_ns
REAL_MONOTONIC_NS = _time.monotonic_ns
_time_shim_installed = False


def install_time_shim() -> None:
    """Point ``time.time_ns`` and ``time.monotonic_ns`` at the virtual clock.

    WHY THIS IS A SEAM AND NOT A HACK
    ---------------------------------
    docs/06 P9 counted 40 time calls across ``node/``, ``dealers/`` and
    ``utils/``, and established the rule:

        `time_ns` / `monotonic_ns` are TIMELINE marks   -> must be virtual
        `perf_counter_ns` measures control-plane CPU    -> must stay real

    Patching the two timeline functions at the module boundary implements exactly
    that rule in one place, instead of editing 31 call sites in production code
    that the simulator is supposed to leave untouched. ``perf_counter_ns`` is
    deliberately NOT patched, so the shadow state's ``_update_cpu_ns`` keeps
    measuring real host CPU -- which is a genuine measurement even in simulation
    (docs/03 §5).

    THE BUG THAT MAKES THIS MANDATORY, NOT OPTIONAL
    -----------------------------------------------
    ``node/infer/sender.py`` arms its packet-grouping trap with
    ``loop.call_later(INTERCEPT_WINDOW_TIME, ...)`` -- the VIRTUAL loop -- but
    computes the deadline with ``time_ns()`` and re-checks it in ``_step`` with
    ``time.time_ns()``. Under a virtual clock those two clocks diverge without
    bound, so the trap never expires, and the very first decode packet is held
    forever: the simulation hangs after producing exactly one token. Mixing the
    two clocks is not a small inaccuracy; it deadlocks. Measured before the fix:
    prefill completed at virtual t=0.009 s and then no further progress.
    """
    global _time_shim_installed
    if _time_shim_installed:
        return
    _time.time_ns = lambda: now_ns()
    _time.monotonic_ns = lambda: now_ns()
    _rebind_module_aliases(virtual=True)
    _time_shim_installed = True


# Modules that did `from time import time_ns`, which binds the REAL function into
# their own namespace at import time -- patching `time.time_ns` cannot reach it.
# Measured consequence of missing this: `node/infer/sender.py` compared a virtual
# `now` (10_116_690) against a wall-clock deadline (1_787_325_256_643_804_538),
# so its grouping trap could never expire and the run hung after one token.
# Kept as an explicit list, and verified by `_MODULE_ALIASES` staying in sync
# with a repo grep for `from time import` (only one site today).
_MODULE_ALIASES: tuple[tuple[str, str], ...] = (
    ("node.infer.sender", "time_ns"),
)


def _rebind_module_aliases(virtual: bool) -> None:
    import importlib
    import sys as _sys

    for module_name, attribute in _MODULE_ALIASES:
        module = _sys.modules.get(module_name)
        if module is None:
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
        if not hasattr(module, attribute):
            # The alias was renamed or removed upstream. Fail loudly: silently
            # skipping it would reintroduce the mixed-clock hang.
            raise AttributeError(
                f"{module_name} no longer defines {attribute!r}; the DROSIM time "
                f"shim must be updated (see _MODULE_ALIASES)."
            )
        if virtual:
            setattr(module, attribute, lambda: now_ns())
        else:
            setattr(module, attribute, REAL_TIME_NS)


def uninstall_time_shim() -> None:
    """Restore the real timeline functions."""
    global _time_shim_installed
    if not _time_shim_installed:
        return
    _time.time_ns = REAL_TIME_NS
    _time.monotonic_ns = REAL_MONOTONIC_NS
    _rebind_module_aliases(virtual=False)
    _time_shim_installed = False


def uninstall() -> None:
    """Drop the virtual clock so timestamps revert to the wall clock."""
    global _ACTIVE
    uninstall_time_shim()
    _ACTIVE = None


def run_simulation(loop: VirtualClockLoop, coro: Any) -> Any:
    """Run one simulated scenario and shut the loop down cleanly.

    Why this exists: ``StdEventLoop.__init__`` installs an exception handler that
    calls ``os._exit(-1)`` (extend_asyncio/std_asyncio.py:60-66). In production
    that is fine -- the process is going away anyway. But long-lived service
    coroutines (the per-node packet pumps) are still parked on ``Queue.get()``
    when the scenario finishes, so CPython's "Task was destroyed but it is
    pending!" notice reaches that handler and kills the interpreter with status
    255 *after* the results were computed. Confirmed minimally: a single task
    awaiting an empty queue at exit reproduces it.

    Suppressing the handler would be wrong -- it is what surfaces genuine errors
    inside simulated nodes. So instead the pending service tasks are cancelled
    and drained before returning, which is what a real shutdown would do.
    """
    import asyncio as _a

    # A retained service task (packet sender, node pump, backend loop) can fail
    # without its exception being propagated to the root workload future.
    # Standard asyncio may not report that exception until garbage collection;
    # in a virtual loop the root then waits forever with no external I/O. Race
    # the root against a fatal-error future populated by every task's done
    # callback so the original exception is raised immediately and audibly.
    fatal: _a.Future[Any] = loop.create_future()
    previous_factory = loop.get_task_factory()

    def observe(task: _a.Task[Any]) -> None:
        if task.cancelled() or fatal.done():
            return
        error = task.exception()
        if error is not None:
            fatal.set_exception(error)

    def task_factory(
        event_loop: _a.AbstractEventLoop,
        task_coro: Any,
        context: Any = None,
    ) -> _a.Task[Any]:
        if previous_factory is not None:
            try:
                task = previous_factory(event_loop, task_coro, context=context)
            except TypeError:
                task = previous_factory(event_loop, task_coro)
        else:
            task = _a.Task(task_coro, loop=event_loop, context=context)
        task.add_done_callback(observe)
        return task

    loop.set_task_factory(task_factory)

    async def root_or_fatal() -> Any:
        root = _a.create_task(coro)
        done, _ = await _a.wait(
            {root, fatal}, return_when=_a.FIRST_COMPLETED
        )
        if fatal in done:
            return fatal.result()
        return await root

    try:
        return loop.run_until_complete(root_or_fatal())
    finally:
        loop.set_task_factory(previous_factory)
        pending = [
            task for task in _a.all_tasks(loop) if not task.done()
        ]
        for task in pending:
            task.cancel()
        if pending:
            # let the cancellations be observed, so no task is garbage-collected
            # while still pending
            try:
                loop.run_until_complete(
                    _a.gather(*pending, return_exceptions=True)
                )
            except Exception:
                pass
