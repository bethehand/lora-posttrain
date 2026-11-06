"""Run sandbox work in a pool of processes, each holding the dev panel in memory.

A report or a score takes around a second of CPU, and an RL step asks for hundreds at once.
Run on the event loop they would queue behind each other and stall the rollouts.
"""
from __future__ import annotations

import asyncio
import multiprocessing
import os
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

# One BLAS thread per process: the pool is the parallelism, and the server is shared.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

_box = None
_pool: ProcessPoolExecutor | None = None


def _exit_with_parent() -> None:
    multiprocessing.parent_process().join()
    os._exit(0)


def _start_worker() -> None:
    global _box
    # The framework may kill the process that owns this pool without running its exit hooks. An
    # orphaned worker would wait for work forever and keep the whole run from exiting.
    threading.Thread(target=_exit_with_parent, daemon=True).start()
    from factor_idea.minutes import load_minutes
    from factor_idea.panel import load_dev
    from factor_idea.sandbox import Sandbox

    # The task is daily bars only. FACTOR_IDEA_MINUTES turns on the minute pilot; see results/factor/minute_pilot.md.
    _box = Sandbox(load_dev(), minutes=load_minutes() if os.environ.get("FACTOR_IDEA_MINUTES") else None)


def _report(text: str) -> str:
    return _box.report(text)


def _score(text: str, key: str, direction: int, also: tuple[str, ...]):
    return _box.score(text, key, direction, also=also)


def _executor() -> ProcessPoolExecutor:
    global _pool
    if _pool is None:
        _pool = ProcessPoolExecutor(
            max_workers=int(os.environ.get("FACTOR_IDEA_WORKERS", "8")),
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_start_worker,
        )
    return _pool


async def _run(function, *args):
    """One killed worker breaks a pool for good, and every later call would fail with it, so start another."""
    global _pool
    loop = asyncio.get_running_loop()
    pool = _executor()
    try:
        return await loop.run_in_executor(pool, function, *args)
    except BrokenProcessPool:
        if _pool is pool:
            _pool = None
        return await loop.run_in_executor(_executor(), function, *args)


async def report(text: str) -> str:
    return await _run(_report, text)


async def score(text: str, key: str, direction: int, also: tuple[str, ...] = ()):
    return await _run(_score, text, key, direction, also)
