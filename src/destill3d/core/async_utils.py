"""
Async utility functions for Destill3D.

Provides helpers for running async code from sync contexts
and async resource management.
"""

import asyncio
from typing import Any, Awaitable, TypeVar

T = TypeVar("T")


def run_sync(coro: Awaitable[T]) -> T:
    """
    Run an async coroutine synchronously.

    Handles the case where an event loop may or may not already be running.

    Args:
        coro: The coroutine to execute.

    Returns:
        The result of the coroutine.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside an existing event loop - use nest_asyncio if available
        try:
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        except ImportError:
            # Create a new thread to run the coroutine
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
    else:
        return asyncio.run(coro)


class AsyncContextManager:
    """Base class for async resource management with sync fallback."""

    async def __aenter__(self):
        await self._async_setup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._async_teardown()
        return False

    def __enter__(self):
        run_sync(self._async_setup())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        run_sync(self._async_teardown())
        return False

    async def _async_setup(self):
        """Override to perform async setup."""
        pass

    async def _async_teardown(self):
        """Override to perform async teardown."""
        pass


async def gather_with_concurrency(
    limit: int,
    *tasks: Awaitable[Any],
) -> list:
    """
    Run tasks concurrently with a concurrency limit.

    Args:
        limit: Maximum number of concurrent tasks.
        tasks: Coroutines to execute.

    Returns:
        List of results in task order.
    """
    semaphore = asyncio.Semaphore(limit)

    async def limited_task(task: Awaitable[Any]) -> Any:
        async with semaphore:
            return await task

    return await asyncio.gather(*(limited_task(t) for t in tasks))
