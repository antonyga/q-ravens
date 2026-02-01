"""
Q-Ravens Asyncio Compatibility Module

Handles platform-specific asyncio configurations, particularly for Windows
where Playwright requires special event loop handling.
"""

import asyncio
import sys
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")


def configure_event_loop() -> None:
    """
    Configure the event loop for the current platform.

    On Windows, Playwright requires the WindowsSelectorEventLoopPolicy
    because the default ProactorEventLoop doesn't support subprocess
    creation in the way Playwright needs.
    """
    if sys.platform == "win32":
        # Windows requires SelectorEventLoop for Playwright subprocess support
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """
    Get the current event loop or create a new one.

    Handles the case where we're running in an environment that
    already has an event loop (like Streamlit).

    Returns:
        The current or newly created event loop
    """
    try:
        loop = asyncio.get_running_loop()
        return loop
    except RuntimeError:
        # No running loop
        configure_event_loop()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run an async coroutine, handling nested event loops.

    This function handles running async code in environments like Streamlit
    that may already have an event loop running.

    Args:
        coro: The coroutine to run

    Returns:
        The result of the coroutine
    """
    # Configure event loop policy first (important for Windows)
    configure_event_loop()

    try:
        # Check if we're already in an async context
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        # We're inside an existing event loop (like Streamlit)
        # Use nest_asyncio to allow nested loops
        try:
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        except ImportError:
            # nest_asyncio not available, try creating a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
    else:
        # No existing loop, we can use asyncio.run
        return asyncio.run(coro)


def run_in_thread(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run an async coroutine in a separate thread.

    This is useful when we need to run Playwright in an environment
    that has event loop conflicts (like Streamlit on Windows).

    Args:
        coro: The coroutine to run

    Returns:
        The result of the coroutine
    """
    import concurrent.futures

    def run_coro():
        configure_event_loop()
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_coro)
        return future.result()


# Auto-configure on import for Windows
if sys.platform == "win32":
    configure_event_loop()
