"""
Q-Ravens Workflow Runner

Main entry point for running the Q-Ravens testing workflow.
Supports checkpointing for state persistence and resume capability.
"""

import asyncio
from typing import Any, AsyncGenerator, Optional
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver

from q_ravens.core.state import QRavensState, WorkflowPhase
from q_ravens.core.graph import compile_graph
from q_ravens.core.checkpoint import CheckpointManager, create_sqlite_checkpointer
from q_ravens.core.human_loop import human_node
from q_ravens.agents import orchestrator_node, analyzer_node, executor_node


class QRavensRunner:
    """
    Runner for the Q-Ravens testing workflow.

    Provides a simple interface to execute the agent swarm with
    optional state persistence via checkpointing.
    """

    def __init__(self, persist: bool = True, session_id: Optional[str] = None):
        """
        Initialize the runner.

        Args:
            persist: Whether to persist state to disk (SQLite)
            session_id: Optional session ID for checkpointing
        """
        self.persist = persist
        self.session_id = session_id or str(uuid4())
        self.checkpoint_manager = CheckpointManager(persist=persist)
        self._graph = None
        self._checkpointer = None

    async def _ensure_graph(self):
        """Ensure the graph is compiled with the appropriate checkpointer."""
        if self._graph is None:
            if self.persist:
                self._checkpointer = await create_sqlite_checkpointer(self.session_id)
            else:
                self._checkpointer = MemorySaver()

            self._graph = compile_graph(
                orchestrator_node=orchestrator_node,
                analyzer_node=analyzer_node,
                executor_node=executor_node,
                human_node=human_node,
                checkpointer=self._checkpointer,
            )
        return self._graph

    @property
    def graph(self):
        """Get the compiled graph (for synchronous access after initialization)."""
        if self._graph is None:
            raise RuntimeError("Graph not initialized. Call _ensure_graph() first or use async methods.")
        return self._graph

    async def run(
        self,
        url: str,
        user_request: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Run the Q-Ravens workflow on a target URL.

        Args:
            url: The URL to test
            user_request: Optional natural language request

        Returns:
            Final state after workflow completion
        """
        graph = await self._ensure_graph()

        # Create initial state
        initial_state = {
            "target_url": url,
            "user_request": user_request or f"Test the website at {url}",
            "session_id": self.session_id,
            "phase": WorkflowPhase.INIT.value,
            "iteration_count": 0,
            "max_iterations": 10,
            "messages": [],
            "errors": [],
            "test_cases": [],
            "test_results": [],
        }

        # Configuration for the run
        config = {"configurable": {"thread_id": self.session_id}}

        # Execute the workflow
        final_state = None
        async for state in graph.astream(initial_state, config, stream_mode="values"):
            if state and isinstance(state, dict):
                final_state = state

        return final_state

    async def stream(
        self,
        url: str,
        user_request: Optional[str] = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream the Q-Ravens workflow, yielding state updates.

        Args:
            url: The URL to test
            user_request: Optional natural language request

        Yields:
            State updates as the workflow progresses
        """
        graph = await self._ensure_graph()

        # Create initial state
        initial_state = {
            "target_url": url,
            "user_request": user_request or f"Test the website at {url}",
            "session_id": self.session_id,
            "phase": WorkflowPhase.INIT.value,
            "iteration_count": 0,
            "max_iterations": 10,
            "messages": [],
            "errors": [],
            "test_cases": [],
            "test_results": [],
        }

        # Configuration for the run
        config = {"configurable": {"thread_id": self.session_id}}

        # Stream the workflow
        async for state in graph.astream(initial_state, config, stream_mode="values"):
            if state and isinstance(state, dict):
                yield state

    async def resume(self, user_input: Optional[str] = None) -> dict[str, Any]:
        """
        Resume a paused workflow (e.g., after human-in-the-loop).

        Args:
            user_input: Optional user input if the workflow was waiting for it

        Returns:
            Final state after workflow completion
        """
        graph = await self._ensure_graph()
        config = {"configurable": {"thread_id": self.session_id}}

        # Get current state
        state = await graph.aget_state(config)
        if not state or not state.values:
            raise ValueError(f"No checkpoint found for session {self.session_id}")

        current_state = dict(state.values)

        # If user input provided, update state
        if user_input:
            current_state["user_input"] = user_input
            current_state["requires_user_input"] = False

        # Resume the workflow
        final_state = None
        async for state in graph.astream(current_state, config, stream_mode="values"):
            if state and isinstance(state, dict):
                final_state = state

        return final_state

    async def get_state(self) -> Optional[dict[str, Any]]:
        """
        Get the current state of the workflow.

        Returns:
            Current state or None if no checkpoint exists
        """
        graph = await self._ensure_graph()
        config = {"configurable": {"thread_id": self.session_id}}

        state = await graph.aget_state(config)
        if state and state.values:
            return dict(state.values)
        return None

    def can_resume(self) -> bool:
        """Check if this session can be resumed."""
        return self.checkpoint_manager.session_exists(self.session_id)

    def run_sync(
        self,
        url: str,
        user_request: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Synchronous wrapper for run().

        Args:
            url: The URL to test
            user_request: Optional natural language request

        Returns:
            Final state after workflow completion
        """
        return asyncio.run(self.run(url, user_request))


async def run_test(url: str) -> None:
    """
    Quick test function to verify the workflow.

    Args:
        url: URL to test
    """
    print(f"\n{'='*60}")
    print(f"Q-Ravens Testing: {url}")
    print(f"{'='*60}\n")

    runner = QRavensRunner(persist=False)  # Don't persist for quick test

    print("Starting workflow...\n")

    async for state in runner.stream(url):
        phase = state.get("phase", "unknown")
        agent = state.get("current_agent", "starting")
        print(f"[{agent.upper()}] Phase: {phase}")

        messages = state.get("messages", [])
        if messages:
            for msg in messages[-1:]:
                content = getattr(msg, "content", str(msg))
                print(f"  Message: {content[:200]}...")
        print()

    print(f"{'='*60}")
    print("Workflow complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        asyncio.run(run_test(sys.argv[1]))
    else:
        print("Usage: python -m q_ravens.core.runner <url>")
