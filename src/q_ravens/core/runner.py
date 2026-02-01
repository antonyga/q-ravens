"""
Q-Ravens Workflow Runner

Main entry point for running the Q-Ravens testing workflow.
Supports checkpointing for state persistence and resume capability.
"""

import asyncio
import logging
from typing import Any, AsyncGenerator, Optional
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver

from q_ravens.core.state import QRavensState, WorkflowPhase
from q_ravens.core.graph import compile_graph
from q_ravens.core.checkpoint import CheckpointManager, create_sqlite_checkpointer
from q_ravens.core.human_loop import human_node
from q_ravens.core.error_handler import create_safe_node
from q_ravens.core.exceptions import (
    QRavensError,
    WorkflowError,
    WorkflowStateError,
    WorkflowCheckpointError,
    MaxIterationsError,
    URLValidationError,
)
from q_ravens.agents import (
    orchestrator_node,
    analyzer_node,
    designer_node,
    executor_node,
    reporter_node,
)

logger = logging.getLogger(__name__)


def _validate_url(url: str) -> None:
    """
    Validate that a URL is properly formatted.

    Args:
        url: URL to validate

    Raises:
        URLValidationError: If URL is invalid
    """
    if not url:
        raise URLValidationError("URL cannot be empty")

    if not url.startswith(("http://", "https://")):
        raise URLValidationError(
            f"URL must start with http:// or https://",
            details={"url": url},
        )


class QRavensRunner:
    """
    Runner for the Q-Ravens testing workflow.

    Provides a simple interface to execute the agent swarm with
    optional state persistence via checkpointing.
    """

    def __init__(
        self,
        persist: bool = True,
        session_id: Optional[str] = None,
        use_safe_nodes: bool = True,
        max_errors: int = 5,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_api_key: Optional[str] = None,
    ):
        """
        Initialize the runner.

        Args:
            persist: Whether to persist state to disk (SQLite)
            session_id: Optional session ID for checkpointing
            use_safe_nodes: Whether to wrap agent nodes with error handling
            max_errors: Maximum errors before failing workflow
            llm_provider: LLM provider to use (e.g., "anthropic", "openai")
            llm_model: Model name to use
            llm_api_key: API key for the LLM provider
        """
        self.persist = persist
        self.session_id = session_id or str(uuid4())
        self.checkpoint_manager = CheckpointManager(persist=persist)
        self.use_safe_nodes = use_safe_nodes
        self.max_errors = max_errors
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        self._graph = None
        self._checkpointer = None

    async def _ensure_graph(self):
        """Ensure the graph is compiled with the appropriate checkpointer."""
        if self._graph is None:
            try:
                if self.persist:
                    self._checkpointer = await create_sqlite_checkpointer(self.session_id)
                else:
                    self._checkpointer = MemorySaver()
            except Exception as e:
                logger.error(f"Failed to create checkpointer: {e}")
                raise WorkflowCheckpointError(
                    f"Failed to initialize checkpointing: {e}",
                    details={"session_id": self.session_id, "persist": self.persist},
                ) from e

            # Optionally wrap nodes with error handling
            orch_node = orchestrator_node
            anlz_node = analyzer_node
            dsgn_node = designer_node
            exec_node = executor_node
            rpt_node = reporter_node

            if self.use_safe_nodes:
                orch_node = create_safe_node(orchestrator_node, "orchestrator", self.max_errors)
                anlz_node = create_safe_node(analyzer_node, "analyzer", self.max_errors)
                dsgn_node = create_safe_node(designer_node, "designer", self.max_errors)
                exec_node = create_safe_node(executor_node, "executor", self.max_errors)
                rpt_node = create_safe_node(reporter_node, "reporter", self.max_errors)

            self._graph = compile_graph(
                orchestrator_node=orch_node,
                analyzer_node=anlz_node,
                executor_node=exec_node,
                designer_node=dsgn_node,
                reporter_node=rpt_node,
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

        Raises:
            URLValidationError: If URL is invalid
            WorkflowError: If workflow fails to execute
            MaxIterationsError: If maximum iterations exceeded
        """
        # Validate URL
        _validate_url(url)

        logger.info(f"Starting Q-Ravens workflow for: {url}")

        try:
            graph = await self._ensure_graph()
        except WorkflowCheckpointError:
            raise
        except Exception as e:
            raise WorkflowError(f"Failed to initialize workflow: {e}") from e

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

        # Add LLM configuration if provided
        if self.llm_provider:
            initial_state["llm_provider"] = self.llm_provider
        if self.llm_model:
            initial_state["llm_model"] = self.llm_model
        if self.llm_api_key:
            initial_state["llm_api_key"] = self.llm_api_key

        # Configuration for the run
        config = {"configurable": {"thread_id": self.session_id}}

        # Execute the workflow
        final_state = None
        try:
            async for state in graph.astream(initial_state, config, stream_mode="values"):
                if state and isinstance(state, dict):
                    final_state = state

                    # Check for error phase
                    if state.get("phase") == WorkflowPhase.ERROR.value:
                        errors = state.get("errors", [])
                        error_msg = state.get("error_message", "Unknown error")
                        logger.error(f"Workflow entered error state: {error_msg}")

                    # Check for max iterations
                    iteration = state.get("iteration_count", 0)
                    max_iter = state.get("max_iterations", 10)
                    if iteration >= max_iter:
                        logger.warning(f"Maximum iterations ({max_iter}) reached")
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise WorkflowError(
                f"Workflow execution failed: {e}",
                details={"url": url, "session_id": self.session_id},
            ) from e

        if final_state is None:
            raise WorkflowStateError(
                "Workflow completed without producing final state",
                details={"url": url, "session_id": self.session_id},
            )

        logger.info(f"Workflow completed with phase: {final_state.get('phase')}")
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

        Raises:
            URLValidationError: If URL is invalid
            WorkflowError: If workflow fails to execute
        """
        # Validate URL
        _validate_url(url)

        logger.info(f"Starting Q-Ravens workflow stream for: {url}")

        try:
            graph = await self._ensure_graph()
        except WorkflowCheckpointError:
            raise
        except Exception as e:
            raise WorkflowError(f"Failed to initialize workflow: {e}") from e

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

        # Add LLM configuration if provided
        if self.llm_provider:
            initial_state["llm_provider"] = self.llm_provider
        if self.llm_model:
            initial_state["llm_model"] = self.llm_model
        if self.llm_api_key:
            initial_state["llm_api_key"] = self.llm_api_key

        # Configuration for the run
        config = {"configurable": {"thread_id": self.session_id}}

        # Stream the workflow
        try:
            async for state in graph.astream(initial_state, config, stream_mode="values"):
                if state and isinstance(state, dict):
                    yield state
        except Exception as e:
            logger.error(f"Workflow stream failed: {e}")
            # Yield an error state so consumers know what happened
            yield {
                "phase": WorkflowPhase.ERROR.value,
                "error_message": str(e),
                "errors": [{"error_type": type(e).__name__, "message": str(e)}],
            }
            raise WorkflowError(
                f"Workflow stream failed: {e}",
                details={"url": url, "session_id": self.session_id},
            ) from e

    async def resume(self, user_input: Optional[str] = None) -> dict[str, Any]:
        """
        Resume a paused workflow (e.g., after human-in-the-loop).

        Args:
            user_input: Optional user input if the workflow was waiting for it

        Returns:
            Final state after workflow completion

        Raises:
            WorkflowCheckpointError: If no checkpoint found
            WorkflowError: If workflow fails to resume
        """
        logger.info(f"Resuming workflow for session: {self.session_id}")

        try:
            graph = await self._ensure_graph()
        except WorkflowCheckpointError:
            raise
        except Exception as e:
            raise WorkflowError(f"Failed to initialize workflow for resume: {e}") from e

        config = {"configurable": {"thread_id": self.session_id}}

        # Get current state
        try:
            state = await graph.aget_state(config)
        except Exception as e:
            raise WorkflowCheckpointError(
                f"Failed to retrieve checkpoint: {e}",
                details={"session_id": self.session_id},
            ) from e

        if not state or not state.values:
            raise WorkflowCheckpointError(
                f"No checkpoint found for session {self.session_id}",
                details={"session_id": self.session_id},
            )

        current_state = dict(state.values)

        # If user input provided, update state
        if user_input:
            current_state["user_input"] = user_input
            current_state["requires_user_input"] = False

        # Resume the workflow
        final_state = None
        try:
            async for state in graph.astream(current_state, config, stream_mode="values"):
                if state and isinstance(state, dict):
                    final_state = state
        except Exception as e:
            logger.error(f"Failed to resume workflow: {e}")
            raise WorkflowError(
                f"Failed to resume workflow: {e}",
                details={"session_id": self.session_id},
            ) from e

        if final_state is None:
            raise WorkflowStateError(
                "Resumed workflow completed without producing final state",
                details={"session_id": self.session_id},
            )

        logger.info(f"Resumed workflow completed with phase: {final_state.get('phase')}")
        return final_state

    async def stream_resume(
        self,
        user_input: Optional[str] = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream a resumed workflow, yielding state updates.

        Args:
            user_input: Optional user input if the workflow was waiting for it

        Yields:
            State updates as the workflow progresses

        Raises:
            WorkflowCheckpointError: If no checkpoint found
            WorkflowError: If workflow fails to resume
        """
        logger.info(f"Streaming resumed workflow for session: {self.session_id}")

        try:
            graph = await self._ensure_graph()
        except WorkflowCheckpointError:
            raise
        except Exception as e:
            raise WorkflowError(f"Failed to initialize workflow for resume: {e}") from e

        config = {"configurable": {"thread_id": self.session_id}}

        # Get current state
        try:
            state = await graph.aget_state(config)
        except Exception as e:
            raise WorkflowCheckpointError(
                f"Failed to retrieve checkpoint: {e}",
                details={"session_id": self.session_id},
            ) from e

        if not state or not state.values:
            raise WorkflowCheckpointError(
                f"No checkpoint found for session {self.session_id}",
                details={"session_id": self.session_id},
            )

        current_state = dict(state.values)

        # If user input provided, update state
        if user_input:
            current_state["user_input"] = user_input
            current_state["requires_user_input"] = False

        # Stream the resumed workflow
        try:
            async for state in graph.astream(current_state, config, stream_mode="values"):
                if state and isinstance(state, dict):
                    yield state
        except Exception as e:
            logger.error(f"Failed to stream resumed workflow: {e}")
            # Yield an error state so consumers know what happened
            yield {
                "phase": WorkflowPhase.ERROR.value,
                "error_message": str(e),
                "errors": [{"error_type": type(e).__name__, "message": str(e)}],
            }
            raise WorkflowError(
                f"Failed to stream resumed workflow: {e}",
                details={"session_id": self.session_id},
            ) from e

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
