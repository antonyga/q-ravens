"""
Q-Ravens Error Handler

Provides error handling utilities for agent nodes, including
wrapping functions to catch exceptions and update workflow state.
"""

import logging
import traceback
from functools import wraps
from typing import Any, Callable, TypeVar

from langchain_core.messages import AIMessage

from q_ravens.core.state import QRavensState, WorkflowPhase
from q_ravens.core.exceptions import (
    QRavensError,
    AgentError,
    AgentExecutionError,
    LLMError,
    ToolError,
    WorkflowError,
    is_retryable,
)

logger = logging.getLogger(__name__)

# Type for the node function
NodeFunc = Callable[[QRavensState], dict[str, Any]]
AsyncNodeFunc = Callable[[QRavensState], Any]  # Returns Awaitable[dict[str, Any]]


class ErrorRecord:
    """Record of an error that occurred during workflow execution."""

    def __init__(
        self,
        error_type: str,
        message: str,
        agent_name: str,
        is_retryable: bool = False,
        stack_trace: str = "",
        details: dict[str, Any] = None,
    ):
        self.error_type = error_type
        self.message = message
        self.agent_name = agent_name
        self.is_retryable = is_retryable
        self.stack_trace = stack_trace
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for state storage."""
        return {
            "error_type": self.error_type,
            "message": self.message,
            "agent_name": self.agent_name,
            "is_retryable": self.is_retryable,
            "stack_trace": self.stack_trace,
            "details": self.details,
        }

    @classmethod
    def from_exception(cls, error: Exception, agent_name: str) -> "ErrorRecord":
        """Create an ErrorRecord from an exception."""
        is_retryable_error = is_retryable(error)

        details = {}
        if isinstance(error, QRavensError):
            details = error.details

        return cls(
            error_type=type(error).__name__,
            message=str(error),
            agent_name=agent_name,
            is_retryable=is_retryable_error,
            stack_trace=traceback.format_exc(),
            details=details,
        )


def handle_agent_error(
    error: Exception,
    agent_name: str,
    state: QRavensState,
    max_errors: int = 5,
) -> dict[str, Any]:
    """
    Handle an error from an agent and return appropriate state updates.

    Args:
        error: The exception that occurred
        agent_name: Name of the agent that failed
        state: Current workflow state
        max_errors: Maximum errors before failing workflow

    Returns:
        State updates dict
    """
    # Create error record
    error_record = ErrorRecord.from_exception(error, agent_name)

    # Get current errors list
    current_errors = state.get("errors", [])
    if not isinstance(current_errors, list):
        current_errors = []

    # Add new error
    updated_errors = current_errors + [error_record.to_dict()]

    # Log the error
    logger.error(
        f"Agent '{agent_name}' encountered error: {error}",
        extra={
            "agent_name": agent_name,
            "error_type": error_record.error_type,
            "is_retryable": error_record.is_retryable,
        },
    )

    # Create error message for workflow
    error_message = AIMessage(
        content=f"Error in {agent_name}: {error_record.message}"
    )

    # Check if we should fail the workflow
    if len(updated_errors) >= max_errors:
        logger.error(f"Maximum errors ({max_errors}) reached, failing workflow")
        return {
            "phase": WorkflowPhase.ERROR.value,
            "errors": updated_errors,
            "messages": [error_message],
            "current_agent": agent_name,
            "error_message": f"Workflow failed after {len(updated_errors)} errors",
        }

    # For retryable errors, we might want to retry the same agent
    if error_record.is_retryable:
        logger.info(f"Error is retryable, will attempt to retry")
        return {
            "errors": updated_errors,
            "messages": [error_message],
            "current_agent": agent_name,
            "next_agent": agent_name,  # Retry same agent
        }

    # Non-retryable error, return to orchestrator for handling
    return {
        "phase": WorkflowPhase.ERROR.value,
        "errors": updated_errors,
        "messages": [error_message],
        "current_agent": agent_name,
        "next_agent": "orchestrator",  # Let orchestrator decide
    }


def with_error_handling(agent_name: str, max_errors: int = 5):
    """
    Decorator to add error handling to an async agent node function.

    Args:
        agent_name: Name of the agent (for error reporting)
        max_errors: Maximum errors before failing workflow

    Returns:
        Decorator function
    """
    def decorator(func: AsyncNodeFunc) -> AsyncNodeFunc:
        @wraps(func)
        async def wrapper(state: QRavensState) -> dict[str, Any]:
            try:
                return await func(state)
            except Exception as e:
                return handle_agent_error(
                    error=e,
                    agent_name=agent_name,
                    state=state,
                    max_errors=max_errors,
                )
        return wrapper
    return decorator


def create_safe_node(
    node_func: AsyncNodeFunc,
    agent_name: str,
    max_errors: int = 5,
) -> AsyncNodeFunc:
    """
    Create a safe version of a node function with error handling.

    This is an alternative to the decorator for when you need to wrap
    an existing function.

    Args:
        node_func: The original node function
        agent_name: Name of the agent
        max_errors: Maximum errors before failing workflow

    Returns:
        Wrapped node function with error handling
    """
    @wraps(node_func)
    async def safe_node(state: QRavensState) -> dict[str, Any]:
        try:
            return await node_func(state)
        except Exception as e:
            return handle_agent_error(
                error=e,
                agent_name=agent_name,
                state=state,
                max_errors=max_errors,
            )
    return safe_node
