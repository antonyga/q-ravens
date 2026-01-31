"""
Q-Ravens Custom Exceptions

Provides a hierarchy of exceptions for proper error handling
throughout the Q-Ravens system.
"""

from typing import Any, Optional


class QRavensError(Exception):
    """Base exception for all Q-Ravens errors."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


# LLM-related errors
class LLMError(QRavensError):
    """Base exception for LLM-related errors."""
    pass


class LLMConnectionError(LLMError):
    """Failed to connect to the LLM provider."""
    pass


class LLMRateLimitError(LLMError):
    """Rate limit exceeded for LLM API."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[float] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, details)
        self.retry_after = retry_after


class LLMResponseError(LLMError):
    """Invalid or unexpected response from LLM."""
    pass


class LLMTimeoutError(LLMError):
    """LLM request timed out."""
    pass


class LLMConfigurationError(LLMError):
    """LLM configuration error (e.g., missing API key)."""
    pass


# Agent-related errors
class AgentError(QRavensError):
    """Base exception for agent-related errors."""

    def __init__(
        self,
        message: str,
        agent_name: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, details)
        self.agent_name = agent_name


class AgentExecutionError(AgentError):
    """Error during agent execution."""
    pass


class AgentTimeoutError(AgentError):
    """Agent execution timed out."""
    pass


# Workflow-related errors
class WorkflowError(QRavensError):
    """Base exception for workflow-related errors."""
    pass


class WorkflowStateError(WorkflowError):
    """Invalid workflow state."""
    pass


class WorkflowCheckpointError(WorkflowError):
    """Error with checkpoint operations."""
    pass


class MaxIterationsError(WorkflowError):
    """Maximum workflow iterations exceeded."""

    def __init__(
        self,
        message: str = "Maximum iterations exceeded",
        iteration_count: int = 0,
        max_iterations: int = 0,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, details)
        self.iteration_count = iteration_count
        self.max_iterations = max_iterations


# Tool-related errors
class ToolError(QRavensError):
    """Base exception for tool-related errors."""

    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, details)
        self.tool_name = tool_name


class BrowserError(ToolError):
    """Error with browser operations."""
    pass


class BrowserTimeoutError(BrowserError):
    """Browser operation timed out."""
    pass


class BrowserNavigationError(BrowserError):
    """Failed to navigate to URL."""
    pass


# Validation errors
class ValidationError(QRavensError):
    """Base exception for validation errors."""
    pass


class URLValidationError(ValidationError):
    """Invalid URL provided."""
    pass


class StateValidationError(ValidationError):
    """Invalid state data."""
    pass


# Retryable error marker
class RetryableError(QRavensError):
    """
    Marker class for errors that can be retried.

    Any exception inheriting from this (in addition to its specific type)
    indicates the operation can be safely retried.
    """
    pass


# Mark certain errors as retryable
class RetryableLLMError(LLMError, RetryableError):
    """LLM error that can be retried."""
    pass


class RetryableConnectionError(LLMConnectionError, RetryableError):
    """Connection error that can be retried."""
    pass


class RetryableRateLimitError(LLMRateLimitError, RetryableError):
    """Rate limit error that can be retried after waiting."""
    pass


class RetryableTimeoutError(LLMTimeoutError, RetryableError):
    """Timeout error that can be retried."""
    pass


def is_retryable(error: Exception) -> bool:
    """
    Check if an error is retryable.

    Args:
        error: The exception to check

    Returns:
        True if the error can be retried
    """
    if isinstance(error, RetryableError):
        return True

    # Also check for common transient errors from underlying libraries
    error_type = type(error).__name__.lower()
    error_msg = str(error).lower()

    retryable_patterns = [
        "rate limit",
        "rate_limit",
        "429",
        "503",
        "502",
        "connection",
        "timeout",
        "temporary",
        "overloaded",
        "capacity",
    ]

    return any(pattern in error_type or pattern in error_msg for pattern in retryable_patterns)
