"""
Q-Ravens Core Module

Contains the agent framework, orchestration logic, state management,
and shared utilities used across all agents.
"""

from q_ravens.core.config import settings, LLMProvider
from q_ravens.core.state import (
    QRavensState,
    WorkflowPhase,
    TestStatus,
    PageInfo,
    AnalysisResult,
    TestCase,
    TestResult,
)
from q_ravens.core.graph import create_graph, compile_graph
from q_ravens.core.runner import QRavensRunner, run_test
from q_ravens.core.exceptions import (
    QRavensError,
    LLMError,
    AgentError,
    WorkflowError,
    ToolError,
    ValidationError,
)
from q_ravens.core.error_handler import (
    ErrorRecord,
    handle_agent_error,
    with_error_handling,
    create_safe_node,
)

__all__ = [
    # Config
    "settings",
    "LLMProvider",
    # State
    "QRavensState",
    "WorkflowPhase",
    "TestStatus",
    "PageInfo",
    "AnalysisResult",
    "TestCase",
    "TestResult",
    # Graph
    "create_graph",
    "compile_graph",
    # Runner
    "QRavensRunner",
    "run_test",
    # Exceptions
    "QRavensError",
    "LLMError",
    "AgentError",
    "WorkflowError",
    "ToolError",
    "ValidationError",
    # Error handling
    "ErrorRecord",
    "handle_agent_error",
    "with_error_handling",
    "create_safe_node",
]
