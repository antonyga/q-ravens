"""
Q-Ravens State Schema

Defines the shared state that flows between all agents in the workflow.
This is the central data structure for LangGraph orchestration.
"""

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class WorkflowPhase(str, Enum):
    """Current phase in the testing workflow."""

    INIT = "init"
    ANALYZING = "analyzing"
    DESIGNING = "designing"
    EXECUTING = "executing"
    REPORTING = "reporting"
    COMPLETED = "completed"
    ERROR = "error"
    WAITING_FOR_USER = "waiting_for_user"


class TestStatus(str, Enum):
    """Status of individual test cases."""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class PageInfo(BaseModel):
    """Information about a discovered page."""

    url: str
    title: Optional[str] = None
    links: list[str] = Field(default_factory=list)
    forms: list[dict[str, Any]] = Field(default_factory=list)
    buttons: list[str] = Field(default_factory=list)
    inputs: list[dict[str, Any]] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    """Results from the Analyzer agent."""

    target_url: str
    pages_discovered: list[PageInfo] = Field(default_factory=list)
    technology_stack: list[str] = Field(default_factory=list)
    has_authentication: bool = False
    auth_type: Optional[str] = None
    sitemap: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class TestCase(BaseModel):
    """A single test case designed by the Designer agent."""

    id: str
    name: str
    description: str
    steps: list[str] = Field(default_factory=list)
    expected_result: str
    priority: str = "medium"  # high, medium, low
    category: str = "functional"  # functional, accessibility, performance
    status: TestStatus = TestStatus.PENDING


class TestResult(BaseModel):
    """Result of executing a test case."""

    test_id: str
    status: TestStatus
    actual_result: Optional[str] = None
    error_message: Optional[str] = None
    screenshot_path: Optional[str] = None
    duration_ms: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)


class QRavensState(TypedDict, total=False):
    """
    The main state object passed between all agents.

    This follows LangGraph patterns for stateful workflows using TypedDict.
    """

    # User Input
    target_url: str
    user_request: str

    # Conversation history (uses LangGraph's message reducer)
    messages: Annotated[list[BaseMessage], add_messages]

    # Workflow Control
    phase: str  # WorkflowPhase value
    current_agent: str
    next_agent: str
    iteration_count: int
    max_iterations: int

    # Analysis Results (from Analyzer)
    analysis: AnalysisResult

    # Test Cases (from Designer)
    test_cases: list[TestCase]

    # Test Results (from Executor)
    test_results: list[TestResult]

    # Error Handling
    errors: list[str]
    requires_user_input: bool
    user_input_prompt: str

    # Metadata
    session_id: str
    started_at: str
