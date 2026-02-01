"""
Q-Ravens Human-in-the-Loop Module

Provides approval gates and human intervention points in the workflow.
"""

from enum import Enum
from typing import Any, Optional

from langchain_core.messages import HumanMessage

from q_ravens.core.state import QRavensState, WorkflowPhase


class ApprovalType(str, Enum):
    """Types of approval that can be requested."""

    TEST_PLAN = "test_plan"  # Approve test cases before execution
    SENSITIVE_ACTION = "sensitive_action"  # Approve actions that could modify data
    CONTINUE = "continue"  # General continue/abort decision
    CREDENTIALS = "credentials"  # Request login credentials
    CUSTOM = "custom"  # Custom approval with message


class ApprovalRequest:
    """Represents a request for human approval."""

    def __init__(
        self,
        approval_type: ApprovalType,
        message: str,
        options: Optional[list[str]] = None,
        context: Optional[dict] = None,
    ):
        self.approval_type = approval_type
        self.message = message
        self.options = options or ["approve", "reject"]
        self.context = context or {}

    def to_dict(self) -> dict:
        """Convert to dictionary for state storage."""
        return {
            "approval_type": self.approval_type.value,
            "message": self.message,
            "options": self.options,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ApprovalRequest":
        """Create from dictionary."""
        return cls(
            approval_type=ApprovalType(data["approval_type"]),
            message=data["message"],
            options=data.get("options"),
            context=data.get("context"),
        )


class ApprovalResponse:
    """Represents a human's response to an approval request."""

    def __init__(
        self,
        approved: bool,
        choice: Optional[str] = None,
        feedback: Optional[str] = None,
    ):
        self.approved = approved
        self.choice = choice
        self.feedback = feedback

    def to_dict(self) -> dict:
        """Convert to dictionary for state storage."""
        return {
            "approved": self.approved,
            "choice": self.choice,
            "feedback": self.feedback,
        }


def create_approval_request(
    approval_type: ApprovalType,
    message: str,
    context: Optional[dict] = None,
) -> dict[str, Any]:
    """
    Create state updates that trigger an approval gate.

    Returns state updates that will pause the workflow for human input.
    """
    request = ApprovalRequest(
        approval_type=approval_type,
        message=message,
        context=context,
    )

    return {
        "requires_user_input": True,
        "user_input_prompt": message,
        "approval_request": request.to_dict(),
        "phase": WorkflowPhase.WAITING_FOR_USER.value,
    }


def create_test_plan_approval(test_cases: list, target_url: str) -> dict[str, Any]:
    """
    Create an approval request for a test plan.

    Args:
        test_cases: List of test cases to be approved
        target_url: The URL being tested

    Returns:
        State updates for approval gate
    """
    test_summary = "\n".join(
        f"  - {tc.name}: {tc.description}"
        for tc in test_cases
    )

    message = f"""Test Plan Approval Required

Target: {target_url}
Tests to execute ({len(test_cases)}):
{test_summary}

Do you approve executing these tests?"""

    return create_approval_request(
        approval_type=ApprovalType.TEST_PLAN,
        message=message,
        context={
            "test_count": len(test_cases),
            "target_url": target_url,
        },
    )


def create_credentials_request(login_url: str) -> dict[str, Any]:
    """
    Request credentials for authenticated testing.

    Args:
        login_url: The login page URL

    Returns:
        State updates for credential request
    """
    message = f"""Authentication Required

The website at {login_url} requires login credentials.
Please provide credentials to continue testing authenticated areas.

(Note: Credentials are only used for this test session and are not stored.)"""

    return create_approval_request(
        approval_type=ApprovalType.CREDENTIALS,
        message=message,
        context={"login_url": login_url},
    )


async def human_node(state: QRavensState) -> dict[str, Any]:
    """
    LangGraph node for human-in-the-loop interaction.

    This node is activated when the workflow requires human input.
    It processes the approval response and returns state updates.

    In CLI mode, this will cause the workflow to pause.
    The CLI will display the prompt and wait for user input,
    then resume the workflow with the response.
    """
    # Check if we have a response already (from resume)
    user_input = state.get("user_input")
    approval_request = state.get("approval_request")

    if user_input:
        # Process the user's response
        approved = user_input.lower() in ["y", "yes", "approve", "ok", "continue"]

        # Determine the correct phase to resume to based on workflow state
        # If we have test_cases, we were waiting for approval to execute -> resume to DESIGNING
        # If we have analysis but no test_cases, resume to ANALYZING
        # Otherwise resume to INIT
        has_test_cases = bool(state.get("test_cases"))
        has_analysis = state.get("analysis") is not None

        if approved:
            if has_test_cases:
                # We were waiting for approval after test design -> resume to DESIGNING
                # so orchestrator can see approval_response and proceed to executor
                resume_phase = WorkflowPhase.DESIGNING.value
            elif has_analysis:
                resume_phase = WorkflowPhase.ANALYZING.value
            else:
                resume_phase = WorkflowPhase.INIT.value
        else:
            resume_phase = WorkflowPhase.COMPLETED.value

        # Clear the approval state
        return {
            "requires_user_input": False,
            "user_input": None,
            "user_input_prompt": None,
            "approval_request": None,
            "approval_response": ApprovalResponse(
                approved=approved,
                feedback=user_input if not approved else None,
            ).to_dict(),
            "phase": resume_phase,
            "messages": [
                HumanMessage(content=f"User response: {user_input}")
            ],
        }

    # No response yet - workflow should pause here
    # This state indicates we're waiting
    return {
        "phase": WorkflowPhase.WAITING_FOR_USER.value,
    }


class HumanLoopManager:
    """
    Manager for human-in-the-loop interactions.

    Provides methods to:
    - Check if approval is needed
    - Get approval prompts
    - Process approval responses
    """

    def __init__(self, auto_approve: bool = False):
        """
        Initialize the manager.

        Args:
            auto_approve: If True, automatically approve all requests
        """
        self.auto_approve = auto_approve

    def needs_approval(self, state: dict) -> bool:
        """Check if the current state requires human approval."""
        return state.get("requires_user_input", False)

    def get_prompt(self, state: dict) -> Optional[str]:
        """Get the approval prompt message."""
        return state.get("user_input_prompt")

    def get_approval_type(self, state: dict) -> Optional[ApprovalType]:
        """Get the type of approval being requested."""
        request = state.get("approval_request")
        if request:
            return ApprovalType(request.get("approval_type"))
        return None

    def create_response_state(self, response: str) -> dict[str, Any]:
        """
        Create state updates from a user response.

        Args:
            response: The user's text response

        Returns:
            State updates to apply
        """
        return {
            "user_input": response,
            "requires_user_input": False,
        }
