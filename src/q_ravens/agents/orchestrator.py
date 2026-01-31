"""
Q-Ravens Orchestrator Agent

The central coordinator (Project Manager) responsible for:
- Understanding user intent
- Delegating tasks to specialist agents
- Managing workflow progression
- Requesting human approval for sensitive actions
- Communicating results
"""

from typing import Any

from langchain_core.messages import AIMessage

from q_ravens.agents.base import BaseAgent
from q_ravens.core.state import QRavensState, WorkflowPhase


class OrchestratorAgent(BaseAgent):
    """
    The Orchestrator is the central coordinator of the Q-Ravens swarm.

    It acts as a Project Manager, interpreting user requests and
    delegating work to specialized agents.
    """

    name = "orchestrator"
    role = "Project Manager"
    description = "Coordinates the agent swarm and manages workflow"

    @property
    def system_prompt(self) -> str:
        return """You are the Orchestrator agent of Q-Ravens, an autonomous QA testing system.
Your role is to act as a Project Manager coordinating a team of specialized QA agents.

Your responsibilities:
1. Understand user requests for testing web applications
2. Break down requests into actionable tasks
3. Decide which specialist agent should handle each task
4. Track progress and report status to the user

Available specialist agents:
- Analyzer: Crawls websites, maps structure, identifies elements and tech stack
- Designer: Creates comprehensive test strategies and test cases
- Executor: Runs automated tests using Playwright
- Reporter: Generates test reports (coming soon)

When given a URL to test, your typical workflow is:
1. First, delegate to Analyzer to understand the website
2. Delegate to Designer to create test cases based on analysis
3. Request user approval for the test plan
4. Delegate to Executor to run the tests
5. Compile results and report to user

Always respond with clear, actionable decisions. Be concise and professional.
When you need to delegate, clearly state which agent and what task."""

    async def process(self, state: QRavensState) -> dict[str, Any]:
        """
        Process the current state and decide next actions.

        Returns state updates including which agent should run next.
        """
        # Get current state values with defaults (dict-based access)
        phase = state.get("phase", WorkflowPhase.INIT.value)
        iteration = state.get("iteration_count", 0) + 1
        target_url = state.get("target_url")
        analysis = state.get("analysis")
        test_results = state.get("test_results", [])
        approval_response = state.get("approval_response")
        require_approval = state.get("require_approval", True)  # Default to requiring approval

        # Handle initial state - need URL from user
        if phase == WorkflowPhase.INIT.value:
            if not target_url:
                return {
                    "phase": WorkflowPhase.WAITING_FOR_USER.value,
                    "requires_user_input": True,
                    "user_input_prompt": "Please provide the URL of the website you want to test.",
                    "iteration_count": iteration,
                    "current_agent": self.name,
                }

            # We have a URL, start analysis
            message = AIMessage(
                content=f"Starting QA analysis for: {target_url}\n"
                f"Delegating to Analyzer agent to explore the website structure."
            )
            return {
                "phase": WorkflowPhase.ANALYZING.value,
                "next_agent": "analyzer",
                "messages": [message],
                "iteration_count": iteration,
                "current_agent": self.name,
            }

        # After analysis is complete - proceed to test design
        if phase == WorkflowPhase.ANALYZING.value and analysis:
            pages_count = len(analysis.pages_discovered)
            tech_stack = ", ".join(analysis.technology_stack) or "Unknown"

            message = AIMessage(
                content=f"Analysis complete!\n"
                f"- Pages discovered: {pages_count}\n"
                f"- Technology stack: {tech_stack}\n"
                f"- Has authentication: {analysis.has_authentication}\n\n"
                f"Delegating to Designer agent to create test cases..."
            )
            return {
                "phase": WorkflowPhase.DESIGNING.value,
                "next_agent": "designer",
                "messages": [message],
                "iteration_count": iteration,
                "current_agent": self.name,
            }

        # After design is complete - request approval before execution
        test_cases = state.get("test_cases", [])
        if phase == WorkflowPhase.DESIGNING.value and test_cases:
            # Summarize test cases for approval
            high_priority = sum(1 for tc in test_cases if tc.priority == "high")
            medium_priority = sum(1 for tc in test_cases if tc.priority == "medium")
            low_priority = sum(1 for tc in test_cases if tc.priority == "low")

            # Check if approval is required and not yet given
            if require_approval and not approval_response:
                # Build test case summary
                test_summary = "\n".join(
                    f"  [{tc.priority[0].upper()}] {tc.id}: {tc.name}"
                    for tc in test_cases[:10]  # Show first 10
                )
                if len(test_cases) > 10:
                    test_summary += f"\n  ... and {len(test_cases) - 10} more"

                # Request approval for test execution
                approval_message = f"""Test Design Complete - Approval Required

Target: {target_url}
Test Cases Created: {len(test_cases)}
  - High priority: {high_priority}
  - Medium priority: {medium_priority}
  - Low priority: {low_priority}

Test Cases:
{test_summary}

Do you approve executing these tests? (yes/no)"""

                return {
                    "phase": WorkflowPhase.WAITING_FOR_USER.value,
                    "requires_user_input": True,
                    "user_input_prompt": approval_message,
                    "iteration_count": iteration,
                    "current_agent": self.name,
                    "messages": [AIMessage(content="Requesting approval to execute designed tests...")],
                }

            # Check approval response
            if approval_response:
                if not approval_response.get("approved", False):
                    # User rejected - complete workflow
                    message = AIMessage(
                        content="Test execution cancelled by user.\n"
                        f"Analysis and {len(test_cases)} designed test cases are available for review."
                    )
                    return {
                        "phase": WorkflowPhase.COMPLETED.value,
                        "next_agent": None,
                        "messages": [message],
                        "iteration_count": iteration,
                        "current_agent": self.name,
                        "approval_response": None,  # Clear the response
                    }

            # Approved (or approval not required) - proceed to execution
            message = AIMessage(
                content=f"Test design approved!\n"
                f"- Total test cases: {len(test_cases)}\n"
                f"- High priority: {high_priority}\n"
                f"- Medium priority: {medium_priority}\n\n"
                f"Proceeding to execute tests..."
            )
            return {
                "phase": WorkflowPhase.EXECUTING.value,
                "next_agent": "executor",
                "messages": [message],
                "iteration_count": iteration,
                "current_agent": self.name,
                "approval_response": None,  # Clear the response
            }

        # After execution is complete
        if phase == WorkflowPhase.EXECUTING.value and test_results:
            passed = sum(1 for r in test_results if r.status.value == "passed")
            failed = sum(1 for r in test_results if r.status.value == "failed")
            total = len(test_results)

            message = AIMessage(
                content=f"Testing complete!\n\n"
                f"Results: {passed}/{total} tests passed, {failed} failed.\n\n"
                f"Test Summary:\n"
                + "\n".join(
                    f"- {r.test_id}: {r.status.value.upper()}"
                    for r in test_results
                )
            )
            return {
                "phase": WorkflowPhase.COMPLETED.value,
                "next_agent": None,
                "messages": [message],
                "iteration_count": iteration,
                "current_agent": self.name,
            }

        # Default: workflow complete
        return {
            "phase": WorkflowPhase.COMPLETED.value,
            "iteration_count": iteration,
            "current_agent": self.name,
        }


# Create node function for LangGraph
async def orchestrator_node(state: QRavensState) -> dict[str, Any]:
    """LangGraph node function for the Orchestrator agent."""
    agent = OrchestratorAgent()
    return await agent.process(state)
