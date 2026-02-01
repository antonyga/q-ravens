"""
Q-Ravens LangGraph Workflow

Defines the agent orchestration graph using LangGraph.
This is the central workflow that coordinates all agents.
"""

from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from q_ravens.core.state import QRavensState, WorkflowPhase


# Type alias for routing decisions
AgentName = Literal["orchestrator", "analyzer", "designer", "executor", "reporter", "human"]


def route_next_agent(state: QRavensState) -> AgentName | str:
    """
    Routing function that determines which agent should run next.

    This is called after each agent to decide the next step.
    """
    phase = state.get("phase", WorkflowPhase.INIT.value)
    next_agent = state.get("next_agent")
    requires_user_input = state.get("requires_user_input", False)
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 10)

    # Check for errors
    if phase == WorkflowPhase.ERROR.value:
        return END

    # Check if we need human input
    if requires_user_input:
        return "human"

    # Check if workflow is complete
    if phase == WorkflowPhase.COMPLETED.value:
        return END

    # Check iteration limit
    if iteration_count >= max_iterations:
        return END

    # Route based on next_agent if explicitly set
    if next_agent:
        return next_agent

    # Handle WAITING_FOR_USER phase - user input has been processed
    # Route to orchestrator to handle the approval response
    if phase == WorkflowPhase.WAITING_FOR_USER.value:
        # If we have an approval response, go to orchestrator to process it
        if state.get("approval_response"):
            return "orchestrator"
        # If we have user_input but it hasn't been processed yet, go to human
        if state.get("user_input"):
            return "human"
        # Otherwise, still waiting - go to orchestrator to re-assert wait
        return "orchestrator"

    # Default routing based on phase
    phase_routing = {
        WorkflowPhase.INIT.value: "orchestrator",
        WorkflowPhase.ANALYZING.value: "analyzer",
        WorkflowPhase.DESIGNING.value: "designer",
        WorkflowPhase.EXECUTING.value: "executor",
        WorkflowPhase.REPORTING.value: "reporter",
    }

    return phase_routing.get(phase, END)


def create_graph(
    orchestrator_node,
    analyzer_node,
    executor_node,
    designer_node=None,
    reporter_node=None,
    human_node=None,
) -> StateGraph:
    """
    Create the Q-Ravens workflow graph.

    Args:
        orchestrator_node: The orchestrator agent function
        analyzer_node: The analyzer agent function
        executor_node: The executor agent function
        designer_node: Optional designer agent function (Test Architect)
        reporter_node: Optional reporter agent function (QA Lead)
        human_node: Optional human-in-the-loop node

    Returns:
        Compiled LangGraph StateGraph
    """
    # Create the graph with our state schema
    workflow = StateGraph(QRavensState)

    # Add agent nodes
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("executor", executor_node)

    # Add designer node if provided
    if designer_node:
        workflow.add_node("designer", designer_node)

    # Add reporter node if provided
    if reporter_node:
        workflow.add_node("reporter", reporter_node)

    # Add human node if provided (for human-in-the-loop)
    if human_node:
        workflow.add_node("human", human_node)

    # Set entry point
    workflow.set_entry_point("orchestrator")

    # Build routing map for conditional edges
    base_routes = {
        "orchestrator": "orchestrator",
        "analyzer": "analyzer",
        "executor": "executor",
        END: END,
    }

    # Add designer route if available
    if designer_node:
        base_routes["designer"] = "designer"

    # Add reporter route if available
    if reporter_node:
        base_routes["reporter"] = "reporter"

    # Add human route if available
    if human_node:
        base_routes["human"] = "human"

    # Add conditional edges from orchestrator
    workflow.add_conditional_edges(
        "orchestrator",
        route_next_agent,
        base_routes,
    )

    # Add conditional edges from analyzer
    workflow.add_conditional_edges(
        "analyzer",
        route_next_agent,
        base_routes,
    )

    # Add conditional edges from designer (if present)
    if designer_node:
        workflow.add_conditional_edges(
            "designer",
            route_next_agent,
            base_routes,
        )

    # Add conditional edges from executor
    workflow.add_conditional_edges(
        "executor",
        route_next_agent,
        base_routes,
    )

    # Add conditional edges from reporter (if present)
    if reporter_node:
        workflow.add_conditional_edges(
            "reporter",
            route_next_agent,
            base_routes,
        )

    # Add edges from human node back to orchestrator
    if human_node:
        workflow.add_edge("human", "orchestrator")

    return workflow


def compile_graph(
    orchestrator_node,
    analyzer_node,
    executor_node,
    designer_node=None,
    reporter_node=None,
    human_node=None,
    checkpointer=None,
):
    """
    Compile the workflow graph with optional checkpointing.

    Args:
        orchestrator_node: The orchestrator agent function
        analyzer_node: The analyzer agent function
        executor_node: The executor agent function
        designer_node: Optional designer agent function (Test Architect)
        reporter_node: Optional reporter agent function (QA Lead)
        human_node: Optional human-in-the-loop node
        checkpointer: Optional checkpointer for state persistence

    Returns:
        Compiled graph ready for execution
    """
    workflow = create_graph(
        orchestrator_node=orchestrator_node,
        analyzer_node=analyzer_node,
        executor_node=executor_node,
        designer_node=designer_node,
        reporter_node=reporter_node,
        human_node=human_node,
    )

    # Use memory saver if no checkpointer provided
    if checkpointer is None:
        checkpointer = MemorySaver()

    return workflow.compile(checkpointer=checkpointer)
