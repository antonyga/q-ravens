"""
Q-Ravens Streamlit Chat UI

Provides a conversational interface for running QA tests on web applications.
Users can input URLs and natural language test requests, view real-time
progress, and interact with human-in-the-loop approval gates.
"""

import streamlit as st
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from q_ravens.core.runner import QRavensRunner
from q_ravens.core.state import WorkflowPhase
from q_ravens.core.asyncio_compat import run_in_thread
from q_ravens.core.config import LLMProvider, settings


# Page configuration
st.set_page_config(
    page_title="Q-Ravens QA Agent",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "workflow_state" not in st.session_state:
        st.session_state.workflow_state = None
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid4())
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "test_results" not in st.session_state:
        st.session_state.test_results = None
    if "report" not in st.session_state:
        st.session_state.report = None
    if "requires_input" not in st.session_state:
        st.session_state.requires_input = False
    if "input_prompt" not in st.session_state:
        st.session_state.input_prompt = ""
    # LLM configuration - persisted in session
    if "llm_api_key" not in st.session_state:
        st.session_state.llm_api_key = ""


def render_sidebar() -> dict[str, Any]:
    """
    Render the sidebar with configuration options.

    Returns:
        Dictionary of configuration values
    """
    with st.sidebar:
        st.title("🦅 Q-Ravens")
        st.caption("Autonomous QA Agent Swarm")

        st.divider()

        # URL Input
        st.subheader("Target Configuration")
        target_url = st.text_input(
            "Target URL",
            placeholder="https://example.com",
            help="The URL of the web application to test",
        )

        # Test scope selection
        test_scope = st.multiselect(
            "Test Scope",
            options=[
                "Smoke Tests",
                "Functional Tests",
                "UI/UX Tests",
                "Performance Tests",
                "Accessibility Tests",
                "Security Tests",
            ],
            default=["Smoke Tests", "Functional Tests"],
            help="Select the types of tests to run",
        )

        st.divider()

        # LLM Configuration
        st.subheader("LLM Configuration")

        # Provider selection
        provider_options = {
            "Anthropic (Claude)": LLMProvider.ANTHROPIC,
            "OpenAI (GPT)": LLMProvider.OPENAI,
            "Google (Gemini)": LLMProvider.GOOGLE,
            "Groq": LLMProvider.GROQ,
            "Ollama (Local)": LLMProvider.OLLAMA,
        }

        # Get default from settings
        default_provider_name = next(
            (name for name, p in provider_options.items()
             if p == settings.default_llm_provider),
            "Anthropic (Claude)"
        )

        selected_provider_name = st.selectbox(
            "LLM Provider",
            options=list(provider_options.keys()),
            index=list(provider_options.keys()).index(default_provider_name),
            help="Select the LLM provider for the AI agents",
        )
        selected_provider = provider_options[selected_provider_name]

        # Model selection based on provider
        model_defaults = {
            LLMProvider.ANTHROPIC: "claude-sonnet-4-20250514",
            LLMProvider.OPENAI: "gpt-4o",
            LLMProvider.GOOGLE: "gemini-2.0-flash",
            LLMProvider.GROQ: "llama-3.3-70b-versatile",
            LLMProvider.OLLAMA: "llama3",
        }

        selected_model = st.text_input(
            "Model Name",
            value=model_defaults.get(selected_provider, ""),
            help="The specific model to use (e.g., claude-sonnet-4-20250514, gpt-4o)",
        )

        # API Key input (not needed for Ollama)
        ollama_url = None  # Initialize for non-Ollama providers

        if selected_provider != LLMProvider.OLLAMA:
            # Check if we have a key in environment
            env_key = settings.get_api_key(selected_provider)
            has_env_key = bool(env_key)

            if has_env_key:
                st.success("API key loaded from environment", icon="✅")
                api_key = env_key
            else:
                api_key = st.text_input(
                    "API Key",
                    type="password",
                    value=st.session_state.llm_api_key,
                    help=f"Enter your {selected_provider_name} API key",
                    placeholder="sk-... or similar",
                )
                # Store in session state to persist across reruns
                if api_key:
                    st.session_state.llm_api_key = api_key

                if not api_key:
                    st.warning("API key required", icon="⚠️")
        else:
            api_key = None
            ollama_url = st.text_input(
                "Ollama URL",
                value=settings.ollama_base_url,
                help="URL of your local Ollama instance",
            )

        st.divider()

        # Advanced Settings
        with st.expander("Advanced Settings"):
            max_iterations = st.slider(
                "Max Iterations",
                min_value=5,
                max_value=50,
                value=10,
                help="Maximum workflow iterations before stopping",
            )

            persist_state = st.checkbox(
                "Persist State",
                value=True,
                help="Save workflow state to disk for resume capability",
            )

            human_approval = st.checkbox(
                "Require Human Approval",
                value=True,
                help="Pause for approval before executing tests",
            )

        st.divider()

        # Session info
        st.subheader("Session Info")
        st.text(f"Session: {st.session_state.session_id[:8]}...")

        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 New Session", use_container_width=True):
                st.session_state.session_id = str(uuid4())
                st.session_state.messages = []
                st.session_state.workflow_state = None
                st.session_state.test_results = None
                st.session_state.report = None
                st.rerun()

        with col2:
            if st.button("📋 Export", use_container_width=True):
                export_results()

        return {
            "target_url": target_url,
            "test_scope": test_scope,
            "max_iterations": max_iterations,
            "persist_state": persist_state,
            "human_approval": human_approval,
            "llm_provider": selected_provider,
            "llm_model": selected_model,
            "llm_api_key": api_key,
            "ollama_url": ollama_url if selected_provider == LLMProvider.OLLAMA else None,
        }


def export_results() -> None:
    """Export test results and report."""
    if st.session_state.report:
        report = st.session_state.report
        if hasattr(report, "markdown_content"):
            st.sidebar.download_button(
                label="Download Report (MD)",
                data=report.markdown_content,
                file_name=f"q_ravens_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
            )
    else:
        st.sidebar.info("No report available to export yet.")


def render_agent_status(phase: str, current_agent: str) -> None:
    """
    Render the current agent status indicator.

    Args:
        phase: Current workflow phase
        current_agent: Name of the active agent
    """
    agent_info = {
        "orchestrator": ("🎯", "Orchestrator", "Coordinating workflow"),
        "analyzer": ("🔍", "Analyzer", "Analyzing target website"),
        "designer": ("📝", "Designer", "Designing test cases"),
        "executor": ("⚡", "Executor", "Executing tests"),
        "reporter": ("📊", "Reporter", "Generating report"),
    }

    phase_colors = {
        WorkflowPhase.INIT.value: "blue",
        WorkflowPhase.ANALYZING.value: "orange",
        WorkflowPhase.DESIGNING.value: "violet",
        WorkflowPhase.EXECUTING.value: "green",
        WorkflowPhase.REPORTING.value: "red",
        WorkflowPhase.COMPLETED.value: "rainbow",
        WorkflowPhase.ERROR.value: "red",
    }

    info = agent_info.get(current_agent, ("🤖", current_agent.title(), "Processing"))
    color = phase_colors.get(phase, "gray")

    # Status container
    status_col1, status_col2, status_col3 = st.columns([1, 2, 1])

    with status_col2:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #1e1e2e 0%, #2d2d3d 100%);
                border-radius: 12px;
                padding: 20px;
                text-align: center;
                border: 1px solid #3d3d4d;
            ">
                <div style="font-size: 48px; margin-bottom: 10px;">{info[0]}</div>
                <div style="font-size: 20px; font-weight: bold; color: #fff;">{info[1]}</div>
                <div style="font-size: 14px; color: #888; margin-top: 5px;">{info[2]}</div>
                <div style="margin-top: 15px;">
                    <span style="
                        background: {'#4CAF50' if phase == 'completed' else '#2196F3'};
                        padding: 5px 15px;
                        border-radius: 20px;
                        font-size: 12px;
                        color: white;
                    ">{phase.upper()}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_progress_timeline(workflow_state: dict[str, Any]) -> None:
    """
    Render a visual progress timeline of the workflow.

    Args:
        workflow_state: Current workflow state
    """
    phases = [
        ("init", "🚀", "Initialize"),
        ("analyzing", "🔍", "Analyze"),
        ("designing", "📝", "Design"),
        ("executing", "⚡", "Execute"),
        ("reporting", "📊", "Report"),
        ("completed", "✅", "Complete"),
    ]

    current_phase = workflow_state.get("phase", "init")

    # Find current phase index
    current_idx = 0
    for i, (phase, _, _) in enumerate(phases):
        if phase == current_phase:
            current_idx = i
            break

    # Render timeline
    cols = st.columns(len(phases))

    for i, (phase, icon, label) in enumerate(phases):
        with cols[i]:
            if i < current_idx:
                # Completed phase
                st.markdown(
                    f"""
                    <div style="text-align: center;">
                        <div style="font-size: 24px; opacity: 0.5;">✅</div>
                        <div style="font-size: 12px; color: #4CAF50;">{label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif i == current_idx:
                # Current phase
                st.markdown(
                    f"""
                    <div style="text-align: center;">
                        <div style="font-size: 28px;">{icon}</div>
                        <div style="font-size: 12px; font-weight: bold; color: #2196F3;">{label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                # Future phase
                st.markdown(
                    f"""
                    <div style="text-align: center;">
                        <div style="font-size: 24px; opacity: 0.3;">{icon}</div>
                        <div style="font-size: 12px; color: #666;">{label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_test_results(test_results: list[dict]) -> None:
    """
    Render test results in an organized view.

    Args:
        test_results: List of test result dictionaries
    """
    if not test_results:
        st.info("No test results available yet.")
        return

    # Summary metrics
    total = len(test_results)
    passed = sum(1 for r in test_results if r.get("status") == "passed")
    failed = sum(1 for r in test_results if r.get("status") == "failed")
    skipped = sum(1 for r in test_results if r.get("status") == "skipped")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Tests", total)
    col2.metric("Passed", passed, delta=None)
    col3.metric("Failed", failed, delta=None if failed == 0 else f"-{failed}")
    col4.metric("Skipped", skipped)

    st.divider()

    # Detailed results
    for i, result in enumerate(test_results):
        status = result.get("status", "unknown")
        status_icon = {"passed": "✅", "failed": "❌", "skipped": "⏭️"}.get(status, "❓")

        with st.expander(f"{status_icon} {result.get('test_name', f'Test {i+1}')}"):
            st.markdown(f"**Status:** {status.upper()}")
            st.markdown(f"**Category:** {result.get('category', 'N/A')}")

            if result.get("description"):
                st.markdown(f"**Description:** {result.get('description')}")

            if result.get("error_message"):
                st.error(f"Error: {result.get('error_message')}")

            if result.get("actual_result"):
                st.markdown("**Actual Result:**")
                st.code(result.get("actual_result"))


def _create_empty_result() -> dict[str, Any]:
    """Create an empty result dictionary for workflow operations."""
    return {
        "success": False,
        "workflow_state": None,
        "test_results": None,
        "report": None,
        "error": None,
        "requires_input": False,
        "input_prompt": "",
        "messages": [],
    }


def render_report(report: Any) -> None:
    """
    Render the final test report.

    Args:
        report: TestReport object
    """
    if not report:
        st.info("Report not yet generated.")
        return

    # Report header
    st.markdown("## 📊 Test Report")

    if hasattr(report, "executive_summary"):
        st.markdown("### Executive Summary")
        st.markdown(report.executive_summary)

    if hasattr(report, "summary"):
        summary = report.summary
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total", getattr(summary, "total_tests", 0))
        col2.metric("Passed", getattr(summary, "passed", 0))
        col3.metric("Failed", getattr(summary, "failed", 0))
        col4.metric("Pass Rate", f"{getattr(summary, 'pass_rate', 0):.1f}%")

    if hasattr(report, "recommendations") and report.recommendations:
        st.markdown("### Recommendations")
        for rec in report.recommendations:
            st.markdown(f"- {rec}")

    if hasattr(report, "markdown_content"):
        with st.expander("View Full Report"):
            st.markdown(report.markdown_content)


def add_message(role: str, content: str, msg_type: str = "text") -> None:
    """
    Add a message to the chat history.

    Args:
        role: Message role (user/assistant/system)
        content: Message content
        msg_type: Type of message (text/status/result)
    """
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "type": msg_type,
        "timestamp": datetime.now().isoformat(),
    })


def render_chat_messages() -> None:
    """Render the chat message history."""
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        msg_type = msg.get("type", "text")

        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        elif role == "assistant":
            with st.chat_message("assistant", avatar="🦅"):
                if msg_type == "status":
                    st.info(content)
                elif msg_type == "error":
                    st.error(content)
                elif msg_type == "success":
                    st.success(content)
                else:
                    st.markdown(content)
        elif role == "system":
            with st.chat_message("assistant", avatar="⚙️"):
                st.caption(content)


async def run_workflow_async(
    url: str,
    user_request: str,
    session_id: str,
    persist_state: bool = True,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_api_key: Optional[str] = None,
) -> dict[str, Any]:
    """
    Run the Q-Ravens workflow asynchronously.

    This function runs in a separate thread and returns results
    instead of modifying Streamlit session state directly.

    Args:
        url: Target URL to test
        user_request: Natural language test request
        session_id: Session ID for the workflow
        persist_state: Whether to persist state to disk
        llm_provider: LLM provider to use
        llm_model: Model name to use
        llm_api_key: API key for the provider

    Returns:
        Dictionary with workflow results
    """
    result = _create_empty_result()

    runner = QRavensRunner(
        persist=persist_state,
        session_id=session_id,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
    )

    result["messages"].append({
        "role": "assistant",
        "content": f"Starting QA workflow for **{url}**...",
        "type": "status",
    })

    try:
        async for state in runner.stream(url, user_request):
            result["workflow_state"] = state

            phase = state.get("phase", "unknown")
            current_agent = state.get("current_agent", "starting")

            # Add status message
            result["messages"].append({
                "role": "assistant",
                "content": f"**{current_agent.title()}** is now active (Phase: {phase})",
                "type": "status",
            })

            # Check for human-in-the-loop
            if state.get("requires_user_input"):
                result["requires_input"] = True
                result["input_prompt"] = state.get(
                    "user_input_prompt", "Please provide input:"
                )
                break

            # Store test results
            if state.get("test_results"):
                result["test_results"] = state.get("test_results")

            # Store report
            if state.get("report"):
                result["report"] = state.get("report")

            # Check for completion
            if phase == WorkflowPhase.COMPLETED.value:
                result["success"] = True
                result["messages"].append({
                    "role": "assistant",
                    "content": "Workflow completed successfully!",
                    "type": "success",
                })
                break

            # Check for errors
            if phase == WorkflowPhase.ERROR.value:
                errors = state.get("errors", [])
                error_msg = errors[-1] if errors else "Unknown error occurred"
                result["error"] = error_msg
                result["messages"].append({
                    "role": "assistant",
                    "content": f"Error: {error_msg}",
                    "type": "error",
                })
                break

    except Exception as e:
        result["error"] = str(e)
        result["messages"].append({
            "role": "assistant",
            "content": f"Workflow failed: {str(e)}",
            "type": "error",
        })

    return result


async def resume_workflow_async(
    session_id: str,
    user_input: str,
    persist_state: bool = True,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_api_key: Optional[str] = None,
) -> dict[str, Any]:
    """
    Resume a paused Q-Ravens workflow with user input.

    Args:
        session_id: Session ID for the workflow
        user_input: User's response to the approval prompt
        persist_state: Whether to persist state to disk
        llm_provider: LLM provider to use
        llm_model: Model name to use
        llm_api_key: API key for the provider

    Returns:
        Dictionary with workflow results
    """
    result = _create_empty_result()

    runner = QRavensRunner(
        persist=persist_state,
        session_id=session_id,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
    )

    result["messages"].append({
        "role": "assistant",
        "content": f"Resuming workflow with your response: **{user_input}**",
        "type": "status",
    })

    try:
        # Resume the workflow with user input
        async for state in runner.stream_resume(user_input):
            result["workflow_state"] = state

            phase = state.get("phase", "unknown")
            current_agent = state.get("current_agent", "processing")

            # Add status message
            result["messages"].append({
                "role": "assistant",
                "content": f"**{current_agent.title()}** is now active (Phase: {phase})",
                "type": "status",
            })

            # Check for human-in-the-loop again
            if state.get("requires_user_input"):
                result["requires_input"] = True
                result["input_prompt"] = state.get(
                    "user_input_prompt", "Please provide input:"
                )
                break

            # Store test results
            if state.get("test_results"):
                result["test_results"] = state.get("test_results")

            # Store report
            if state.get("report"):
                result["report"] = state.get("report")

            # Check for completion
            if phase == WorkflowPhase.COMPLETED.value:
                result["success"] = True
                result["messages"].append({
                    "role": "assistant",
                    "content": "Workflow completed successfully!",
                    "type": "success",
                })
                break

            # Check for errors
            if phase == WorkflowPhase.ERROR.value:
                errors = state.get("errors", [])
                error_msg = errors[-1] if errors else "Unknown error occurred"
                result["error"] = error_msg
                result["messages"].append({
                    "role": "assistant",
                    "content": f"Error: {error_msg}",
                    "type": "error",
                })
                break

    except Exception as e:
        result["error"] = str(e)
        result["messages"].append({
            "role": "assistant",
            "content": f"Workflow resume failed: {str(e)}",
            "type": "error",
        })

    return result


def run_workflow(url: str, user_request: str, config: dict[str, Any]) -> None:
    """
    Run the workflow and update Streamlit session state with results.

    Uses run_in_thread to handle Windows asyncio compatibility issues
    with Playwright subprocess creation.

    Args:
        url: Target URL
        user_request: User's test request
        config: Configuration options
    """
    # Extract session state values BEFORE running in thread
    # (st.session_state is not accessible from other threads)
    session_id = st.session_state.session_id
    persist_state = config.get("persist_state", True)

    # Extract LLM configuration
    llm_provider = config.get("llm_provider")
    llm_provider_str = llm_provider.value if llm_provider else None
    llm_model = config.get("llm_model")
    llm_api_key = config.get("llm_api_key")

    try:
        # Run the workflow in a separate thread with proper event loop
        result = run_in_thread(
            run_workflow_async(
                url,
                user_request,
                session_id,
                persist_state,
                llm_provider_str,
                llm_model,
                llm_api_key,
            )
        )

        # Update session state with results (back in main thread)
        if result:
            # Add all messages from the workflow
            for msg in result.get("messages", []):
                st.session_state.messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                    "type": msg["type"],
                    "timestamp": datetime.now().isoformat(),
                })

            # Update workflow state
            if result.get("workflow_state"):
                st.session_state.workflow_state = result["workflow_state"]

            # Update test results
            if result.get("test_results"):
                st.session_state.test_results = result["test_results"]

            # Update report
            if result.get("report"):
                st.session_state.report = result["report"]

            # Handle human-in-the-loop
            if result.get("requires_input"):
                st.session_state.requires_input = True
                st.session_state.input_prompt = result["input_prompt"]

    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Workflow execution failed: {str(e)}",
            "type": "error",
            "timestamp": datetime.now().isoformat(),
        })

    finally:
        st.session_state.is_running = False


def resume_workflow(user_input: str, config: dict[str, Any]) -> None:
    """
    Resume a paused workflow with user input and update Streamlit session state.

    Args:
        user_input: User's response to the approval prompt
        config: Configuration options
    """
    session_id = st.session_state.session_id
    persist_state = config.get("persist_state", True)

    # Extract LLM configuration
    llm_provider = config.get("llm_provider")
    llm_provider_str = llm_provider.value if llm_provider else None
    llm_model = config.get("llm_model")
    llm_api_key = config.get("llm_api_key")

    try:
        # Resume the workflow in a separate thread with proper event loop
        result = run_in_thread(
            resume_workflow_async(
                session_id,
                user_input,
                persist_state,
                llm_provider_str,
                llm_model,
                llm_api_key,
            )
        )

        # Update session state with results (back in main thread)
        if result:
            # Add all messages from the workflow
            for msg in result.get("messages", []):
                st.session_state.messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                    "type": msg["type"],
                    "timestamp": datetime.now().isoformat(),
                })

            # Update workflow state
            if result.get("workflow_state"):
                st.session_state.workflow_state = result["workflow_state"]

            # Update test results
            if result.get("test_results"):
                st.session_state.test_results = result["test_results"]

            # Update report
            if result.get("report"):
                st.session_state.report = result["report"]

            # Handle human-in-the-loop (if more approval needed)
            if result.get("requires_input"):
                st.session_state.requires_input = True
                st.session_state.input_prompt = result["input_prompt"]

    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Workflow resume failed: {str(e)}",
            "type": "error",
            "timestamp": datetime.now().isoformat(),
        })

    finally:
        st.session_state.is_running = False


def main() -> None:
    """Main entry point for the Streamlit chat UI."""
    init_session_state()

    # Render sidebar and get config
    config = render_sidebar()

    # Main content area
    st.title("🦅 Q-Ravens Chat")
    st.caption("Autonomous QA Agent Swarm - Chat Interface")

    # Show workflow progress if running
    if st.session_state.workflow_state:
        with st.container():
            render_progress_timeline(st.session_state.workflow_state)

            phase = st.session_state.workflow_state.get("phase", "init")
            current_agent = st.session_state.workflow_state.get("current_agent", "")

            if phase not in [WorkflowPhase.COMPLETED.value, WorkflowPhase.ERROR.value]:
                render_agent_status(phase, current_agent)

    st.divider()

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📋 Results", "📊 Report"])

    with tab1:
        # Chat interface
        render_chat_messages()

        # Human-in-the-loop input
        if st.session_state.requires_input:
            st.warning(st.session_state.input_prompt)
            user_input = st.text_input("Your response:", key="human_input")
            if st.button("Submit Response"):
                if user_input:
                    add_message("user", user_input)
                    st.session_state.requires_input = False
                    st.session_state.is_running = True
                    # Resume workflow with user input
                    resume_workflow(user_input, config)
                    st.rerun()

        # Chat input
        if prompt := st.chat_input(
            "Describe what you want to test...",
            disabled=st.session_state.is_running,
        ):
            # Add user message
            add_message("user", prompt)

            # Check if URL is provided
            target_url = config.get("target_url", "").strip()

            if not target_url:
                add_message(
                    "assistant",
                    "Please enter a target URL in the sidebar before starting tests.",
                    "error",
                )
            elif not target_url.startswith(("http://", "https://")):
                add_message(
                    "assistant",
                    "Please enter a valid URL starting with http:// or https://",
                    "error",
                )
            else:
                # Build the test request from user input and scope
                test_scope = config.get("test_scope", [])
                full_request = f"{prompt}\n\nTest scope: {', '.join(test_scope)}"

                # Start workflow
                st.session_state.is_running = True
                run_workflow(target_url, full_request, config)

            st.rerun()

    with tab2:
        # Test results view
        if st.session_state.test_results:
            render_test_results(st.session_state.test_results)
        else:
            st.info("Test results will appear here once tests are executed.")

    with tab3:
        # Report view
        if st.session_state.report:
            render_report(st.session_state.report)
        else:
            st.info("The test report will appear here after the workflow completes.")

    # Footer
    st.divider()
    st.caption(
        "Q-Ravens - Autonomous QA Agent Swarm | "
        f"Session: {st.session_state.session_id[:8]}..."
    )


if __name__ == "__main__":
    main()
