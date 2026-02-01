"""
Q-Ravens UI Components

Reusable Streamlit components for the Q-Ravens chat interface.
Includes widgets for displaying test progress, results, and reports.
"""

import streamlit as st
from typing import Any, Optional
from dataclasses import dataclass
from enum import Enum


class StatusColor(Enum):
    """Status indicator colors."""

    SUCCESS = "#4CAF50"
    WARNING = "#FF9800"
    ERROR = "#F44336"
    INFO = "#2196F3"
    NEUTRAL = "#9E9E9E"


@dataclass
class AgentInfo:
    """Information about an agent for display."""

    name: str
    icon: str
    title: str
    description: str
    color: str = "#2196F3"


# Agent display configurations
AGENT_CONFIGS = {
    "orchestrator": AgentInfo(
        name="orchestrator",
        icon="🎯",
        title="Orchestrator",
        description="Coordinating workflow and delegating tasks",
        color="#9C27B0",
    ),
    "analyzer": AgentInfo(
        name="analyzer",
        icon="🔍",
        title="Analyzer",
        description="Analyzing target website structure and functionality",
        color="#FF9800",
    ),
    "designer": AgentInfo(
        name="designer",
        icon="📝",
        title="Designer",
        description="Designing comprehensive test cases",
        color="#3F51B5",
    ),
    "executor": AgentInfo(
        name="executor",
        icon="⚡",
        title="Executor",
        description="Executing automated tests",
        color="#4CAF50",
    ),
    "reporter": AgentInfo(
        name="reporter",
        icon="📊",
        title="Reporter",
        description="Generating detailed test reports",
        color="#E91E63",
    ),
}


def render_metric_card(
    label: str,
    value: Any,
    icon: str = "",
    delta: Optional[str] = None,
    color: str = "#2196F3",
) -> None:
    """
    Render a styled metric card.

    Args:
        label: Metric label
        value: Metric value
        icon: Optional icon
        delta: Optional delta indicator
        color: Card accent color
    """
    delta_html = ""
    if delta:
        delta_color = StatusColor.SUCCESS.value if delta.startswith("+") else StatusColor.ERROR.value
        delta_html = f'<div style="color: {delta_color}; font-size: 12px;">{delta}</div>'

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #1e1e2e 0%, #2d2d3d 100%);
            border-radius: 10px;
            padding: 15px;
            border-left: 4px solid {color};
        ">
            <div style="font-size: 12px; color: #888; text-transform: uppercase;">{icon} {label}</div>
            <div style="font-size: 28px; font-weight: bold; color: #fff; margin-top: 5px;">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_agent_card(agent_name: str, is_active: bool = False) -> None:
    """
    Render an agent status card.

    Args:
        agent_name: Name of the agent
        is_active: Whether the agent is currently active
    """
    config = AGENT_CONFIGS.get(agent_name, AgentInfo(
        name=agent_name,
        icon="🤖",
        title=agent_name.title(),
        description="Processing",
        color="#9E9E9E",
    ))

    border_style = f"2px solid {config.color}" if is_active else "1px solid #3d3d4d"
    opacity = "1" if is_active else "0.6"

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #1e1e2e 0%, #2d2d3d 100%);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            border: {border_style};
            opacity: {opacity};
            transition: all 0.3s ease;
        ">
            <div style="font-size: 36px; margin-bottom: 10px;">{config.icon}</div>
            <div style="font-size: 16px; font-weight: bold; color: #fff;">{config.title}</div>
            <div style="font-size: 12px; color: #888; margin-top: 5px;">{config.description}</div>
            {'<div style="margin-top: 10px;"><span style="background: ' + config.color + '; padding: 3px 10px; border-radius: 10px; font-size: 10px; color: white;">ACTIVE</span></div>' if is_active else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_test_result_card(
    test_name: str,
    status: str,
    category: str = "",
    description: str = "",
    error_message: str = "",
) -> None:
    """
    Render a test result card.

    Args:
        test_name: Name of the test
        status: Test status (passed/failed/skipped)
        category: Test category
        description: Test description
        error_message: Error message if failed
    """
    status_config = {
        "passed": ("✅", StatusColor.SUCCESS.value, "PASSED"),
        "failed": ("❌", StatusColor.ERROR.value, "FAILED"),
        "skipped": ("⏭️", StatusColor.WARNING.value, "SKIPPED"),
    }

    icon, color, label = status_config.get(status, ("❓", StatusColor.NEUTRAL.value, "UNKNOWN"))

    error_html = ""
    if error_message:
        error_html = f"""
        <div style="
            background: rgba(244, 67, 54, 0.1);
            border-radius: 5px;
            padding: 10px;
            margin-top: 10px;
            border-left: 3px solid {StatusColor.ERROR.value};
        ">
            <div style="font-size: 12px; color: #F44336;">{error_message}</div>
        </div>
        """

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #1e1e2e 0%, #2d2d3d 100%);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px;
            border-left: 4px solid {color};
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 18px; margin-right: 10px;">{icon}</span>
                    <span style="font-size: 14px; font-weight: bold; color: #fff;">{test_name}</span>
                </div>
                <span style="
                    background: {color};
                    padding: 3px 10px;
                    border-radius: 10px;
                    font-size: 10px;
                    color: white;
                ">{label}</span>
            </div>
            {f'<div style="font-size: 12px; color: #888; margin-top: 5px;">Category: {category}</div>' if category else ''}
            {f'<div style="font-size: 12px; color: #aaa; margin-top: 5px;">{description}</div>' if description else ''}
            {error_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_progress_bar(
    current: int,
    total: int,
    label: str = "",
    color: str = "#2196F3",
) -> None:
    """
    Render a custom progress bar.

    Args:
        current: Current progress value
        total: Total value
        label: Optional label
        color: Bar color
    """
    percentage = (current / total * 100) if total > 0 else 0

    st.markdown(
        f"""
        <div style="margin-bottom: 10px;">
            {f'<div style="font-size: 12px; color: #888; margin-bottom: 5px;">{label}</div>' if label else ''}
            <div style="
                background: #3d3d4d;
                border-radius: 10px;
                height: 8px;
                overflow: hidden;
            ">
                <div style="
                    background: linear-gradient(90deg, {color} 0%, {color}88 100%);
                    width: {percentage}%;
                    height: 100%;
                    border-radius: 10px;
                    transition: width 0.5s ease;
                "></div>
            </div>
            <div style="font-size: 10px; color: #666; margin-top: 3px; text-align: right;">
                {current}/{total} ({percentage:.0f}%)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workflow_step(
    step_name: str,
    status: str,
    icon: str = "⏳",
    details: str = "",
) -> None:
    """
    Render a workflow step indicator.

    Args:
        step_name: Name of the step
        status: Step status (pending/active/completed/error)
        icon: Step icon
        details: Additional details
    """
    status_styles = {
        "pending": {"bg": "#3d3d4d", "border": "#555", "opacity": "0.5"},
        "active": {"bg": "#2196F3", "border": "#2196F3", "opacity": "1"},
        "completed": {"bg": "#4CAF50", "border": "#4CAF50", "opacity": "0.8"},
        "error": {"bg": "#F44336", "border": "#F44336", "opacity": "1"},
    }

    style = status_styles.get(status, status_styles["pending"])

    st.markdown(
        f"""
        <div style="
            display: flex;
            align-items: center;
            padding: 10px;
            background: {style['bg']}22;
            border-radius: 8px;
            border: 1px solid {style['border']};
            margin-bottom: 8px;
            opacity: {style['opacity']};
        ">
            <span style="font-size: 20px; margin-right: 12px;">{icon}</span>
            <div>
                <div style="font-size: 14px; font-weight: bold; color: #fff;">{step_name}</div>
                {f'<div style="font-size: 11px; color: #888;">{details}</div>' if details else ''}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stat_row(stats: list[tuple[str, Any, str]]) -> None:
    """
    Render a row of statistics.

    Args:
        stats: List of (label, value, icon) tuples
    """
    cols = st.columns(len(stats))
    for col, (label, value, icon) in zip(cols, stats):
        with col:
            st.markdown(
                f"""
                <div style="text-align: center; padding: 10px;">
                    <div style="font-size: 24px;">{icon}</div>
                    <div style="font-size: 20px; font-weight: bold; color: #fff;">{value}</div>
                    <div style="font-size: 11px; color: #888; text-transform: uppercase;">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_message_bubble(
    content: str,
    role: str = "assistant",
    timestamp: str = "",
) -> None:
    """
    Render a chat message bubble.

    Args:
        content: Message content
        role: Message role (user/assistant)
        timestamp: Optional timestamp
    """
    if role == "user":
        align = "flex-end"
        bg_color = "#2196F3"
        text_color = "#fff"
    else:
        align = "flex-start"
        bg_color = "#2d2d3d"
        text_color = "#fff"

    st.markdown(
        f"""
        <div style="display: flex; justify-content: {align}; margin-bottom: 10px;">
            <div style="
                background: {bg_color};
                color: {text_color};
                padding: 12px 16px;
                border-radius: 18px;
                max-width: 70%;
                font-size: 14px;
            ">
                {content}
                {f'<div style="font-size: 10px; color: #888; margin-top: 5px;">{timestamp}</div>' if timestamp else ''}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_loading_spinner(message: str = "Processing...") -> None:
    """
    Render a custom loading spinner with message.

    Args:
        message: Loading message to display
    """
    st.markdown(
        f"""
        <div style="
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        ">
            <div style="
                width: 20px;
                height: 20px;
                border: 3px solid #3d3d4d;
                border-top: 3px solid #2196F3;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-right: 12px;
            "></div>
            <span style="color: #888;">{message}</span>
        </div>
        <style>
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_error_banner(message: str, details: str = "") -> None:
    """
    Render an error banner.

    Args:
        message: Error message
        details: Additional error details
    """
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #5c1e1e 0%, #3d1515 100%);
            border-radius: 10px;
            padding: 15px;
            border-left: 4px solid #F44336;
            margin-bottom: 15px;
        ">
            <div style="display: flex; align-items: center;">
                <span style="font-size: 24px; margin-right: 12px;">⚠️</span>
                <div>
                    <div style="font-size: 14px; font-weight: bold; color: #fff;">{message}</div>
                    {f'<div style="font-size: 12px; color: #aaa; margin-top: 5px;">{details}</div>' if details else ''}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_success_banner(message: str, details: str = "") -> None:
    """
    Render a success banner.

    Args:
        message: Success message
        details: Additional details
    """
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #1e5c1e 0%, #153d15 100%);
            border-radius: 10px;
            padding: 15px;
            border-left: 4px solid #4CAF50;
            margin-bottom: 15px;
        ">
            <div style="display: flex; align-items: center;">
                <span style="font-size: 24px; margin-right: 12px;">✅</span>
                <div>
                    <div style="font-size: 14px; font-weight: bold; color: #fff;">{message}</div>
                    {f'<div style="font-size: 12px; color: #aaa; margin-top: 5px;">{details}</div>' if details else ''}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_custom_css() -> None:
    """Apply custom CSS styles to the Streamlit app."""
    st.markdown(
        """
        <style>
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #1e1e2e;
        }
        ::-webkit-scrollbar-thumb {
            background: #3d3d4d;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #555;
        }

        /* Chat input styling */
        .stChatInput {
            border-radius: 20px;
        }

        /* Button styling */
        .stButton button {
            border-radius: 8px;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
        }

        /* Expander styling */
        .streamlit-expanderHeader {
            border-radius: 8px;
        }

        /* Metric styling */
        [data-testid="stMetricValue"] {
            font-size: 28px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
