"""
Q-Ravens UI Module

Contains user interface components:
- CLI: Command-line interface
- Web: Streamlit chat interface (Phase 3)
"""

from q_ravens.ui.chat import main as run_chat_ui
from q_ravens.ui.components import (
    AGENT_CONFIGS,
    AgentInfo,
    StatusColor,
    apply_custom_css,
    render_agent_card,
    render_error_banner,
    render_loading_spinner,
    render_message_bubble,
    render_metric_card,
    render_progress_bar,
    render_stat_row,
    render_success_banner,
    render_test_result_card,
    render_workflow_step,
)
from q_ravens.ui.settings import (
    DEFAULT_SETTINGS,
    PAGE_CONFIG,
    THEMES,
    TestScope,
    Theme,
    UISettings,
    WorkflowConfig,
)

__all__ = [
    # Main entry point
    "run_chat_ui",
    # Components
    "AGENT_CONFIGS",
    "AgentInfo",
    "StatusColor",
    "apply_custom_css",
    "render_agent_card",
    "render_error_banner",
    "render_loading_spinner",
    "render_message_bubble",
    "render_metric_card",
    "render_progress_bar",
    "render_stat_row",
    "render_success_banner",
    "render_test_result_card",
    "render_workflow_step",
    # Settings
    "DEFAULT_SETTINGS",
    "PAGE_CONFIG",
    "THEMES",
    "TestScope",
    "Theme",
    "UISettings",
    "WorkflowConfig",
]
