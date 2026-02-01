"""
Q-Ravens UI Settings

Configuration and settings management for the Streamlit chat interface.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum


class Theme(Enum):
    """UI theme options."""

    DARK = "dark"
    LIGHT = "light"
    SYSTEM = "system"


class TestScope(Enum):
    """Available test scope options."""

    SMOKE = "Smoke Tests"
    FUNCTIONAL = "Functional Tests"
    UI_UX = "UI/UX Tests"
    PERFORMANCE = "Performance Tests"
    ACCESSIBILITY = "Accessibility Tests"
    SECURITY = "Security Tests"


@dataclass
class UISettings:
    """
    UI configuration settings.

    Stores user preferences for the chat interface.
    """

    # Theme settings
    theme: Theme = Theme.DARK
    show_agent_cards: bool = True
    show_progress_timeline: bool = True
    compact_mode: bool = False

    # Workflow settings
    default_test_scopes: list[TestScope] = field(
        default_factory=lambda: [TestScope.SMOKE, TestScope.FUNCTIONAL]
    )
    max_iterations: int = 10
    persist_state: bool = True
    require_human_approval: bool = True
    auto_export_reports: bool = False

    # Performance settings
    refresh_interval_ms: int = 1000
    max_chat_history: int = 100

    # Export settings
    export_format: str = "markdown"
    include_screenshots: bool = True
    include_raw_data: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            "theme": self.theme.value,
            "show_agent_cards": self.show_agent_cards,
            "show_progress_timeline": self.show_progress_timeline,
            "compact_mode": self.compact_mode,
            "default_test_scopes": [s.value for s in self.default_test_scopes],
            "max_iterations": self.max_iterations,
            "persist_state": self.persist_state,
            "require_human_approval": self.require_human_approval,
            "auto_export_reports": self.auto_export_reports,
            "refresh_interval_ms": self.refresh_interval_ms,
            "max_chat_history": self.max_chat_history,
            "export_format": self.export_format,
            "include_screenshots": self.include_screenshots,
            "include_raw_data": self.include_raw_data,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UISettings":
        """Create settings from dictionary."""
        return cls(
            theme=Theme(data.get("theme", "dark")),
            show_agent_cards=data.get("show_agent_cards", True),
            show_progress_timeline=data.get("show_progress_timeline", True),
            compact_mode=data.get("compact_mode", False),
            default_test_scopes=[
                TestScope(s) for s in data.get("default_test_scopes", ["Smoke Tests", "Functional Tests"])
            ],
            max_iterations=data.get("max_iterations", 10),
            persist_state=data.get("persist_state", True),
            require_human_approval=data.get("require_human_approval", True),
            auto_export_reports=data.get("auto_export_reports", False),
            refresh_interval_ms=data.get("refresh_interval_ms", 1000),
            max_chat_history=data.get("max_chat_history", 100),
            export_format=data.get("export_format", "markdown"),
            include_screenshots=data.get("include_screenshots", True),
            include_raw_data=data.get("include_raw_data", False),
        )


@dataclass
class WorkflowConfig:
    """
    Workflow configuration for a test run.

    Passed to the QRavensRunner when starting a new workflow.
    """

    target_url: str
    user_request: str
    test_scopes: list[TestScope] = field(default_factory=list)
    max_iterations: int = 10
    persist_state: bool = True
    require_human_approval: bool = True
    session_id: Optional[str] = None

    def get_full_request(self) -> str:
        """Generate full test request including scope."""
        scope_str = ", ".join(s.value for s in self.test_scopes)
        if scope_str:
            return f"{self.user_request}\n\nTest scope: {scope_str}"
        return self.user_request


# Default configuration
DEFAULT_SETTINGS = UISettings()


# Streamlit page configuration
PAGE_CONFIG = {
    "page_title": "Q-Ravens QA Agent",
    "page_icon": "🦅",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "menu_items": {
        "Get Help": "https://github.com/your-org/q-ravens",
        "Report a bug": "https://github.com/your-org/q-ravens/issues",
        "About": """
        # Q-Ravens
        Autonomous QA Agent Swarm

        An intelligent multi-agent system for comprehensive web application testing.

        Built with LangGraph, Playwright, and Streamlit.
        """,
    },
}


# CSS Theme configurations
THEMES = {
    "dark": {
        "background": "#0e1117",
        "secondary_background": "#1e1e2e",
        "text": "#ffffff",
        "text_secondary": "#888888",
        "primary": "#2196F3",
        "success": "#4CAF50",
        "warning": "#FF9800",
        "error": "#F44336",
        "border": "#3d3d4d",
    },
    "light": {
        "background": "#ffffff",
        "secondary_background": "#f5f5f5",
        "text": "#1e1e2e",
        "text_secondary": "#666666",
        "primary": "#1976D2",
        "success": "#388E3C",
        "warning": "#F57C00",
        "error": "#D32F2F",
        "border": "#e0e0e0",
    },
}
