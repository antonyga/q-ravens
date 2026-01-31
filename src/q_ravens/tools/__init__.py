"""
Q-Ravens Tools Module

Contains integrations with testing tools:
- Playwright: Browser automation
- Vision: Screenshot capture and SoM overlay
- Lighthouse: Performance testing
- Axe-core: Accessibility testing
- OWASP ZAP: Security scanning (future)
"""

from q_ravens.tools.browser import BrowserTool
from q_ravens.tools.vision import (
    VisionService,
    VisualState,
    InteractiveElement,
    BoundingBox,
    capture_page_state,
)
from q_ravens.tools.actions import (
    ActionExecutor,
    ActionResult,
    ClickAction,
    TypeAction,
    ScrollAction,
    WaitAction,
    HotkeyAction,
    DragAction,
    MouseButton,
    ScrollDirection,
    create_action_from_dict,
)

__all__ = [
    # Browser
    "BrowserTool",
    # Vision
    "VisionService",
    "VisualState",
    "InteractiveElement",
    "BoundingBox",
    "capture_page_state",
    # Actions
    "ActionExecutor",
    "ActionResult",
    "ClickAction",
    "TypeAction",
    "ScrollAction",
    "WaitAction",
    "HotkeyAction",
    "DragAction",
    "MouseButton",
    "ScrollDirection",
    "create_action_from_dict",
]
