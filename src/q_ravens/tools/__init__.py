"""
Q-Ravens Tools Module

Contains integrations with testing tools:
- Playwright: Browser automation
- Vision: Screenshot capture and SoM overlay
- Lighthouse: Performance testing (Core Web Vitals)
- Axe-core: Accessibility testing (WCAG 2.1)
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
from q_ravens.tools.lighthouse import (
    LighthouseTool,
    LighthouseResult,
    CoreWebVitals,
    PerformanceAudit,
    PerformanceCategory,
    run_lighthouse_audit,
)
from q_ravens.tools.accessibility import (
    AccessibilityTool,
    AccessibilityResult,
    AccessibilityViolation,
    WCAGLevel,
    ImpactLevel,
    run_accessibility_audit,
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
    # Lighthouse (Performance)
    "LighthouseTool",
    "LighthouseResult",
    "CoreWebVitals",
    "PerformanceAudit",
    "PerformanceCategory",
    "run_lighthouse_audit",
    # Accessibility (axe-core)
    "AccessibilityTool",
    "AccessibilityResult",
    "AccessibilityViolation",
    "WCAGLevel",
    "ImpactLevel",
    "run_accessibility_audit",
]
