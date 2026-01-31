"""
Q-Ravens Action Primitives

Implements the "Act" component of the See-Think-Act-Reflect loop.
Provides platform-agnostic action primitives that translate decisions
into executable input commands.

Action Types (from PRD):
- click: Mouse click at coordinates
- type: Keyboard text entry
- scroll: Scroll within viewport or element
- wait: Wait for visual condition
- hotkey: Keyboard shortcuts
- drag: Drag and drop operations
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Optional, Union

from playwright.async_api import Page


class MouseButton(str, Enum):
    """Mouse button options."""

    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


class ScrollDirection(str, Enum):
    """Scroll direction options."""

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


@dataclass
class ActionResult:
    """Result of executing an action."""

    success: bool
    action_type: str
    message: str
    details: dict = None
    screenshot_after: Optional[str] = None  # Base64 screenshot after action

    def __post_init__(self):
        if self.details is None:
            self.details = {}


@dataclass
class ClickAction:
    """Click action at coordinates."""

    x: float
    y: float
    button: MouseButton = MouseButton.LEFT
    click_count: int = 1
    delay_ms: int = 0  # Delay between multiple clicks


@dataclass
class TypeAction:
    """Type text action."""

    text: str
    clear_first: bool = False
    delay_ms: int = 50  # Delay between keystrokes


@dataclass
class ScrollAction:
    """Scroll action."""

    direction: ScrollDirection
    amount: int = 300  # Pixels to scroll
    x: Optional[float] = None  # Target x coordinate (for element scroll)
    y: Optional[float] = None  # Target y coordinate


@dataclass
class WaitAction:
    """Wait for condition action."""

    condition: Literal["visible", "hidden", "stable", "network_idle", "timeout"]
    timeout_ms: int = 5000
    selector: Optional[str] = None  # For selector-based conditions
    x: Optional[float] = None  # For coordinate-based waiting
    y: Optional[float] = None


@dataclass
class HotkeyAction:
    """Keyboard hotkey action."""

    keys: list[str]  # e.g., ["Control", "c"] for Ctrl+C


@dataclass
class DragAction:
    """Drag and drop action."""

    start_x: float
    start_y: float
    end_x: float
    end_y: float
    steps: int = 10  # Number of intermediate steps


# Union type for all actions
Action = Union[ClickAction, TypeAction, ScrollAction, WaitAction, HotkeyAction, DragAction]


class ActionExecutor:
    """
    Executes action primitives on a Playwright page.

    This is the "Act" layer that translates high-level actions
    into actual browser interactions using coordinates.
    """

    def __init__(
        self,
        page: Page,
        action_delay_ms: int = 100,
        capture_screenshots: bool = True,
    ):
        """
        Initialize the action executor.

        Args:
            page: Playwright Page instance
            action_delay_ms: Default delay after each action
            capture_screenshots: Whether to capture screenshots after actions
        """
        self.page = page
        self.action_delay_ms = action_delay_ms
        self.capture_screenshots = capture_screenshots
        self._action_history: list[ActionResult] = []

    @property
    def action_history(self) -> list[ActionResult]:
        """Get the history of executed actions."""
        return self._action_history.copy()

    async def execute(self, action: Action) -> ActionResult:
        """
        Execute a single action.

        Args:
            action: The action to execute

        Returns:
            ActionResult with success status and details
        """
        if isinstance(action, ClickAction):
            result = await self._execute_click(action)
        elif isinstance(action, TypeAction):
            result = await self._execute_type(action)
        elif isinstance(action, ScrollAction):
            result = await self._execute_scroll(action)
        elif isinstance(action, WaitAction):
            result = await self._execute_wait(action)
        elif isinstance(action, HotkeyAction):
            result = await self._execute_hotkey(action)
        elif isinstance(action, DragAction):
            result = await self._execute_drag(action)
        else:
            result = ActionResult(
                success=False,
                action_type="unknown",
                message=f"Unknown action type: {type(action).__name__}",
            )

        # Add delay after action
        if result.success and self.action_delay_ms > 0:
            await asyncio.sleep(self.action_delay_ms / 1000)

        # Record in history
        self._action_history.append(result)

        return result

    async def execute_sequence(
        self,
        actions: list[Action],
        stop_on_failure: bool = True,
    ) -> list[ActionResult]:
        """
        Execute a sequence of actions.

        Args:
            actions: List of actions to execute
            stop_on_failure: Whether to stop on first failure

        Returns:
            List of ActionResults
        """
        results = []
        for action in actions:
            result = await self.execute(action)
            results.append(result)

            if not result.success and stop_on_failure:
                break

        return results

    async def _execute_click(self, action: ClickAction) -> ActionResult:
        """Execute a click action."""
        try:
            # Map button to playwright button name
            button_map = {
                MouseButton.LEFT: "left",
                MouseButton.RIGHT: "right",
                MouseButton.MIDDLE: "middle",
            }

            await self.page.mouse.click(
                action.x,
                action.y,
                button=button_map[action.button],
                click_count=action.click_count,
                delay=action.delay_ms,
            )

            return ActionResult(
                success=True,
                action_type="click",
                message=f"Clicked at ({action.x}, {action.y})",
                details={
                    "x": action.x,
                    "y": action.y,
                    "button": action.button.value,
                    "click_count": action.click_count,
                },
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action_type="click",
                message=f"Click failed: {str(e)}",
                details={"error": str(e), "x": action.x, "y": action.y},
            )

    async def _execute_type(self, action: TypeAction) -> ActionResult:
        """Execute a type action."""
        try:
            if action.clear_first:
                # Select all and delete
                await self.page.keyboard.press("Control+a")
                await self.page.keyboard.press("Backspace")
                await asyncio.sleep(0.05)

            # Type the text with delay between keystrokes
            await self.page.keyboard.type(action.text, delay=action.delay_ms)

            return ActionResult(
                success=True,
                action_type="type",
                message=f"Typed {len(action.text)} characters",
                details={
                    "text_length": len(action.text),
                    "cleared_first": action.clear_first,
                },
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action_type="type",
                message=f"Type failed: {str(e)}",
                details={"error": str(e)},
            )

    async def _execute_scroll(self, action: ScrollAction) -> ActionResult:
        """Execute a scroll action."""
        try:
            # Calculate scroll delta
            delta_x = 0
            delta_y = 0

            if action.direction == ScrollDirection.UP:
                delta_y = -action.amount
            elif action.direction == ScrollDirection.DOWN:
                delta_y = action.amount
            elif action.direction == ScrollDirection.LEFT:
                delta_x = -action.amount
            elif action.direction == ScrollDirection.RIGHT:
                delta_x = action.amount

            # If coordinates provided, move mouse there first
            if action.x is not None and action.y is not None:
                await self.page.mouse.move(action.x, action.y)

            # Execute scroll
            await self.page.mouse.wheel(delta_x, delta_y)

            return ActionResult(
                success=True,
                action_type="scroll",
                message=f"Scrolled {action.direction.value} by {action.amount}px",
                details={
                    "direction": action.direction.value,
                    "amount": action.amount,
                    "delta_x": delta_x,
                    "delta_y": delta_y,
                },
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action_type="scroll",
                message=f"Scroll failed: {str(e)}",
                details={"error": str(e)},
            )

    async def _execute_wait(self, action: WaitAction) -> ActionResult:
        """Execute a wait action."""
        try:
            if action.condition == "network_idle":
                await self.page.wait_for_load_state(
                    "networkidle", timeout=action.timeout_ms
                )
            elif action.condition == "timeout":
                await asyncio.sleep(action.timeout_ms / 1000)
            elif action.condition == "stable":
                # Wait for page to be visually stable (no layout shifts)
                await self._wait_for_stable(timeout_ms=action.timeout_ms)
            elif action.selector:
                # Selector-based waiting
                if action.condition == "visible":
                    await self.page.wait_for_selector(
                        action.selector, state="visible", timeout=action.timeout_ms
                    )
                elif action.condition == "hidden":
                    await self.page.wait_for_selector(
                        action.selector, state="hidden", timeout=action.timeout_ms
                    )

            return ActionResult(
                success=True,
                action_type="wait",
                message=f"Wait for {action.condition} completed",
                details={
                    "condition": action.condition,
                    "timeout_ms": action.timeout_ms,
                },
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action_type="wait",
                message=f"Wait failed: {str(e)}",
                details={"error": str(e), "condition": action.condition},
            )

    async def _execute_hotkey(self, action: HotkeyAction) -> ActionResult:
        """Execute a hotkey action."""
        try:
            # Build key combination string
            key_combo = "+".join(action.keys)
            await self.page.keyboard.press(key_combo)

            return ActionResult(
                success=True,
                action_type="hotkey",
                message=f"Pressed {key_combo}",
                details={"keys": action.keys, "combo": key_combo},
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action_type="hotkey",
                message=f"Hotkey failed: {str(e)}",
                details={"error": str(e), "keys": action.keys},
            )

    async def _execute_drag(self, action: DragAction) -> ActionResult:
        """Execute a drag and drop action."""
        try:
            # Move to start position
            await self.page.mouse.move(action.start_x, action.start_y)

            # Press mouse button
            await self.page.mouse.down()

            # Move to end position with intermediate steps
            step_x = (action.end_x - action.start_x) / action.steps
            step_y = (action.end_y - action.start_y) / action.steps

            for i in range(1, action.steps + 1):
                x = action.start_x + step_x * i
                y = action.start_y + step_y * i
                await self.page.mouse.move(x, y)
                await asyncio.sleep(0.01)

            # Release mouse button
            await self.page.mouse.up()

            return ActionResult(
                success=True,
                action_type="drag",
                message=f"Dragged from ({action.start_x}, {action.start_y}) to ({action.end_x}, {action.end_y})",
                details={
                    "start": (action.start_x, action.start_y),
                    "end": (action.end_x, action.end_y),
                    "steps": action.steps,
                },
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action_type="drag",
                message=f"Drag failed: {str(e)}",
                details={"error": str(e)},
            )

    async def _wait_for_stable(self, timeout_ms: int = 5000, stability_ms: int = 500):
        """
        Wait for the page to become visually stable.

        Args:
            timeout_ms: Maximum time to wait
            stability_ms: Time without changes to consider stable
        """
        start_time = asyncio.get_event_loop().time()
        last_change_time = start_time
        last_screenshot = None

        while (asyncio.get_event_loop().time() - start_time) * 1000 < timeout_ms:
            # Take a screenshot
            screenshot = await self.page.screenshot(type="png")

            if last_screenshot is not None and screenshot != last_screenshot:
                last_change_time = asyncio.get_event_loop().time()

            last_screenshot = screenshot

            # Check if stable
            if (asyncio.get_event_loop().time() - last_change_time) * 1000 >= stability_ms:
                return

            await asyncio.sleep(0.1)

    # Convenience methods for common actions

    async def click_at(
        self,
        x: float,
        y: float,
        button: MouseButton = MouseButton.LEFT,
    ) -> ActionResult:
        """Click at specific coordinates."""
        return await self.execute(ClickAction(x=x, y=y, button=button))

    async def double_click_at(self, x: float, y: float) -> ActionResult:
        """Double-click at specific coordinates."""
        return await self.execute(ClickAction(x=x, y=y, click_count=2))

    async def right_click_at(self, x: float, y: float) -> ActionResult:
        """Right-click at specific coordinates."""
        return await self.execute(ClickAction(x=x, y=y, button=MouseButton.RIGHT))

    async def type_text(self, text: str, clear_first: bool = False) -> ActionResult:
        """Type text at current cursor position."""
        return await self.execute(TypeAction(text=text, clear_first=clear_first))

    async def scroll_down(self, amount: int = 300) -> ActionResult:
        """Scroll down by specified amount."""
        return await self.execute(ScrollAction(direction=ScrollDirection.DOWN, amount=amount))

    async def scroll_up(self, amount: int = 300) -> ActionResult:
        """Scroll up by specified amount."""
        return await self.execute(ScrollAction(direction=ScrollDirection.UP, amount=amount))

    async def press_keys(self, *keys: str) -> ActionResult:
        """Press a key combination."""
        return await self.execute(HotkeyAction(keys=list(keys)))

    async def wait_for_network_idle(self, timeout_ms: int = 5000) -> ActionResult:
        """Wait for network to be idle."""
        return await self.execute(
            WaitAction(condition="network_idle", timeout_ms=timeout_ms)
        )

    async def wait_ms(self, ms: int) -> ActionResult:
        """Wait for specified milliseconds."""
        return await self.execute(WaitAction(condition="timeout", timeout_ms=ms))


def create_action_from_dict(data: dict) -> Action:
    """
    Create an Action object from a dictionary.

    This is useful for deserializing actions from LLM outputs.

    Args:
        data: Dictionary with action parameters

    Returns:
        Appropriate Action instance
    """
    action_type = data.get("action", data.get("type"))

    if action_type == "click":
        return ClickAction(
            x=data["x"],
            y=data["y"],
            button=MouseButton(data.get("button", "left")),
            click_count=data.get("count", data.get("click_count", 1)),
        )
    elif action_type == "type":
        return TypeAction(
            text=data["text"],
            clear_first=data.get("clear_first", False),
        )
    elif action_type == "scroll":
        return ScrollAction(
            direction=ScrollDirection(data["direction"]),
            amount=data.get("amount", 300),
            x=data.get("x"),
            y=data.get("y"),
        )
    elif action_type == "wait":
        return WaitAction(
            condition=data["condition"],
            timeout_ms=data.get("timeout", data.get("timeout_ms", 5000)),
            selector=data.get("selector"),
        )
    elif action_type == "hotkey":
        return HotkeyAction(keys=data["keys"])
    elif action_type == "drag":
        return DragAction(
            start_x=data["start_x"],
            start_y=data["start_y"],
            end_x=data["end_x"],
            end_y=data["end_y"],
        )
    else:
        raise ValueError(f"Unknown action type: {action_type}")


if __name__ == "__main__":
    import asyncio
    from playwright.async_api import async_playwright

    async def demo():
        """Demo of action execution."""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            page = await browser.new_page()

            await page.goto("https://example.com")

            executor = ActionExecutor(page)

            # Demo actions
            await executor.scroll_down(200)
            await executor.wait_ms(500)
            await executor.scroll_up(200)

            # Click at center of page
            await executor.click_at(640, 360)

            print(f"Executed {len(executor.action_history)} actions")
            for result in executor.action_history:
                print(f"  {result.action_type}: {result.message}")

            await browser.close()

    asyncio.run(demo())
