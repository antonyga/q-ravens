"""
Q-Ravens Visual Agent

Implements the complete See-Think-Act-Reflect cognitive loop
for autonomous visual-based web testing.

This is the core agent that:
1. SEE: Captures visual state with SoM overlay
2. THINK: Uses multimodal LLM to understand and plan
3. ACT: Executes coordinate-based actions
4. REFLECT: Verifies outcomes and adapts
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from playwright.async_api import Page, Browser, async_playwright

from q_ravens.core.config import settings
from q_ravens.core.visual_reasoning import (
    VisualReasoningEngine,
    ReasoningResult,
    ReasoningConfidence,
    GoalDecomposition,
)
from q_ravens.tools.vision import VisionService, VisualState
from q_ravens.tools.actions import (
    ActionExecutor,
    ActionResult,
    ClickAction,
    TypeAction,
    ScrollAction,
    WaitAction,
    ScrollDirection,
    create_action_from_dict,
)

logger = logging.getLogger(__name__)


class LoopPhase(str, Enum):
    """Current phase in the cognitive loop."""

    SEE = "see"
    THINK = "think"
    ACT = "act"
    REFLECT = "reflect"
    COMPLETE = "complete"
    ERROR = "error"
    WAITING_FOR_HUMAN = "waiting_for_human"


@dataclass
class LoopIteration:
    """Record of a single iteration of the cognitive loop."""

    iteration_number: int
    phase: LoopPhase
    timestamp: datetime = field(default_factory=datetime.now)

    # See phase
    visual_state: Optional[VisualState] = None

    # Think phase
    reasoning: Optional[ReasoningResult] = None
    planned_action: Optional[dict] = None

    # Act phase
    action_result: Optional[ActionResult] = None

    # Reflect phase
    verification: Optional[dict] = None
    success: bool = False
    error: Optional[str] = None


@dataclass
class LoopState:
    """Complete state of the cognitive loop execution."""

    goal: str
    url: str
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    # Progress
    current_phase: LoopPhase = LoopPhase.SEE
    iteration_count: int = 0
    max_iterations: int = 20
    consecutive_failures: int = 0
    max_consecutive_failures: int = 3

    # Goal decomposition
    goal_decomposition: Optional[GoalDecomposition] = None

    # History
    iterations: list[LoopIteration] = field(default_factory=list)

    # Final status
    success: bool = False
    error_message: Optional[str] = None

    @property
    def is_complete(self) -> bool:
        """Check if the loop has completed."""
        return self.current_phase in [LoopPhase.COMPLETE, LoopPhase.ERROR]

    @property
    def should_continue(self) -> bool:
        """Check if the loop should continue."""
        if self.is_complete:
            return False
        if self.iteration_count >= self.max_iterations:
            return False
        if self.consecutive_failures >= self.max_consecutive_failures:
            return False
        return True


class VisualAgent:
    """
    Visual-based testing agent implementing the See-Think-Act-Reflect loop.

    This is the main cognitive agent that can autonomously interact with
    web applications using visual understanding.
    """

    def __init__(
        self,
        headless: bool = True,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        action_delay_ms: int = 500,
        capture_all_screenshots: bool = True,
    ):
        """
        Initialize the visual agent.

        Args:
            headless: Whether to run browser in headless mode
            viewport_width: Browser viewport width
            viewport_height: Browser viewport height
            action_delay_ms: Delay between actions
            capture_all_screenshots: Whether to capture screenshots at each step
        """
        self.headless = headless
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.action_delay_ms = action_delay_ms
        self.capture_all_screenshots = capture_all_screenshots

        # Components
        self.vision_service = VisionService(
            viewport_width=viewport_width,
            viewport_height=viewport_height,
        )
        self.reasoning_engine = VisualReasoningEngine()

        # Runtime state
        self._browser: Optional[Browser] = None
        self._page: Optional[Page] = None
        self._action_executor: Optional[ActionExecutor] = None
        self._current_state: Optional[LoopState] = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self._start_browser()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._stop_browser()

    async def _start_browser(self):
        """Start the browser."""
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self.headless)
        context = await self._browser.new_context(
            viewport={"width": self.viewport_width, "height": self.viewport_height},
            user_agent="Q-Ravens/0.1.0 (Visual Testing Agent)",
        )
        self._page = await context.new_page()
        self._action_executor = ActionExecutor(
            self._page,
            action_delay_ms=self.action_delay_ms,
        )

    async def _stop_browser(self):
        """Stop the browser."""
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        self._browser = None
        self._page = None
        self._action_executor = None

    async def execute_goal(
        self,
        url: str,
        goal: str,
        decompose_goal: bool = True,
    ) -> LoopState:
        """
        Execute a goal using the See-Think-Act-Reflect loop.

        Args:
            url: Starting URL
            goal: Goal to accomplish
            decompose_goal: Whether to decompose the goal into steps

        Returns:
            LoopState with execution history and results
        """
        if not self._page:
            raise RuntimeError("Browser not started. Use 'async with' context manager.")

        # Initialize state
        self._current_state = LoopState(goal=goal, url=url)

        logger.info(f"Starting visual agent for goal: {goal}")

        # Navigate to starting URL
        await self._page.goto(url, wait_until="networkidle")

        # Optionally decompose the goal
        if decompose_goal:
            initial_visual_state = await self.vision_service.capture_visual_state(self._page)
            self._current_state.goal_decomposition = await self.reasoning_engine.decompose_goal(
                goal, initial_visual_state
            )
            logger.info(f"Goal decomposed into {len(self._current_state.goal_decomposition.steps)} steps")

        # Run the cognitive loop
        while self._current_state.should_continue:
            await self._run_iteration()

        # Finalize
        self._current_state.completed_at = datetime.now()

        return self._current_state

    async def _run_iteration(self):
        """Run a single iteration of the See-Think-Act-Reflect loop."""
        state = self._current_state
        state.iteration_count += 1

        iteration = LoopIteration(
            iteration_number=state.iteration_count,
            phase=LoopPhase.SEE,
        )

        logger.info(f"=== Iteration {state.iteration_count} ===")

        try:
            # ===== SEE PHASE =====
            iteration.phase = LoopPhase.SEE
            logger.info("Phase: SEE - Capturing visual state")

            visual_state = await self.vision_service.capture_visual_state(self._page)
            iteration.visual_state = visual_state

            logger.info(f"  Found {len(visual_state.elements)} interactive elements")

            # ===== THINK PHASE =====
            iteration.phase = LoopPhase.THINK
            logger.info("Phase: THINK - Analyzing and planning")

            # Determine current goal/step
            current_goal = state.goal
            if state.goal_decomposition and not state.goal_decomposition.is_complete:
                current_goal = state.goal_decomposition.current_step
                logger.info(f"  Current step: {current_goal}")

            # Get reasoning from multimodal LLM
            reasoning = await self.reasoning_engine.analyze_state(
                visual_state,
                goal=current_goal,
                context=self._build_context(),
            )
            iteration.reasoning = reasoning

            logger.info(f"  Scene: {reasoning.scene_description[:100]}...")
            logger.info(f"  Confidence: {reasoning.confidence.value}")

            # Check if clarification needed
            if reasoning.needs_clarification:
                logger.warning(f"  Clarification needed: {reasoning.clarification_question}")
                state.current_phase = LoopPhase.WAITING_FOR_HUMAN
                iteration.error = "Clarification needed"
                state.iterations.append(iteration)
                return

            # Check confidence threshold
            if reasoning.confidence == ReasoningConfidence.UNCLEAR:
                logger.warning("  Low confidence - skipping action")
                state.consecutive_failures += 1
                iteration.error = "Low confidence"
                state.iterations.append(iteration)
                return

            # No action recommended (might be complete)
            if not reasoning.recommended_action:
                logger.info("  No action recommended - checking if goal complete")
                if state.goal_decomposition:
                    state.goal_decomposition.advance()
                    if state.goal_decomposition.is_complete:
                        state.current_phase = LoopPhase.COMPLETE
                        state.success = True
                else:
                    state.current_phase = LoopPhase.COMPLETE
                    state.success = True
                state.iterations.append(iteration)
                return

            iteration.planned_action = reasoning.recommended_action

            # ===== ACT PHASE =====
            iteration.phase = LoopPhase.ACT
            logger.info("Phase: ACT - Executing action")

            action_result = await self._execute_action(reasoning.recommended_action, visual_state)
            iteration.action_result = action_result

            logger.info(f"  Action: {action_result.action_type}")
            logger.info(f"  Result: {'Success' if action_result.success else 'Failed'}")

            if not action_result.success:
                state.consecutive_failures += 1
                iteration.error = action_result.message
                state.iterations.append(iteration)
                return

            # ===== REFLECT PHASE =====
            iteration.phase = LoopPhase.REFLECT
            logger.info("Phase: REFLECT - Verifying outcome")

            # Wait for any transitions
            await asyncio.sleep(0.5)
            await self._page.wait_for_load_state("networkidle", timeout=5000)

            # Capture new state
            after_state = await self.vision_service.capture_visual_state(self._page)

            # Verify the action result
            verification = await self.reasoning_engine.verify_action_result(
                before_state=visual_state,
                after_state=after_state,
                action_taken=reasoning.recommended_action,
                expected_outcome=reasoning.action_reasoning,
            )
            iteration.verification = verification

            logger.info(f"  Expected outcome achieved: {verification.get('expected_outcome_achieved', False)}")

            if verification.get("expected_outcome_achieved", False):
                iteration.success = True
                state.consecutive_failures = 0

                # Advance goal if using decomposition
                if state.goal_decomposition:
                    state.goal_decomposition.advance()
                    if state.goal_decomposition.is_complete:
                        logger.info("All steps completed!")
                        state.current_phase = LoopPhase.COMPLETE
                        state.success = True
            else:
                state.consecutive_failures += 1
                iteration.error = verification.get("observed_changes", "Unexpected outcome")

                # Check recommendation
                recommendation = verification.get("recommendation", "proceed")
                if recommendation == "abort":
                    state.current_phase = LoopPhase.ERROR
                    state.error_message = "Agent recommended abort"
                elif recommendation == "human_intervention":
                    state.current_phase = LoopPhase.WAITING_FOR_HUMAN

        except Exception as e:
            logger.error(f"Error in iteration: {e}")
            iteration.error = str(e)
            iteration.phase = LoopPhase.ERROR
            state.consecutive_failures += 1

            if state.consecutive_failures >= state.max_consecutive_failures:
                state.current_phase = LoopPhase.ERROR
                state.error_message = f"Too many consecutive failures: {e}"

        finally:
            state.iterations.append(iteration)

    async def _execute_action(
        self,
        action_dict: dict,
        visual_state: VisualState,
    ) -> ActionResult:
        """
        Execute an action based on the reasoning output.

        Args:
            action_dict: Action dictionary from reasoning
            visual_state: Current visual state for element lookup

        Returns:
            ActionResult from execution
        """
        action_type = action_dict.get("action", "")

        # Handle click action
        if action_type == "click":
            # Get coordinates - prefer explicit x,y, fall back to element lookup
            x = action_dict.get("x")
            y = action_dict.get("y")

            if x is None or y is None:
                # Look up by mark ID
                mark_id = action_dict.get("target_mark_id")
                if mark_id:
                    element = visual_state.get_element_by_mark(mark_id)
                    if element:
                        x, y = element.bounding_box.center
                    else:
                        return ActionResult(
                            success=False,
                            action_type="click",
                            message=f"Element with mark {mark_id} not found",
                        )
                else:
                    return ActionResult(
                        success=False,
                        action_type="click",
                        message="No coordinates or mark ID provided",
                    )

            return await self._action_executor.click_at(x, y)

        # Handle type action
        elif action_type == "type":
            text = action_dict.get("text", "")

            # If target specified, click first
            target_mark_id = action_dict.get("target_mark_id")
            if target_mark_id:
                element = visual_state.get_element_by_mark(target_mark_id)
                if element:
                    x, y = element.bounding_box.center
                    click_result = await self._action_executor.click_at(x, y)
                    if not click_result.success:
                        return click_result
                    await asyncio.sleep(0.1)

            clear_first = action_dict.get("clear_first", False)
            return await self._action_executor.type_text(text, clear_first=clear_first)

        # Handle scroll action
        elif action_type == "scroll":
            direction = action_dict.get("direction", "down")
            amount = action_dict.get("amount", 300)

            direction_map = {
                "up": ScrollDirection.UP,
                "down": ScrollDirection.DOWN,
                "left": ScrollDirection.LEFT,
                "right": ScrollDirection.RIGHT,
            }

            if direction == "down":
                return await self._action_executor.scroll_down(amount)
            elif direction == "up":
                return await self._action_executor.scroll_up(amount)
            else:
                return await self._action_executor.execute(
                    ScrollAction(direction=direction_map.get(direction, ScrollDirection.DOWN), amount=amount)
                )

        # Handle wait action
        elif action_type == "wait":
            timeout = action_dict.get("timeout", 1000)
            return await self._action_executor.wait_ms(timeout)

        # Handle hotkey action
        elif action_type == "hotkey":
            keys = action_dict.get("keys", [])
            return await self._action_executor.press_keys(*keys)

        else:
            return ActionResult(
                success=False,
                action_type=action_type,
                message=f"Unknown action type: {action_type}",
            )

    def _build_context(self) -> str:
        """Build context string for reasoning."""
        state = self._current_state
        if not state:
            return ""

        lines = [
            f"Iteration: {state.iteration_count}/{state.max_iterations}",
            f"Consecutive failures: {state.consecutive_failures}",
        ]

        if state.goal_decomposition:
            lines.append(f"Step {state.goal_decomposition.current_step_index + 1}/{len(state.goal_decomposition.steps)}")
            if state.goal_decomposition.completed_steps:
                lines.append(f"Completed: {', '.join(state.goal_decomposition.completed_steps[-3:])}")

        # Add recent history
        recent = state.iterations[-3:] if state.iterations else []
        if recent:
            lines.append("\nRecent actions:")
            for it in recent:
                if it.action_result:
                    lines.append(f"  - {it.action_result.action_type}: {it.action_result.message}")

        return "\n".join(lines)


async def run_visual_test(url: str, goal: str) -> LoopState:
    """
    Convenience function to run a visual test.

    Args:
        url: Starting URL
        goal: Goal to accomplish

    Returns:
        LoopState with results
    """
    async with VisualAgent(headless=True) as agent:
        return await agent.execute_goal(url, goal)


if __name__ == "__main__":
    import sys

    async def main():
        url = sys.argv[1] if len(sys.argv) > 1 else "https://example.com"
        goal = sys.argv[2] if len(sys.argv) > 2 else "Click the 'More information' link"

        print(f"Running visual test")
        print(f"URL: {url}")
        print(f"Goal: {goal}")
        print("=" * 50)

        result = await run_visual_test(url, goal)

        print("\n" + "=" * 50)
        print(f"Result: {'SUCCESS' if result.success else 'FAILED'}")
        print(f"Iterations: {result.iteration_count}")
        print(f"Duration: {(result.completed_at - result.started_at).total_seconds():.1f}s")

        if result.error_message:
            print(f"Error: {result.error_message}")

    asyncio.run(main())
