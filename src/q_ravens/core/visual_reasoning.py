"""
Q-Ravens Visual Reasoning Module

Implements the "Think" component of the See-Think-Act-Reflect loop.
Uses multimodal LLMs to:
- Understand the current visual state
- Identify relevant elements for the task
- Decide on the next action
- Generate reasoning traces
"""

import base64
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from q_ravens.core.config import LLMProvider, settings
from q_ravens.tools.vision import VisualState, InteractiveElement


class ReasoningConfidence(str, Enum):
    """Confidence level for reasoning decisions."""

    HIGH = "high"  # > 90% confident
    MEDIUM = "medium"  # 70-90% confident
    LOW = "low"  # < 70% confident
    UNCLEAR = "unclear"  # Cannot determine


@dataclass
class ReasoningResult:
    """Result of visual reasoning about a page state."""

    # What the agent sees/understands
    scene_description: str
    identified_elements: list[dict]  # Elements relevant to the task

    # Recommended action
    recommended_action: Optional[dict]  # Action to take
    target_element: Optional[int]  # Mark ID of target element
    action_reasoning: str  # Why this action was chosen

    # Confidence and status
    confidence: ReasoningConfidence
    needs_clarification: bool
    clarification_question: Optional[str]

    # Raw response for debugging
    raw_response: str


@dataclass
class GoalDecomposition:
    """Decomposition of a high-level goal into steps."""

    original_goal: str
    steps: list[str]
    current_step_index: int = 0
    completed_steps: list[str] = field(default_factory=list)

    @property
    def current_step(self) -> Optional[str]:
        """Get the current step."""
        if self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    @property
    def is_complete(self) -> bool:
        """Check if all steps are completed."""
        return self.current_step_index >= len(self.steps)

    def advance(self) -> None:
        """Move to the next step."""
        if self.current_step:
            self.completed_steps.append(self.current_step)
        self.current_step_index += 1


class VisualReasoningEngine:
    """
    Visual reasoning engine using multimodal LLMs.

    This implements the "Think" phase, analyzing screenshots
    and deciding what actions to take.
    """

    SYSTEM_PROMPT = """You are a visual reasoning agent for web application testing.
Your role is to analyze screenshots of web pages and decide what actions to take.

You will receive:
1. A screenshot of a web page with numbered markers (Set-of-Mark) on interactive elements
2. A list of detected interactive elements with their mark numbers and positions
3. The current goal or task to accomplish

Your job is to:
1. Understand the current state of the page
2. Identify which element(s) are relevant to the current goal
3. Decide on the next action to take
4. Explain your reasoning

IMPORTANT GUIDELINES:
- Always reference elements by their mark number (e.g., "element [3]")
- Provide coordinates when specifying actions (x, y)
- If the goal is unclear or requires clarification, ask
- If you're not confident about an action, say so
- If you see an error message or unexpected state, report it

OUTPUT FORMAT:
Respond in JSON format with this structure:
{
    "scene_description": "Brief description of what you see on the page",
    "identified_elements": [
        {"mark_id": 1, "description": "Login button", "relevance": "high"}
    ],
    "recommended_action": {
        "action": "click",
        "target_mark_id": 1,
        "x": 640,
        "y": 360,
        "reasoning": "Clicking the login button to proceed with authentication"
    },
    "confidence": "high",
    "needs_clarification": false,
    "clarification_question": null
}

For action types use: click, type, scroll, wait, hotkey
"""

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize the visual reasoning engine.

        Args:
            provider: LLM provider to use (defaults to settings)
            model: Model name (must support vision)
        """
        self.provider = provider or settings.default_llm_provider
        self.model = model
        self.llm = self._create_llm()
        self._conversation_history: list = []

    def _create_llm(self):
        """Create a vision-capable LLM instance."""
        if self.provider == LLMProvider.ANTHROPIC:
            api_key = settings.get_api_key(LLMProvider.ANTHROPIC)
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not configured")
            # Use Claude Sonnet for vision (Opus 4 also works)
            model = self.model or "claude-sonnet-4-20250514"
            return ChatAnthropic(
                model=model,
                api_key=api_key,
                max_tokens=4096,
            )

        elif self.provider == LLMProvider.OPENAI:
            api_key = settings.get_api_key(LLMProvider.OPENAI)
            if not api_key:
                raise ValueError("OPENAI_API_KEY not configured")
            model = self.model or "gpt-4o"
            return ChatOpenAI(
                model=model,
                api_key=api_key,
                max_tokens=4096,
            )

        else:
            raise ValueError(f"Provider {self.provider} does not support vision")

    async def analyze_state(
        self,
        visual_state: VisualState,
        goal: str,
        context: Optional[str] = None,
    ) -> ReasoningResult:
        """
        Analyze the current visual state and decide next action.

        Args:
            visual_state: Current visual state with SoM overlay
            goal: The goal or task to accomplish
            context: Additional context about the workflow

        Returns:
            ReasoningResult with recommended action
        """
        # Build the prompt with visual context
        prompt_text = self._build_analysis_prompt(visual_state, goal, context)

        # Create message with image
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{visual_state.som_screenshot_base64}",
                        },
                    },
                    {"type": "text", "text": prompt_text},
                ]
            ),
        ]

        # Invoke the LLM
        response = await self.llm.ainvoke(messages)
        raw_response = response.content

        # Parse the response
        return self._parse_reasoning_response(raw_response)

    async def decompose_goal(self, goal: str, visual_state: VisualState) -> GoalDecomposition:
        """
        Decompose a high-level goal into executable steps.

        Args:
            goal: The high-level goal (e.g., "Purchase the cheapest laptop")
            visual_state: Current visual state for context

        Returns:
            GoalDecomposition with steps
        """
        prompt = f"""Given the current webpage state, decompose this goal into specific, executable steps:

Goal: {goal}

Page Context:
{visual_state.to_prompt_context()}

Provide a list of steps that can be executed one at a time.
Each step should be a single, atomic action.

Respond in JSON format:
{{
    "steps": [
        "Navigate to the search bar",
        "Type 'laptop' in the search field",
        "Click the search button",
        ...
    ]
}}
"""

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{visual_state.som_screenshot_base64}",
                        },
                    },
                    {"type": "text", "text": prompt},
                ]
            ),
        ]

        response = await self.llm.ainvoke(messages)

        try:
            # Extract JSON from response
            response_text = response.content
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                data = json.loads(response_text[json_start:json_end])
                steps = data.get("steps", [])
            else:
                steps = [goal]  # Fallback to single step
        except json.JSONDecodeError:
            steps = [goal]

        return GoalDecomposition(original_goal=goal, steps=steps)

    async def verify_action_result(
        self,
        before_state: VisualState,
        after_state: VisualState,
        action_taken: dict,
        expected_outcome: str,
    ) -> dict:
        """
        Verify the result of an action (Reflect phase).

        Args:
            before_state: Visual state before action
            after_state: Visual state after action
            action_taken: The action that was executed
            expected_outcome: What was expected to happen

        Returns:
            Dictionary with verification results
        """
        prompt = f"""Compare these two screenshots (before and after an action) and verify the result.

Action taken: {json.dumps(action_taken)}
Expected outcome: {expected_outcome}

Analyze:
1. Did the action have the expected effect?
2. Is the page now in the expected state?
3. Are there any errors or unexpected changes?
4. Should we proceed or take corrective action?

Respond in JSON:
{{
    "action_successful": true/false,
    "state_changed": true/false,
    "expected_outcome_achieved": true/false,
    "observed_changes": "description of what changed",
    "errors_detected": ["list of any errors"],
    "recommendation": "proceed/retry/abort/human_intervention",
    "reasoning": "explanation"
}}
"""

        messages = [
            SystemMessage(content="You are verifying the result of an action on a web page."),
            HumanMessage(
                content=[
                    {"type": "text", "text": "BEFORE:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{before_state.som_screenshot_base64}",
                        },
                    },
                    {"type": "text", "text": "AFTER:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{after_state.som_screenshot_base64}",
                        },
                    },
                    {"type": "text", "text": prompt},
                ]
            ),
        ]

        response = await self.llm.ainvoke(messages)

        try:
            response_text = response.content
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                return json.loads(response_text[json_start:json_end])
        except json.JSONDecodeError:
            pass

        return {
            "action_successful": False,
            "state_changed": True,
            "expected_outcome_achieved": False,
            "observed_changes": "Unable to parse response",
            "errors_detected": [],
            "recommendation": "human_intervention",
            "reasoning": response.content,
        }

    def _build_analysis_prompt(
        self,
        visual_state: VisualState,
        goal: str,
        context: Optional[str],
    ) -> str:
        """Build the prompt for visual analysis."""
        lines = [
            f"CURRENT GOAL: {goal}",
            "",
            "PAGE INFORMATION:",
            visual_state.to_prompt_context(),
            "",
        ]

        if context:
            lines.extend(["ADDITIONAL CONTEXT:", context, ""])

        lines.extend([
            "Based on the screenshot with numbered markers and the element list above,",
            "analyze the page and recommend the next action to achieve the goal.",
            "",
            "Remember to:",
            "- Reference elements by their mark number [N]",
            "- Provide specific coordinates for actions",
            "- Explain your reasoning",
        ])

        return "\n".join(lines)

    def _parse_reasoning_response(self, raw_response: str) -> ReasoningResult:
        """Parse the LLM response into a ReasoningResult."""
        try:
            # Extract JSON from response
            json_start = raw_response.find("{")
            json_end = raw_response.rfind("}") + 1

            if json_start != -1 and json_end > json_start:
                data = json.loads(raw_response[json_start:json_end])
            else:
                # No JSON found, create a basic result
                return ReasoningResult(
                    scene_description="Unable to parse response",
                    identified_elements=[],
                    recommended_action=None,
                    target_element=None,
                    action_reasoning=raw_response,
                    confidence=ReasoningConfidence.UNCLEAR,
                    needs_clarification=True,
                    clarification_question="Could not understand the response. Please clarify.",
                    raw_response=raw_response,
                )

            # Map confidence string to enum
            confidence_str = data.get("confidence", "medium").lower()
            confidence_map = {
                "high": ReasoningConfidence.HIGH,
                "medium": ReasoningConfidence.MEDIUM,
                "low": ReasoningConfidence.LOW,
                "unclear": ReasoningConfidence.UNCLEAR,
            }
            confidence = confidence_map.get(confidence_str, ReasoningConfidence.MEDIUM)

            # Extract target element from recommended action
            recommended_action = data.get("recommended_action")
            target_element = None
            if recommended_action:
                target_element = recommended_action.get("target_mark_id")

            return ReasoningResult(
                scene_description=data.get("scene_description", ""),
                identified_elements=data.get("identified_elements", []),
                recommended_action=recommended_action,
                target_element=target_element,
                action_reasoning=recommended_action.get("reasoning", "") if recommended_action else "",
                confidence=confidence,
                needs_clarification=data.get("needs_clarification", False),
                clarification_question=data.get("clarification_question"),
                raw_response=raw_response,
            )

        except json.JSONDecodeError as e:
            return ReasoningResult(
                scene_description="JSON parse error",
                identified_elements=[],
                recommended_action=None,
                target_element=None,
                action_reasoning=f"Failed to parse: {e}",
                confidence=ReasoningConfidence.UNCLEAR,
                needs_clarification=True,
                clarification_question="Response format was invalid. Please try again.",
                raw_response=raw_response,
            )


async def analyze_page_and_decide(
    visual_state: VisualState,
    goal: str,
) -> ReasoningResult:
    """
    Convenience function to analyze a page and get action recommendation.

    Args:
        visual_state: Visual state from VisionService
        goal: Goal to accomplish

    Returns:
        ReasoningResult with recommended action
    """
    engine = VisualReasoningEngine()
    return await engine.analyze_state(visual_state, goal)


if __name__ == "__main__":
    import asyncio
    from q_ravens.tools.vision import capture_page_state

    async def demo():
        """Demo of visual reasoning."""
        url = "https://example.com"
        print(f"Capturing visual state for: {url}")

        visual_state = await capture_page_state(url)
        print(f"Found {len(visual_state.elements)} interactive elements")

        print("\nAnalyzing page...")
        engine = VisualReasoningEngine()

        result = await engine.analyze_state(
            visual_state,
            goal="Find and click on the 'More information' link",
        )

        print(f"\nScene: {result.scene_description}")
        print(f"Confidence: {result.confidence.value}")

        if result.recommended_action:
            print(f"\nRecommended action: {result.recommended_action}")
        else:
            print(f"\nNo action recommended. Clarification needed: {result.clarification_question}")

    asyncio.run(demo())
