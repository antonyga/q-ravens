"""
Q-Ravens Designer Agent

The Test Architect responsible for:
- Generating test scenarios from analysis results
- Creating comprehensive test strategies
- Designing test data sets
- Prioritizing tests by risk and business impact
- Creating BDD-style specifications (Gherkin)
"""

import json
import re
from typing import Any
from datetime import datetime

from langchain_core.messages import AIMessage

from q_ravens.agents.base import BaseAgent
from q_ravens.core.state import (
    QRavensState,
    WorkflowPhase,
    AnalysisResult,
    TestCase,
    TestStatus,
)


class DesignerAgent(BaseAgent):
    """
    The Designer creates comprehensive test strategies and test cases.

    It analyzes the results from the Analyzer agent and produces
    detailed test specifications including positive, negative, and edge cases.
    """

    name = "designer"
    role = "Test Architect"
    description = "Designs comprehensive test strategies and detailed test cases"

    @property
    def system_prompt(self) -> str:
        return """You are the Designer agent of Q-Ravens, a Test Architect specialist.
Your role is to create test cases that DIRECTLY address the user's specific testing request.

CRITICAL RULES:
1. ONLY generate test cases that are directly relevant to the user's instruction
2. DO NOT generate generic tests (page load, broken images, accessibility, performance) unless explicitly requested
3. Focus on EXACTLY what the user asked for - nothing more, nothing less
4. If the user asks to verify a specific item exists, create a test ONLY for that verification
5. Be precise and targeted - quality over quantity

Output Format:
You MUST respond with a JSON array of test cases. Each test case must have:
- id: Unique identifier (e.g., "TC-001")
- name: Short descriptive name that reflects the user's request
- description: What the test validates (should match user's intent)
- steps: Array of specific step descriptions to accomplish the user's goal
- expected_result: What success looks like based on user's requirement
- priority: "high" (since it's what the user specifically asked for)
- category: "functional", "accessibility", "performance", "security", or "edge_case"

Example - If user asks "Verify 'Tequila Ocho Blanco' is on the Menu page":
```json
[
  {
    "id": "TC-001",
    "name": "Verify 'Tequila Ocho Blanco' on Menu Page",
    "description": "Verify that the item 'Tequila Ocho Blanco' is present and visible on the Menu page",
    "steps": [
      "Navigate to the website homepage",
      "Find and click on the Menu link/section",
      "Wait for Menu page to load",
      "Search for 'Tequila Ocho Blanco' text on the page",
      "Verify the item is visible"
    ],
    "expected_result": "'Tequila Ocho Blanco' is found and visible on the Menu page",
    "priority": "high",
    "category": "functional"
  }
]
```

IMPORTANT: Generate ONLY tests that fulfill the user's specific request. Do not add extra generic tests."""

    async def process(self, state: QRavensState) -> dict[str, Any]:
        """
        Design test cases based on analysis results.

        Uses LLM to generate comprehensive test strategies.
        """
        analysis = state.get("analysis")
        errors = state.get("errors", [])
        user_request = state.get("user_request", "")

        if not analysis:
            return {
                "phase": WorkflowPhase.ERROR.value,
                "errors": errors + ["No analysis results available for test design"],
                "current_agent": self.name,
            }

        try:
            # Generate test cases based on analysis
            test_cases = await self._design_test_cases(analysis, user_request)

            message = AIMessage(
                content=f"Test design complete. Created {len(test_cases)} test cases:\n"
                + self._summarize_test_cases(test_cases)
            )

            return {
                "test_cases": test_cases,
                "phase": WorkflowPhase.DESIGNING.value,
                "next_agent": "orchestrator",
                "messages": [message],
                "current_agent": self.name,
            }

        except Exception as e:
            return {
                "phase": WorkflowPhase.ERROR.value,
                "errors": errors + [f"Test design failed: {str(e)}"],
                "current_agent": self.name,
            }

    async def _design_test_cases(
        self, analysis: AnalysisResult, user_request: str
    ) -> list[TestCase]:
        """
        Generate test cases using LLM based on analysis.

        Args:
            analysis: The analysis results from the Analyzer
            user_request: Original user request for context

        Returns:
            List of designed TestCase objects
        """
        # Build context from analysis
        context = self._build_analysis_context(analysis)

        # Create the prompt for test design - FOCUSED on user's specific request
        prompt = f"""You are designing test cases for a specific user request.

CRITICAL: Generate ONLY test cases that DIRECTLY address the user's request below.
DO NOT generate generic tests like "page load", "broken images", "accessibility", or "performance" unless the user specifically asked for them.

User's Specific Request: "{user_request}"

Website Analysis Context:
{context}

Based on the user's request above, create test cases that:
1. DIRECTLY verify what the user asked for
2. Include specific steps to accomplish the user's goal
3. Focus ONLY on the user's requirement - nothing extra

If the user asks to verify something exists on a page, create a test that:
- Navigates to the correct page
- Searches for the specific item/element
- Verifies its presence

Respond with ONLY a JSON array of test cases that fulfill the user's specific request.
Do NOT add generic tests that the user did not ask for."""

        # Get LLM response
        response = await self.invoke_llm(prompt)

        # Parse the test cases from response
        test_cases = self._parse_test_cases(response)

        # If LLM parsing fails, generate targeted fallback tests based on user request
        if not test_cases:
            test_cases = self._generate_fallback_tests(analysis, user_request)

        return test_cases

    def _build_analysis_context(self, analysis: AnalysisResult) -> str:
        """Build a text context from analysis results for the LLM."""
        lines = [
            f"Target URL: {analysis.target_url}",
            f"Technology Stack: {', '.join(analysis.technology_stack) or 'Unknown'}",
            f"Has Authentication: {analysis.has_authentication}",
            f"Auth Type: {analysis.auth_type or 'N/A'}",
            f"\nPages Discovered: {len(analysis.pages_discovered)}",
        ]

        for i, page in enumerate(analysis.pages_discovered[:5]):  # Limit to 5 pages
            lines.append(f"\n--- Page {i+1}: {page.url} ---")
            lines.append(f"Title: {page.title or 'N/A'}")
            lines.append(f"Links: {len(page.links)}")
            lines.append(f"Forms: {len(page.forms)}")
            lines.append(f"Buttons: {len(page.buttons)}")
            lines.append(f"Input Fields: {len(page.inputs)}")

            # Add form details if present
            if page.forms:
                lines.append("Form Details:")
                for form in page.forms[:3]:  # Limit to 3 forms
                    lines.append(f"  - Action: {form.get('action', 'N/A')}, Method: {form.get('method', 'N/A')}")

            # Add input details if present
            if page.inputs:
                lines.append("Input Fields:")
                for inp in page.inputs[:5]:  # Limit to 5 inputs
                    lines.append(f"  - Type: {inp.get('type', 'text')}, Name: {inp.get('name', 'N/A')}")

        return "\n".join(lines)

    def _parse_test_cases(self, response: str) -> list[TestCase]:
        """Parse LLM response into TestCase objects."""
        test_cases = []

        try:
            # Try to extract JSON from response
            # Handle cases where LLM wraps JSON in markdown code blocks
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find raw JSON array
                json_match = re.search(r'\[[\s\S]*\]', response)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    return []

            data = json.loads(json_str)

            if not isinstance(data, list):
                return []

            for item in data:
                if isinstance(item, dict) and "id" in item and "name" in item:
                    test_case = TestCase(
                        id=item.get("id", f"TC-{len(test_cases)+1:03d}"),
                        name=item.get("name", "Unnamed Test"),
                        description=item.get("description", ""),
                        steps=item.get("steps", []),
                        expected_result=item.get("expected_result", "Test passes"),
                        priority=item.get("priority", "medium"),
                        category=item.get("category", "functional"),
                        status=TestStatus.PENDING,
                    )
                    test_cases.append(test_case)

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Return empty list - caller will use fallback
            pass

        return test_cases

    def _generate_fallback_tests(self, analysis: AnalysisResult, user_request: str = "") -> list[TestCase]:
        """
        Generate targeted tests based on user request when LLM parsing fails.

        Only generates tests relevant to the user's specific instruction.
        """
        test_cases = []
        test_id = 1
        user_request_lower = user_request.lower() if user_request else ""

        # Parse user request to understand what they want
        # Check for specific item verification requests
        import re
        item_match = re.search(r"['\"]([^'\"]+)['\"]", user_request)
        search_item = item_match.group(1) if item_match else None

        # Determine the target page from user request
        target_page = None
        if "menu" in user_request_lower:
            target_page = "Menu"
        elif "about" in user_request_lower:
            target_page = "About"
        elif "contact" in user_request_lower:
            target_page = "Contact"
        elif "home" in user_request_lower:
            target_page = "Home"

        # Check what type of test the user wants
        wants_verification = any(word in user_request_lower for word in ["verify", "check", "confirm", "test if", "is on", "exists", "present", "find"])
        wants_performance = any(word in user_request_lower for word in ["performance", "speed", "load time", "web vitals", "lighthouse"])
        wants_accessibility = any(word in user_request_lower for word in ["accessibility", "wcag", "a11y", "screen reader"])
        wants_links = any(word in user_request_lower for word in ["links", "broken links", "navigation links"])
        wants_forms = any(word in user_request_lower for word in ["form", "input", "submit"])

        # Generate test based on user's specific request
        if search_item and target_page:
            # User wants to verify a specific item on a specific page
            test_cases.append(TestCase(
                id=f"TC-{test_id:03d}",
                name=f"Verify '{search_item}' on {target_page} Page",
                description=f"Verify that '{search_item}' is present and visible on the {target_page} page",
                steps=[
                    f"Navigate to {analysis.target_url}",
                    f"Find and click on the {target_page} link/section",
                    f"Wait for {target_page} page to load completely",
                    f"Search for '{search_item}' text on the page",
                    f"Verify '{search_item}' is visible on the page",
                ],
                expected_result=f"'{search_item}' is found and visible on the {target_page} page",
                priority="high",
                category="functional",
                status=TestStatus.PENDING,
            ))
            test_id += 1

        elif search_item and wants_verification:
            # User wants to verify a specific item exists (page not specified)
            test_cases.append(TestCase(
                id=f"TC-{test_id:03d}",
                name=f"Verify '{search_item}' Exists",
                description=f"Verify that '{search_item}' is present on the website",
                steps=[
                    f"Navigate to {analysis.target_url}",
                    "Explore relevant pages to find the item",
                    f"Search for '{search_item}' text",
                    f"Verify '{search_item}' is visible",
                ],
                expected_result=f"'{search_item}' is found on the website",
                priority="high",
                category="functional",
                status=TestStatus.PENDING,
            ))
            test_id += 1

        elif target_page and wants_verification:
            # User wants to verify something about a specific page
            test_cases.append(TestCase(
                id=f"TC-{test_id:03d}",
                name=f"Verify {target_page} Page",
                description=f"Verify the {target_page} page loads and displays correctly",
                steps=[
                    f"Navigate to {analysis.target_url}",
                    f"Find and click on the {target_page} link",
                    f"Wait for {target_page} page to load",
                    f"Verify {target_page} page content is visible",
                ],
                expected_result=f"{target_page} page loads successfully with expected content",
                priority="high",
                category="functional",
                status=TestStatus.PENDING,
            ))
            test_id += 1

        elif wants_performance:
            # User explicitly asked for performance testing
            test_cases.append(TestCase(
                id=f"TC-{test_id:03d}",
                name="Performance - Core Web Vitals",
                description="Measure Core Web Vitals using Lighthouse audit",
                steps=[
                    f"Navigate to {analysis.target_url}",
                    "Run Lighthouse performance audit",
                    "Measure LCP (Largest Contentful Paint)",
                    "Measure TBT (Total Blocking Time)",
                    "Measure CLS (Cumulative Layout Shift)",
                ],
                expected_result="Core Web Vitals meet 'Good' thresholds",
                priority="high",
                category="performance",
                status=TestStatus.PENDING,
            ))
            test_id += 1

        elif wants_accessibility:
            # User explicitly asked for accessibility testing
            test_cases.append(TestCase(
                id=f"TC-{test_id:03d}",
                name="Accessibility - WCAG Compliance",
                description="Check WCAG 2.1 Level AA compliance",
                steps=[
                    f"Navigate to {analysis.target_url}",
                    "Run axe-core accessibility audit",
                    "Check for accessibility violations",
                ],
                expected_result="No critical accessibility violations",
                priority="high",
                category="accessibility",
                status=TestStatus.PENDING,
            ))
            test_id += 1

        elif wants_links:
            # User explicitly asked for link testing
            test_cases.append(TestCase(
                id=f"TC-{test_id:03d}",
                name="Link Validation",
                description="Verify all links on the page are valid",
                steps=[
                    f"Navigate to {analysis.target_url}",
                    "Find all links on the page",
                    "Check each link for validity",
                ],
                expected_result="All links are valid and working",
                priority="high",
                category="functional",
                status=TestStatus.PENDING,
            ))
            test_id += 1

        elif wants_forms:
            # User explicitly asked for form testing
            test_cases.append(TestCase(
                id=f"TC-{test_id:03d}",
                name="Form Functionality Test",
                description="Verify forms work correctly",
                steps=[
                    f"Navigate to {analysis.target_url}",
                    "Locate form elements",
                    "Test form interaction",
                ],
                expected_result="Forms are functional and accessible",
                priority="high",
                category="functional",
                status=TestStatus.PENDING,
            ))
            test_id += 1

        else:
            # Generic fallback only if we couldn't understand the request
            # Create a single test based on the user's request text
            test_cases.append(TestCase(
                id=f"TC-{test_id:03d}",
                name=f"User Request: {user_request[:50]}..." if len(user_request) > 50 else f"User Request: {user_request}",
                description=f"Execute test based on user instruction: {user_request}",
                steps=[
                    f"Navigate to {analysis.target_url}",
                    "Perform actions as specified in user request",
                    "Verify expected outcome",
                ],
                expected_result="User's testing requirement is satisfied",
                priority="high",
                category="functional",
                status=TestStatus.PENDING,
            ))

        return test_cases

    def _summarize_test_cases(self, test_cases: list[TestCase]) -> str:
        """Create a summary of designed test cases."""
        if not test_cases:
            return "No test cases generated."

        # Count by priority
        priority_counts = {"high": 0, "medium": 0, "low": 0}
        category_counts = {}

        for tc in test_cases:
            priority_counts[tc.priority] = priority_counts.get(tc.priority, 0) + 1
            category_counts[tc.category] = category_counts.get(tc.category, 0) + 1

        lines = [
            f"  - High priority: {priority_counts['high']}",
            f"  - Medium priority: {priority_counts['medium']}",
            f"  - Low priority: {priority_counts['low']}",
            "\nCategories:",
        ]

        for category, count in sorted(category_counts.items()):
            lines.append(f"  - {category}: {count}")

        lines.append("\nTest Cases:")
        for tc in test_cases[:10]:  # Show first 10
            lines.append(f"  [{tc.priority[0].upper()}] {tc.id}: {tc.name}")

        if len(test_cases) > 10:
            lines.append(f"  ... and {len(test_cases) - 10} more")

        return "\n".join(lines)


# Create node function for LangGraph
async def designer_node(state: QRavensState) -> dict[str, Any]:
    """LangGraph node function for the Designer agent."""
    agent = DesignerAgent.from_state(state)
    return await agent.process(state)
