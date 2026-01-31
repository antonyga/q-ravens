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
Your role is to create comprehensive test strategies and detailed test cases based on analysis results.

Your responsibilities:
1. Generate test scenarios from the analysis data
2. Design test cases covering functional, accessibility, and edge cases
3. Prioritize tests by risk and business impact
4. Create BDD-style specifications when appropriate
5. Design test data requirements
6. Identify cross-browser and device coverage needs

When designing tests, consider:
- Happy path scenarios (main user flows)
- Negative test cases (error handling, invalid inputs)
- Edge cases (boundary values, special characters)
- Accessibility concerns (WCAG compliance areas)
- Performance-sensitive areas

Output Format:
You MUST respond with a JSON array of test cases. Each test case must have:
- id: Unique identifier (e.g., "TC-001")
- name: Short descriptive name
- description: What the test validates
- steps: Array of step descriptions
- expected_result: What success looks like
- priority: "high", "medium", or "low"
- category: "functional", "accessibility", "performance", "security", or "edge_case"

Example:
```json
[
  {
    "id": "TC-001",
    "name": "Homepage loads successfully",
    "description": "Verify the homepage loads and displays expected content",
    "steps": ["Navigate to homepage URL", "Wait for page to load completely"],
    "expected_result": "Page loads with status 200, title is visible",
    "priority": "high",
    "category": "functional"
  }
]
```

Be thorough but practical. Focus on tests that provide the most value."""

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

        # Create the prompt for test design
        prompt = f"""Based on the following website analysis, design comprehensive test cases.

User's Testing Goal: {user_request or "General comprehensive testing"}

{context}

Design test cases that cover:
1. Core functionality tests (page loads, navigation, forms)
2. Input validation tests (if forms/inputs exist)
3. Link verification tests (if links exist)
4. Edge cases and negative scenarios
5. Accessibility basics (if applicable)

Respond with ONLY a JSON array of test cases, no additional text."""

        # Get LLM response
        response = await self.invoke_llm(prompt)

        # Parse the test cases from response
        test_cases = self._parse_test_cases(response)

        # If LLM parsing fails, generate basic tests programmatically
        if not test_cases:
            test_cases = self._generate_fallback_tests(analysis)

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

    def _generate_fallback_tests(self, analysis: AnalysisResult) -> list[TestCase]:
        """Generate basic tests programmatically when LLM parsing fails."""
        test_cases = []
        test_id = 1

        # Basic page load test
        test_cases.append(TestCase(
            id=f"TC-{test_id:03d}",
            name="Homepage Load Test",
            description=f"Verify {analysis.target_url} loads successfully",
            steps=[
                f"Navigate to {analysis.target_url}",
                "Wait for page to fully load",
                "Verify HTTP status is successful (2xx)",
            ],
            expected_result="Page loads with status 200, no JavaScript errors",
            priority="high",
            category="functional",
            status=TestStatus.PENDING,
        ))
        test_id += 1

        # Title verification test
        for page in analysis.pages_discovered[:3]:
            if page.title:
                test_cases.append(TestCase(
                    id=f"TC-{test_id:03d}",
                    name=f"Title Verification - {page.title[:30]}",
                    description=f"Verify page title is present and correct",
                    steps=[
                        f"Navigate to {page.url}",
                        "Get page title",
                        f"Verify title contains expected text",
                    ],
                    expected_result=f"Page title matches: {page.title}",
                    priority="medium",
                    category="functional",
                    status=TestStatus.PENDING,
                ))
                test_id += 1

        # Link tests
        for page in analysis.pages_discovered:
            if page.links:
                test_cases.append(TestCase(
                    id=f"TC-{test_id:03d}",
                    name=f"Link Validation - {page.url[:40]}",
                    description=f"Verify all links on page are valid",
                    steps=[
                        f"Navigate to {page.url}",
                        f"Find all {len(page.links)} links",
                        "Verify links have valid href attributes",
                    ],
                    expected_result="All links have valid href values",
                    priority="medium",
                    category="functional",
                    status=TestStatus.PENDING,
                ))
                test_id += 1
                break  # Only one link test

        # Form tests
        for page in analysis.pages_discovered:
            if page.forms:
                test_cases.append(TestCase(
                    id=f"TC-{test_id:03d}",
                    name=f"Form Presence Test",
                    description=f"Verify forms are present and have required elements",
                    steps=[
                        f"Navigate to {page.url}",
                        f"Locate form elements",
                        "Verify form has action and method attributes",
                    ],
                    expected_result="Forms are properly structured with required attributes",
                    priority="high",
                    category="functional",
                    status=TestStatus.PENDING,
                ))
                test_id += 1
                break  # Only one form test

        # Input validation tests
        for page in analysis.pages_discovered:
            if page.inputs:
                # Find text inputs for validation test
                text_inputs = [i for i in page.inputs if i.get("type") in ["text", "email", "password", None]]
                if text_inputs:
                    test_cases.append(TestCase(
                        id=f"TC-{test_id:03d}",
                        name="Input Field Presence Test",
                        description="Verify input fields are present and accessible",
                        steps=[
                            f"Navigate to {page.url}",
                            f"Locate {len(page.inputs)} input fields",
                            "Verify inputs are visible and enabled",
                        ],
                        expected_result="All input fields are present and interactive",
                        priority="medium",
                        category="functional",
                        status=TestStatus.PENDING,
                    ))
                    test_id += 1
                    break

        # Authentication test if detected
        if analysis.has_authentication:
            test_cases.append(TestCase(
                id=f"TC-{test_id:03d}",
                name="Authentication Flow Presence",
                description=f"Verify {analysis.auth_type or 'authentication'} flow is present",
                steps=[
                    f"Navigate to {analysis.target_url}",
                    "Look for login/authentication elements",
                    "Verify authentication UI is accessible",
                ],
                expected_result="Authentication flow is present and accessible",
                priority="high",
                category="functional",
                status=TestStatus.PENDING,
            ))
            test_id += 1

        # JavaScript error check
        test_cases.append(TestCase(
            id=f"TC-{test_id:03d}",
            name="JavaScript Console Error Check",
            description="Verify no JavaScript errors on page load",
            steps=[
                f"Navigate to {analysis.target_url}",
                "Monitor browser console",
                "Check for JavaScript errors",
            ],
            expected_result="No critical JavaScript errors in console",
            priority="medium",
            category="functional",
            status=TestStatus.PENDING,
        ))
        test_id += 1

        # Basic accessibility test
        test_cases.append(TestCase(
            id=f"TC-{test_id:03d}",
            name="Basic Accessibility Check",
            description="Verify basic accessibility attributes are present",
            steps=[
                f"Navigate to {analysis.target_url}",
                "Check for alt attributes on images",
                "Verify form labels are present",
                "Check heading hierarchy",
            ],
            expected_result="Basic accessibility requirements are met",
            priority="medium",
            category="accessibility",
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
    agent = DesignerAgent()
    return await agent.process(state)
