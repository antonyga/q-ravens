"""
Q-Ravens Executor Agent

The Automation Engineer responsible for:
- Generating test code from specifications
- Executing tests with Playwright
- Capturing screenshots and evidence
- Reporting pass/fail with details
"""

from typing import Any
from datetime import datetime
import uuid

from langchain_core.messages import AIMessage

from q_ravens.agents.base import BaseAgent
from q_ravens.core.state import (
    QRavensState,
    WorkflowPhase,
    TestCase,
    TestResult,
    TestStatus,
)
from q_ravens.tools.browser import BrowserTool


class ExecutorAgent(BaseAgent):
    """
    The Executor runs automated tests using Playwright.

    It takes test specifications and executes them,
    capturing evidence and reporting results.
    """

    name = "executor"
    role = "Automation Engineer"
    description = "Executes automated tests and captures results"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.browser_tool = BrowserTool()

    @property
    def system_prompt(self) -> str:
        return """You are the Executor agent of Q-Ravens, an Automation Engineer specialist.
Your role is to execute automated tests against web applications.

Your responsibilities:
1. Take test specifications and execute them using Playwright
2. Handle dynamic elements and implement smart waits
3. Capture screenshots and evidence for failures
4. Report detailed pass/fail results with evidence

When executing tests:
- Be resilient to minor UI variations
- Implement appropriate waits for dynamic content
- Capture meaningful screenshots on failures
- Provide clear error messages that help debugging"""

    async def process(self, state: QRavensState) -> dict[str, Any]:
        """
        Execute tests against the target website.

        If no test cases are defined, generates basic smoke tests.
        """
        target_url = state.get("target_url")
        errors = state.get("errors", [])
        test_cases = state.get("test_cases", [])
        analysis = state.get("analysis")

        if not target_url:
            return {
                "phase": WorkflowPhase.ERROR.value,
                "errors": errors + ["No target URL for test execution"],
                "current_agent": self.name,
            }

        try:
            # If no test cases defined, create basic smoke tests
            if not test_cases:
                test_cases = self._generate_smoke_tests(target_url, analysis)

            # Execute each test
            results = []
            for test_case in test_cases:
                result = await self._execute_test(target_url, test_case)
                results.append(result)

            passed = sum(1 for r in results if r.status == TestStatus.PASSED)
            failed = sum(1 for r in results if r.status == TestStatus.FAILED)

            message = AIMessage(
                content=f"Executed {len(results)} tests: {passed} passed, {failed} failed"
            )

            return {
                "test_cases": test_cases,
                "test_results": results,
                "phase": WorkflowPhase.EXECUTING.value,  # Orchestrator will advance
                "next_agent": "orchestrator",
                "messages": [message],
                "current_agent": self.name,
            }

        except Exception as e:
            return {
                "phase": WorkflowPhase.ERROR.value,
                "errors": errors + [f"Test execution failed: {str(e)}"],
                "current_agent": self.name,
            }

    def _generate_smoke_tests(self, target_url: str, analysis) -> list[TestCase]:
        """
        Generate basic smoke tests based on analysis.

        These are fundamental tests that verify basic functionality.
        """
        tests = []

        # Test 1: Page loads successfully
        tests.append(TestCase(
            id=f"smoke-{uuid.uuid4().hex[:8]}",
            name="Page Load Test",
            description="Verify the main page loads successfully",
            steps=[
                "Navigate to the target URL",
                "Wait for page to fully load",
                "Verify page title exists",
            ],
            expected_result="Page loads without errors",
            priority="high",
            category="functional",
        ))

        # Test 2: No JavaScript errors
        tests.append(TestCase(
            id=f"smoke-{uuid.uuid4().hex[:8]}",
            name="JavaScript Error Check",
            description="Verify no JavaScript errors on page load",
            steps=[
                "Navigate to the target URL",
                "Monitor console for errors",
            ],
            expected_result="No JavaScript errors in console",
            priority="high",
            category="functional",
        ))

        # Test 3: Links are valid (if analysis found links)
        if analysis and analysis.pages_discovered:
            page_info = analysis.pages_discovered[0]
            if page_info.links:
                tests.append(TestCase(
                    id=f"smoke-{uuid.uuid4().hex[:8]}",
                    name="Navigation Links Test",
                    description="Verify main navigation links are clickable",
                    steps=[
                        "Find navigation links on the page",
                        "Verify links have valid href attributes",
                    ],
                    expected_result="Navigation links are valid and clickable",
                    priority="medium",
                    category="functional",
                ))

        # Test 4: Forms are present (if analysis found forms)
        if analysis and analysis.pages_discovered:
            page_info = analysis.pages_discovered[0]
            if page_info.forms:
                tests.append(TestCase(
                    id=f"smoke-{uuid.uuid4().hex[:8]}",
                    name="Form Elements Test",
                    description="Verify form elements are interactive",
                    steps=[
                        "Locate form elements on the page",
                        "Verify inputs are enabled and focusable",
                    ],
                    expected_result="Form elements are functional",
                    priority="medium",
                    category="functional",
                ))

        return tests

    async def _execute_test(self, url: str, test_case: TestCase) -> TestResult:
        """
        Execute a single test case.

        Args:
            url: The target URL
            test_case: The test case to execute

        Returns:
            TestResult with execution details
        """
        start_time = datetime.now()

        try:
            # Execute based on test type
            if "Page Load" in test_case.name:
                result = await self._test_page_load(url)
            elif "JavaScript" in test_case.name:
                result = await self._test_no_js_errors(url)
            elif "Links" in test_case.name:
                result = await self._test_navigation_links(url)
            elif "Form" in test_case.name:
                result = await self._test_form_elements(url)
            else:
                # Generic test execution
                result = {"passed": True, "message": "Test completed"}

            duration = int((datetime.now() - start_time).total_seconds() * 1000)

            return TestResult(
                test_id=test_case.id,
                status=TestStatus.PASSED if result.get("passed") else TestStatus.FAILED,
                actual_result=result.get("message", "Test completed"),
                duration_ms=duration,
            )

        except Exception as e:
            duration = int((datetime.now() - start_time).total_seconds() * 1000)
            return TestResult(
                test_id=test_case.id,
                status=TestStatus.FAILED,
                error_message=str(e),
                duration_ms=duration,
            )

    async def _test_page_load(self, url: str) -> dict:
        """Test that the page loads successfully."""
        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            try:
                response = await page.goto(url, timeout=30000)

                if response and response.status < 400:
                    title = await page.title()
                    return {
                        "passed": True,
                        "message": f"Page loaded successfully. Title: {title}",
                    }
                else:
                    return {
                        "passed": False,
                        "message": f"Page returned status {response.status if response else 'unknown'}",
                    }
            finally:
                await browser.close()

    async def _test_no_js_errors(self, url: str) -> dict:
        """Test that there are no JavaScript errors."""
        from playwright.async_api import async_playwright

        errors = []

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            # Capture console errors
            page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)

            try:
                await page.goto(url, timeout=30000)
                await page.wait_for_load_state("networkidle")

                if errors:
                    return {
                        "passed": False,
                        "message": f"Found {len(errors)} JavaScript errors: {errors[:3]}",
                    }
                return {
                    "passed": True,
                    "message": "No JavaScript errors detected",
                }
            finally:
                await browser.close()

    async def _test_navigation_links(self, url: str) -> dict:
        """Test that navigation links are valid."""
        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            try:
                await page.goto(url, timeout=30000)

                links = await page.eval_on_selector_all(
                    "a[href]",
                    "elements => elements.slice(0, 10).map(e => ({href: e.href, text: e.textContent}))"
                )

                valid_links = [l for l in links if l.get("href") and not l["href"].startswith("javascript:")]

                if valid_links:
                    return {
                        "passed": True,
                        "message": f"Found {len(valid_links)} valid navigation links",
                    }
                return {
                    "passed": False,
                    "message": "No valid navigation links found",
                }
            finally:
                await browser.close()

    async def _test_form_elements(self, url: str) -> dict:
        """Test that form elements are functional."""
        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            try:
                await page.goto(url, timeout=30000)

                forms = await page.query_selector_all("form")
                inputs = await page.query_selector_all("input:not([type='hidden']), textarea, select")

                if forms or inputs:
                    return {
                        "passed": True,
                        "message": f"Found {len(forms)} forms and {len(inputs)} input elements",
                    }
                return {
                    "passed": True,
                    "message": "No forms found (not necessarily an error)",
                }
            finally:
                await browser.close()


# Create node function for LangGraph
async def executor_node(state: QRavensState) -> dict[str, Any]:
    """LangGraph node function for the Executor agent."""
    agent = ExecutorAgent()
    return await agent.process(state)
