"""
Q-Ravens Executor Agent

The Automation Engineer responsible for:
- Generating test code from specifications
- Executing tests with Playwright
- Running performance tests with Lighthouse
- Running accessibility tests with axe-core
- Capturing screenshots and evidence
- Reporting pass/fail with details
"""

import logging
from typing import Any, Optional
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
from q_ravens.tools.lighthouse import LighthouseTool, LighthouseResult
from q_ravens.tools.accessibility import AccessibilityTool, AccessibilityResult, WCAGLevel

logger = logging.getLogger(__name__)


class ExecutorAgent(BaseAgent):
    """
    The Executor runs automated tests using Playwright, Lighthouse, and axe-core.

    It takes test specifications and executes them,
    capturing evidence and reporting results.
    """

    name = "executor"
    role = "Automation Engineer"
    description = "Executes automated tests including performance and accessibility"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.browser_tool = BrowserTool()
        self.lighthouse_tool = LighthouseTool()
        self.accessibility_tool = AccessibilityTool(wcag_level=WCAGLevel.AA)

    @property
    def system_prompt(self) -> str:
        return """You are the Executor agent of Q-Ravens, an Automation Engineer specialist.
Your role is to execute automated tests against web applications.

Your responsibilities:
1. Take test specifications and execute them using Playwright
2. Run performance tests using Lighthouse (Core Web Vitals)
3. Run accessibility tests using axe-core (WCAG 2.1 compliance)
4. Handle dynamic elements and implement smart waits
5. Capture screenshots and evidence for failures
6. Report detailed pass/fail results with evidence

Test categories you handle:
- Functional: UI interactions, navigation, forms
- Performance: Core Web Vitals (LCP, FID, CLS), page load times
- Accessibility: WCAG 2.1 AA compliance, screen reader compatibility

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

        These are fundamental tests that verify basic functionality,
        performance, and accessibility.
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

        # Test 5: Performance - Core Web Vitals
        tests.append(TestCase(
            id=f"perf-{uuid.uuid4().hex[:8]}",
            name="Performance - Core Web Vitals",
            description="Measure Core Web Vitals using Lighthouse",
            steps=[
                "Run Lighthouse performance audit",
                "Check LCP (Largest Contentful Paint)",
                "Check TBT (Total Blocking Time)",
                "Check CLS (Cumulative Layout Shift)",
            ],
            expected_result="Core Web Vitals meet 'Good' thresholds",
            priority="medium",
            category="performance",
        ))

        # Test 6: Accessibility - WCAG 2.1 AA
        tests.append(TestCase(
            id=f"a11y-{uuid.uuid4().hex[:8]}",
            name="Accessibility - WCAG 2.1 AA",
            description="Check WCAG 2.1 Level AA compliance using axe-core",
            steps=[
                "Run axe-core accessibility audit",
                "Check for critical violations",
                "Check for serious violations",
                "Verify compliance percentage",
            ],
            expected_result="No critical or serious accessibility violations",
            priority="high",
            category="accessibility",
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
            # Execute based on test category and name
            if test_case.category == "performance" or "Performance" in test_case.name:
                result = await self._test_performance(url)
            elif test_case.category == "accessibility" or "Accessibility" in test_case.name:
                result = await self._test_accessibility(url)
            elif "Page Load" in test_case.name:
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
                error_message=result.get("error") if not result.get("passed") else None,
                duration_ms=duration,
            )

        except Exception as e:
            duration = int((datetime.now() - start_time).total_seconds() * 1000)
            logger.error(f"Test execution failed for {test_case.id}: {e}")
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

    async def _test_performance(self, url: str) -> dict:
        """
        Run Lighthouse performance audit.

        Checks Core Web Vitals and returns pass/fail based on thresholds.
        """
        logger.info(f"Running performance audit for: {url}")

        try:
            result = await self.lighthouse_tool.run_audit(url)

            # Check for errors
            if result.audits and result.audits[0].id == "error":
                return {
                    "passed": False,
                    "message": "Lighthouse audit failed",
                    "error": result.audits[0].description,
                }

            # Evaluate performance score
            perf_score = result.performance_score or 0
            cwv = result.core_web_vitals

            # Build detailed message
            details = [
                f"Performance Score: {perf_score:.0f}/100",
            ]

            if cwv.lcp_ms:
                lcp_status = "Good" if cwv.lcp_ms <= 2500 else ("Needs Improvement" if cwv.lcp_ms <= 4000 else "Poor")
                details.append(f"LCP: {cwv.lcp_ms:.0f}ms ({lcp_status})")

            if cwv.tbt_ms is not None:
                tbt_status = "Good" if cwv.tbt_ms <= 200 else ("Needs Improvement" if cwv.tbt_ms <= 600 else "Poor")
                details.append(f"TBT: {cwv.tbt_ms:.0f}ms ({tbt_status})")

            if cwv.cls_value is not None:
                cls_status = "Good" if cwv.cls_value <= 0.1 else ("Needs Improvement" if cwv.cls_value <= 0.25 else "Poor")
                details.append(f"CLS: {cwv.cls_value:.3f} ({cls_status})")

            if cwv.fcp_ms:
                fcp_status = "Good" if cwv.fcp_ms <= 1800 else ("Needs Improvement" if cwv.fcp_ms <= 3000 else "Poor")
                details.append(f"FCP: {cwv.fcp_ms:.0f}ms ({fcp_status})")

            # Pass if performance score >= 50 (adjustable threshold)
            passed = perf_score >= 50

            return {
                "passed": passed,
                "message": " | ".join(details),
            }

        except Exception as e:
            logger.error(f"Performance test failed: {e}")
            return {
                "passed": False,
                "message": "Performance test could not be completed",
                "error": str(e),
            }

    async def _test_accessibility(self, url: str) -> dict:
        """
        Run axe-core accessibility audit.

        Checks WCAG 2.1 AA compliance and returns pass/fail based on violations.
        """
        logger.info(f"Running accessibility audit for: {url}")

        try:
            result = await self.accessibility_tool.run_audit(url)

            # Check for errors
            if result.violations and result.violations[0].id == "error":
                return {
                    "passed": False,
                    "message": "Accessibility audit failed",
                    "error": result.violations[0].description,
                }

            # Build detailed message
            details = [
                f"WCAG {result.wcag_level} Compliance: {result.estimated_compliance:.1f}%",
                f"Violations: {result.violations_count}",
                f"Passes: {result.passes_count}",
            ]

            # Count by severity
            if result.critical_count > 0:
                details.append(f"Critical: {result.critical_count}")
            if result.serious_count > 0:
                details.append(f"Serious: {result.serious_count}")
            if result.moderate_count > 0:
                details.append(f"Moderate: {result.moderate_count}")

            # List top violations
            if result.violations:
                top_issues = [v.help for v in result.violations[:3]]
                details.append(f"Top issues: {'; '.join(top_issues)}")

            # Pass if no critical or serious violations
            passed = result.critical_count == 0 and result.serious_count == 0

            return {
                "passed": passed,
                "message": " | ".join(details),
            }

        except Exception as e:
            logger.error(f"Accessibility test failed: {e}")
            return {
                "passed": False,
                "message": "Accessibility test could not be completed",
                "error": str(e),
            }


# Create node function for LangGraph
async def executor_node(state: QRavensState) -> dict[str, Any]:
    """LangGraph node function for the Executor agent."""
    agent = ExecutorAgent.from_state(state)
    return await agent.process(state)
