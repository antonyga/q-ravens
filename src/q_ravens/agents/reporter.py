"""
Q-Ravens Reporter Agent

The QA Lead responsible for:
- Generating executive summaries (non-technical)
- Creating detailed technical reports
- Producing visual dashboards
- Tracking trends across test runs
- Highlighting critical issues and blockers
- Recommending next steps and priorities
"""

import json
import re
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from langchain_core.messages import AIMessage

from q_ravens.agents.base import BaseAgent
from q_ravens.core.state import (
    QRavensState,
    WorkflowPhase,
    AnalysisResult,
    TestCase,
    TestResult,
    TestStatus,
    TestReport,
    TestSummary,
    ReportFormat,
)


class ReporterAgent(BaseAgent):
    """
    The Reporter compiles test results into actionable reports.

    It generates executive summaries for non-technical stakeholders,
    detailed technical reports for developers, and provides recommendations
    for next steps based on the test findings.
    """

    name = "reporter"
    role = "QA Lead"
    description = "Compiles test results into actionable reports for different audiences"

    @property
    def system_prompt(self) -> str:
        return """You are the Reporter agent of Q-Ravens, a QA Lead specialist.
Your role is to analyze test results and generate comprehensive, actionable reports.

Your responsibilities:
1. Generate executive summaries for non-technical stakeholders
2. Create detailed technical reports for developers
3. Highlight critical issues and blockers
4. Recommend next steps and priorities
5. Track patterns and trends in test results
6. Provide actionable insights, not just data

When writing reports, consider your audience:
- Executive Summary: Brief, business-focused, no jargon
- Technical Details: Specific, actionable, with reproduction steps
- Recommendations: Prioritized, practical, with clear next actions

Output Format for Executive Summary:
Write a clear, concise summary (2-3 paragraphs) that:
- States the overall quality status (healthy/concerning/critical)
- Highlights the most important findings
- Recommends immediate actions if needed

Output Format for Critical Issues (JSON array):
```json
[
  {
    "issue": "Brief description of the issue",
    "severity": "critical|high|medium|low",
    "impact": "Business impact description",
    "recommendation": "What to do about it"
  }
]
```

Be direct and actionable. Focus on what matters most."""

    async def process(self, state: QRavensState) -> dict[str, Any]:
        """
        Generate a comprehensive test report from execution results.

        Analyzes test results and produces reports suitable for
        different audiences (executive, technical).
        """
        analysis = state.get("analysis")
        test_cases = state.get("test_cases", [])
        test_results = state.get("test_results", [])
        target_url = state.get("target_url", "")
        user_request = state.get("user_request", "")
        errors = state.get("errors", [])

        if not test_results:
            # No test results - generate a report explaining why
            report = self._generate_empty_report(
                target_url=target_url,
                user_request=user_request,
                reason="No test results available. Tests may not have been executed.",
            )

            message = AIMessage(
                content="Report generated. Note: No test results were available to analyze."
            )

            return {
                "report": report,
                "phase": WorkflowPhase.REPORTING.value,
                "next_agent": "orchestrator",
                "messages": [message],
                "current_agent": self.name,
            }

        try:
            # Generate the comprehensive report
            report = await self._generate_report(
                analysis=analysis,
                test_cases=test_cases,
                test_results=test_results,
                target_url=target_url,
                user_request=user_request,
            )

            # Create summary message
            summary = report.summary
            status_emoji = self._get_status_indicator(summary.pass_rate)

            message = AIMessage(
                content=f"""Test Report Generated {status_emoji}

**Summary:**
- Total Tests: {summary.total_tests}
- Passed: {summary.passed} | Failed: {summary.failed} | Skipped: {summary.skipped}
- Pass Rate: {summary.pass_rate:.1f}%
- Critical Issues: {len(report.critical_issues)}

{report.executive_summary[:500]}{'...' if len(report.executive_summary) > 500 else ''}"""
            )

            return {
                "report": report,
                "phase": WorkflowPhase.REPORTING.value,
                "next_agent": "orchestrator",
                "messages": [message],
                "current_agent": self.name,
            }

        except Exception as e:
            return {
                "phase": WorkflowPhase.ERROR.value,
                "errors": errors + [f"Report generation failed: {str(e)}"],
                "current_agent": self.name,
            }

    async def _generate_report(
        self,
        analysis: Optional[AnalysisResult],
        test_cases: list[TestCase],
        test_results: list[TestResult],
        target_url: str,
        user_request: str,
    ) -> TestReport:
        """
        Generate a comprehensive test report.

        Args:
            analysis: Analysis results from Analyzer
            test_cases: Test cases from Designer
            test_results: Test results from Executor
            target_url: Target URL being tested
            user_request: Original user request

        Returns:
            Complete TestReport object
        """
        # Calculate summary statistics
        summary = self._calculate_summary(test_results)

        # Build test details
        test_details = self._build_test_details(test_cases, test_results)

        # Generate executive summary using LLM
        executive_summary = await self._generate_executive_summary(
            summary=summary,
            test_details=test_details,
            target_url=target_url,
            user_request=user_request,
        )

        # Identify critical issues and recommendations
        critical_issues, warnings = self._identify_issues(test_results, test_cases)
        recommendations = await self._generate_recommendations(
            summary=summary,
            critical_issues=critical_issues,
            warnings=warnings,
            user_request=user_request,
        )

        # Generate markdown content
        markdown_content = self._generate_markdown_report(
            summary=summary,
            executive_summary=executive_summary,
            test_details=test_details,
            critical_issues=critical_issues,
            warnings=warnings,
            recommendations=recommendations,
            target_url=target_url,
            user_request=user_request,
        )

        # Create the report
        report = TestReport(
            report_id=f"RPT-{uuid4().hex[:8].upper()}",
            title=f"Q-Ravens Test Report: {target_url}",
            target_url=target_url,
            user_request=user_request,
            executive_summary=executive_summary,
            summary=summary,
            critical_issues=critical_issues,
            warnings=warnings,
            recommendations=recommendations,
            test_details=test_details,
            markdown_content=markdown_content,
            format=ReportFormat.MARKDOWN,
        )

        return report

    def _calculate_summary(self, test_results: list[TestResult]) -> TestSummary:
        """Calculate summary statistics from test results."""
        total = len(test_results)
        if total == 0:
            return TestSummary()

        passed = sum(1 for r in test_results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in test_results if r.status == TestStatus.FAILED)
        skipped = sum(1 for r in test_results if r.status == TestStatus.SKIPPED)
        errors = sum(1 for r in test_results if r.status == TestStatus.ERROR)
        total_duration = sum(r.duration_ms for r in test_results)

        pass_rate = (passed / total * 100) if total > 0 else 0.0

        return TestSummary(
            total_tests=total,
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
            pass_rate=pass_rate,
            total_duration_ms=total_duration,
        )

    def _build_test_details(
        self, test_cases: list[TestCase], test_results: list[TestResult]
    ) -> list[dict[str, Any]]:
        """Build detailed test information by matching cases with results."""
        # Create a map of test_id to result
        result_map = {r.test_id: r for r in test_results}

        details = []
        for tc in test_cases:
            result = result_map.get(tc.id)
            detail = {
                "id": tc.id,
                "name": tc.name,
                "description": tc.description,
                "category": tc.category,
                "priority": tc.priority,
                "status": result.status.value if result else "not_executed",
                "actual_result": result.actual_result if result else None,
                "error_message": result.error_message if result else None,
                "duration_ms": result.duration_ms if result else 0,
                "screenshot_path": result.screenshot_path if result else None,
            }
            details.append(detail)

        return details

    async def _generate_executive_summary(
        self,
        summary: TestSummary,
        test_details: list[dict[str, Any]],
        target_url: str,
        user_request: str,
    ) -> str:
        """Generate an executive summary using LLM."""
        # Build context for LLM
        failed_tests = [d for d in test_details if d["status"] == "failed"]
        error_tests = [d for d in test_details if d["status"] == "error"]

        context = f"""Test Results Summary:
- Target URL: {target_url}
- User Request: {user_request or "General testing"}
- Total Tests: {summary.total_tests}
- Passed: {summary.passed} ({summary.pass_rate:.1f}%)
- Failed: {summary.failed}
- Errors: {summary.errors}
- Skipped: {summary.skipped}
- Total Duration: {summary.total_duration_ms}ms

Failed Tests:
{self._format_failed_tests(failed_tests[:5])}

Error Tests:
{self._format_failed_tests(error_tests[:5])}"""

        prompt = f"""Based on the test results below, write a brief executive summary (2-3 paragraphs).

{context}

The summary should:
1. State the overall quality status clearly (e.g., "Tests are passing well" or "Critical issues found")
2. Highlight the most important findings (failures, patterns)
3. Be understandable by non-technical stakeholders
4. Avoid jargon and technical details

Write ONLY the summary text, no headers or formatting."""

        try:
            response = await self.invoke_llm(prompt)
            return response.strip()
        except Exception:
            # Fallback to programmatic summary
            return self._generate_fallback_executive_summary(summary, failed_tests)

    def _format_failed_tests(self, tests: list[dict[str, Any]]) -> str:
        """Format failed tests for context."""
        if not tests:
            return "None"

        lines = []
        for t in tests:
            error = t.get("error_message", "No error message")
            lines.append(f"- {t['name']}: {error[:100]}")
        return "\n".join(lines)

    def _generate_fallback_executive_summary(
        self, summary: TestSummary, failed_tests: list[dict[str, Any]]
    ) -> str:
        """Generate a basic executive summary when LLM fails."""
        if summary.pass_rate >= 90:
            status = "The application is in good health"
            concern_level = "minor"
        elif summary.pass_rate >= 70:
            status = "The application has some quality concerns"
            concern_level = "moderate"
        elif summary.pass_rate >= 50:
            status = "The application has significant quality issues"
            concern_level = "significant"
        else:
            status = "The application has critical quality issues requiring immediate attention"
            concern_level = "critical"

        summary_text = f"""{status}. Out of {summary.total_tests} tests executed, {summary.passed} passed ({summary.pass_rate:.1f}% pass rate) and {summary.failed} failed.

"""

        if failed_tests:
            summary_text += f"The {concern_level} concerns primarily involve: "
            failure_names = [t["name"] for t in failed_tests[:3]]
            summary_text += ", ".join(failure_names)
            if len(failed_tests) > 3:
                summary_text += f", and {len(failed_tests) - 3} other test(s)"
            summary_text += ".\n\n"

        if summary.pass_rate < 70:
            summary_text += "Immediate attention is recommended to address the failing tests before deployment."
        elif summary.pass_rate < 90:
            summary_text += "Review of failing tests is recommended before the next release."
        else:
            summary_text += "The application meets quality standards for deployment."

        return summary_text

    def _identify_issues(
        self, test_results: list[TestResult], test_cases: list[TestCase]
    ) -> tuple[list[str], list[str]]:
        """Identify critical issues and warnings from test results."""
        critical_issues = []
        warnings = []

        # Create case map for priority lookup
        case_map = {tc.id: tc for tc in test_cases}

        for result in test_results:
            if result.status in [TestStatus.FAILED, TestStatus.ERROR]:
                test_case = case_map.get(result.test_id)
                priority = test_case.priority if test_case else "medium"
                name = test_case.name if test_case else result.test_id

                issue = f"{name}: {result.error_message or 'Test failed'}"

                if priority == "high" or result.status == TestStatus.ERROR:
                    critical_issues.append(issue)
                else:
                    warnings.append(issue)

        return critical_issues, warnings

    async def _generate_recommendations(
        self,
        summary: TestSummary,
        critical_issues: list[str],
        warnings: list[str],
        user_request: str,
    ) -> list[str]:
        """Generate actionable recommendations using LLM."""
        if summary.pass_rate >= 95 and not critical_issues:
            return [
                "All tests passing - consider expanding test coverage",
                "Monitor for regression in future releases",
            ]

        context = f"""Test Summary:
- Pass Rate: {summary.pass_rate:.1f}%
- Failed: {summary.failed}
- Errors: {summary.errors}

Critical Issues ({len(critical_issues)}):
{chr(10).join(f'- {i}' for i in critical_issues[:5]) or 'None'}

Warnings ({len(warnings)}):
{chr(10).join(f'- {w}' for w in warnings[:5]) or 'None'}

User's Goal: {user_request or 'General testing'}"""

        prompt = f"""Based on these test results, provide 3-5 actionable recommendations.

{context}

Each recommendation should:
1. Be specific and actionable
2. Include priority (immediate, short-term, long-term)
3. Focus on improving quality

Return as a simple list, one recommendation per line, no numbering."""

        try:
            response = await self.invoke_llm(prompt)
            # Parse response into list
            recommendations = [
                line.strip().lstrip("- •*")
                for line in response.strip().split("\n")
                if line.strip() and not line.strip().startswith("#")
            ]
            return recommendations[:5]  # Limit to 5
        except Exception:
            return self._generate_fallback_recommendations(summary, critical_issues)

    def _generate_fallback_recommendations(
        self, summary: TestSummary, critical_issues: list[str]
    ) -> list[str]:
        """Generate basic recommendations when LLM fails."""
        recommendations = []

        if critical_issues:
            recommendations.append(
                f"[Immediate] Address {len(critical_issues)} critical issue(s) before deployment"
            )

        if summary.failed > 0:
            recommendations.append(
                f"[Short-term] Investigate and fix {summary.failed} failing test(s)"
            )

        if summary.errors > 0:
            recommendations.append(
                f"[Immediate] Review {summary.errors} test error(s) - may indicate infrastructure issues"
            )

        if summary.pass_rate < 80:
            recommendations.append(
                "[Short-term] Consider a dedicated bug-fixing sprint to improve stability"
            )

        if summary.skipped > 0:
            recommendations.append(
                f"[Long-term] Review {summary.skipped} skipped test(s) for relevance"
            )

        if not recommendations:
            recommendations.append("[Long-term] Consider expanding test coverage")
            recommendations.append("[Long-term] Set up continuous monitoring")

        return recommendations

    def _generate_markdown_report(
        self,
        summary: TestSummary,
        executive_summary: str,
        test_details: list[dict[str, Any]],
        critical_issues: list[str],
        warnings: list[str],
        recommendations: list[str],
        target_url: str,
        user_request: str,
    ) -> str:
        """Generate a complete markdown report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status_badge = self._get_status_badge(summary.pass_rate)

        report = f"""# Q-Ravens Test Report

**Generated:** {timestamp}
**Target:** {target_url}
**Status:** {status_badge}

---

## Executive Summary

{executive_summary}

---

## Test Results Overview

| Metric | Value |
|--------|-------|
| Total Tests | {summary.total_tests} |
| Passed | {summary.passed} |
| Failed | {summary.failed} |
| Errors | {summary.errors} |
| Skipped | {summary.skipped} |
| Pass Rate | {summary.pass_rate:.1f}% |
| Total Duration | {summary.total_duration_ms}ms |

"""

        # Critical Issues Section
        if critical_issues:
            report += """---

## Critical Issues

The following issues require immediate attention:

"""
            for issue in critical_issues:
                report += f"- {issue}\n"
            report += "\n"

        # Warnings Section
        if warnings:
            report += """---

## Warnings

The following issues should be reviewed:

"""
            for warning in warnings:
                report += f"- {warning}\n"
            report += "\n"

        # Recommendations Section
        report += """---

## Recommendations

"""
        for i, rec in enumerate(recommendations, 1):
            report += f"{i}. {rec}\n"

        # Detailed Test Results
        report += """
---

## Detailed Test Results

"""
        # Group by status
        for status in ["failed", "error", "passed", "skipped", "not_executed"]:
            tests_with_status = [d for d in test_details if d["status"] == status]
            if tests_with_status:
                status_header = status.replace("_", " ").title()
                report += f"### {status_header} ({len(tests_with_status)})\n\n"

                for test in tests_with_status:
                    report += f"**{test['id']}: {test['name']}**\n"
                    report += f"- Category: {test['category']} | Priority: {test['priority']}\n"
                    if test.get("error_message"):
                        report += f"- Error: {test['error_message']}\n"
                    if test.get("duration_ms"):
                        report += f"- Duration: {test['duration_ms']}ms\n"
                    report += "\n"

        # Footer
        report += f"""---

*Report generated by Q-Ravens Autonomous QA Agent Swarm*
*Testing Goal: {user_request or 'General comprehensive testing'}*
"""

        return report

    def _get_status_badge(self, pass_rate: float) -> str:
        """Get a status badge based on pass rate."""
        if pass_rate >= 90:
            return "HEALTHY"
        elif pass_rate >= 70:
            return "NEEDS ATTENTION"
        elif pass_rate >= 50:
            return "CONCERNING"
        else:
            return "CRITICAL"

    def _get_status_indicator(self, pass_rate: float) -> str:
        """Get a status indicator emoji based on pass rate."""
        if pass_rate >= 90:
            return "[PASS]"
        elif pass_rate >= 70:
            return "[WARN]"
        elif pass_rate >= 50:
            return "[ALERT]"
        else:
            return "[FAIL]"

    def _generate_empty_report(
        self, target_url: str, user_request: str, reason: str
    ) -> TestReport:
        """Generate a report when no test results are available."""
        return TestReport(
            report_id=f"RPT-{uuid4().hex[:8].upper()}",
            title=f"Q-Ravens Test Report: {target_url}",
            target_url=target_url,
            user_request=user_request,
            executive_summary=f"No test results to report. {reason}",
            summary=TestSummary(),
            critical_issues=[],
            warnings=[reason],
            recommendations=[
                "Verify that tests were designed and executed",
                "Check for errors in the testing workflow",
                "Re-run the test suite if needed",
            ],
            test_details=[],
            markdown_content=f"# Q-Ravens Test Report\n\n**Status:** No Results\n\n{reason}",
            format=ReportFormat.MARKDOWN,
        )


# Create node function for LangGraph
async def reporter_node(state: QRavensState) -> dict[str, Any]:
    """LangGraph node function for the Reporter agent."""
    agent = ReporterAgent.from_state(state)
    return await agent.process(state)
