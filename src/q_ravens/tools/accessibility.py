"""
Q-Ravens Accessibility Tool

WCAG 2.1 accessibility testing using axe-core via Playwright.
Detects accessibility violations and provides remediation guidance.
"""

import logging
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, Optional

from playwright.async_api import async_playwright, Browser, Page
from pydantic import BaseModel, Field

from q_ravens.core.config import settings

logger = logging.getLogger(__name__)


# Axe-core CDN URL (auto-injects into page)
AXE_CORE_CDN = "https://cdnjs.cloudflare.com/ajax/libs/axe-core/4.8.4/axe.min.js"


class WCAGLevel(str, Enum):
    """WCAG conformance levels."""

    A = "A"
    AA = "AA"
    AAA = "AAA"


class ImpactLevel(str, Enum):
    """Accessibility violation impact levels."""

    CRITICAL = "critical"
    SERIOUS = "serious"
    MODERATE = "moderate"
    MINOR = "minor"


class AccessibilityViolation(BaseModel):
    """Individual accessibility violation."""

    id: str
    impact: ImpactLevel
    description: str
    help: str
    help_url: str
    tags: list[str] = Field(default_factory=list)
    nodes: list[dict[str, Any]] = Field(default_factory=list)
    wcag_criteria: list[str] = Field(default_factory=list)


class AccessibilityResult(BaseModel):
    """Complete accessibility audit result."""

    url: str
    timestamp: str

    # Summary counts
    violations_count: int = 0
    passes_count: int = 0
    incomplete_count: int = 0
    inapplicable_count: int = 0

    # By impact level
    critical_count: int = 0
    serious_count: int = 0
    moderate_count: int = 0
    minor_count: int = 0

    # WCAG compliance estimate
    wcag_level: Optional[str] = None
    estimated_compliance: float = 0.0  # 0-100

    # Detailed violations
    violations: list[AccessibilityViolation] = Field(default_factory=list)

    # Passed rules (for reporting)
    passes: list[str] = Field(default_factory=list)

    # Incomplete (needs manual review)
    incomplete: list[dict[str, Any]] = Field(default_factory=list)


class AccessibilityTool:
    """
    Accessibility testing tool using axe-core.

    Injects axe-core into web pages via Playwright and runs
    WCAG 2.1 compliance checks.
    """

    def __init__(
        self,
        wcag_level: WCAGLevel = WCAGLevel.AA,
        headless: bool = True,
    ):
        """
        Initialize accessibility tool.

        Args:
            wcag_level: Target WCAG conformance level
            headless: Run browser in headless mode
        """
        self.wcag_level = wcag_level
        self.headless = headless

    @asynccontextmanager
    async def _get_browser(self):
        """Context manager for browser instance."""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.headless)
            try:
                yield browser
            finally:
                await browser.close()

    @asynccontextmanager
    async def _get_page(self, browser: Browser):
        """Context manager for page instance."""
        context = await browser.new_context(
            viewport={"width": 1280, "height": 720},
            user_agent="Q-Ravens/0.1.0 (Accessibility Audit)",
        )
        page = await context.new_page()
        try:
            yield page
        finally:
            await context.close()

    async def run_audit(
        self,
        url: str,
        include_passes: bool = False,
        include_incomplete: bool = True,
        rules: Optional[list[str]] = None,
        exclude_selectors: Optional[list[str]] = None,
    ) -> AccessibilityResult:
        """
        Run accessibility audit on a URL.

        Args:
            url: URL to audit
            include_passes: Include passed rules in results
            include_incomplete: Include incomplete checks
            rules: Specific rules to run (None = all)
            exclude_selectors: CSS selectors to exclude from audit

        Returns:
            AccessibilityResult with findings
        """
        logger.info(f"Running accessibility audit for: {url}")

        async with self._get_browser() as browser:
            async with self._get_page(browser) as page:
                try:
                    # Navigate to page
                    await page.goto(url, timeout=settings.default_timeout * 1000)
                    await page.wait_for_load_state("networkidle")

                    # Inject axe-core
                    await self._inject_axe(page)

                    # Build axe options
                    options = self._build_options(rules, exclude_selectors)

                    # Run axe analysis
                    results = await page.evaluate(
                        f"window.axe.run(document, {options})"
                    )

                    # Parse results
                    return self._parse_results(url, results, include_passes, include_incomplete)

                except Exception as e:
                    logger.error(f"Accessibility audit failed: {e}")
                    return AccessibilityResult(
                        url=url,
                        timestamp="",
                        violations=[
                            AccessibilityViolation(
                                id="error",
                                impact=ImpactLevel.CRITICAL,
                                description=f"Audit failed: {str(e)}",
                                help="Check if the URL is accessible",
                                help_url="",
                                tags=["error"],
                            )
                        ],
                        violations_count=1,
                        critical_count=1,
                    )

    async def _inject_axe(self, page: Page) -> None:
        """Inject axe-core library into the page."""
        # Check if axe is already loaded
        has_axe = await page.evaluate("typeof window.axe !== 'undefined'")
        if has_axe:
            return

        # Fetch and inject axe-core
        axe_script = f"""
            (function() {{
                return new Promise((resolve, reject) => {{
                    const script = document.createElement('script');
                    script.src = '{AXE_CORE_CDN}';
                    script.onload = () => resolve(true);
                    script.onerror = () => reject(new Error('Failed to load axe-core'));
                    document.head.appendChild(script);
                }});
            }})()
        """
        await page.evaluate(axe_script)

        # Wait for axe to be available
        await page.wait_for_function("typeof window.axe !== 'undefined'")
        logger.debug("axe-core injected successfully")

    def _build_options(
        self,
        rules: Optional[list[str]],
        exclude_selectors: Optional[list[str]],
    ) -> str:
        """Build axe-core run options."""
        import json

        options: dict[str, Any] = {
            "resultTypes": ["violations", "passes", "incomplete", "inapplicable"],
        }

        # Filter by WCAG level
        wcag_tags = self._get_wcag_tags()
        options["runOnly"] = {
            "type": "tag",
            "values": wcag_tags,
        }

        # Specific rules override
        if rules:
            options["runOnly"] = {
                "type": "rule",
                "values": rules,
            }

        # Exclude elements
        if exclude_selectors:
            options["exclude"] = [[sel] for sel in exclude_selectors]

        return json.dumps(options)

    def _get_wcag_tags(self) -> list[str]:
        """Get axe-core tags for the target WCAG level."""
        tags = ["wcag2a", "wcag21a", "best-practice"]

        if self.wcag_level in [WCAGLevel.AA, WCAGLevel.AAA]:
            tags.extend(["wcag2aa", "wcag21aa"])

        if self.wcag_level == WCAGLevel.AAA:
            tags.extend(["wcag2aaa", "wcag21aaa"])

        return tags

    def _parse_results(
        self,
        url: str,
        results: dict,
        include_passes: bool,
        include_incomplete: bool,
    ) -> AccessibilityResult:
        """Parse axe-core results into AccessibilityResult."""
        from datetime import datetime

        violations = []
        critical_count = 0
        serious_count = 0
        moderate_count = 0
        minor_count = 0

        # Parse violations
        for v in results.get("violations", []):
            impact = ImpactLevel(v.get("impact", "minor"))

            # Count by impact
            if impact == ImpactLevel.CRITICAL:
                critical_count += 1
            elif impact == ImpactLevel.SERIOUS:
                serious_count += 1
            elif impact == ImpactLevel.MODERATE:
                moderate_count += 1
            else:
                minor_count += 1

            # Extract WCAG criteria from tags
            wcag_criteria = [
                tag for tag in v.get("tags", [])
                if tag.startswith("wcag")
            ]

            # Parse nodes (affected elements)
            nodes = []
            for node in v.get("nodes", [])[:5]:  # Limit to 5 examples
                nodes.append({
                    "html": node.get("html", "")[:200],  # Truncate
                    "target": node.get("target", []),
                    "failure_summary": node.get("failureSummary", ""),
                })

            violation = AccessibilityViolation(
                id=v.get("id", ""),
                impact=impact,
                description=v.get("description", ""),
                help=v.get("help", ""),
                help_url=v.get("helpUrl", ""),
                tags=v.get("tags", []),
                nodes=nodes,
                wcag_criteria=wcag_criteria,
            )
            violations.append(violation)

        # Sort violations by impact severity
        impact_order = {
            ImpactLevel.CRITICAL: 0,
            ImpactLevel.SERIOUS: 1,
            ImpactLevel.MODERATE: 2,
            ImpactLevel.MINOR: 3,
        }
        violations.sort(key=lambda x: impact_order.get(x.impact, 4))

        # Parse passes
        passes = []
        if include_passes:
            passes = [p.get("id", "") for p in results.get("passes", [])]

        # Parse incomplete
        incomplete = []
        if include_incomplete:
            for inc in results.get("incomplete", []):
                incomplete.append({
                    "id": inc.get("id", ""),
                    "description": inc.get("description", ""),
                    "help": inc.get("help", ""),
                    "nodes_count": len(inc.get("nodes", [])),
                })

        # Calculate compliance estimate
        total_rules = (
            len(results.get("violations", []))
            + len(results.get("passes", []))
        )
        passes_count = len(results.get("passes", []))
        compliance = (passes_count / total_rules * 100) if total_rules > 0 else 100.0

        return AccessibilityResult(
            url=url,
            timestamp=datetime.now().isoformat(),
            violations_count=len(violations),
            passes_count=len(results.get("passes", [])),
            incomplete_count=len(results.get("incomplete", [])),
            inapplicable_count=len(results.get("inapplicable", [])),
            critical_count=critical_count,
            serious_count=serious_count,
            moderate_count=moderate_count,
            minor_count=minor_count,
            wcag_level=self.wcag_level.value,
            estimated_compliance=round(compliance, 1),
            violations=violations,
            passes=passes,
            incomplete=incomplete,
        )

    def get_impact_rating(self, impact: ImpactLevel) -> str:
        """Get human-readable impact description."""
        ratings = {
            ImpactLevel.CRITICAL: "Blocks access for some users",
            ImpactLevel.SERIOUS: "Significantly impacts user experience",
            ImpactLevel.MODERATE: "Moderately impacts user experience",
            ImpactLevel.MINOR: "Minor impact on user experience",
        }
        return ratings.get(impact, "Unknown impact")

    def format_summary(self, result: AccessibilityResult) -> str:
        """Format a human-readable summary of results."""
        lines = [
            f"Accessibility Audit Report: {result.url}",
            "=" * 50,
            "",
            f"WCAG {result.wcag_level} Compliance: {result.estimated_compliance:.1f}%",
            "",
            "Summary:",
            f"  Violations: {result.violations_count}",
            f"  Passes: {result.passes_count}",
            f"  Needs Review: {result.incomplete_count}",
            "",
            "Violations by Impact:",
            f"  Critical: {result.critical_count}",
            f"  Serious:  {result.serious_count}",
            f"  Moderate: {result.moderate_count}",
            f"  Minor:    {result.minor_count}",
        ]

        if result.violations:
            lines.extend(["", "Top Issues:"])
            for v in result.violations[:5]:
                lines.append(f"  [{v.impact.value.upper()}] {v.id}: {v.help}")
                if v.wcag_criteria:
                    lines.append(f"    WCAG: {', '.join(v.wcag_criteria)}")

        if result.incomplete:
            lines.extend(["", "Needs Manual Review:"])
            for inc in result.incomplete[:3]:
                lines.append(f"  - {inc['id']}: {inc['description'][:60]}...")

        return "\n".join(lines)

    def get_remediation_priority(self, result: AccessibilityResult) -> list[dict[str, Any]]:
        """
        Get prioritized remediation recommendations.

        Returns violations sorted by impact and frequency.
        """
        priorities = []

        for v in result.violations:
            priority_score = {
                ImpactLevel.CRITICAL: 100,
                ImpactLevel.SERIOUS: 75,
                ImpactLevel.MODERATE: 50,
                ImpactLevel.MINOR: 25,
            }.get(v.impact, 0)

            # Boost score by number of affected elements
            affected_count = len(v.nodes)
            priority_score += min(affected_count * 5, 25)

            priorities.append({
                "id": v.id,
                "priority_score": priority_score,
                "impact": v.impact.value,
                "affected_elements": affected_count,
                "description": v.help,
                "help_url": v.help_url,
                "wcag_criteria": v.wcag_criteria,
            })

        return sorted(priorities, key=lambda x: x["priority_score"], reverse=True)


# Convenience function for quick audits
async def run_accessibility_audit(
    url: str,
    wcag_level: WCAGLevel = WCAGLevel.AA,
) -> AccessibilityResult:
    """
    Run a quick accessibility audit.

    Args:
        url: URL to audit
        wcag_level: Target WCAG level (default: AA)

    Returns:
        AccessibilityResult
    """
    tool = AccessibilityTool(wcag_level=wcag_level)
    return await tool.run_audit(url)
