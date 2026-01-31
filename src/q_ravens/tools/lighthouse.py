"""
Q-Ravens Lighthouse Tool

Performance testing using Google Lighthouse via Playwright.
Provides Core Web Vitals and performance metrics for web pages.
"""

import asyncio
import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PerformanceCategory(str, Enum):
    """Lighthouse audit categories."""

    PERFORMANCE = "performance"
    ACCESSIBILITY = "accessibility"
    BEST_PRACTICES = "best-practices"
    SEO = "seo"
    PWA = "pwa"


class CoreWebVitals(BaseModel):
    """Core Web Vitals metrics."""

    # Largest Contentful Paint (LCP) - loading performance
    lcp_ms: Optional[float] = Field(default=None, description="Largest Contentful Paint in ms")
    lcp_score: Optional[float] = Field(default=None, description="LCP score (0-1)")

    # First Input Delay (FID) / Total Blocking Time (TBT) - interactivity
    tbt_ms: Optional[float] = Field(default=None, description="Total Blocking Time in ms")
    tbt_score: Optional[float] = Field(default=None, description="TBT score (0-1)")

    # Cumulative Layout Shift (CLS) - visual stability
    cls_value: Optional[float] = Field(default=None, description="Cumulative Layout Shift value")
    cls_score: Optional[float] = Field(default=None, description="CLS score (0-1)")

    # First Contentful Paint (FCP)
    fcp_ms: Optional[float] = Field(default=None, description="First Contentful Paint in ms")
    fcp_score: Optional[float] = Field(default=None, description="FCP score (0-1)")

    # Speed Index
    speed_index_ms: Optional[float] = Field(default=None, description="Speed Index in ms")
    speed_index_score: Optional[float] = Field(default=None, description="Speed Index score (0-1)")

    # Time to Interactive
    tti_ms: Optional[float] = Field(default=None, description="Time to Interactive in ms")
    tti_score: Optional[float] = Field(default=None, description="TTI score (0-1)")


class PerformanceAudit(BaseModel):
    """Individual performance audit result."""

    id: str
    title: str
    description: str
    score: Optional[float] = None  # 0-1, None means not applicable
    display_value: Optional[str] = None
    numeric_value: Optional[float] = None
    numeric_unit: Optional[str] = None


class LighthouseResult(BaseModel):
    """Complete Lighthouse audit result."""

    url: str
    fetch_time: str

    # Category scores (0-100)
    performance_score: Optional[float] = None
    accessibility_score: Optional[float] = None
    best_practices_score: Optional[float] = None
    seo_score: Optional[float] = None

    # Core Web Vitals
    core_web_vitals: CoreWebVitals = Field(default_factory=CoreWebVitals)

    # Detailed audits
    audits: list[PerformanceAudit] = Field(default_factory=list)

    # Opportunities for improvement
    opportunities: list[dict[str, Any]] = Field(default_factory=list)

    # Diagnostics
    diagnostics: list[dict[str, Any]] = Field(default_factory=list)

    # Raw JSON (for detailed analysis)
    raw_json_path: Optional[str] = None


class LighthouseTool:
    """
    Lighthouse performance testing tool.

    Uses Google Lighthouse via CLI to analyze web page performance,
    accessibility, SEO, and best practices.

    Requires: npm install -g lighthouse
    """

    def __init__(
        self,
        chrome_flags: Optional[list[str]] = None,
        throttling: bool = True,
        device: str = "desktop",  # "desktop" or "mobile"
    ):
        """
        Initialize Lighthouse tool.

        Args:
            chrome_flags: Additional Chrome flags
            throttling: Whether to apply network/CPU throttling
            device: Device emulation ("desktop" or "mobile")
        """
        self.chrome_flags = chrome_flags or ["--headless", "--no-sandbox"]
        self.throttling = throttling
        self.device = device

    async def run_audit(
        self,
        url: str,
        categories: Optional[list[PerformanceCategory]] = None,
        save_report: bool = False,
        report_path: Optional[str] = None,
    ) -> LighthouseResult:
        """
        Run Lighthouse audit on a URL.

        Args:
            url: URL to audit
            categories: Categories to audit (default: all)
            save_report: Whether to save the JSON report
            report_path: Path for the report (auto-generated if not provided)

        Returns:
            LighthouseResult with audit findings
        """
        categories = categories or [
            PerformanceCategory.PERFORMANCE,
            PerformanceCategory.ACCESSIBILITY,
            PerformanceCategory.BEST_PRACTICES,
            PerformanceCategory.SEO,
        ]

        # Create temp file for output
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            output_path = tmp_file.name

        try:
            # Build Lighthouse command
            cmd = self._build_command(url, categories, output_path)

            logger.info(f"Running Lighthouse audit for: {url}")
            logger.debug(f"Command: {' '.join(cmd)}")

            # Run Lighthouse
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                logger.error(f"Lighthouse failed: {error_msg}")
                # Return partial result with error
                return LighthouseResult(
                    url=url,
                    fetch_time="",
                    audits=[
                        PerformanceAudit(
                            id="error",
                            title="Lighthouse Error",
                            description=f"Audit failed: {error_msg}",
                            score=0,
                        )
                    ],
                )

            # Parse results
            result = self._parse_results(url, output_path)

            # Optionally save report
            if save_report:
                final_path = report_path or f"lighthouse_report_{Path(url).stem}.json"
                Path(output_path).rename(final_path)
                result.raw_json_path = final_path
            else:
                Path(output_path).unlink(missing_ok=True)

            return result

        except FileNotFoundError:
            logger.error("Lighthouse not found. Install with: npm install -g lighthouse")
            return LighthouseResult(
                url=url,
                fetch_time="",
                audits=[
                    PerformanceAudit(
                        id="error",
                        title="Lighthouse Not Installed",
                        description="Install Lighthouse: npm install -g lighthouse",
                        score=0,
                    )
                ],
            )
        except Exception as e:
            logger.error(f"Lighthouse audit failed: {e}")
            return LighthouseResult(
                url=url,
                fetch_time="",
                audits=[
                    PerformanceAudit(
                        id="error",
                        title="Audit Error",
                        description=str(e),
                        score=0,
                    )
                ],
            )

    def _build_command(
        self,
        url: str,
        categories: list[PerformanceCategory],
        output_path: str,
    ) -> list[str]:
        """Build the Lighthouse CLI command."""
        cmd = [
            "lighthouse",
            url,
            "--output=json",
            f"--output-path={output_path}",
            "--quiet",
        ]

        # Add categories
        cat_str = ",".join(c.value for c in categories)
        cmd.append(f"--only-categories={cat_str}")

        # Add Chrome flags
        if self.chrome_flags:
            cmd.append(f"--chrome-flags={' '.join(self.chrome_flags)}")

        # Device preset
        if self.device == "mobile":
            cmd.append("--preset=perf")
            cmd.append("--form-factor=mobile")
        else:
            cmd.append("--preset=desktop")
            cmd.append("--form-factor=desktop")

        # Throttling
        if not self.throttling:
            cmd.append("--throttling-method=provided")

        return cmd

    def _parse_results(self, url: str, json_path: str) -> LighthouseResult:
        """Parse Lighthouse JSON results."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract category scores
        categories = data.get("categories", {})
        performance_score = self._get_category_score(categories, "performance")
        accessibility_score = self._get_category_score(categories, "accessibility")
        best_practices_score = self._get_category_score(categories, "best-practices")
        seo_score = self._get_category_score(categories, "seo")

        # Extract audits
        audits_data = data.get("audits", {})
        audits = []
        opportunities = []
        diagnostics = []

        # Parse Core Web Vitals
        cwv = self._extract_core_web_vitals(audits_data)

        # Parse individual audits
        for audit_id, audit in audits_data.items():
            parsed_audit = PerformanceAudit(
                id=audit_id,
                title=audit.get("title", ""),
                description=audit.get("description", ""),
                score=audit.get("score"),
                display_value=audit.get("displayValue"),
                numeric_value=audit.get("numericValue"),
                numeric_unit=audit.get("numericUnit"),
            )
            audits.append(parsed_audit)

            # Categorize as opportunity or diagnostic
            if audit.get("details", {}).get("type") == "opportunity":
                opportunities.append({
                    "id": audit_id,
                    "title": audit.get("title"),
                    "savings_ms": audit.get("details", {}).get("overallSavingsMs", 0),
                    "savings_bytes": audit.get("details", {}).get("overallSavingsBytes", 0),
                })

        return LighthouseResult(
            url=url,
            fetch_time=data.get("fetchTime", ""),
            performance_score=performance_score,
            accessibility_score=accessibility_score,
            best_practices_score=best_practices_score,
            seo_score=seo_score,
            core_web_vitals=cwv,
            audits=audits,
            opportunities=sorted(
                opportunities,
                key=lambda x: x.get("savings_ms", 0),
                reverse=True,
            )[:10],
            diagnostics=diagnostics,
        )

    def _get_category_score(self, categories: dict, category_id: str) -> Optional[float]:
        """Extract category score (0-100)."""
        cat = categories.get(category_id, {})
        score = cat.get("score")
        return round(score * 100, 1) if score is not None else None

    def _extract_core_web_vitals(self, audits: dict) -> CoreWebVitals:
        """Extract Core Web Vitals from audits."""
        def get_metric(audit_id: str) -> tuple[Optional[float], Optional[float]]:
            audit = audits.get(audit_id, {})
            value = audit.get("numericValue")
            score = audit.get("score")
            return value, score

        lcp_ms, lcp_score = get_metric("largest-contentful-paint")
        tbt_ms, tbt_score = get_metric("total-blocking-time")
        cls_value, cls_score = get_metric("cumulative-layout-shift")
        fcp_ms, fcp_score = get_metric("first-contentful-paint")
        si_ms, si_score = get_metric("speed-index")
        tti_ms, tti_score = get_metric("interactive")

        return CoreWebVitals(
            lcp_ms=lcp_ms,
            lcp_score=lcp_score,
            tbt_ms=tbt_ms,
            tbt_score=tbt_score,
            cls_value=cls_value,
            cls_score=cls_score,
            fcp_ms=fcp_ms,
            fcp_score=fcp_score,
            speed_index_ms=si_ms,
            speed_index_score=si_score,
            tti_ms=tti_ms,
            tti_score=tti_score,
        )

    def get_score_rating(self, score: Optional[float]) -> str:
        """Get human-readable rating for a score."""
        if score is None:
            return "N/A"
        if score >= 90:
            return "Good"
        if score >= 50:
            return "Needs Improvement"
        return "Poor"

    def format_summary(self, result: LighthouseResult) -> str:
        """Format a human-readable summary of results."""
        lines = [
            f"Lighthouse Performance Report: {result.url}",
            "=" * 50,
            "",
            "Category Scores:",
            f"  Performance:    {result.performance_score or 'N/A':>5} ({self.get_score_rating(result.performance_score)})",
            f"  Accessibility:  {result.accessibility_score or 'N/A':>5} ({self.get_score_rating(result.accessibility_score)})",
            f"  Best Practices: {result.best_practices_score or 'N/A':>5} ({self.get_score_rating(result.best_practices_score)})",
            f"  SEO:            {result.seo_score or 'N/A':>5} ({self.get_score_rating(result.seo_score)})",
            "",
            "Core Web Vitals:",
        ]

        cwv = result.core_web_vitals
        if cwv.lcp_ms:
            lines.append(f"  LCP: {cwv.lcp_ms:.0f}ms ({self.get_score_rating((cwv.lcp_score or 0) * 100)})")
        if cwv.tbt_ms:
            lines.append(f"  TBT: {cwv.tbt_ms:.0f}ms ({self.get_score_rating((cwv.tbt_score or 0) * 100)})")
        if cwv.cls_value is not None:
            lines.append(f"  CLS: {cwv.cls_value:.3f} ({self.get_score_rating((cwv.cls_score or 0) * 100)})")
        if cwv.fcp_ms:
            lines.append(f"  FCP: {cwv.fcp_ms:.0f}ms ({self.get_score_rating((cwv.fcp_score or 0) * 100)})")

        if result.opportunities:
            lines.extend(["", "Top Opportunities:"])
            for opp in result.opportunities[:5]:
                savings = opp.get("savings_ms", 0)
                if savings > 0:
                    lines.append(f"  - {opp['title']}: Save {savings:.0f}ms")

        return "\n".join(lines)


# Convenience function for quick audits
async def run_lighthouse_audit(
    url: str,
    device: str = "desktop",
) -> LighthouseResult:
    """
    Run a quick Lighthouse audit.

    Args:
        url: URL to audit
        device: "desktop" or "mobile"

    Returns:
        LighthouseResult
    """
    tool = LighthouseTool(device=device)
    return await tool.run_audit(url)
