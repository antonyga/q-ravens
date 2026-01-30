"""
Q-Ravens Analyzer Agent

The QA Analyst responsible for:
- Crawling and mapping website structure
- Identifying interactive elements
- Detecting technology stack
- Discovering authentication flows
"""

from typing import Any
from datetime import datetime

from langchain_core.messages import AIMessage

from q_ravens.agents.base import BaseAgent
from q_ravens.core.state import (
    QRavensState,
    WorkflowPhase,
    AnalysisResult,
)
from q_ravens.tools.browser import BrowserTool


class AnalyzerAgent(BaseAgent):
    """
    The Analyzer crawls and analyzes web applications.

    It maps the structure of websites, identifies interactive elements,
    and provides insights for test case design.
    """

    name = "analyzer"
    role = "QA Analyst"
    description = "Analyzes websites to understand structure and capabilities"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.browser_tool = BrowserTool()

    @property
    def system_prompt(self) -> str:
        return """You are the Analyzer agent of Q-Ravens, a QA Analyst specialist.
Your role is to thoroughly analyze web applications to understand their structure.

Your responsibilities:
1. Crawl website pages starting from the target URL
2. Identify all interactive elements (forms, buttons, links, inputs)
3. Detect the technology stack being used
4. Discover authentication mechanisms
5. Map the site structure and navigation flows

When analyzing, be thorough but efficient. Focus on:
- Main navigation and page structure
- Forms and user input areas
- Authentication/login flows
- Key user journeys

Provide structured, actionable insights that will help design comprehensive tests."""

    async def process(self, state: QRavensState) -> dict[str, Any]:
        """
        Analyze the target website.

        Uses Playwright to crawl and extract information.
        """
        target_url = state.get("target_url")
        errors = state.get("errors", [])

        if not target_url:
            return {
                "phase": WorkflowPhase.ERROR.value,
                "errors": errors + ["No target URL provided for analysis"],
                "current_agent": self.name,
            }

        try:
            # Perform the analysis
            analysis_result = await self._analyze_website(target_url)

            message = AIMessage(
                content=f"Analysis of {target_url} complete.\n"
                f"Found {len(analysis_result.pages_discovered)} pages, "
                f"detected technologies: {', '.join(analysis_result.technology_stack) or 'None detected'}"
            )

            return {
                "analysis": analysis_result,
                "phase": WorkflowPhase.ANALYZING.value,  # Orchestrator will advance
                "next_agent": "orchestrator",
                "messages": [message],
                "current_agent": self.name,
            }

        except Exception as e:
            return {
                "phase": WorkflowPhase.ERROR.value,
                "errors": errors + [f"Analysis failed: {str(e)}"],
                "current_agent": self.name,
            }

    async def _analyze_website(self, url: str) -> AnalysisResult:
        """
        Perform website analysis using Playwright.

        Args:
            url: The URL to analyze

        Returns:
            AnalysisResult with discovered information
        """
        page_info = await self.browser_tool.analyze_page(url)

        # Detect technology stack from page content
        tech_stack = await self.browser_tool.detect_technologies(url)

        # Check for authentication
        has_auth, auth_type = await self.browser_tool.detect_authentication(url)

        return AnalysisResult(
            target_url=url,
            pages_discovered=[page_info],
            technology_stack=tech_stack,
            has_authentication=has_auth,
            auth_type=auth_type,
            timestamp=datetime.now(),
        )


# Create node function for LangGraph
async def analyzer_node(state: QRavensState) -> dict[str, Any]:
    """LangGraph node function for the Analyzer agent."""
    agent = AnalyzerAgent()
    return await agent.process(state)
