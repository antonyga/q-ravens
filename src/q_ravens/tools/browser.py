"""
Q-Ravens Browser Tool

Playwright-based browser automation for website analysis and test execution.
"""

from typing import Optional
from contextlib import asynccontextmanager

from playwright.async_api import async_playwright, Browser, Page

from q_ravens.core.state import PageInfo
from q_ravens.core.config import settings


class BrowserTool:
    """
    Browser automation tool using Playwright.

    Provides methods for:
    - Page analysis and element discovery
    - Technology detection
    - Test execution
    - Screenshot capture
    """

    def __init__(self, headless: bool = True):
        self.headless = headless
        self._browser: Optional[Browser] = None

    @asynccontextmanager
    async def get_browser(self):
        """Context manager for browser instance."""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.headless)
            try:
                yield browser
            finally:
                await browser.close()

    @asynccontextmanager
    async def get_page(self, browser: Browser):
        """Context manager for page instance."""
        context = await browser.new_context(
            viewport={"width": 1280, "height": 720},
            user_agent="Q-Ravens/0.1.0 (Autonomous QA Agent)",
        )
        page = await context.new_page()
        try:
            yield page
        finally:
            await context.close()

    async def analyze_page(self, url: str) -> PageInfo:
        """
        Analyze a single page and extract information.

        Args:
            url: The URL to analyze

        Returns:
            PageInfo with discovered elements
        """
        async with self.get_browser() as browser:
            async with self.get_page(browser) as page:
                # Navigate to the page
                await page.goto(url, timeout=settings.default_timeout * 1000)
                await page.wait_for_load_state("networkidle")

                # Get page title
                title = await page.title()

                # Find all links
                links = await page.eval_on_selector_all(
                    "a[href]",
                    "elements => elements.map(e => e.href)"
                )

                # Find all forms
                forms = await page.eval_on_selector_all(
                    "form",
                    """elements => elements.map(form => ({
                        action: form.action,
                        method: form.method,
                        id: form.id,
                        inputs: Array.from(form.querySelectorAll('input')).map(i => ({
                            type: i.type,
                            name: i.name,
                            id: i.id
                        }))
                    }))"""
                )

                # Find all buttons
                buttons = await page.eval_on_selector_all(
                    "button, input[type='submit'], input[type='button']",
                    "elements => elements.map(e => e.textContent || e.value || 'unnamed')"
                )

                # Find all input elements
                inputs = await page.eval_on_selector_all(
                    "input, textarea, select",
                    """elements => elements.map(e => ({
                        type: e.type || e.tagName.toLowerCase(),
                        name: e.name,
                        id: e.id,
                        placeholder: e.placeholder
                    }))"""
                )

                return PageInfo(
                    url=url,
                    title=title,
                    links=links[:50],  # Limit to first 50 links
                    forms=forms,
                    buttons=buttons,
                    inputs=inputs,
                )

    async def detect_technologies(self, url: str) -> list[str]:
        """
        Detect technologies used by the website.

        Basic detection based on common patterns.
        """
        technologies = []

        async with self.get_browser() as browser:
            async with self.get_page(browser) as page:
                await page.goto(url, timeout=settings.default_timeout * 1000)

                # Get page content for analysis
                content = await page.content()
                content_lower = content.lower()

                # Common framework detection
                tech_patterns = {
                    "React": ["react", "_reactroot", "data-reactroot"],
                    "Vue.js": ["vue", "data-v-", "__vue__"],
                    "Angular": ["ng-", "angular", "ng-version"],
                    "Next.js": ["_next", "__next"],
                    "Nuxt.js": ["__nuxt", "_nuxt"],
                    "jQuery": ["jquery"],
                    "Bootstrap": ["bootstrap"],
                    "Tailwind CSS": ["tailwind"],
                    "WordPress": ["wp-content", "wordpress"],
                }

                for tech, patterns in tech_patterns.items():
                    if any(p in content_lower for p in patterns):
                        technologies.append(tech)

                # Check meta generator tag
                generator = await page.eval_on_selector_all(
                    "meta[name='generator']",
                    "elements => elements.map(e => e.content)"
                )
                technologies.extend(generator)

        return list(set(technologies))

    async def detect_authentication(self, url: str) -> tuple[bool, Optional[str]]:
        """
        Detect if the page has authentication mechanisms.

        Returns:
            Tuple of (has_auth, auth_type)
        """
        async with self.get_browser() as browser:
            async with self.get_page(browser) as page:
                await page.goto(url, timeout=settings.default_timeout * 1000)

                # Look for login forms
                login_indicators = [
                    "input[type='password']",
                    "form[action*='login']",
                    "form[action*='signin']",
                    "button:has-text('Log in')",
                    "button:has-text('Sign in')",
                    "a:has-text('Login')",
                    "a:has-text('Sign in')",
                ]

                for selector in login_indicators:
                    try:
                        element = await page.query_selector(selector)
                        if element:
                            return True, "form-based"
                    except Exception:
                        continue

                # Check for OAuth buttons
                oauth_indicators = [
                    "google",
                    "facebook",
                    "github",
                    "twitter",
                    "oauth",
                ]
                content = (await page.content()).lower()
                for provider in oauth_indicators:
                    if f"sign in with {provider}" in content or f"login with {provider}" in content:
                        return True, "oauth"

        return False, None

    async def take_screenshot(self, url: str, path: str) -> str:
        """
        Take a screenshot of a page.

        Args:
            url: The URL to screenshot
            path: Where to save the screenshot

        Returns:
            Path to the saved screenshot
        """
        async with self.get_browser() as browser:
            async with self.get_page(browser) as page:
                await page.goto(url, timeout=settings.default_timeout * 1000)
                await page.screenshot(path=path, full_page=True)
                return path

    async def run_test(
        self,
        url: str,
        test_steps: list[dict],
    ) -> dict:
        """
        Execute a test with given steps.

        Args:
            url: Starting URL
            test_steps: List of test steps to execute

        Returns:
            Test result dictionary
        """
        results = {
            "passed": True,
            "steps_completed": 0,
            "error": None,
        }

        async with self.get_browser() as browser:
            async with self.get_page(browser) as page:
                try:
                    await page.goto(url, timeout=settings.default_timeout * 1000)

                    for i, step in enumerate(test_steps):
                        action = step.get("action")
                        selector = step.get("selector")
                        value = step.get("value")

                        if action == "click":
                            await page.click(selector)
                        elif action == "fill":
                            await page.fill(selector, value)
                        elif action == "wait":
                            await page.wait_for_selector(selector)
                        elif action == "assert_visible":
                            await page.wait_for_selector(selector, state="visible")
                        elif action == "assert_text":
                            element = await page.query_selector(selector)
                            text = await element.text_content()
                            if value not in text:
                                raise AssertionError(f"Expected '{value}' in text")

                        results["steps_completed"] = i + 1

                except Exception as e:
                    results["passed"] = False
                    results["error"] = str(e)

        return results
