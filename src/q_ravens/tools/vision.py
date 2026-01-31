"""
Q-Ravens Vision Service

Implements the "See" component of the See-Think-Act-Reflect loop.
Provides:
- Screenshot capture with high resolution
- Set-of-Mark (SoM) overlay for interactive element marking
- Grid-based coordinate system for precise targeting
- Visual state extraction for multimodal LLM understanding
"""

import base64
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont
from playwright.async_api import Page, async_playwright


@dataclass
class BoundingBox:
    """Bounding box for an element."""

    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> tuple[float, float]:
        """Get the center point of the bounding box."""
        return (self.x + self.width / 2, self.y + self.height / 2)

    @property
    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return as (x, y, width, height) tuple."""
        return (self.x, self.y, self.width, self.height)


@dataclass
class InteractiveElement:
    """Represents an interactive element on the page."""

    mark_id: int  # The SoM marker number
    tag_name: str
    element_type: Optional[str]  # e.g., 'button', 'text', 'link', 'input'
    text: str
    bounding_box: BoundingBox
    attributes: dict = field(default_factory=dict)
    is_visible: bool = True
    is_enabled: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "mark_id": self.mark_id,
            "tag_name": self.tag_name,
            "element_type": self.element_type,
            "text": self.text[:100] if self.text else "",
            "center": self.bounding_box.center,
            "bounds": self.bounding_box.as_tuple,
            "attributes": self.attributes,
            "is_visible": self.is_visible,
            "is_enabled": self.is_enabled,
        }


@dataclass
class VisualState:
    """
    Complete visual state of a page at a point in time.

    This is the output of the "See" phase.
    """

    url: str
    title: str
    viewport_width: int
    viewport_height: int
    screenshot_base64: str  # Original screenshot
    som_screenshot_base64: str  # Screenshot with SoM overlay
    elements: list[InteractiveElement]
    grid_overlay_base64: Optional[str] = None  # Optional grid overlay

    def get_element_by_mark(self, mark_id: int) -> Optional[InteractiveElement]:
        """Get an element by its mark ID."""
        for elem in self.elements:
            if elem.mark_id == mark_id:
                return elem
        return None

    def get_elements_by_type(self, element_type: str) -> list[InteractiveElement]:
        """Get all elements of a specific type."""
        return [e for e in self.elements if e.element_type == element_type]

    def to_prompt_context(self) -> str:
        """
        Generate a text description for LLM context.

        This provides structured information about the visual state
        that can be used alongside the screenshot for multimodal LLMs.
        """
        lines = [
            f"Page: {self.title}",
            f"URL: {self.url}",
            f"Viewport: {self.viewport_width}x{self.viewport_height}",
            "",
            "Interactive Elements (marked with numbers in the screenshot):",
        ]

        for elem in self.elements:
            center = elem.bounding_box.center
            lines.append(
                f"  [{elem.mark_id}] {elem.element_type or elem.tag_name}: "
                f'"{elem.text[:50]}" at ({int(center[0])}, {int(center[1])})'
            )

        return "\n".join(lines)


class VisionService:
    """
    Vision service for capturing and analyzing page visual state.

    Implements the Set-of-Mark (SoM) technique for visual grounding,
    placing numbered markers on interactive elements to enable
    coordinate-based interaction via multimodal LLM understanding.
    """

    # Colors for SoM markers (RGB)
    MARKER_COLORS = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 165, 0),  # Orange
        (128, 0, 128),  # Purple
        (0, 255, 255),  # Cyan
        (255, 192, 203),  # Pink
        (255, 255, 0),  # Yellow
    ]

    # Interactive element selectors
    INTERACTIVE_SELECTORS = [
        "button",
        "a[href]",
        "input:not([type='hidden'])",
        "textarea",
        "select",
        "[role='button']",
        "[role='link']",
        "[role='checkbox']",
        "[role='radio']",
        "[role='tab']",
        "[role='menuitem']",
        "[onclick]",
        "[tabindex]:not([tabindex='-1'])",
    ]

    def __init__(
        self,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        marker_size: int = 24,
        marker_font_size: int = 14,
    ):
        """
        Initialize the vision service.

        Args:
            viewport_width: Width of the browser viewport
            viewport_height: Height of the browser viewport
            marker_size: Size of the SoM marker circles
            marker_font_size: Font size for marker numbers
        """
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.marker_size = marker_size
        self.marker_font_size = marker_font_size

    async def capture_visual_state(
        self,
        page: Page,
        include_grid: bool = False,
        max_elements: int = 50,
    ) -> VisualState:
        """
        Capture the complete visual state of a page.

        Args:
            page: Playwright Page instance
            include_grid: Whether to include a coordinate grid overlay
            max_elements: Maximum number of elements to mark

        Returns:
            VisualState containing screenshots and element information
        """
        # Get page info
        url = page.url
        title = await page.title()

        # Capture original screenshot
        screenshot_bytes = await page.screenshot(type="png")
        screenshot_base64 = base64.b64encode(screenshot_bytes).decode("utf-8")

        # Find and analyze interactive elements
        elements = await self._find_interactive_elements(page, max_elements)

        # Create SoM overlay
        som_image = self._create_som_overlay(screenshot_bytes, elements)
        som_bytes = io.BytesIO()
        som_image.save(som_bytes, format="PNG")
        som_screenshot_base64 = base64.b64encode(som_bytes.getvalue()).decode("utf-8")

        # Optionally create grid overlay
        grid_overlay_base64 = None
        if include_grid:
            grid_image = self._create_grid_overlay(screenshot_bytes)
            grid_bytes = io.BytesIO()
            grid_image.save(grid_bytes, format="PNG")
            grid_overlay_base64 = base64.b64encode(grid_bytes.getvalue()).decode(
                "utf-8"
            )

        return VisualState(
            url=url,
            title=title,
            viewport_width=self.viewport_width,
            viewport_height=self.viewport_height,
            screenshot_base64=screenshot_base64,
            som_screenshot_base64=som_screenshot_base64,
            elements=elements,
            grid_overlay_base64=grid_overlay_base64,
        )

    async def _find_interactive_elements(
        self,
        page: Page,
        max_elements: int,
    ) -> list[InteractiveElement]:
        """
        Find all interactive elements on the page.

        Args:
            page: Playwright Page instance
            max_elements: Maximum number of elements to return

        Returns:
            List of InteractiveElement objects
        """
        elements = []
        mark_id = 1

        # Combined selector for all interactive elements
        combined_selector = ", ".join(self.INTERACTIVE_SELECTORS)

        # Execute JavaScript to get element information
        element_data = await page.evaluate(
            """
            (selector) => {
                const elements = document.querySelectorAll(selector);
                const results = [];

                for (const el of elements) {
                    const rect = el.getBoundingClientRect();

                    // Skip elements that are not visible
                    if (rect.width === 0 || rect.height === 0) continue;
                    if (rect.top > window.innerHeight || rect.bottom < 0) continue;
                    if (rect.left > window.innerWidth || rect.right < 0) continue;

                    // Get computed style to check visibility
                    const style = window.getComputedStyle(el);
                    if (style.display === 'none' || style.visibility === 'hidden') continue;
                    if (parseFloat(style.opacity) < 0.1) continue;

                    // Determine element type
                    let elementType = el.tagName.toLowerCase();
                    if (el.tagName === 'INPUT') {
                        elementType = el.type || 'text';
                    } else if (el.tagName === 'A') {
                        elementType = 'link';
                    } else if (el.tagName === 'BUTTON' || el.getAttribute('role') === 'button') {
                        elementType = 'button';
                    }

                    // Get text content
                    let text = el.textContent?.trim() || '';
                    if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {
                        text = el.placeholder || el.value || el.getAttribute('aria-label') || '';
                    }
                    if (el.tagName === 'IMG') {
                        text = el.alt || '';
                    }

                    results.push({
                        tagName: el.tagName.toLowerCase(),
                        elementType: elementType,
                        text: text.substring(0, 200),
                        x: rect.x,
                        y: rect.y,
                        width: rect.width,
                        height: rect.height,
                        id: el.id || null,
                        className: el.className || null,
                        href: el.href || null,
                        disabled: el.disabled || false,
                        ariaLabel: el.getAttribute('aria-label') || null,
                    });
                }

                return results;
            }
            """,
            combined_selector,
        )

        # Convert to InteractiveElement objects
        for data in element_data[:max_elements]:
            bbox = BoundingBox(
                x=data["x"],
                y=data["y"],
                width=data["width"],
                height=data["height"],
            )

            attributes = {}
            if data.get("id"):
                attributes["id"] = data["id"]
            if data.get("href"):
                attributes["href"] = data["href"]
            if data.get("ariaLabel"):
                attributes["aria-label"] = data["ariaLabel"]

            element = InteractiveElement(
                mark_id=mark_id,
                tag_name=data["tagName"],
                element_type=data["elementType"],
                text=data["text"],
                bounding_box=bbox,
                attributes=attributes,
                is_visible=True,
                is_enabled=not data.get("disabled", False),
            )

            elements.append(element)
            mark_id += 1

        return elements

    def _create_som_overlay(
        self,
        screenshot_bytes: bytes,
        elements: list[InteractiveElement],
    ) -> Image.Image:
        """
        Create a screenshot with Set-of-Mark overlay.

        Places numbered circles on each interactive element.

        Args:
            screenshot_bytes: Original screenshot as bytes
            elements: List of interactive elements to mark

        Returns:
            PIL Image with SoM overlay
        """
        # Load the screenshot
        image = Image.open(io.BytesIO(screenshot_bytes)).convert("RGBA")
        draw = ImageDraw.Draw(image)

        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", self.marker_font_size)
        except (IOError, OSError):
            try:
                # Try common Linux font paths
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                    self.marker_font_size,
                )
            except (IOError, OSError):
                font = ImageFont.load_default()

        for elem in elements:
            # Get marker color (cycle through colors)
            color = self.MARKER_COLORS[(elem.mark_id - 1) % len(self.MARKER_COLORS)]

            # Calculate marker position (top-left corner of element)
            center_x, center_y = elem.bounding_box.center
            marker_x = elem.bounding_box.x
            marker_y = elem.bounding_box.y

            # Draw a semi-transparent rectangle over the element
            overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle(
                [
                    elem.bounding_box.x,
                    elem.bounding_box.y,
                    elem.bounding_box.x + elem.bounding_box.width,
                    elem.bounding_box.y + elem.bounding_box.height,
                ],
                outline=color,
                width=2,
            )
            image = Image.alpha_composite(image, overlay)
            draw = ImageDraw.Draw(image)

            # Draw marker circle with number
            marker_radius = self.marker_size // 2
            circle_bbox = [
                marker_x - marker_radius,
                marker_y - marker_radius,
                marker_x + marker_radius,
                marker_y + marker_radius,
            ]

            # Draw filled circle
            draw.ellipse(circle_bbox, fill=color + (220,), outline=(255, 255, 255))

            # Draw the number
            text = str(elem.mark_id)
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_x = marker_x - text_width // 2
            text_y = marker_y - text_height // 2 - 2  # Slight adjustment for centering

            draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)

        return image

    def _create_grid_overlay(
        self,
        screenshot_bytes: bytes,
        grid_size: int = 100,
    ) -> Image.Image:
        """
        Create a screenshot with a coordinate grid overlay.

        Args:
            screenshot_bytes: Original screenshot as bytes
            grid_size: Size of grid cells in pixels

        Returns:
            PIL Image with grid overlay
        """
        image = Image.open(io.BytesIO(screenshot_bytes)).convert("RGBA")
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype("arial.ttf", 10)
        except (IOError, OSError):
            font = ImageFont.load_default()

        width, height = image.size

        # Draw vertical lines
        for x in range(0, width, grid_size):
            draw.line([(x, 0), (x, height)], fill=(128, 128, 128, 100), width=1)
            draw.text((x + 2, 2), str(x), fill=(128, 128, 128), font=font)

        # Draw horizontal lines
        for y in range(0, height, grid_size):
            draw.line([(0, y), (width, y)], fill=(128, 128, 128, 100), width=1)
            draw.text((2, y + 2), str(y), fill=(128, 128, 128), font=font)

        return image

    def save_visual_state(
        self,
        visual_state: VisualState,
        output_dir: str | Path,
        prefix: str = "visual_state",
    ) -> dict[str, str]:
        """
        Save visual state artifacts to disk.

        Args:
            visual_state: VisualState to save
            output_dir: Directory to save files
            prefix: Prefix for filenames

        Returns:
            Dictionary of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {}

        # Save original screenshot
        orig_path = output_dir / f"{prefix}_original.png"
        with open(orig_path, "wb") as f:
            f.write(base64.b64decode(visual_state.screenshot_base64))
        paths["original"] = str(orig_path)

        # Save SoM screenshot
        som_path = output_dir / f"{prefix}_som.png"
        with open(som_path, "wb") as f:
            f.write(base64.b64decode(visual_state.som_screenshot_base64))
        paths["som"] = str(som_path)

        # Save grid overlay if present
        if visual_state.grid_overlay_base64:
            grid_path = output_dir / f"{prefix}_grid.png"
            with open(grid_path, "wb") as f:
                f.write(base64.b64decode(visual_state.grid_overlay_base64))
            paths["grid"] = str(grid_path)

        # Save element data as JSON
        elements_path = output_dir / f"{prefix}_elements.json"
        with open(elements_path, "w") as f:
            json.dump(
                {
                    "url": visual_state.url,
                    "title": visual_state.title,
                    "viewport": {
                        "width": visual_state.viewport_width,
                        "height": visual_state.viewport_height,
                    },
                    "elements": [e.to_dict() for e in visual_state.elements],
                },
                f,
                indent=2,
            )
        paths["elements"] = str(elements_path)

        return paths


async def capture_page_state(url: str, output_dir: Optional[str] = None) -> VisualState:
    """
    Convenience function to capture visual state from a URL.

    Args:
        url: URL to capture
        output_dir: Optional directory to save artifacts

    Returns:
        VisualState of the page
    """
    vision = VisionService()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": vision.viewport_width, "height": vision.viewport_height}
        )
        page = await context.new_page()

        await page.goto(url, wait_until="networkidle")
        visual_state = await vision.capture_visual_state(page, include_grid=True)

        if output_dir:
            vision.save_visual_state(visual_state, output_dir)

        await browser.close()

    return visual_state


if __name__ == "__main__":
    import asyncio
    import sys

    async def main():
        url = sys.argv[1] if len(sys.argv) > 1 else "https://example.com"
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "./vision_output"

        print(f"Capturing visual state for: {url}")
        state = await capture_page_state(url, output_dir)

        print(f"\nPage: {state.title}")
        print(f"Found {len(state.elements)} interactive elements")
        print(f"\nSaved artifacts to: {output_dir}")
        print("\nElement summary:")
        print(state.to_prompt_context())

    asyncio.run(main())
