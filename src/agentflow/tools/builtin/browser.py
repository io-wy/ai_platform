"""Browser automation tool using Playwright."""

from typing import Any, Optional

from pydantic import BaseModel, Field

from agentflow.tools.base import BaseTool, ToolResult


class BrowserParameters(BaseModel):
    """Parameters for browser tool."""
    
    action: str = Field(
        description="Action to perform: 'navigate', 'click', 'type', 'screenshot', 'get_content', 'get_text', 'wait', 'close'"
    )
    url: Optional[str] = Field(default=None, description="URL to navigate to")
    selector: Optional[str] = Field(default=None, description="CSS selector for element")
    text: Optional[str] = Field(default=None, description="Text to type or search for")
    timeout: Optional[int] = Field(default=30000, description="Timeout in milliseconds")
    wait_until: Optional[str] = Field(
        default="domcontentloaded",
        description="Wait until: 'load', 'domcontentloaded', 'networkidle'"
    )


class BrowserTool(BaseTool):
    """Browser automation tool using Playwright.
    
    Supports:
    - Navigation
    - Clicking elements
    - Typing text
    - Taking screenshots
    - Extracting content
    - Waiting for elements
    """
    
    name = "browser"
    description = "Control a web browser to navigate, interact with elements, and extract content from web pages"
    parameters = BrowserParameters
    category = "web"
    
    def __init__(self, headless: bool = True, **config: Any):
        super().__init__(**config)
        self.headless = headless
        self._browser = None
        self._context = None
        self._page = None
    
    async def _ensure_browser(self):
        """Ensure browser is initialized."""
        if self._browser is None:
            try:
                from playwright.async_api import async_playwright
            except ImportError:
                raise ImportError(
                    "playwright is required for BrowserTool. "
                    "Install it with: pip install playwright && playwright install"
                )
            
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=self.headless)
            self._context = await self._browser.new_context()
            self._page = await self._context.new_page()
    
    async def execute(
        self,
        action: str,
        url: Optional[str] = None,
        selector: Optional[str] = None,
        text: Optional[str] = None,
        timeout: int = 30000,
        wait_until: str = "domcontentloaded",
        **kwargs: Any,
    ) -> ToolResult:
        """Execute browser action."""
        try:
            await self._ensure_browser()
            
            if action == "navigate":
                if not url:
                    return ToolResult(success=False, output=None, error="URL is required for navigate action")
                await self._page.goto(url, wait_until=wait_until, timeout=timeout)
                return ToolResult(
                    success=True,
                    output=f"Navigated to {url}",
                    metadata={"url": self._page.url, "title": await self._page.title()},
                )
            
            elif action == "click":
                if not selector:
                    return ToolResult(success=False, output=None, error="Selector is required for click action")
                await self._page.click(selector, timeout=timeout)
                return ToolResult(success=True, output=f"Clicked element: {selector}")
            
            elif action == "type":
                if not selector or text is None:
                    return ToolResult(success=False, output=None, error="Selector and text are required for type action")
                await self._page.fill(selector, text, timeout=timeout)
                return ToolResult(success=True, output=f"Typed text into {selector}")
            
            elif action == "screenshot":
                screenshot_bytes = await self._page.screenshot()
                import base64
                screenshot_b64 = base64.b64encode(screenshot_bytes).decode()
                return ToolResult(
                    success=True,
                    output=f"Screenshot taken (base64 encoded, {len(screenshot_bytes)} bytes)",
                    metadata={"screenshot_base64": screenshot_b64},
                )
            
            elif action == "get_content":
                content = await self._page.content()
                return ToolResult(success=True, output=content[:50000])  # Limit content size
            
            elif action == "get_text":
                if selector:
                    element = await self._page.query_selector(selector)
                    if element:
                        text_content = await element.text_content()
                        return ToolResult(success=True, output=text_content or "")
                    return ToolResult(success=False, output=None, error=f"Element not found: {selector}")
                else:
                    text_content = await self._page.inner_text("body")
                    return ToolResult(success=True, output=text_content[:50000])
            
            elif action == "wait":
                if selector:
                    await self._page.wait_for_selector(selector, timeout=timeout)
                    return ToolResult(success=True, output=f"Element appeared: {selector}")
                else:
                    await self._page.wait_for_timeout(timeout)
                    return ToolResult(success=True, output=f"Waited {timeout}ms")
            
            elif action == "close":
                await self._close_browser()
                return ToolResult(success=True, output="Browser closed")
            
            else:
                return ToolResult(success=False, output=None, error=f"Unknown action: {action}")
        
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
    
    async def _close_browser(self):
        """Close the browser."""
        if self._page:
            await self._page.close()
            self._page = None
        if self._context:
            await self._context.close()
            self._context = None
        if self._browser:
            await self._browser.close()
            self._browser = None
        if hasattr(self, '_playwright') and self._playwright:
            await self._playwright.stop()
            self._playwright = None
    
    def __del__(self):
        """Cleanup on deletion."""
        import asyncio
        if self._browser:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._close_browser())
                else:
                    loop.run_until_complete(self._close_browser())
            except Exception:
                pass
