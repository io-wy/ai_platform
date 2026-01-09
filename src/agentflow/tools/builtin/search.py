"""Web search tools."""

from typing import Any, Optional

import aiohttp
from pydantic import BaseModel, Field

from agentflow.tools.base import BaseTool, ToolResult


class WebSearchParameters(BaseModel):
    """Parameters for web search."""
    
    query: str = Field(description="Search query")
    num_results: int = Field(default=5, description="Number of results to return", ge=1, le=20)
    search_type: str = Field(
        default="web",
        description="Type of search: 'web', 'news', 'images'"
    )


class WebSearchTool(BaseTool):
    """Web search tool supporting multiple search providers.
    
    Supports:
    - Google Custom Search API
    - SerpAPI
    - DuckDuckGo (free, no API key required)
    """
    
    name = "web_search"
    description = "Search the web for information. Returns relevant search results with titles, URLs, and snippets."
    parameters = WebSearchParameters
    category = "web"
    
    def __init__(
        self,
        provider: str = "duckduckgo",
        api_key: Optional[str] = None,
        google_cse_id: Optional[str] = None,
        **config: Any,
    ):
        super().__init__(**config)
        self.provider = provider
        self.api_key = api_key
        self.google_cse_id = google_cse_id
    
    async def execute(
        self,
        query: str,
        num_results: int = 5,
        search_type: str = "web",
        **kwargs: Any,
    ) -> ToolResult:
        """Execute web search."""
        try:
            if self.provider == "google" and self.api_key and self.google_cse_id:
                results = await self._search_google(query, num_results, search_type)
            elif self.provider == "serpapi" and self.api_key:
                results = await self._search_serpapi(query, num_results)
            else:
                results = await self._search_duckduckgo(query, num_results)
            
            return ToolResult(
                success=True,
                output=results,
                metadata={"provider": self.provider, "query": query},
            )
        
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
    
    async def _search_google(
        self,
        query: str,
        num_results: int,
        search_type: str,
    ) -> list[dict[str, Any]]:
        """Search using Google Custom Search API."""
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.api_key,
            "cx": self.google_cse_id,
            "q": query,
            "num": min(num_results, 10),  # Google API max is 10
        }
        
        if search_type == "images":
            params["searchType"] = "image"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
        
        results = []
        for item in data.get("items", []):
            results.append({
                "title": item.get("title"),
                "url": item.get("link"),
                "snippet": item.get("snippet"),
            })
        
        return results
    
    async def _search_serpapi(self, query: str, num_results: int) -> list[dict[str, Any]]:
        """Search using SerpAPI."""
        url = "https://serpapi.com/search"
        params = {
            "api_key": self.api_key,
            "q": query,
            "num": num_results,
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
        
        results = []
        for item in data.get("organic_results", []):
            results.append({
                "title": item.get("title"),
                "url": item.get("link"),
                "snippet": item.get("snippet"),
            })
        
        return results
    
    async def _search_duckduckgo(self, query: str, num_results: int) -> list[dict[str, Any]]:
        """Search using DuckDuckGo (no API key required)."""
        # Use DuckDuckGo's HTML interface
        url = "https://html.duckduckgo.com/html/"
        data = {"q": query}
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data, headers=headers) as response:
                html = await response.text()
        
        # Parse results
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        
        results = []
        for result in soup.select(".result")[:num_results]:
            title_elem = result.select_one(".result__title")
            link_elem = result.select_one(".result__url")
            snippet_elem = result.select_one(".result__snippet")
            
            if title_elem:
                # Extract actual URL from DuckDuckGo redirect
                href = title_elem.find("a")
                actual_url = ""
                if href and href.get("href"):
                    href_value = href.get("href", "")
                    if "uddg=" in href_value:
                        import urllib.parse
                        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href_value).query)
                        actual_url = parsed.get("uddg", [""])[0]
                    else:
                        actual_url = href_value
                
                results.append({
                    "title": title_elem.get_text(strip=True),
                    "url": actual_url or (link_elem.get_text(strip=True) if link_elem else ""),
                    "snippet": snippet_elem.get_text(strip=True) if snippet_elem else "",
                })
        
        return results
