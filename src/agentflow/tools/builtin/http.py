"""HTTP request tool."""

from typing import Any, Optional

import aiohttp
from pydantic import BaseModel, Field

from agentflow.tools.base import BaseTool, ToolResult


class HTTPParameters(BaseModel):
    """Parameters for HTTP requests."""
    
    method: str = Field(
        default="GET",
        description="HTTP method: GET, POST, PUT, DELETE, PATCH"
    )
    url: str = Field(description="URL to request")
    headers: Optional[dict[str, str]] = Field(default=None, description="HTTP headers")
    params: Optional[dict[str, str]] = Field(default=None, description="Query parameters")
    json_body: Optional[dict[str, Any]] = Field(default=None, description="JSON body for POST/PUT")
    data: Optional[str] = Field(default=None, description="Raw body data")
    timeout: int = Field(default=30, description="Request timeout in seconds")


class HTTPTool(BaseTool):
    """HTTP request tool for API calls and web fetching."""
    
    name = "http_request"
    description = "Make HTTP requests to APIs and web services. Supports GET, POST, PUT, DELETE, and PATCH methods."
    parameters = HTTPParameters
    category = "web"
    
    def __init__(
        self,
        default_headers: Optional[dict[str, str]] = None,
        max_response_size: int = 100000,
        **config: Any,
    ):
        super().__init__(**config)
        self.default_headers = default_headers or {
            "User-Agent": "AgentFlow/1.0"
        }
        self.max_response_size = max_response_size
    
    async def execute(
        self,
        method: str = "GET",
        url: str = "",
        headers: Optional[dict[str, str]] = None,
        params: Optional[dict[str, str]] = None,
        json_body: Optional[dict[str, Any]] = None,
        data: Optional[str] = None,
        timeout: int = 30,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute HTTP request."""
        if not url:
            return ToolResult(success=False, output=None, error="URL is required")
        
        # Merge headers
        request_headers = {**self.default_headers}
        if headers:
            request_headers.update(headers)
        
        try:
            async with aiohttp.ClientSession() as session:
                request_kwargs: dict[str, Any] = {
                    "headers": request_headers,
                    "timeout": aiohttp.ClientTimeout(total=timeout),
                }
                
                if params:
                    request_kwargs["params"] = params
                
                if json_body:
                    request_kwargs["json"] = json_body
                elif data:
                    request_kwargs["data"] = data
                
                async with session.request(method.upper(), url, **request_kwargs) as response:
                    # Get response content
                    content_type = response.headers.get("Content-Type", "")
                    
                    if "application/json" in content_type:
                        body = await response.json()
                    else:
                        body = await response.text()
                        # Truncate if too large
                        if len(body) > self.max_response_size:
                            body = body[:self.max_response_size] + "\n... (response truncated)"
                    
                    return ToolResult(
                        success=response.status < 400,
                        output=body,
                        error=None if response.status < 400 else f"HTTP {response.status}",
                        metadata={
                            "status_code": response.status,
                            "headers": dict(response.headers),
                            "url": str(response.url),
                        },
                    )
        
        except aiohttp.ClientError as e:
            return ToolResult(success=False, output=None, error=f"Request failed: {str(e)}")
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
