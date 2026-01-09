"""Tool executor for running tools safely."""

import asyncio
from typing import Any, Optional

import structlog

from agentflow.core.config import ToolConfig
from agentflow.core.message import ToolCall, ToolResult as MessageToolResult
from agentflow.tools.base import BaseTool, ToolRegistry, ToolResult

logger = structlog.get_logger()


class ToolExecutor:
    """Executes tools safely with timeout, retry, and error handling.
    
    Example:
        ```python
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        
        executor = ToolExecutor(registry)
        
        # Execute a single tool
        result = await executor.execute("calculator", expression="2 + 2")
        
        # Execute tool calls from LLM
        tool_calls = [ToolCall(name="calculator", arguments={"expression": "2+2"})]
        results = await executor.execute_tool_calls(tool_calls)
        ```
    """
    
    def __init__(
        self,
        registry: ToolRegistry,
        config: Optional[ToolConfig] = None,
    ):
        self.registry = registry
        self.config = config or ToolConfig()
    
    async def execute(
        self,
        tool_name: str,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute a tool by name.
        
        Args:
            tool_name: Name of the tool to execute.
            timeout: Optional timeout in seconds.
            **kwargs: Tool parameters.
            
        Returns:
            ToolResult from the tool execution.
        """
        tool = self.registry.get(tool_name)
        if tool is None:
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool '{tool_name}' not found",
            )
        
        # Check if tool requires confirmation
        if tool.requires_confirmation:
            logger.warning(
                "Tool requires confirmation",
                tool=tool_name,
                is_dangerous=tool.is_dangerous,
            )
        
        timeout = timeout or self.config.tool_timeout
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                tool(**kwargs),
                timeout=timeout,
            )
            
            logger.info(
                "Tool executed",
                tool=tool_name,
                success=result.success,
            )
            
            return result
            
        except asyncio.TimeoutError:
            logger.error("Tool execution timed out", tool=tool_name, timeout=timeout)
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool execution timed out after {timeout}s",
            )
        except Exception as e:
            logger.error("Tool execution failed", tool=tool_name, error=str(e))
            return ToolResult(
                success=False,
                output=None,
                error=str(e),
            )
    
    async def execute_tool_calls(
        self,
        tool_calls: list[ToolCall],
        parallel: Optional[bool] = None,
    ) -> list[MessageToolResult]:
        """Execute multiple tool calls.
        
        Args:
            tool_calls: List of tool calls from the LLM.
            parallel: Whether to execute tools in parallel.
            
        Returns:
            List of ToolResult for message construction.
        """
        parallel = parallel if parallel is not None else self.config.parallel_tool_calls
        
        # Check max tool calls limit
        if len(tool_calls) > self.config.max_tool_calls_per_turn:
            logger.warning(
                "Too many tool calls",
                count=len(tool_calls),
                max=self.config.max_tool_calls_per_turn,
            )
            tool_calls = tool_calls[:self.config.max_tool_calls_per_turn]
        
        if parallel:
            # Execute in parallel
            tasks = [
                self.execute(tc.name, **tc.arguments)
                for tc in tool_calls
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Execute sequentially
            results = []
            for tc in tool_calls:
                result = await self.execute(tc.name, **tc.arguments)
                results.append(result)
        
        # Convert to message tool results
        message_results = []
        for tc, result in zip(tool_calls, results):
            if isinstance(result, Exception):
                result = ToolResult(
                    success=False,
                    output=None,
                    error=str(result),
                )
            
            message_results.append(MessageToolResult(
                tool_call_id=tc.id,
                name=tc.name,
                content=result.to_string(),
                is_error=not result.success,
                metadata=result.metadata,
            ))
        
        return message_results
    
    def get_available_tools(self) -> list[str]:
        """Get names of all available tools."""
        return self.registry.get_names()
    
    def get_tool_schemas(self, names: Optional[list[str]] = None) -> list[dict[str, Any]]:
        """Get OpenAI-compatible schemas for tools."""
        # Filter by enabled tools if configured
        if self.config.enabled_tools:
            names = names or self.config.enabled_tools
            names = [n for n in names if n in self.config.enabled_tools]
        
        return self.registry.get_schemas(names)
