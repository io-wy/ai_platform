"""Tests for tool system."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from agentflow.tools.base import BaseTool, tool, ToolResult, ToolRegistry
from agentflow.tools.executor import ToolExecutor
from agentflow.core.message import ToolCall
from pydantic import BaseModel, Field


class TestToolResult:
    """Tests for ToolResult."""
    
    def test_success_result(self):
        """Test successful result."""
        result = ToolResult(success=True, output="Result data")
        
        assert result.success is True
        assert result.output == "Result data"
        assert result.error is None
    
    def test_error_result(self):
        """Test error result."""
        result = ToolResult(success=False, output=None, error="Something failed")
        
        assert result.success is False
        assert result.error == "Something failed"
    
    def test_to_string(self):
        """Test converting result to string."""
        result = ToolResult(success=True, output={"key": "value"})
        string = result.to_string()
        
        assert "key" in string
        assert "value" in string
        
        error_result = ToolResult(success=False, output=None, error="Error!")
        assert "Error: Error!" == error_result.to_string()


class TestBaseTool:
    """Tests for BaseTool."""
    
    def test_tool_creation(self):
        """Test creating a tool subclass."""
        class TestParams(BaseModel):
            value: str = Field(description="Test value")
        
        class TestTool(BaseTool):
            name = "test_tool"
            description = "A test tool"
            parameters = TestParams
            
            async def execute(self, value: str) -> ToolResult:
                return ToolResult(success=True, output=f"Got: {value}")
        
        tool = TestTool()
        
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
    
    def test_get_schema(self):
        """Test getting tool schema."""
        class TestParams(BaseModel):
            query: str = Field(description="Search query")
            limit: int = Field(default=10, description="Max results")
        
        class SearchTool(BaseTool):
            name = "search"
            description = "Search for information"
            parameters = TestParams
            
            async def execute(self, query: str, limit: int = 10) -> ToolResult:
                return ToolResult(success=True, output="results")
        
        tool = SearchTool()
        schema = tool.get_schema()
        
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "search"
        assert schema["function"]["description"] == "Search for information"
        assert "query" in schema["function"]["parameters"]["properties"]
    
    @pytest.mark.asyncio
    async def test_tool_execution(self):
        """Test executing a tool."""
        class AddTool(BaseTool):
            name = "add"
            description = "Add two numbers"
            
            async def execute(self, a: int, b: int) -> ToolResult:
                return ToolResult(success=True, output=a + b)
        
        tool = AddTool()
        result = await tool(a=2, b=3)
        
        assert result.success is True
        assert result.output == 5


class TestToolDecorator:
    """Tests for @tool decorator."""
    
    def test_basic_decorator(self):
        """Test basic tool decorator."""
        @tool(name="greet", description="Greet someone")
        async def greet(name: str) -> str:
            return f"Hello, {name}!"
        
        # The decorator returns a class
        tool_instance = greet()
        
        assert tool_instance.name == "greet"
        assert tool_instance.description == "Greet someone"
    
    @pytest.mark.asyncio
    async def test_decorated_execution(self):
        """Test executing decorated tool."""
        @tool(name="multiply", description="Multiply numbers")
        async def multiply(a: int, b: int) -> int:
            return a * b
        
        tool_instance = multiply()
        result = await tool_instance(a=3, b=4)
        
        assert result.success is True
        assert result.output == 12
    
    def test_decorator_with_category(self):
        """Test decorator with category."""
        @tool(name="fetch", description="Fetch URL", category="web")
        async def fetch(url: str) -> str:
            return "content"
        
        tool_instance = fetch()
        
        assert tool_instance.category == "web"


class TestToolRegistry:
    """Tests for ToolRegistry."""
    
    def test_register_tool(self):
        """Test registering a tool."""
        registry = ToolRegistry()
        
        class DummyTool(BaseTool):
            name = "dummy"
            description = "A dummy tool"
            
            async def execute(self) -> ToolResult:
                return ToolResult(success=True, output="done")
        
        registry.register(DummyTool())
        
        assert "dummy" in registry
        assert len(registry) == 1
    
    def test_get_tool(self):
        """Test getting a tool by name."""
        registry = ToolRegistry()
        
        class MyTool(BaseTool):
            name = "my_tool"
            description = "My tool"
            
            async def execute(self) -> ToolResult:
                return ToolResult(success=True, output="done")
        
        tool = MyTool()
        registry.register(tool)
        
        retrieved = registry.get("my_tool")
        assert retrieved is tool
        
        assert registry.get("nonexistent") is None
    
    def test_get_by_category(self):
        """Test getting tools by category."""
        registry = ToolRegistry()
        
        class WebTool(BaseTool):
            name = "web"
            description = "Web tool"
            category = "web"
            
            async def execute(self) -> ToolResult:
                return ToolResult(success=True, output="done")
        
        class FileTool(BaseTool):
            name = "file"
            description = "File tool"
            category = "file"
            
            async def execute(self) -> ToolResult:
                return ToolResult(success=True, output="done")
        
        registry.register(WebTool())
        registry.register(FileTool())
        
        web_tools = registry.get_by_category("web")
        assert len(web_tools) == 1
        assert web_tools[0].name == "web"
    
    def test_get_schemas(self):
        """Test getting schemas for tools."""
        registry = ToolRegistry()
        
        class Tool1(BaseTool):
            name = "tool1"
            description = "Tool 1"
            
            async def execute(self) -> ToolResult:
                return ToolResult(success=True, output="done")
        
        class Tool2(BaseTool):
            name = "tool2"
            description = "Tool 2"
            
            async def execute(self) -> ToolResult:
                return ToolResult(success=True, output="done")
        
        registry.register(Tool1())
        registry.register(Tool2())
        
        schemas = registry.get_schemas()
        assert len(schemas) == 2
        
        # Filter by names
        schemas = registry.get_schemas(names=["tool1"])
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "tool1"
    
    def test_unregister_tool(self):
        """Test unregistering a tool."""
        registry = ToolRegistry()
        
        class TempTool(BaseTool):
            name = "temp"
            description = "Temporary"
            
            async def execute(self) -> ToolResult:
                return ToolResult(success=True, output="done")
        
        registry.register(TempTool())
        assert "temp" in registry
        
        registry.unregister("temp")
        assert "temp" not in registry


class TestToolExecutor:
    """Tests for ToolExecutor."""
    
    @pytest.fixture
    def registry_with_tools(self):
        """Create a registry with test tools."""
        registry = ToolRegistry()
        
        class EchoTool(BaseTool):
            name = "echo"
            description = "Echo input"
            
            async def execute(self, message: str) -> ToolResult:
                return ToolResult(success=True, output=message)
        
        class FailTool(BaseTool):
            name = "fail"
            description = "Always fails"
            
            async def execute(self) -> ToolResult:
                return ToolResult(success=False, output=None, error="Failed!")
        
        registry.register(EchoTool())
        registry.register(FailTool())
        
        return registry
    
    @pytest.mark.asyncio
    async def test_execute_tool(self, registry_with_tools):
        """Test executing a tool through executor."""
        executor = ToolExecutor(registry_with_tools)
        
        result = await executor.execute("echo", message="Hello!")
        
        assert result.success is True
        assert result.output == "Hello!"
    
    @pytest.mark.asyncio
    async def test_execute_nonexistent_tool(self, registry_with_tools):
        """Test executing a nonexistent tool."""
        executor = ToolExecutor(registry_with_tools)
        
        result = await executor.execute("nonexistent")
        
        assert result.success is False
        assert "not found" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_execute_tool_calls(self, registry_with_tools):
        """Test executing multiple tool calls."""
        executor = ToolExecutor(registry_with_tools)
        
        tool_calls = [
            ToolCall(id="call_1", name="echo", arguments={"message": "First"}),
            ToolCall(id="call_2", name="echo", arguments={"message": "Second"}),
        ]
        
        results = await executor.execute_tool_calls(tool_calls)
        
        assert len(results) == 2
        assert results[0].content == "First"
        assert results[1].content == "Second"
