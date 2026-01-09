"""Tests for Agent class."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agentflow.core.agent import Agent
from agentflow.core.config import AgentConfig, LLMConfig, MemoryConfig, ReasoningPattern
from agentflow.core.message import Message, MessageRole
from agentflow.tools.base import BaseTool, ToolResult
from agentflow.llm.providers import LLMResponse


class DummyTool(BaseTool):
    """A dummy tool for testing."""
    
    name = "dummy"
    description = "A dummy tool"
    
    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output="Dummy result")


class TestAgent:
    """Tests for Agent class."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM client."""
        llm = MagicMock()
        
        async def mock_chat(messages, tools=None, **kwargs):
            return LLMResponse(
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content="Agent response",
                ),
                finish_reason="stop",
                usage={"total_tokens": 100},
            )
        
        async def mock_embed(texts, **kwargs):
            return [[0.1] * 1536 for _ in texts]
        
        llm.chat = AsyncMock(side_effect=mock_chat)
        llm.embed = AsyncMock(side_effect=mock_embed)
        llm.close = AsyncMock()
        
        return llm
    
    @pytest.fixture
    def sample_config(self):
        """Create sample configuration without long-term memory."""
        return AgentConfig(
            name="TestAgent",
            llm=LLMConfig(model="gpt-4o-mini"),
            memory=MemoryConfig(enable_long_term=False),  # Disable to avoid chromadb
            pattern=ReasoningPattern.REACT,
            max_iterations=5,
        )
    
    def test_agent_creation(self, sample_config, mock_llm):
        """Test creating an agent."""
        agent = Agent(config=sample_config, llm=mock_llm)
        
        assert agent.config.name == "TestAgent"
        assert agent.pattern is not None
    
    def test_register_tool(self, sample_config, mock_llm):
        """Test registering a tool."""
        agent = Agent(config=sample_config, llm=mock_llm)
        
        tool = DummyTool()
        agent.register_tool(tool)
        
        assert "dummy" in agent.get_available_tools()
    
    def test_register_tools(self, sample_config, mock_llm):
        """Test registering multiple tools."""
        agent = Agent(config=sample_config, llm=mock_llm)
        
        class Tool1(BaseTool):
            name = "tool1"
            description = "Tool 1"
            async def execute(self): return ToolResult(success=True, output="1")
        
        class Tool2(BaseTool):
            name = "tool2"
            description = "Tool 2"
            async def execute(self): return ToolResult(success=True, output="2")
        
        agent.register_tools([Tool1(), Tool2()])
        
        tools = agent.get_available_tools()
        assert "tool1" in tools
        assert "tool2" in tools
    
    @pytest.mark.asyncio
    async def test_run_task(self, sample_config, mock_llm):
        """Test running a task."""
        agent = Agent(config=sample_config, llm=mock_llm)
        
        result = await agent.run("Test task")
        
        assert result.success is True
        assert result.output == "Agent response"
    
    @pytest.mark.asyncio
    async def test_chat_interface(self, sample_config, mock_llm):
        """Test chat interface."""
        agent = Agent(config=sample_config, llm=mock_llm)
        
        response = await agent.chat("Hello!")
        
        assert response == "Agent response"
    
    def test_set_pattern(self, sample_config, mock_llm):
        """Test changing pattern."""
        agent = Agent(config=sample_config, llm=mock_llm)
        
        assert agent.config.pattern == ReasoningPattern.REACT
        
        agent.set_pattern(ReasoningPattern.COT)
        
        assert agent.config.pattern == ReasoningPattern.COT
    
    def test_set_system_prompt(self, sample_config, mock_llm):
        """Test setting system prompt."""
        agent = Agent(config=sample_config, llm=mock_llm)
        
        agent.set_system_prompt("You are a helpful assistant.")
        
        assert agent.config.system_prompt == "You are a helpful assistant."
    
    @pytest.mark.asyncio
    async def test_clear_memory(self, sample_config, mock_llm):
        """Test clearing memory."""
        agent = Agent(config=sample_config, llm=mock_llm)
        
        # Add some messages
        await agent.run("First task")
        await agent.run("Second task")
        
        await agent.clear_memory()
        
        stats = await agent.get_memory_stats()
        # After clearing, short-term should be minimal
        assert stats["short_term_entries"] <= 1  # May keep system prompt
    
    @pytest.mark.asyncio
    async def test_context_manager(self, sample_config, mock_llm):
        """Test agent as context manager."""
        async with Agent(config=sample_config, llm=mock_llm) as agent:
            result = await agent.run("Test")
            assert result.success is True
        
        # LLM should be closed
        mock_llm.close.assert_called()
    
    def test_quick_start(self):
        """Test quick start class method."""
        with patch('agentflow.core.agent.LLMClient'):
            agent = Agent.quick_start(
                model="gpt-4o-mini",
                pattern=ReasoningPattern.COT,
            )
            
            assert agent.config.llm.model == "gpt-4o-mini"
            assert agent.config.pattern == ReasoningPattern.COT
    
    def test_get_tool_schemas(self, sample_config, mock_llm):
        """Test getting tool schemas."""
        agent = Agent(config=sample_config, llm=mock_llm)
        agent.register_tool(DummyTool())
        
        schemas = agent.get_tool_schemas()
        
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "dummy"
    
    def test_unregister_tool(self, sample_config, mock_llm):
        """Test unregistering a tool."""
        agent = Agent(config=sample_config, llm=mock_llm)
        agent.register_tool(DummyTool())
        
        assert "dummy" in agent.get_available_tools()
        
        agent.unregister_tool("dummy")
        
        assert "dummy" not in agent.get_available_tools()
