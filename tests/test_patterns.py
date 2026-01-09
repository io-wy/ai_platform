"""Tests for reasoning patterns."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agentflow.patterns.base import PatternResult
from agentflow.patterns.react import ReActPattern
from agentflow.patterns.cot import ChainOfThoughtPattern
from agentflow.patterns.auto import AutoPattern
from agentflow.core.message import Message, MessageRole
from agentflow.llm.providers import LLMResponse


class TestPatternResult:
    """Tests for PatternResult."""
    
    def test_success_result(self):
        """Test successful pattern result."""
        result = PatternResult(
            success=True,
            output="Task completed",
            iterations=3,
            tool_calls_made=2,
        )
        
        assert result.success is True
        assert result.output == "Task completed"
        assert result.iterations == 3
        assert result.tool_calls_made == 2
    
    def test_error_result(self):
        """Test error pattern result."""
        result = PatternResult(
            success=False,
            output="",
            error="Something went wrong",
        )
        
        assert result.success is False
        assert result.error == "Something went wrong"


class TestReActPattern:
    """Tests for ReActPattern."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM client."""
        llm = MagicMock()
        
        async def mock_chat(messages, tools=None, **kwargs):
            return LLMResponse(
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content="This is my response.",
                ),
                finish_reason="stop",
                usage={"total_tokens": 100},
            )
        
        llm.chat = AsyncMock(side_effect=mock_chat)
        return llm
    
    @pytest.mark.asyncio
    async def test_simple_response(self, mock_llm):
        """Test ReAct with simple response (no tool calls)."""
        pattern = ReActPattern(llm=mock_llm, max_iterations=5)
        
        result = await pattern.run("What is 2+2?")
        
        assert result.success is True
        assert result.output == "This is my response."
        assert result.iterations == 1
        assert result.tool_calls_made == 0
    
    @pytest.mark.asyncio
    async def test_max_iterations(self, mock_llm):
        """Test max iterations limit."""
        # Mock to always return tool calls
        async def mock_with_tools(messages, tools=None, **kwargs):
            return LLMResponse(
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content=None,
                    tool_calls=[],  # Empty tool calls still trigger loop
                ),
                finish_reason="stop",
            )
        
        mock_llm.chat = AsyncMock(side_effect=mock_with_tools)
        
        pattern = ReActPattern(llm=mock_llm, max_iterations=3)
        
        result = await pattern.run("Keep trying")
        
        # Should stop at max iterations
        assert result.iterations <= 3
    
    def test_system_prompt(self):
        """Test ReAct system prompt."""
        pattern = ReActPattern(llm=MagicMock())
        prompt = pattern.get_system_prompt()
        
        assert "ReAct" in prompt
        assert "Thought" in prompt
        assert "Action" in prompt


class TestChainOfThoughtPattern:
    """Tests for ChainOfThoughtPattern."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM client."""
        llm = MagicMock()
        
        async def mock_chat(messages, **kwargs):
            return LLMResponse(
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content="""**Step 1:** Understand the problem
**Step 2:** Break it down
**Step 3:** Solve each part
**Final Answer:** 42""",
                ),
                finish_reason="stop",
                usage={"total_tokens": 50},
            )
        
        llm.chat = AsyncMock(side_effect=mock_chat)
        return llm
    
    @pytest.mark.asyncio
    async def test_cot_response(self, mock_llm):
        """Test CoT pattern response."""
        pattern = ChainOfThoughtPattern(llm=mock_llm)
        
        result = await pattern.run("Solve this complex problem")
        
        assert result.success is True
        assert "Step 1" in result.output
        assert "Final Answer" in result.output
    
    def test_system_prompt(self):
        """Test CoT system prompt."""
        pattern = ChainOfThoughtPattern(llm=MagicMock())
        prompt = pattern.get_system_prompt()
        
        assert "step" in prompt.lower()
        assert "reasoning" in prompt.lower()
    
    @pytest.mark.asyncio
    async def test_zero_shot_prompt(self, mock_llm):
        """Test zero-shot CoT adds the magic phrase."""
        pattern = ChainOfThoughtPattern(llm=mock_llm, use_zero_shot=True)
        
        await pattern.run("Calculate something")
        
        # Check that the prompt includes the zero-shot phrase
        call_args = mock_llm.chat.call_args
        messages = call_args[1].get("messages") or call_args[0][0]
        user_content = next(
            (m.content for m in messages if m.role == MessageRole.USER),
            ""
        )
        assert "step by step" in user_content.lower()


class TestAutoPattern:
    """Tests for AutoPattern."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM client."""
        llm = MagicMock()
        call_count = [0]
        
        async def mock_chat(messages, **kwargs):
            call_count[0] += 1
            # First call is pattern selection
            if call_count[0] == 1:
                return LLMResponse(
                    message=Message(
                        role=MessageRole.ASSISTANT,
                        content="cot",  # Select CoT pattern
                    ),
                    finish_reason="stop",
                )
            # Subsequent calls are pattern execution
            return LLMResponse(
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content="Executed with selected pattern",
                ),
                finish_reason="stop",
            )
        
        llm.chat = AsyncMock(side_effect=mock_chat)
        return llm
    
    @pytest.mark.asyncio
    async def test_pattern_selection(self, mock_llm):
        """Test auto pattern selects appropriate pattern."""
        pattern = AutoPattern(llm=mock_llm)
        
        selected = await pattern.select_pattern("Calculate 15 * 23")
        
        # Should call LLM for selection
        assert mock_llm.chat.called
    
    @pytest.mark.asyncio
    async def test_force_pattern(self, mock_llm):
        """Test forcing a specific pattern."""
        pattern = AutoPattern(llm=mock_llm)
        
        result = await pattern.run(
            "Do something",
            force_pattern="react",
        )
        
        assert result.metadata.get("selected_pattern") == "react"
    
    def test_available_patterns(self):
        """Test getting available patterns."""
        pattern = AutoPattern(llm=MagicMock())
        
        patterns = pattern.get_available_patterns()
        
        assert "react" in patterns
        assert "cot" in patterns
        assert "tot" in patterns
        assert "reflexion" in patterns
        assert "plan_execute" in patterns
