"""Test configuration for AgentFlow."""

import os
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message = MagicMock()
    response.choices[0].message.content = "Test response"
    response.choices[0].message.tool_calls = None
    response.choices[0].finish_reason = "stop"
    response.usage = MagicMock()
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 20
    response.usage.total_tokens = 30
    return response


@pytest.fixture
def mock_llm_client(mock_openai_response):
    """Create a mock LLM client."""
    from agentflow.llm.client import LLMClient
    from agentflow.core.message import Message, MessageRole
    from agentflow.llm.providers import LLMResponse
    
    client = MagicMock(spec=LLMClient)
    
    async def mock_chat(*args, **kwargs):
        return LLMResponse(
            message=Message(role=MessageRole.ASSISTANT, content="Test response"),
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )
    
    async def mock_embed(texts, **kwargs):
        return [[0.1] * 1536 for _ in texts]
    
    client.chat = AsyncMock(side_effect=mock_chat)
    client.embed = AsyncMock(side_effect=mock_embed)
    client.close = AsyncMock()
    
    return client


@pytest.fixture
def sample_config():
    """Create a sample agent configuration."""
    from agentflow.core.config import AgentConfig, LLMConfig, ReasoningPattern
    
    return AgentConfig(
        name="TestAgent",
        llm=LLMConfig(model="gpt-4o-mini"),
        pattern=ReasoningPattern.REACT,
        max_iterations=5,
    )
