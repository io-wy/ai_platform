"""Base LLM provider interface."""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Optional

from agentflow.core.message import Message


class LLMResponse:
    """Response from LLM."""
    
    def __init__(
        self,
        message: Message,
        finish_reason: Optional[str] = None,
        usage: Optional[dict[str, int]] = None,
        raw_response: Optional[Any] = None,
    ):
        self.message = message
        self.finish_reason = finish_reason
        self.usage = usage or {}
        self.raw_response = raw_response
    
    @property
    def content(self) -> Optional[str]:
        """Get the response content."""
        return self.message.content
    
    @property
    def tool_calls(self):
        """Get tool calls from the response."""
        return self.message.tool_calls
    
    @property
    def prompt_tokens(self) -> int:
        """Get prompt tokens used."""
        return self.usage.get("prompt_tokens", 0)
    
    @property
    def completion_tokens(self) -> int:
        """Get completion tokens used."""
        return self.usage.get("completion_tokens", 0)
    
    @property
    def total_tokens(self) -> int:
        """Get total tokens used."""
        return self.usage.get("total_tokens", 0)


class BaseLLMProvider(ABC):
    """Base class for LLM providers."""
    
    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request.
        
        Args:
            messages: List of messages in the conversation.
            tools: Optional list of tools in OpenAI format.
            **kwargs: Additional provider-specific parameters.
            
        Returns:
            LLMResponse containing the model's response.
        """
        pass
    
    @abstractmethod
    async def chat_stream(
        self,
        messages: list[Message],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a chat completion request.
        
        Args:
            messages: List of messages in the conversation.
            tools: Optional list of tools in OpenAI format.
            **kwargs: Additional provider-specific parameters.
            
        Yields:
            String chunks of the response.
        """
        pass
    
    @abstractmethod
    async def embed(
        self,
        texts: list[str],
        model: Optional[str] = None,
    ) -> list[list[float]]:
        """Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed.
            model: Optional embedding model to use.
            
        Returns:
            List of embedding vectors.
        """
        pass
    
    @abstractmethod
    async def count_tokens(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> int:
        """Count tokens in text.
        
        Args:
            text: Text to count tokens in.
            model: Optional model for tokenizer selection.
            
        Returns:
            Number of tokens.
        """
        pass
    
    async def close(self) -> None:
        """Close the provider and release resources."""
        pass
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
