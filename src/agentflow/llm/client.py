"""Unified LLM client for all providers."""

from typing import Any, AsyncIterator, Optional

from agentflow.core.config import LLMConfig, LLMProvider, Settings, get_settings
from agentflow.core.message import Message
from agentflow.llm.providers import BaseLLMProvider, LLMResponse
from agentflow.llm.providers.openai_provider import (
    OpenAIProvider,
    VLLMProvider,
    OllamaProvider,
)


class LLMClient:
    """Unified LLM client that supports multiple providers.
    
    This client provides a consistent interface across different LLM providers
    including OpenAI, Azure OpenAI, vLLM, Ollama, Anthropic, and custom providers.
    
    Example:
        ```python
        from agentflow.llm import LLMClient
        from agentflow.core.config import LLMConfig, LLMProvider
        
        # Using OpenAI
        config = LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4o-mini")
        client = LLMClient(config)
        
        # Using vLLM with a fine-tuned model
        config = LLMConfig(
            provider=LLMProvider.VLLM,
            model="my-finetuned-model",
            api_base="http://localhost:8000/v1"
        )
        client = LLMClient(config)
        
        # Chat
        messages = [Message.user("Hello!")]
        response = await client.chat(messages)
        print(response.content)
        ```
    """
    
    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        settings: Optional[Settings] = None,
    ):
        """Initialize the LLM client.
        
        Args:
            config: LLM configuration. If not provided, uses default settings.
            settings: Global settings for API keys, etc.
        """
        self.config = config or LLMConfig()
        self.settings = settings or get_settings()
        self._provider: Optional[BaseLLMProvider] = None
        
        # Apply settings to config
        self._apply_settings()
    
    def _apply_settings(self) -> None:
        """Apply global settings to config if not already set."""
        if self.config.api_key is None:
            if self.config.provider == LLMProvider.OPENAI:
                self.config.api_key = self.settings.openai_api_key
                if self.config.api_base is None:
                    self.config.api_base = self.settings.openai_api_base
            elif self.config.provider == LLMProvider.AZURE_OPENAI:
                self.config.api_key = self.settings.azure_openai_api_key
                if self.config.api_base is None:
                    self.config.api_base = self.settings.azure_openai_endpoint
                if self.config.api_version is None:
                    self.config.api_version = self.settings.azure_openai_api_version
            elif self.config.provider == LLMProvider.VLLM:
                self.config.api_key = self.settings.vllm_api_key
                if self.config.api_base is None:
                    self.config.api_base = self.settings.vllm_api_base
            elif self.config.provider == LLMProvider.OLLAMA:
                if self.config.api_base is None:
                    self.config.api_base = self.settings.ollama_api_base
            elif self.config.provider == LLMProvider.ANTHROPIC:
                self.config.api_key = self.settings.anthropic_api_key
    
    @property
    def provider(self) -> BaseLLMProvider:
        """Get or create the provider instance."""
        if self._provider is None:
            self._provider = self._create_provider()
        return self._provider
    
    def _create_provider(self) -> BaseLLMProvider:
        """Create the appropriate provider based on config."""
        provider_type = self.config.provider
        
        if provider_type == LLMProvider.VLLM:
            return VLLMProvider(self.config)
        elif provider_type == LLMProvider.OLLAMA:
            return OllamaProvider(self.config)
        elif provider_type == LLMProvider.ANTHROPIC:
            from agentflow.llm.providers.anthropic_provider import AnthropicProvider
            return AnthropicProvider(self.config)
        elif provider_type in (LLMProvider.OPENAI, LLMProvider.AZURE_OPENAI, LLMProvider.CUSTOM):
            return OpenAIProvider(self.config)
        else:
            raise ValueError(f"Unknown provider: {provider_type}")
    
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
        return await self.provider.chat(messages, tools, **kwargs)
    
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
        async for chunk in self.provider.chat_stream(messages, tools, **kwargs):
            yield chunk
    
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
        return await self.provider.embed(texts, model)
    
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
        return await self.provider.count_tokens(text, model)
    
    async def close(self) -> None:
        """Close the client and release resources."""
        if self._provider is not None:
            await self._provider.close()
            self._provider = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    def with_config(self, **kwargs: Any) -> "LLMClient":
        """Create a new client with updated config.
        
        Args:
            **kwargs: Configuration parameters to update.
            
        Returns:
            New LLMClient instance with updated config.
        """
        new_config = self.config.model_copy(update=kwargs)
        return LLMClient(config=new_config, settings=self.settings)
