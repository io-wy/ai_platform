"""OpenAI-compatible LLM provider."""

import json
from typing import Any, AsyncIterator, Optional

import httpx
import tiktoken
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from agentflow.core.config import LLMConfig, LLMProvider
from agentflow.core.message import Message, MessageRole, ToolCall
from agentflow.llm.providers import BaseLLMProvider, LLMResponse


class OpenAIProvider(BaseLLMProvider):
    """OpenAI and OpenAI-compatible API provider.
    
    Supports:
    - OpenAI API
    - Azure OpenAI
    - vLLM (OpenAI-compatible mode)
    - Ollama (OpenAI-compatible mode)
    - Any OpenAI-compatible API
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._client: Optional[AsyncOpenAI] = None
        self._tokenizer_cache: dict[str, tiktoken.Encoding] = {}
    
    @property
    def client(self) -> AsyncOpenAI:
        """Get or create the OpenAI client."""
        if self._client is None:
            self._client = self._create_client()
        return self._client
    
    def _create_client(self) -> AsyncOpenAI:
        """Create the appropriate OpenAI client based on provider."""
        api_key = self.config.api_key.get_secret_value() if self.config.api_key else "dummy"
        
        if self.config.provider == LLMProvider.AZURE_OPENAI:
            from openai import AsyncAzureOpenAI
            return AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=self.config.api_base or "",
                azure_deployment=self.config.azure_deployment,
                api_version=self.config.api_version or "2024-02-01",
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
        
        # For OpenAI, vLLM, Ollama, and custom providers
        base_url = self.config.api_base
        if self.config.provider == LLMProvider.VLLM:
            base_url = base_url or "http://localhost:8000/v1"
        elif self.config.provider == LLMProvider.OLLAMA:
            base_url = base_url or "http://localhost:11434/v1"
        
        return AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
        )
    
    def _get_generation_params(self, **kwargs: Any) -> dict[str, Any]:
        """Get generation parameters from config and kwargs."""
        params: dict[str, Any] = {
            "model": kwargs.pop("model", self.config.model),
            "temperature": kwargs.pop("temperature", self.config.temperature),
            "top_p": kwargs.pop("top_p", self.config.top_p),
            "frequency_penalty": kwargs.pop("frequency_penalty", self.config.frequency_penalty),
            "presence_penalty": kwargs.pop("presence_penalty", self.config.presence_penalty),
        }
        
        if self.config.max_tokens is not None:
            params["max_tokens"] = kwargs.pop("max_tokens", self.config.max_tokens)
        
        if self.config.stop:
            params["stop"] = kwargs.pop("stop", self.config.stop)
        
        # Add any extra params
        params.update(self.config.extra_params)
        params.update(kwargs)
        
        return params
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def chat(
        self,
        messages: list[Message],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request."""
        params = self._get_generation_params(**kwargs)
        
        # Convert messages to OpenAI format
        openai_messages = [msg.to_openai_format() for msg in messages]
        
        # Add tools if provided
        if tools:
            params["tools"] = tools
            params["tool_choice"] = kwargs.get("tool_choice", "auto")
        
        response = await self.client.chat.completions.create(
            messages=openai_messages,
            **params,
        )
        
        # Parse response
        choice = response.choices[0]
        assistant_message = choice.message
        
        # Parse tool calls
        tool_calls = None
        if assistant_message.tool_calls:
            tool_calls = []
            for tc in assistant_message.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))
        
        message = Message(
            role=MessageRole.ASSISTANT,
            content=assistant_message.content,
            tool_calls=tool_calls,
        )
        
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        
        return LLMResponse(
            message=message,
            finish_reason=choice.finish_reason,
            usage=usage,
            raw_response=response,
        )
    
    async def chat_stream(
        self,
        messages: list[Message],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a chat completion request."""
        params = self._get_generation_params(**kwargs)
        params["stream"] = True
        
        # Convert messages to OpenAI format
        openai_messages = [msg.to_openai_format() for msg in messages]
        
        # Note: streaming with tools is limited
        if tools:
            params["tools"] = tools
        
        response = await self.client.chat.completions.create(
            messages=openai_messages,
            **params,
        )
        
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    async def embed(
        self,
        texts: list[str],
        model: Optional[str] = None,
    ) -> list[list[float]]:
        """Generate embeddings for texts."""
        model = model or "text-embedding-3-small"
        
        response = await self.client.embeddings.create(
            input=texts,
            model=model,
        )
        
        return [item.embedding for item in response.data]
    
    async def count_tokens(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> int:
        """Count tokens in text."""
        model = model or self.config.model
        
        # Get or create tokenizer
        if model not in self._tokenizer_cache:
            try:
                self._tokenizer_cache[model] = tiktoken.encoding_for_model(model)
            except KeyError:
                # Fallback to cl100k_base for unknown models
                self._tokenizer_cache[model] = tiktoken.get_encoding("cl100k_base")
        
        return len(self._tokenizer_cache[model].encode(text))
    
    async def close(self) -> None:
        """Close the client."""
        if self._client is not None:
            await self._client.close()
            self._client = None


class VLLMProvider(OpenAIProvider):
    """vLLM-specific provider with additional features."""
    
    def __init__(self, config: LLMConfig):
        # Ensure vLLM settings
        config.provider = LLMProvider.VLLM
        if not config.api_base:
            config.api_base = "http://localhost:8000/v1"
        super().__init__(config)
    
    async def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.config.api_base}/models")
            return response.json()


class OllamaProvider(OpenAIProvider):
    """Ollama-specific provider."""
    
    def __init__(self, config: LLMConfig):
        # Ensure Ollama settings
        config.provider = LLMProvider.OLLAMA
        if not config.api_base:
            config.api_base = "http://localhost:11434/v1"
        super().__init__(config)
    
    async def pull_model(self, model: str) -> None:
        """Pull a model from Ollama."""
        base = self.config.api_base or "http://localhost:11434"
        base = base.replace("/v1", "")  # Use native Ollama API
        
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{base}/api/pull",
                json={"name": model},
                timeout=None,  # Pulling can take a long time
            )
    
    async def list_models(self) -> list[dict[str, Any]]:
        """List available models."""
        base = self.config.api_base or "http://localhost:11434"
        base = base.replace("/v1", "")
        
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base}/api/tags")
            return response.json().get("models", [])
