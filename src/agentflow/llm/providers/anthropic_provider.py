"""Anthropic LLM provider."""

from typing import Any, AsyncIterator, Optional

from agentflow.core.config import LLMConfig, LLMProvider
from agentflow.core.message import Message, MessageRole, ToolCall
from agentflow.llm.providers import BaseLLMProvider, LLMResponse


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API provider."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None
    
    @property
    def client(self):
        """Get or create the Anthropic client."""
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package is required for Anthropic provider. "
                    "Install it with: pip install anthropic"
                )
            
            api_key = self.config.api_key.get_secret_value() if self.config.api_key else None
            self._client = anthropic.AsyncAnthropic(api_key=api_key)
        return self._client
    
    def _convert_messages(self, messages: list[Message]) -> tuple[Optional[str], list[dict]]:
        """Convert messages to Anthropic format.
        
        Returns:
            Tuple of (system_prompt, messages_list)
        """
        system_prompt = None
        anthropic_messages = []
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_prompt = msg.content
            elif msg.role == MessageRole.USER:
                anthropic_messages.append({
                    "role": "user",
                    "content": msg.content or "",
                })
            elif msg.role == MessageRole.ASSISTANT:
                content = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        content.append({
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        })
                anthropic_messages.append({
                    "role": "assistant",
                    "content": content if content else msg.content or "",
                })
            elif msg.role == MessageRole.TOOL:
                anthropic_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content or "",
                    }],
                })
        
        return system_prompt, anthropic_messages
    
    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI tool format to Anthropic format."""
        anthropic_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                })
        return anthropic_tools
    
    async def chat(
        self,
        messages: list[Message],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request."""
        system_prompt, anthropic_messages = self._convert_messages(messages)
        
        params: dict[str, Any] = {
            "model": kwargs.pop("model", self.config.model or "claude-3-5-sonnet-20241022"),
            "messages": anthropic_messages,
            "max_tokens": kwargs.pop("max_tokens", self.config.max_tokens or 4096),
            "temperature": kwargs.pop("temperature", self.config.temperature),
        }
        
        if system_prompt:
            params["system"] = system_prompt
        
        if tools:
            params["tools"] = self._convert_tools(tools)
        
        response = await self.client.messages.create(**params)
        
        # Parse response
        content = None
        tool_calls = []
        
        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                ))
        
        message = Message(
            role=MessageRole.ASSISTANT,
            content=content,
            tool_calls=tool_calls if tool_calls else None,
        )
        
        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }
        
        return LLMResponse(
            message=message,
            finish_reason=response.stop_reason,
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
        system_prompt, anthropic_messages = self._convert_messages(messages)
        
        params: dict[str, Any] = {
            "model": kwargs.pop("model", self.config.model or "claude-3-5-sonnet-20241022"),
            "messages": anthropic_messages,
            "max_tokens": kwargs.pop("max_tokens", self.config.max_tokens or 4096),
            "temperature": kwargs.pop("temperature", self.config.temperature),
        }
        
        if system_prompt:
            params["system"] = system_prompt
        
        async with self.client.messages.stream(**params) as stream:
            async for text in stream.text_stream:
                yield text
    
    async def embed(
        self,
        texts: list[str],
        model: Optional[str] = None,
    ) -> list[list[float]]:
        """Generate embeddings - not directly supported by Anthropic."""
        raise NotImplementedError(
            "Anthropic does not provide an embedding API. "
            "Use OpenAI or another provider for embeddings."
        )
    
    async def count_tokens(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> int:
        """Count tokens in text."""
        # Use Anthropic's token counting
        response = await self.client.count_tokens(text)
        return response
    
    async def close(self) -> None:
        """Close the client."""
        if self._client is not None:
            await self._client.close()
            self._client = None
