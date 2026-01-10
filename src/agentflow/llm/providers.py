"""
轻量级 LLM 提供者
==================

统一的 LLM 接口，支持多种后端。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from agentflow.core.types import (
    LLMConfig,
    LLMProvider,
    LLMResponse,
    Message,
    ToolCall,
)


class OpenAIProvider:
    """OpenAI 兼容的 LLM 提供者.
    
    支持: OpenAI, Azure OpenAI, vLLM, Ollama (兼容模式), 本地服务
    
    Example:
        ```python
        # OpenAI
        llm = OpenAIProvider(model="gpt-4o-mini")
        
        # vLLM
        llm = OpenAIProvider(
            model="Qwen/Qwen2.5-7B-Instruct",
            api_base="http://localhost:8000/v1",
        )
        
        # Ollama
        llm = OpenAIProvider(
            model="llama3",
            api_base="http://localhost:11434/v1",
        )
        ```
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: float = 60.0,
        **kwargs: Any,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.api_base = api_base or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.extra = kwargs
        
        self._client = None
    
    @classmethod
    def from_config(cls, config: LLMConfig) -> OpenAIProvider:
        """从配置创建."""
        return cls(
            model=config.model,
            api_key=config.api_key,
            api_base=config.api_base,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
            **config.extra,
        )
    
    def _get_client(self):
        """懒加载客户端."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError("openai 包未安装。运行: uv pip install openai")
            
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                timeout=self.timeout,
            )
        return self._client
    
    async def complete(
        self,
        messages: list[Message],
        tools: Optional[list[dict]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """生成补全."""
        client = self._get_client()
        
        # 转换消息格式
        openai_messages = []
        for msg in messages:
            m = {"role": msg.role.value, "content": msg.content}
            if msg.name:
                m["name"] = msg.name
            if msg.tool_call_id:
                m["tool_call_id"] = msg.tool_call_id
            openai_messages.append(m)
        
        # 准备参数
        params = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        
        if tools:
            params["tools"] = tools
            params["tool_choice"] = kwargs.get("tool_choice", "auto")
        
        # 调用 API
        response = await client.chat.completions.create(**params)
        
        # 解析响应
        choice = response.choices[0]
        content = choice.message.content or ""
        
        # 解析工具调用
        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                ))
        
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            raw=response,
        )
    
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """生成嵌入."""
        client = self._get_client()
        
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        
        return [item.embedding for item in response.data]
    
    async def stream(
        self,
        messages: list[Message],
        **kwargs: Any,
    ):
        """流式生成."""
        client = self._get_client()
        
        openai_messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]
        
        stream = await client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stream=True,
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class MockProvider:
    """模拟 LLM 提供者 - 用于测试."""
    
    def __init__(self, responses: Optional[list[str]] = None):
        self.responses = responses or ["Mock response"]
        self._index = 0
    
    async def complete(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> LLMResponse:
        response = self.responses[self._index % len(self.responses)]
        self._index += 1
        return LLMResponse(content=response)
    
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 384 for _ in texts]
