"""
AgentFlow 极简核心
==================

设计原则:
1. Protocol-first: 使用协议定义接口
2. Composition: 组合优于继承  
3. Minimal API: 暴露最少必要接口
4. Lazy Loading: 按需加载
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, TypeVar, runtime_checkable
from enum import Enum


# ============================================================================
# 类型和枚举
# ============================================================================

T = TypeVar("T")


class Role(str, Enum):
    """消息角色."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Pattern(str, Enum):
    """推理模式."""
    REACT = "react"
    COT = "cot"
    TOT = "tot"
    REFLEXION = "reflexion"
    PLAN_EXECUTE = "plan_execute"
    SIMPLE = "simple"


# ============================================================================
# 核心数据结构
# ============================================================================

@dataclass(slots=True, frozen=True)
class Message:
    """消息 - 不可变."""
    role: Role
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    meta: dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def system(cls, content: str) -> Message:
        return cls(Role.SYSTEM, content)
    
    @classmethod
    def user(cls, content: str) -> Message:
        return cls(Role.USER, content)
    
    @classmethod
    def assistant(cls, content: str) -> Message:
        return cls(Role.ASSISTANT, content)
    
    @classmethod
    def tool(cls, content: str, tool_call_id: str, name: str) -> Message:
        return cls(Role.TOOL, content, name=name, tool_call_id=tool_call_id)


@dataclass
class ToolCall:
    """工具调用."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """LLM 响应."""
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None


@dataclass
class AgentResult:
    """Agent 执行结果."""
    output: str
    success: bool = True
    steps: list[dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    meta: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# 协议定义
# ============================================================================

@runtime_checkable
class LLMProvider(Protocol):
    """LLM 提供者协议."""
    
    async def complete(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> LLMResponse:
        """生成补全."""
        ...
    
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """生成嵌入."""
        ...


@runtime_checkable
class Tool(Protocol):
    """工具协议."""
    
    @property
    def name(self) -> str:
        """工具名称."""
        ...
    
    @property
    def description(self) -> str:
        """工具描述."""
        ...
    
    @property
    def parameters(self) -> dict[str, Any]:
        """参数 JSON Schema."""
        ...
    
    async def execute(self, **kwargs: Any) -> Any:
        """执行工具."""
        ...


@runtime_checkable
class MemoryProvider(Protocol):
    """记忆提供者协议."""
    
    async def remember(self, content: str, **meta: Any) -> Any:
        """记忆."""
        ...
    
    async def recall(self, query: str, limit: int = 10) -> list[Any]:
        """回忆."""
        ...
    
    async def context(self, query: Optional[str] = None) -> str:
        """获取上下文."""
        ...


@runtime_checkable
class PatternExecutor(Protocol):
    """推理模式执行器协议."""
    
    async def execute(
        self,
        task: str,
        llm: LLMProvider,
        tools: list[Tool],
        memory: Optional[MemoryProvider] = None,
        **kwargs: Any,
    ) -> AgentResult:
        """执行推理."""
        ...


# ============================================================================
# 配置
# ============================================================================

@dataclass
class LLMConfig:
    """LLM 配置."""
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: float = 60.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentConfig:
    """Agent 配置."""
    name: str = "Agent"
    llm: LLMConfig = field(default_factory=LLMConfig)
    pattern: Pattern = Pattern.SIMPLE
    system_prompt: str = ""
    max_iterations: int = 10
    verbose: bool = False


# ============================================================================
# 工具注册表
# ============================================================================

class ToolRegistry:
    """工具注册表."""
    
    def __init__(self):
        self._tools: dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """注册工具."""
        self._tools[tool.name] = tool
    
    def unregister(self, name: str) -> None:
        """注销工具."""
        self._tools.pop(name, None)
    
    def get(self, name: str) -> Optional[Tool]:
        """获取工具."""
        return self._tools.get(name)
    
    def list(self) -> list[Tool]:
        """列出所有工具."""
        return list(self._tools.values())
    
    def to_openai_tools(self) -> list[dict]:
        """转换为 OpenAI 工具格式."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]
    
    def __len__(self) -> int:
        return len(self._tools)
    
    def __iter__(self):
        return iter(self._tools.values())


# ============================================================================
# 工具装饰器
# ============================================================================

def tool(
    name: Optional[str] = None,
    description: str = "",
    parameters: Optional[dict] = None,
):
    """工具装饰器.
    
    Example:
        ```python
        @tool(description="Add two numbers")
        async def add(a: int, b: int) -> int:
            return a + b
        ```
    """
    def decorator(func: Callable) -> Tool:
        import inspect
        
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or ""
        
        # 自动生成参数 schema
        if parameters:
            tool_params = parameters
        else:
            sig = inspect.signature(func)
            hints = func.__annotations__
            
            properties = {}
            required = []
            
            for param_name, param in sig.parameters.items():
                if param_name in ("self", "cls"):
                    continue
                
                param_type = hints.get(param_name, str)
                json_type = _python_type_to_json(param_type)
                
                properties[param_name] = {"type": json_type}
                
                if param.default is inspect.Parameter.empty:
                    required.append(param_name)
            
            tool_params = {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        
        class FuncTool:
            @property
            def name(self) -> str:
                return tool_name
            
            @property
            def description(self) -> str:
                return tool_desc
            
            @property
            def parameters(self) -> dict:
                return tool_params
            
            async def execute(self, **kwargs) -> Any:
                import asyncio
                result = func(**kwargs)
                if asyncio.iscoroutine(result):
                    return await result
                return result
        
        return FuncTool()
    
    return decorator


def _python_type_to_json(py_type: type) -> str:
    """Python 类型转 JSON Schema 类型."""
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }
    return type_map.get(py_type, "string")


# ============================================================================
# 事件系统
# ============================================================================

class Event:
    """事件."""
    
    def __init__(self, name: str, data: Any = None):
        self.name = name
        self.data = data


class EventEmitter:
    """事件发射器."""
    
    def __init__(self):
        self._handlers: dict[str, list[Callable]] = {}
    
    def on(self, event: str, handler: Callable) -> None:
        """注册事件处理器."""
        if event not in self._handlers:
            self._handlers[event] = []
        self._handlers[event].append(handler)
    
    def off(self, event: str, handler: Callable) -> None:
        """移除事件处理器."""
        if event in self._handlers:
            self._handlers[event].remove(handler)
    
    async def emit(self, event: str, data: Any = None) -> None:
        """发射事件."""
        if event in self._handlers:
            import asyncio
            for handler in self._handlers[event]:
                result = handler(Event(event, data))
                if asyncio.iscoroutine(result):
                    await result
