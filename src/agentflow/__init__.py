"""
AgentFlow - A flexible LLM agent platform supporting multiple inference patterns.

This platform provides:
- Multiple LLM backend support (OpenAI, vLLM, local models)
- Flexible reasoning patterns (ReAct, CoT, ToT, etc.)
- Comprehensive tool system
- Memory and context management with database support
- vLLM module for high-throughput form processing
- Easy extensibility
"""

from agentflow.core.agent import Agent
from agentflow.core.config import (
    AgentConfig, 
    Settings, 
    LLMConfig,
    LLMProvider,
    ReasoningPattern,
)
from agentflow.llm.client import LLMClient
from agentflow.tools.base import BaseTool, tool, ToolResult
from agentflow.memory.base import BaseMemory, MemoryEntry
from agentflow.patterns.base import BasePattern

__version__ = "0.1.0"
__all__ = [
    # Core
    "Agent",
    "AgentConfig",
    "Settings",
    "LLMConfig",
    "LLMProvider",
    "ReasoningPattern",
    # LLM
    "LLMClient",
    # Tools
    "BaseTool",
    "tool",
    "ToolResult",
    # Memory
    "BaseMemory",
    "MemoryEntry",
    # Patterns
    "BasePattern",
]
