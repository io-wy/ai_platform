"""Memory system for AgentFlow.

分层记忆系统设计（基于最新论文）
================================

基于以下论文的设计：
- MemGPT: Towards LLMs as Operating Systems (2023)
- Generative Agents: Interactive Simulacra of Human Behavior (2023)
- RecallM: An Architecture for Temporal Context Understanding (2023)
- Cognitive Architectures for Language Agents (CoALA, 2023)

记忆层级：
1. 感知缓冲 (SensoryBuffer) - 原始输入缓存
2. 工作记忆 (WorkingMemory) - 当前上下文，有限容量
3. 情景记忆 (EpisodicMemory) - 具体事件和经历
4. 语义记忆 (SemanticMemory) - 抽象知识和概念

核心功能：
- 分层存储和检索
- 自动重要性评估
- 记忆整合和反思
- 多策略混合检索
"""

# 旧版兼容接口
from agentflow.memory.base import BaseMemory, MemoryEntry as LegacyMemoryEntry
from agentflow.memory.short_term import ShortTermMemory
from agentflow.memory.long_term import LongTermMemory
from agentflow.memory.context import ContextManager
from agentflow.memory.database import DatabaseMemory, ConversationStore

# 新版分层记忆系统
from agentflow.memory.hierarchical import (
    MemoryEntry,
    MemoryType,
    RetrievalResult,
    SensoryBuffer,
    WorkingMemory,
    EpisodicMemory,
    SemanticMemory,
    HierarchicalMemory,
)

from agentflow.memory.consolidation import (
    MemoryConsolidator,
    MemoryScheduler,
    ConsolidationConfig,
)

from agentflow.memory.retrieval import (
    QueryType,
    RetrievalStrategy,
    RetrievalConfig,
    QueryAnalyzer,
    HybridRetriever,
    ContextAwareRetriever,
)

__all__ = [
    # 旧版兼容
    "BaseMemory",
    "LegacyMemoryEntry",
    "ShortTermMemory",
    "LongTermMemory",
    "ContextManager",
    "DatabaseMemory",
    "ConversationStore",
    
    # 新版分层记忆
    "MemoryEntry",
    "MemoryType",
    "RetrievalResult",
    "SensoryBuffer",
    "WorkingMemory", 
    "EpisodicMemory",
    "SemanticMemory",
    "HierarchicalMemory",
    
    # 记忆整合
    "MemoryConsolidator",
    "MemoryScheduler",
    "ConsolidationConfig",
    
    # 检索系统
    "QueryType",
    "RetrievalStrategy",
    "RetrievalConfig",
    "QueryAnalyzer",
    "HybridRetriever",
    "ContextAwareRetriever",
]
