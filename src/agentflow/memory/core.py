"""
极简记忆系统核心
================

基于 2024-2025 年最新论文设计的轻量级记忆系统：

参考论文:
- A Survey on Memory Mechanism of LLM-based Agents (2024, arXiv:2404.13501)
- MemGPT: LLMs as Operating Systems (2024)
- MemoryBank: Long-Term Memory for LLMs (2023)
- Generative Agents (2023) - 反思与重要性评分

设计原则:
1. 极简接口 - 仅暴露必要的 API
2. 协议优先 - 使用 Protocol 而非继承
3. 组合优于继承 - 功能通过组合实现
4. 懒加载 - 按需初始化存储后端
"""

from __future__ import annotations

import hashlib
import math
import json
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Callable,
    Generic,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)
from uuid import uuid4


# ============================================================================
# 类型定义
# ============================================================================

T = TypeVar("T")
Embedding = list[float]
EmbedFn = Callable[[list[str]], list[Embedding]]


class MemKind(str, Enum):
    """记忆种类 - 参考认知科学分类."""
    
    EPISODIC = "episodic"      # 情景记忆: 具体事件
    SEMANTIC = "semantic"      # 语义记忆: 通用知识
    PROCEDURAL = "procedural"  # 程序记忆: 如何做
    WORKING = "working"        # 工作记忆: 当前上下文


class MemRole(str, Enum):
    """记忆角色 - 对话中的来源."""
    
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
    OBSERVATION = "observation"
    THOUGHT = "thought"
    REFLECTION = "reflection"


# ============================================================================
# 核心数据结构
# ============================================================================

@dataclass(slots=True)
class MemoryItem:
    """单条记忆项 - 不可变数据结构.
    
    基于 Generative Agents 论文的记忆流设计，每条记忆包含:
    - 内容和元数据
    - 时间戳和访问统计
    - 重要性评分 (用于检索排序)
    """
    
    id: str = field(default_factory=lambda: uuid4().hex[:16])
    content: str = ""
    kind: MemKind = MemKind.EPISODIC
    role: MemRole = MemRole.OBSERVATION
    
    # 时间
    created_at: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    accessed_at: float = 0.0
    
    # 评分
    importance: float = 0.5  # [0, 1]
    access_count: int = 0
    
    # 向量 (可选)
    embedding: Optional[Embedding] = None
    
    # 关联
    parent_id: Optional[str] = None
    tags: tuple[str, ...] = ()
    
    # 扩展
    meta: dict[str, Any] = field(default_factory=dict)
    
    @property
    def hash(self) -> str:
        """内容哈希 (用于去重)."""
        return hashlib.md5(self.content.encode()).hexdigest()[:12]
    
    def recency(self, decay: float = 0.99, hours_cap: float = 168) -> float:
        """时间衰减分数 - Generative Agents 公式.
        
        score = decay ^ min(hours_since_creation, hours_cap)
        """
        hours = (datetime.now(timezone.utc).timestamp() - self.created_at) / 3600
        return math.pow(decay, min(hours, hours_cap))
    
    def score(
        self,
        relevance: float = 0.0,
        w_recency: float = 1.0,
        w_importance: float = 1.0,
        w_relevance: float = 1.0,
    ) -> float:
        """综合检索分数 - 三维评分函数.
        
        score = w_r * recency + w_i * importance + w_s * relevance
        """
        return (
            w_recency * self.recency()
            + w_importance * self.importance
            + w_relevance * relevance
        )


@dataclass
class SearchResult:
    """检索结果."""
    
    item: MemoryItem
    score: float
    match_type: str = "semantic"


# ============================================================================
# 协议定义 (Protocol-based Design)
# ============================================================================

@runtime_checkable
class MemoryStore(Protocol):
    """记忆存储协议 - 定义存储后端接口."""
    
    async def put(self, item: MemoryItem) -> str:
        """存储记忆项，返回 ID."""
        ...
    
    async def get(self, id: str) -> Optional[MemoryItem]:
        """按 ID 获取记忆."""
        ...
    
    async def delete(self, id: str) -> bool:
        """删除记忆."""
        ...
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        **filters: Any,
    ) -> list[SearchResult]:
        """搜索记忆."""
        ...
    
    async def recent(self, limit: int = 20) -> list[MemoryItem]:
        """获取最近记忆."""
        ...
    
    async def clear(self) -> int:
        """清空所有记忆，返回删除数量."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """嵌入器协议."""
    
    async def embed(self, texts: list[str]) -> list[Embedding]:
        """生成文本嵌入向量."""
        ...


@runtime_checkable
class ImportanceEvaluator(Protocol):
    """重要性评估器协议."""
    
    async def evaluate(self, content: str, context: Optional[str] = None) -> float:
        """评估内容重要性，返回 [0, 1]."""
        ...


# ============================================================================
# 内置实现
# ============================================================================

class InMemoryStore:
    """内存存储 - 用于测试和轻量级场景."""
    
    def __init__(self, max_size: int = 1000):
        self._store: dict[str, MemoryItem] = {}
        self._max_size = max_size
    
    async def put(self, item: MemoryItem) -> str:
        # LRU 淘汰
        if len(self._store) >= self._max_size:
            oldest = min(self._store.values(), key=lambda x: x.accessed_at or x.created_at)
            del self._store[oldest.id]
        
        self._store[item.id] = item
        return item.id
    
    async def get(self, id: str) -> Optional[MemoryItem]:
        item = self._store.get(id)
        if item:
            # 更新访问时间
            item.accessed_at = datetime.now(timezone.utc).timestamp()
            item.access_count += 1
        return item
    
    async def delete(self, id: str) -> bool:
        if id in self._store:
            del self._store[id]
            return True
        return False
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        **filters: Any,
    ) -> list[SearchResult]:
        """简单关键词搜索."""
        query_lower = query.lower()
        results = []
        
        for item in self._store.values():
            # 过滤
            if "kind" in filters and item.kind != filters["kind"]:
                continue
            if "role" in filters and item.role != filters["role"]:
                continue
            
            # 关键词匹配
            content_lower = item.content.lower()
            if query_lower in content_lower:
                # 计算简单相关性
                relevance = content_lower.count(query_lower) / max(1, len(content_lower.split()))
                score = item.score(relevance=min(1.0, relevance))
                results.append(SearchResult(item=item, score=score, match_type="keyword"))
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]
    
    async def recent(self, limit: int = 20) -> list[MemoryItem]:
        items = sorted(self._store.values(), key=lambda x: x.created_at, reverse=True)
        return items[:limit]
    
    async def clear(self) -> int:
        count = len(self._store)
        self._store.clear()
        return count
    
    def __len__(self) -> int:
        return len(self._store)


class HeuristicImportance:
    """启发式重要性评估 - 无需 LLM."""
    
    IMPORTANT_WORDS = frozenset([
        "重要", "关键", "必须", "紧急", "核心", "记住", "注意",
        "决定", "承诺", "约定", "目标", "计划", "问题", "解决",
        "important", "critical", "must", "key", "remember",
        "urgent", "goal", "plan", "problem", "solution",
    ])
    
    async def evaluate(self, content: str, context: Optional[str] = None) -> float:
        score = 0.5
        content_lower = content.lower()
        
        # 长度因子
        length = len(content)
        if length > 200:
            score += 0.1
        if length > 500:
            score += 0.1
        
        # 关键词因子
        for word in self.IMPORTANT_WORDS:
            if word in content_lower:
                score += 0.15
                break
        
        # 情感强度 (感叹号/问号)
        if "!" in content or "！" in content:
            score += 0.05
        if "?" in content or "？" in content:
            score += 0.05
        
        # 数字/代码因子 (可能是具体信息)
        if any(c.isdigit() for c in content):
            score += 0.05
        
        return min(1.0, score)


# ============================================================================
# 主记忆管理器
# ============================================================================

class Memory:
    """统一记忆管理器.
    
    极简 API:
    - remember(content, **meta) -> MemoryItem
    - recall(query, limit) -> list[SearchResult]
    - forget(id) -> bool
    - context(query) -> str  # 用于 LLM 提示
    
    Example:
        ```python
        mem = Memory()
        
        # 记忆
        item = await mem.remember("用户喜欢 Python", role=MemRole.OBSERVATION)
        
        # 回忆
        results = await mem.recall("Python", limit=5)
        
        # 获取上下文
        ctx = await mem.context("编程建议")
        ```
    """
    
    def __init__(
        self,
        store: Optional[MemoryStore] = None,
        embedder: Optional[Embedder] = None,
        importance: Optional[ImportanceEvaluator] = None,
        *,
        auto_embed: bool = True,
        dedup: bool = True,
    ):
        """初始化记忆管理器.
        
        Args:
            store: 存储后端，默认使用内存存储
            embedder: 嵌入器，用于语义搜索
            importance: 重要性评估器
            auto_embed: 是否自动生成嵌入
            dedup: 是否自动去重
        """
        self._store = store or InMemoryStore()
        self._embedder = embedder
        self._importance = importance or HeuristicImportance()
        self._auto_embed = auto_embed
        self._dedup = dedup
        self._seen_hashes: set[str] = set()
    
    async def remember(
        self,
        content: str,
        kind: MemKind = MemKind.EPISODIC,
        role: MemRole = MemRole.OBSERVATION,
        importance: Optional[float] = None,
        tags: tuple[str, ...] = (),
        parent_id: Optional[str] = None,
        meta: Optional[dict] = None,
    ) -> Optional[MemoryItem]:
        """记住一条内容.
        
        Returns:
            MemoryItem 或 None (如果是重复内容)
        """
        # 去重检查
        content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
        if self._dedup and content_hash in self._seen_hashes:
            return None
        self._seen_hashes.add(content_hash)
        
        # 评估重要性
        if importance is None:
            importance = await self._importance.evaluate(content)
        
        # 创建记忆项
        item = MemoryItem(
            content=content,
            kind=kind,
            role=role,
            importance=importance,
            tags=tags,
            parent_id=parent_id,
            meta=meta or {},
        )
        
        # 生成嵌入
        if self._auto_embed and self._embedder:
            try:
                embeddings = await self._embedder.embed([content])
                item.embedding = embeddings[0]
            except Exception:
                pass
        
        # 存储
        await self._store.put(item)
        return item
    
    async def recall(
        self,
        query: str,
        limit: int = 10,
        kind: Optional[MemKind] = None,
        role: Optional[MemRole] = None,
    ) -> list[SearchResult]:
        """回忆相关记忆."""
        filters = {}
        if kind:
            filters["kind"] = kind
        if role:
            filters["role"] = role
        
        return await self._store.search(query, limit=limit, **filters)
    
    async def forget(self, id: str) -> bool:
        """忘记一条记忆."""
        return await self._store.delete(id)
    
    async def recent(self, limit: int = 20) -> list[MemoryItem]:
        """获取最近的记忆."""
        return await self._store.recent(limit)
    
    async def context(
        self,
        query: Optional[str] = None,
        max_items: int = 10,
        max_chars: int = 4000,
    ) -> str:
        """生成用于 LLM 的上下文字符串.
        
        结合最近记忆和相关记忆。
        """
        items: list[MemoryItem] = []
        
        # 相关记忆
        if query:
            results = await self.recall(query, limit=max_items // 2)
            items.extend(r.item for r in results)
        
        # 最近记忆
        recent = await self.recent(limit=max_items // 2)
        for item in recent:
            if item not in items:
                items.append(item)
        
        # 格式化
        lines = []
        total_chars = 0
        
        for item in items[:max_items]:
            line = f"[{item.role.value}] {item.content}"
            if total_chars + len(line) > max_chars:
                break
            lines.append(line)
            total_chars += len(line)
        
        return "\n".join(lines)
    
    async def clear(self) -> int:
        """清空所有记忆."""
        self._seen_hashes.clear()
        return await self._store.clear()


# ============================================================================
# 工作记忆 (Working Memory) - 基于 MemGPT
# ============================================================================

@dataclass
class WorkingMemory:
    """工作记忆 - 管理当前上下文窗口.
    
    基于 MemGPT 论文:
    - core: 持久核心信息 (角色设定、用户画像)
    - buffer: 当前对话缓冲
    - scratch: 临时工作区
    """
    
    # 核心记忆 (持久)
    persona: str = ""
    user_profile: str = ""
    instructions: str = ""
    
    # 对话缓冲 (滑动窗口)
    buffer: list[MemoryItem] = field(default_factory=list)
    buffer_limit: int = 50
    
    # 临时工作区
    scratch: dict[str, Any] = field(default_factory=dict)
    
    def add(self, item: MemoryItem) -> None:
        """添加到缓冲区."""
        self.buffer.append(item)
        if len(self.buffer) > self.buffer_limit:
            self.buffer.pop(0)
    
    def set_core(self, key: str, value: str) -> None:
        """设置核心记忆."""
        if key == "persona":
            self.persona = value
        elif key == "user_profile":
            self.user_profile = value
        elif key == "instructions":
            self.instructions = value
    
    def to_prompt(self, include_buffer: int = 20) -> str:
        """转换为 LLM 提示."""
        parts = []
        
        if self.persona:
            parts.append(f"[System]\n{self.persona}")
        if self.user_profile:
            parts.append(f"[User Profile]\n{self.user_profile}")
        if self.instructions:
            parts.append(f"[Instructions]\n{self.instructions}")
        
        if self.buffer:
            recent = self.buffer[-include_buffer:]
            conv = "\n".join(f"[{m.role.value}] {m.content}" for m in recent)
            parts.append(f"[Conversation]\n{conv}")
        
        return "\n\n".join(parts)
    
    def clear_buffer(self) -> None:
        """清空对话缓冲."""
        self.buffer.clear()
        self.scratch.clear()
