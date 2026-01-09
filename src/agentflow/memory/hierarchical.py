"""
分层记忆系统
============

基于以下论文的设计：
- MemGPT: Towards LLMs as Operating Systems (2023)
- Generative Agents: Interactive Simulacra of Human Behavior (2023)  
- RecallM: An Architecture for Temporal Context Understanding (2023)
- Cognitive Architectures for Language Agents (CoALA, 2023)

核心设计原则：
1. 分层存储：感知缓冲 -> 工作记忆 -> 情景记忆 -> 语义记忆
2. 记忆整合：通过反思和摘要将短期记忆转化为长期记忆
3. 重要性评估：基于访问频率、时间衰减、情感强度的动态评分
4. 检索增强：结合时间相关性、语义相似度、重要性的多维检索
"""

import asyncio
import json
import hashlib
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from typing import Any, Optional, Callable, TypeVar
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4
import math

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """记忆类型."""
    OBSERVATION = "observation"      # 观察（感知）
    THOUGHT = "thought"              # 思考
    REFLECTION = "reflection"        # 反思（整合后的记忆）
    PLAN = "plan"                    # 计划
    ACTION = "action"                # 行动
    RESULT = "result"                # 结果
    CONVERSATION = "conversation"    # 对话


class MemoryEntry(BaseModel):
    """统一的记忆条目.
    
    基于 Generative Agents 论文的记忆流设计，每个记忆都有：
    - 内容描述
    - 时间戳
    - 重要性分数（用于决定是否被检索和保留）
    - 关联记忆（用于构建记忆网络）
    """
    
    id: str = Field(default_factory=lambda: uuid4().hex)
    content: str
    memory_type: MemoryType = MemoryType.OBSERVATION
    
    # 时间信息
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: Optional[datetime] = None
    
    # 重要性和访问统计
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    access_count: int = 0
    
    # 关联信息
    related_ids: list[str] = Field(default_factory=list)
    source_id: Optional[str] = None  # 来源记忆ID（如反思是从哪些记忆生成的）
    
    # 嵌入向量（用于语义搜索）
    embedding: Optional[list[float]] = None
    
    # 元数据
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    # 内容哈希（用于去重）
    content_hash: str = ""
    
    def __init__(self, **data):
        super().__init__(**data)
        if not self.content_hash:
            self.content_hash = hashlib.md5(self.content.encode()).hexdigest()[:16]
    
    def recency_score(self, decay_factor: float = 0.995) -> float:
        """计算时间衰减分数.
        
        基于 Generative Agents 论文：越近的记忆分数越高
        使用指数衰减：score = decay^hours_since_creation
        """
        hours_ago = (datetime.now(timezone.utc) - self.created_at).total_seconds() / 3600
        return math.pow(decay_factor, hours_ago)
    
    def retrieval_score(
        self,
        semantic_score: float = 0.0,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
    ) -> float:
        """计算综合检索分数.
        
        基于 Generative Agents 论文的检索函数：
        score = α * recency + β * importance + γ * relevance
        
        Args:
            semantic_score: 语义相似度分数 (0-1)
            alpha: 时间衰减权重
            beta: 重要性权重  
            gamma: 相关性权重
        """
        recency = self.recency_score()
        return alpha * recency + beta * self.importance + gamma * semantic_score


@dataclass
class RetrievalResult:
    """检索结果."""
    entry: MemoryEntry
    score: float
    match_type: str = "semantic"  # semantic, keyword, temporal


class BaseMemoryStore(ABC):
    """记忆存储基类."""
    
    @abstractmethod
    async def add(self, entry: MemoryEntry) -> MemoryEntry:
        """添加记忆."""
        pass
    
    @abstractmethod
    async def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """获取记忆."""
        pass
    
    @abstractmethod
    async def search(
        self,
        query: str,
        limit: int = 10,
        **kwargs,
    ) -> list[RetrievalResult]:
        """搜索记忆."""
        pass
    
    @abstractmethod
    async def get_recent(self, limit: int = 10) -> list[MemoryEntry]:
        """获取最近的记忆."""
        pass
    
    @abstractmethod
    async def delete(self, entry_id: str) -> bool:
        """删除记忆."""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """清空所有记忆."""
        pass
    
    @abstractmethod
    async def count(self) -> int:
        """统计记忆数量."""
        pass


class SensoryBuffer:
    """感知缓冲区 (Sensory Buffer).
    
    基于 CoALA 论文的设计：
    - 存储最近的原始感知输入
    - 容量有限，采用 FIFO 策略
    - 作为工作记忆的输入来源
    """
    
    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self._buffer: list[MemoryEntry] = []
    
    def add(self, entry: MemoryEntry) -> None:
        """添加感知."""
        self._buffer.append(entry)
        if len(self._buffer) > self.capacity:
            self._buffer.pop(0)
    
    def get_all(self) -> list[MemoryEntry]:
        """获取所有感知."""
        return list(self._buffer)
    
    def clear(self) -> None:
        """清空缓冲区."""
        self._buffer.clear()
    
    def __len__(self) -> int:
        return len(self._buffer)


class WorkingMemory:
    """工作记忆 (Working Memory).
    
    基于 MemGPT 论文的设计：
    - 有限容量（对应 LLM 的上下文窗口）
    - 支持主动管理（压缩、淘汰）
    - 分为核心记忆（始终保持）和工作区（动态管理）
    
    结构：
    - core_memory: 核心记忆（系统提示、角色设定等）
    - working_context: 当前工作上下文
    - scratch_pad: 临时工作区
    """
    
    def __init__(
        self,
        max_tokens: int = 8000,
        core_ratio: float = 0.3,
    ):
        self.max_tokens = max_tokens
        self.core_ratio = core_ratio
        
        # 核心记忆（持久）
        self.core_memory: dict[str, str] = {
            "persona": "",      # 角色设定
            "user_info": "",    # 用户信息
            "instructions": "", # 特殊指令
        }
        
        # 工作上下文
        self.working_context: list[MemoryEntry] = []
        
        # 临时工作区
        self.scratch_pad: dict[str, Any] = {}
    
    def set_core(self, key: str, value: str) -> None:
        """设置核心记忆."""
        if key in self.core_memory:
            self.core_memory[key] = value
    
    def get_core(self, key: str) -> str:
        """获取核心记忆."""
        return self.core_memory.get(key, "")
    
    def add_to_context(self, entry: MemoryEntry) -> None:
        """添加到工作上下文."""
        self.working_context.append(entry)
        self._enforce_limit()
    
    def _enforce_limit(self) -> None:
        """强制执行容量限制."""
        # 简单策略：保留最近的条目
        # 实际使用时应该基于 token 计数
        max_entries = 50  # 近似值
        if len(self.working_context) > max_entries:
            # 保留最近的 80%
            keep_count = int(max_entries * 0.8)
            self.working_context = self.working_context[-keep_count:]
    
    def get_context(self) -> list[MemoryEntry]:
        """获取当前上下文."""
        return list(self.working_context)
    
    def to_prompt(self) -> str:
        """转换为提示文本."""
        parts = []
        
        # 核心记忆
        if self.core_memory["persona"]:
            parts.append(f"[角色设定]\n{self.core_memory['persona']}")
        if self.core_memory["user_info"]:
            parts.append(f"[用户信息]\n{self.core_memory['user_info']}")
        if self.core_memory["instructions"]:
            parts.append(f"[特殊指令]\n{self.core_memory['instructions']}")
        
        # 工作上下文
        if self.working_context:
            context_str = "\n".join([
                f"[{e.memory_type.value}] {e.content}"
                for e in self.working_context[-20:]  # 最近 20 条
            ])
            parts.append(f"[上下文]\n{context_str}")
        
        return "\n\n".join(parts)
    
    def clear_context(self) -> None:
        """清空工作上下文."""
        self.working_context.clear()
        self.scratch_pad.clear()


class EpisodicMemory(BaseMemoryStore):
    """情景记忆 (Episodic Memory).
    
    基于 Generative Agents 和 RecallM 论文的设计：
    - 存储具体的事件和经历
    - 支持时间序列检索
    - 支持基于内容的语义检索
    - 自动进行重要性评估
    
    使用 SQLite + FTS5 实现持久化和全文搜索。
    """
    
    def __init__(
        self,
        db_path: str = "episodic_memory.db",
        embedding_func: Optional[Callable] = None,
    ):
        self.db_path = db_path
        self.embedding_func = embedding_func
        self._conn = None
        self._initialized = False
    
    async def _ensure_initialized(self):
        """确保数据库已初始化."""
        if self._initialized:
            return
        
        try:
            import aiosqlite
        except ImportError:
            raise ImportError("aiosqlite 是必需的依赖。安装命令: uv pip install aiosqlite")
        
        self._conn = await aiosqlite.connect(self.db_path)
        
        # 主表
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_accessed TEXT,
                importance REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                related_ids TEXT,
                source_id TEXT,
                embedding TEXT,
                metadata TEXT,
                content_hash TEXT UNIQUE
            )
        """)
        
        # 索引
        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_episodes_created 
            ON episodes(created_at DESC)
        """)
        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_episodes_importance 
            ON episodes(importance DESC)
        """)
        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_episodes_type 
            ON episodes(memory_type)
        """)
        
        # FTS 全文搜索
        await self._conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts 
            USING fts5(id, content, memory_type, tokenize='unicode61')
        """)
        
        await self._conn.commit()
        self._initialized = True
    
    async def add(self, entry: MemoryEntry) -> MemoryEntry:
        """添加情景记忆."""
        await self._ensure_initialized()
        
        # 检查去重
        cursor = await self._conn.execute(
            "SELECT id FROM episodes WHERE content_hash = ?",
            (entry.content_hash,)
        )
        existing = await cursor.fetchone()
        if existing:
            # 更新访问计数而不是重复添加
            await self._update_access(existing[0])
            entry.id = existing[0]
            return entry
        
        # 生成嵌入（如果有嵌入函数）
        if self.embedding_func and entry.embedding is None:
            try:
                embeddings = await self.embedding_func([entry.content])
                entry.embedding = embeddings[0]
            except Exception:
                pass  # 嵌入失败时继续，不影响存储
        
        # 插入主表
        await self._conn.execute("""
            INSERT INTO episodes 
            (id, content, memory_type, created_at, last_accessed, importance, 
             access_count, related_ids, source_id, embedding, metadata, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.id,
            entry.content,
            entry.memory_type.value,
            entry.created_at.isoformat(),
            entry.last_accessed.isoformat() if entry.last_accessed else None,
            entry.importance,
            entry.access_count,
            json.dumps(entry.related_ids),
            entry.source_id,
            json.dumps(entry.embedding) if entry.embedding else None,
            json.dumps(entry.metadata),
            entry.content_hash,
        ))
        
        # 插入 FTS
        await self._conn.execute("""
            INSERT INTO episodes_fts (id, content, memory_type)
            VALUES (?, ?, ?)
        """, (entry.id, entry.content, entry.memory_type.value))
        
        await self._conn.commit()
        return entry
    
    async def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """获取记忆."""
        await self._ensure_initialized()
        
        cursor = await self._conn.execute(
            "SELECT * FROM episodes WHERE id = ?",
            (entry_id,)
        )
        row = await cursor.fetchone()
        
        if not row:
            return None
        
        # 更新访问
        await self._update_access(entry_id)
        
        return self._row_to_entry(row)
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        memory_types: Optional[list[MemoryType]] = None,
        time_range: Optional[tuple[datetime, datetime]] = None,
        min_importance: float = 0.0,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        **kwargs,
    ) -> list[RetrievalResult]:
        """多维度检索记忆.
        
        结合三种检索策略：
        1. 语义相似度（基于嵌入向量）
        2. 关键词匹配（基于 FTS）
        3. 时间相关性
        
        Args:
            query: 查询文本
            limit: 返回数量
            memory_types: 筛选记忆类型
            time_range: 时间范围筛选
            min_importance: 最小重要性
            alpha: 时间衰减权重
            beta: 重要性权重
            gamma: 语义相关性权重
        """
        await self._ensure_initialized()
        
        results = []
        
        # 1. FTS 关键词搜索
        fts_sql = """
            SELECT e.*, bm25(episodes_fts) as fts_score
            FROM episodes e
            JOIN episodes_fts f ON e.id = f.id
            WHERE episodes_fts MATCH ?
        """
        params = [query]
        
        if memory_types:
            placeholders = ",".join("?" * len(memory_types))
            fts_sql += f" AND e.memory_type IN ({placeholders})"
            params.extend([t.value for t in memory_types])
        
        if time_range:
            fts_sql += " AND e.created_at BETWEEN ? AND ?"
            params.extend([time_range[0].isoformat(), time_range[1].isoformat()])
        
        fts_sql += " AND e.importance >= ?"
        params.append(min_importance)
        
        fts_sql += f" ORDER BY fts_score LIMIT {limit * 2}"  # 多取一些用于重排序
        
        try:
            cursor = await self._conn.execute(fts_sql, params)
            rows = await cursor.fetchall()
            
            for row in rows:
                entry = self._row_to_entry(row[:-1])  # 最后一列是 fts_score
                fts_score = abs(row[-1]) / 10  # 归一化
                
                # 计算综合分数
                semantic_score = fts_score  # FTS 分数作为语义相似度的近似
                total_score = entry.retrieval_score(semantic_score, alpha, beta, gamma)
                
                results.append(RetrievalResult(
                    entry=entry,
                    score=total_score,
                    match_type="keyword",
                ))
        except Exception:
            pass  # FTS 查询失败时跳过
        
        # 2. 如果有嵌入函数，进行向量搜索
        if self.embedding_func and len(results) < limit:
            # 向量搜索实现（需要额外的向量数据库支持）
            # 这里简化处理，实际应该使用专门的向量数据库
            pass
        
        # 3. 按分数排序并返回
        results.sort(key=lambda x: x.score, reverse=True)
        
        # 更新访问统计
        for r in results[:limit]:
            await self._update_access(r.entry.id)
        
        return results[:limit]
    
    async def get_recent(
        self,
        limit: int = 10,
        memory_types: Optional[list[MemoryType]] = None,
    ) -> list[MemoryEntry]:
        """获取最近的记忆."""
        await self._ensure_initialized()
        
        sql = "SELECT * FROM episodes"
        params = []
        
        if memory_types:
            placeholders = ",".join("?" * len(memory_types))
            sql += f" WHERE memory_type IN ({placeholders})"
            params.extend([t.value for t in memory_types])
        
        sql += f" ORDER BY created_at DESC LIMIT {limit}"
        
        cursor = await self._conn.execute(sql, params)
        rows = await cursor.fetchall()
        
        return [self._row_to_entry(row) for row in rows]
    
    async def delete(self, entry_id: str) -> bool:
        """删除记忆."""
        await self._ensure_initialized()
        
        cursor = await self._conn.execute(
            "DELETE FROM episodes WHERE id = ?",
            (entry_id,)
        )
        await self._conn.execute(
            "DELETE FROM episodes_fts WHERE id = ?",
            (entry_id,)
        )
        await self._conn.commit()
        
        return cursor.rowcount > 0
    
    async def clear(self) -> None:
        """清空所有记忆."""
        await self._ensure_initialized()
        
        await self._conn.execute("DELETE FROM episodes")
        await self._conn.execute("DELETE FROM episodes_fts")
        await self._conn.commit()
    
    async def count(self) -> int:
        """统计记忆数量."""
        await self._ensure_initialized()
        
        cursor = await self._conn.execute("SELECT COUNT(*) FROM episodes")
        row = await cursor.fetchone()
        return row[0] if row else 0
    
    async def get_by_importance(
        self,
        limit: int = 10,
        min_importance: float = 0.5,
    ) -> list[MemoryEntry]:
        """获取重要记忆."""
        await self._ensure_initialized()
        
        cursor = await self._conn.execute("""
            SELECT * FROM episodes
            WHERE importance >= ?
            ORDER BY importance DESC, access_count DESC
            LIMIT ?
        """, (min_importance, limit))
        
        rows = await cursor.fetchall()
        return [self._row_to_entry(row) for row in rows]
    
    async def update_importance(self, entry_id: str, importance: float) -> bool:
        """更新重要性分数."""
        await self._ensure_initialized()
        
        cursor = await self._conn.execute("""
            UPDATE episodes SET importance = ? WHERE id = ?
        """, (importance, entry_id))
        
        await self._conn.commit()
        return cursor.rowcount > 0
    
    async def _update_access(self, entry_id: str):
        """更新访问统计."""
        now = datetime.now(timezone.utc).isoformat()
        await self._conn.execute("""
            UPDATE episodes 
            SET access_count = access_count + 1, last_accessed = ?
            WHERE id = ?
        """, (now, entry_id))
        await self._conn.commit()
    
    def _row_to_entry(self, row: tuple) -> MemoryEntry:
        """转换数据库行为记忆条目."""
        return MemoryEntry(
            id=row[0],
            content=row[1],
            memory_type=MemoryType(row[2]),
            created_at=datetime.fromisoformat(row[3]),
            last_accessed=datetime.fromisoformat(row[4]) if row[4] else None,
            importance=row[5],
            access_count=row[6],
            related_ids=json.loads(row[7]) if row[7] else [],
            source_id=row[8],
            embedding=json.loads(row[9]) if row[9] else None,
            metadata=json.loads(row[10]) if row[10] else {},
            content_hash=row[11] or "",
        )
    
    async def close(self):
        """关闭连接."""
        if self._conn:
            await self._conn.close()
            self._conn = None
            self._initialized = False


class SemanticMemory:
    """语义记忆 (Semantic Memory).
    
    存储从情景记忆中提取的通用知识和事实：
    - 通过反思（reflection）从情景记忆中生成
    - 更加抽象和通用
    - 不绑定具体时间点
    
    基于 Generative Agents 论文的反思机制。
    """
    
    def __init__(
        self,
        db_path: str = "semantic_memory.db",
        reflection_threshold: float = 0.7,
    ):
        self.db_path = db_path
        self.reflection_threshold = reflection_threshold
        self._conn = None
        self._initialized = False
    
    async def _ensure_initialized(self):
        """确保初始化."""
        if self._initialized:
            return
        
        try:
            import aiosqlite
        except ImportError:
            raise ImportError("aiosqlite 是必需的依赖")
        
        self._conn = await aiosqlite.connect(self.db_path)
        
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS semantic (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                importance REAL DEFAULT 0.5,
                source_ids TEXT,
                created_at TEXT NOT NULL,
                access_count INTEGER DEFAULT 0,
                metadata TEXT
            )
        """)
        
        await self._conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS semantic_fts 
            USING fts5(id, content, tokenize='unicode61')
        """)
        
        await self._conn.commit()
        self._initialized = True
    
    async def add_reflection(
        self,
        content: str,
        source_entries: list[MemoryEntry],
        importance: float = 0.7,
    ) -> str:
        """添加反思（从情景记忆中生成的抽象知识）."""
        await self._ensure_initialized()
        
        entry_id = uuid4().hex
        source_ids = [e.id for e in source_entries]
        
        await self._conn.execute("""
            INSERT INTO semantic (id, content, importance, source_ids, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            entry_id,
            content,
            importance,
            json.dumps(source_ids),
            datetime.now(timezone.utc).isoformat(),
            json.dumps({}),
        ))
        
        await self._conn.execute("""
            INSERT INTO semantic_fts (id, content)
            VALUES (?, ?)
        """, (entry_id, content))
        
        await self._conn.commit()
        return entry_id
    
    async def search(self, query: str, limit: int = 5) -> list[dict]:
        """搜索语义记忆."""
        await self._ensure_initialized()
        
        cursor = await self._conn.execute("""
            SELECT s.* FROM semantic s
            JOIN semantic_fts f ON s.id = f.id
            WHERE semantic_fts MATCH ?
            ORDER BY s.importance DESC
            LIMIT ?
        """, (query, limit))
        
        rows = await cursor.fetchall()
        
        return [
            {
                "id": row[0],
                "content": row[1],
                "importance": row[2],
                "source_ids": json.loads(row[3]) if row[3] else [],
            }
            for row in rows
        ]
    
    async def close(self):
        """关闭连接."""
        if self._conn:
            await self._conn.close()


class HierarchicalMemory:
    """分层记忆系统.
    
    整合所有记忆层级：
    1. 感知缓冲区 (Sensory Buffer) - 最近的原始输入
    2. 工作记忆 (Working Memory) - 当前上下文
    3. 情景记忆 (Episodic Memory) - 具体事件
    4. 语义记忆 (Semantic Memory) - 抽象知识
    
    支持：
    - 自动记忆整合（从短期到长期）
    - 重要性评估
    - 反思生成
    - 记忆检索
    """
    
    def __init__(
        self,
        db_path: str = "memory.db",
        embedding_func: Optional[Callable] = None,
        importance_evaluator: Optional[Callable] = None,
        reflection_generator: Optional[Callable] = None,
    ):
        """初始化分层记忆系统.
        
        Args:
            db_path: 数据库路径
            embedding_func: 嵌入函数 async def embed(texts: list[str]) -> list[list[float]]
            importance_evaluator: 重要性评估函数 async def eval(content: str) -> float
            reflection_generator: 反思生成函数 async def reflect(entries: list) -> str
        """
        self.sensory = SensoryBuffer(capacity=10)
        self.working = WorkingMemory(max_tokens=8000)
        self.episodic = EpisodicMemory(
            db_path=db_path.replace(".db", "_episodic.db"),
            embedding_func=embedding_func,
        )
        self.semantic = SemanticMemory(
            db_path=db_path.replace(".db", "_semantic.db"),
        )
        
        self.embedding_func = embedding_func
        self.importance_evaluator = importance_evaluator
        self.reflection_generator = reflection_generator
        
        # 反思计数器
        self._observation_count = 0
        self._reflection_interval = 10  # 每 10 个观察进行一次反思
    
    async def observe(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.OBSERVATION,
        importance: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> MemoryEntry:
        """记录观察（主要的记忆入口）.
        
        流程：
        1. 评估重要性
        2. 添加到感知缓冲区
        3. 如果重要，添加到工作记忆和情景记忆
        4. 定期触发反思
        """
        # 评估重要性
        if importance is None:
            if self.importance_evaluator:
                importance = await self.importance_evaluator(content)
            else:
                importance = self._estimate_importance(content)
        
        # 创建记忆条目
        entry = MemoryEntry(
            content=content,
            memory_type=memory_type,
            importance=importance,
            metadata=metadata or {},
        )
        
        # 添加到感知缓冲区
        self.sensory.add(entry)
        
        # 添加到工作记忆
        self.working.add_to_context(entry)
        
        # 添加到情景记忆（持久化）
        await self.episodic.add(entry)
        
        # 检查是否需要反思
        self._observation_count += 1
        if self._observation_count >= self._reflection_interval:
            await self._maybe_reflect()
            self._observation_count = 0
        
        return entry
    
    async def recall(
        self,
        query: str,
        limit: int = 10,
        include_semantic: bool = True,
        **kwargs,
    ) -> list[RetrievalResult]:
        """回忆相关记忆.
        
        从情景记忆和语义记忆中检索。
        """
        results = []
        
        # 从情景记忆检索
        episodic_results = await self.episodic.search(
            query,
            limit=limit,
            **kwargs,
        )
        results.extend(episodic_results)
        
        # 从语义记忆检索
        if include_semantic:
            semantic_results = await self.semantic.search(query, limit=limit // 2)
            for r in semantic_results:
                # 转换为统一格式
                entry = MemoryEntry(
                    id=r["id"],
                    content=r["content"],
                    memory_type=MemoryType.REFLECTION,
                    importance=r["importance"],
                )
                results.append(RetrievalResult(
                    entry=entry,
                    score=r["importance"],
                    match_type="semantic",
                ))
        
        # 按分数排序
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:limit]
    
    async def get_context(self, query: Optional[str] = None) -> str:
        """获取当前上下文（用于构建 LLM 提示）.
        
        结合：
        1. 核心记忆
        2. 工作记忆
        3. 相关的长期记忆
        """
        parts = []
        
        # 核心记忆
        core_prompt = self.working.to_prompt()
        if core_prompt:
            parts.append(core_prompt)
        
        # 相关长期记忆
        if query:
            relevant = await self.recall(query, limit=5)
            if relevant:
                relevant_str = "\n".join([
                    f"- {r.entry.content[:200]}"
                    for r in relevant
                ])
                parts.append(f"[相关记忆]\n{relevant_str}")
        
        return "\n\n".join(parts)
    
    def _estimate_importance(self, content: str) -> float:
        """估算重要性（简单启发式方法）.
        
        基于：
        - 内容长度
        - 关键词存在
        - 情感强度（简化版）
        """
        score = 0.5
        
        # 长度因子
        if len(content) > 200:
            score += 0.1
        if len(content) > 500:
            score += 0.1
        
        # 关键词因子
        important_keywords = [
            "重要", "必须", "紧急", "关键", "核心",
            "记住", "注意", "决定", "承诺", "约定",
            "important", "must", "critical", "key", "remember",
        ]
        for keyword in important_keywords:
            if keyword in content.lower():
                score += 0.1
                break
        
        # 情感强度（简化：感叹号和问号）
        if "!" in content or "！" in content:
            score += 0.05
        if "?" in content or "？" in content:
            score += 0.05
        
        return min(1.0, score)
    
    async def _maybe_reflect(self):
        """尝试进行反思.
        
        基于 Generative Agents 论文的反思机制：
        1. 获取最近的重要记忆
        2. 生成高层次的抽象总结
        3. 存储为语义记忆
        """
        if not self.reflection_generator:
            return
        
        # 获取最近的重要记忆
        recent = await self.episodic.get_recent(limit=20)
        important = [e for e in recent if e.importance > 0.6]
        
        if len(important) < 3:
            return
        
        try:
            # 生成反思
            reflection = await self.reflection_generator(important)
            
            # 存储为语义记忆
            await self.semantic.add_reflection(
                content=reflection,
                source_entries=important,
                importance=0.8,
            )
        except Exception:
            pass  # 反思生成失败不影响正常流程
    
    async def close(self):
        """关闭所有连接."""
        await self.episodic.close()
        await self.semantic.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
