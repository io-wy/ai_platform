"""
记忆检索模块
============

基于以下论文的检索策略：
- Generative Agents (2023): 多维度检索评分
- RAG 相关研究: 检索增强生成
- Memory^3 (2023): 多级记忆协调

主要功能：
1. 多策略检索：语义、关键词、时间
2. 检索融合：RRF (Reciprocal Rank Fusion)
3. 上下文感知检索
4. 自适应检索（基于查询类型调整策略）
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import re

from .hierarchical import (
    MemoryEntry,
    MemoryType,
    RetrievalResult,
    EpisodicMemory,
    SemanticMemory,
    WorkingMemory,
)


class QueryType(str, Enum):
    """查询类型."""
    FACTUAL = "factual"         # 事实查询："什么是X"
    TEMPORAL = "temporal"       # 时间查询："昨天发生了什么"
    SEMANTIC = "semantic"       # 语义查询：相关概念
    PROCEDURAL = "procedural"   # 过程查询："如何做X"
    PERSONAL = "personal"       # 个人信息："我说过什么"


class RetrievalStrategy(str, Enum):
    """检索策略."""
    SEMANTIC = "semantic"       # 语义相似度
    KEYWORD = "keyword"         # 关键词匹配
    TEMPORAL = "temporal"       # 时间相关
    IMPORTANCE = "importance"   # 重要性优先
    HYBRID = "hybrid"          # 混合策略


@dataclass
class RetrievalConfig:
    """检索配置."""
    
    # 默认返回数量
    default_limit: int = 10
    
    # 检索权重
    alpha: float = 1.0  # 时间衰减权重
    beta: float = 1.0   # 重要性权重
    gamma: float = 1.0  # 语义相关性权重
    
    # 最小分数阈值
    min_score: float = 0.1
    
    # RRF 参数
    rrf_k: int = 60
    
    # 时间窗口（用于时间查询）
    time_windows: dict[str, int] = field(default_factory=lambda: {
        "today": 1,
        "yesterday": 2,
        "this_week": 7,
        "this_month": 30,
        "recent": 3,
    })


class QueryAnalyzer:
    """查询分析器.
    
    分析查询意图，确定最佳检索策略。
    """
    
    # 时间关键词模式
    TEMPORAL_PATTERNS = [
        (r"昨[天日]|yesterday", "yesterday"),
        (r"今[天日]|today", "today"),
        (r"前[天日]|day before", "day_before"),
        (r"上周|last week", "last_week"),
        (r"本周|this week", "this_week"),
        (r"上个?月|last month", "last_month"),
        (r"最近|recently|近期", "recent"),
        (r"\d+天前|\d+ days? ago", "n_days_ago"),
        (r"\d+周前|\d+ weeks? ago", "n_weeks_ago"),
    ]
    
    # 个人信息关键词
    PERSONAL_KEYWORDS = [
        "我说", "我的", "我想", "我要", "我们",
        "i said", "my ", "i want", "i need",
    ]
    
    # 过程关键词
    PROCEDURAL_KEYWORDS = [
        "如何", "怎么", "怎样", "步骤", "方法",
        "how to", "how do", "steps", "guide",
    ]
    
    def analyze(self, query: str) -> tuple[QueryType, dict[str, Any]]:
        """分析查询.
        
        Returns:
            (查询类型, 附加信息)
        """
        query_lower = query.lower()
        info = {}
        
        # 检查时间相关
        for pattern, time_type in self.TEMPORAL_PATTERNS:
            if re.search(pattern, query_lower):
                info["time_type"] = time_type
                return QueryType.TEMPORAL, info
        
        # 检查个人信息
        for keyword in self.PERSONAL_KEYWORDS:
            if keyword in query_lower:
                return QueryType.PERSONAL, info
        
        # 检查过程查询
        for keyword in self.PROCEDURAL_KEYWORDS:
            if keyword in query_lower:
                return QueryType.PROCEDURAL, info
        
        # 默认为语义查询
        return QueryType.SEMANTIC, info


class HybridRetriever:
    """混合检索器.
    
    结合多种检索策略，使用 RRF 融合结果。
    """
    
    def __init__(
        self,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        working: Optional[WorkingMemory] = None,
        embedding_func: Optional[Callable] = None,
        config: Optional[RetrievalConfig] = None,
    ):
        self.episodic = episodic
        self.semantic = semantic
        self.working = working
        self.embedding_func = embedding_func
        self.config = config or RetrievalConfig()
        
        self.analyzer = QueryAnalyzer()
    
    async def retrieve(
        self,
        query: str,
        limit: Optional[int] = None,
        strategy: Optional[RetrievalStrategy] = None,
        memory_types: Optional[list[MemoryType]] = None,
        time_range: Optional[tuple[datetime, datetime]] = None,
        **kwargs,
    ) -> list[RetrievalResult]:
        """检索记忆.
        
        Args:
            query: 查询文本
            limit: 返回数量限制
            strategy: 检索策略（None 表示自动选择）
            memory_types: 筛选记忆类型
            time_range: 时间范围
        """
        limit = limit or self.config.default_limit
        
        # 分析查询
        query_type, query_info = self.analyzer.analyze(query)
        
        # 确定策略
        if strategy is None:
            strategy = self._select_strategy(query_type, query_info)
        
        # 根据查询类型调整时间范围
        if query_type == QueryType.TEMPORAL and time_range is None:
            time_range = self._get_time_range(query_info.get("time_type", "recent"))
        
        # 执行检索
        if strategy == RetrievalStrategy.HYBRID:
            return await self._hybrid_retrieve(
                query, limit, memory_types, time_range, **kwargs
            )
        elif strategy == RetrievalStrategy.SEMANTIC:
            return await self._semantic_retrieve(
                query, limit, memory_types, time_range, **kwargs
            )
        elif strategy == RetrievalStrategy.KEYWORD:
            return await self._keyword_retrieve(
                query, limit, memory_types, time_range, **kwargs
            )
        elif strategy == RetrievalStrategy.TEMPORAL:
            return await self._temporal_retrieve(
                query, limit, memory_types, time_range, **kwargs
            )
        elif strategy == RetrievalStrategy.IMPORTANCE:
            return await self._importance_retrieve(
                query, limit, memory_types, time_range, **kwargs
            )
        else:
            return await self._hybrid_retrieve(
                query, limit, memory_types, time_range, **kwargs
            )
    
    async def retrieve_for_context(
        self,
        query: str,
        max_tokens: int = 2000,
    ) -> str:
        """检索并格式化为上下文文本.
        
        用于构建 LLM 提示。
        """
        results = await self.retrieve(query, limit=10)
        
        if not results:
            return ""
        
        # 估算 token 并截断
        context_parts = []
        estimated_tokens = 0
        
        for r in results:
            entry_text = f"[{r.entry.memory_type.value}] {r.entry.content}"
            entry_tokens = len(entry_text) // 2  # 粗略估算
            
            if estimated_tokens + entry_tokens > max_tokens:
                break
            
            context_parts.append(entry_text)
            estimated_tokens += entry_tokens
        
        return "\n".join(context_parts)
    
    def _select_strategy(
        self,
        query_type: QueryType,
        query_info: dict,
    ) -> RetrievalStrategy:
        """根据查询类型选择策略."""
        strategy_map = {
            QueryType.FACTUAL: RetrievalStrategy.HYBRID,
            QueryType.TEMPORAL: RetrievalStrategy.TEMPORAL,
            QueryType.SEMANTIC: RetrievalStrategy.SEMANTIC,
            QueryType.PROCEDURAL: RetrievalStrategy.KEYWORD,
            QueryType.PERSONAL: RetrievalStrategy.HYBRID,
        }
        return strategy_map.get(query_type, RetrievalStrategy.HYBRID)
    
    def _get_time_range(
        self,
        time_type: str,
    ) -> tuple[datetime, datetime]:
        """获取时间范围."""
        now = datetime.now(timezone.utc)
        
        days_map = {
            "today": 1,
            "yesterday": 2,
            "day_before": 3,
            "recent": 3,
            "this_week": 7,
            "last_week": 14,
            "this_month": 30,
            "last_month": 60,
        }
        
        days = days_map.get(time_type, 7)
        start = now - timedelta(days=days)
        
        return (start, now)
    
    async def _hybrid_retrieve(
        self,
        query: str,
        limit: int,
        memory_types: Optional[list[MemoryType]],
        time_range: Optional[tuple[datetime, datetime]],
        **kwargs,
    ) -> list[RetrievalResult]:
        """混合检索（RRF 融合）."""
        # 并行执行多种检索
        tasks = [
            self._keyword_retrieve(query, limit * 2, memory_types, time_range),
            self._temporal_retrieve(query, limit * 2, memory_types, time_range),
        ]
        
        # 添加语义检索（如果有嵌入函数）
        if self.embedding_func:
            tasks.append(
                self._semantic_retrieve(query, limit * 2, memory_types, time_range)
            )
        
        all_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 收集所有结果
        ranked_lists: list[list[RetrievalResult]] = []
        for result in all_results:
            if isinstance(result, list):
                ranked_lists.append(result)
        
        # RRF 融合
        return self._rrf_fusion(ranked_lists, limit)
    
    async def _semantic_retrieve(
        self,
        query: str,
        limit: int,
        memory_types: Optional[list[MemoryType]],
        time_range: Optional[tuple[datetime, datetime]],
        **kwargs,
    ) -> list[RetrievalResult]:
        """语义检索."""
        # 使用情景记忆的搜索（基于 FTS，近似语义）
        results = await self.episodic.search(
            query,
            limit=limit,
            memory_types=memory_types,
            time_range=time_range,
            alpha=self.config.alpha,
            beta=self.config.beta,
            gamma=self.config.gamma,
        )
        
        # 添加语义记忆的结果
        semantic_results = await self.semantic.search(query, limit=limit // 2)
        for r in semantic_results:
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
        
        return results
    
    async def _keyword_retrieve(
        self,
        query: str,
        limit: int,
        memory_types: Optional[list[MemoryType]],
        time_range: Optional[tuple[datetime, datetime]],
        **kwargs,
    ) -> list[RetrievalResult]:
        """关键词检索."""
        return await self.episodic.search(
            query,
            limit=limit,
            memory_types=memory_types,
            time_range=time_range,
            alpha=0,  # 不考虑时间衰减
            beta=0.5,  # 较低的重要性权重
            gamma=2.0,  # 更高的关键词匹配权重
        )
    
    async def _temporal_retrieve(
        self,
        query: str,
        limit: int,
        memory_types: Optional[list[MemoryType]],
        time_range: Optional[tuple[datetime, datetime]],
        **kwargs,
    ) -> list[RetrievalResult]:
        """时间优先检索."""
        # 获取时间范围内的记忆
        entries = await self.episodic.get_recent(limit=limit * 2, memory_types=memory_types)
        
        # 筛选时间范围
        if time_range:
            entries = [
                e for e in entries
                if time_range[0] <= e.created_at <= time_range[1]
            ]
        
        # 按时间排序
        entries.sort(key=lambda e: e.created_at, reverse=True)
        
        # 转换为结果
        results = []
        for i, entry in enumerate(entries[:limit]):
            # 简单关键词匹配评分
            keyword_score = self._simple_keyword_score(query, entry.content)
            score = entry.retrieval_score(keyword_score, alpha=2.0, beta=0.5, gamma=0.5)
            results.append(RetrievalResult(
                entry=entry,
                score=score,
                match_type="temporal",
            ))
        
        return results
    
    async def _importance_retrieve(
        self,
        query: str,
        limit: int,
        memory_types: Optional[list[MemoryType]],
        time_range: Optional[tuple[datetime, datetime]],
        **kwargs,
    ) -> list[RetrievalResult]:
        """重要性优先检索."""
        entries = await self.episodic.get_by_importance(limit=limit * 2)
        
        # 筛选
        if memory_types:
            entries = [e for e in entries if e.memory_type in memory_types]
        if time_range:
            entries = [
                e for e in entries
                if time_range[0] <= e.created_at <= time_range[1]
            ]
        
        results = []
        for entry in entries[:limit]:
            keyword_score = self._simple_keyword_score(query, entry.content)
            score = entry.retrieval_score(keyword_score, alpha=0.5, beta=2.0, gamma=1.0)
            results.append(RetrievalResult(
                entry=entry,
                score=score,
                match_type="importance",
            ))
        
        return results
    
    def _simple_keyword_score(self, query: str, content: str) -> float:
        """简单的关键词匹配评分."""
        query_lower = query.lower()
        content_lower = content.lower()
        
        # 提取查询词
        words = re.findall(r'\w+', query_lower)
        if not words:
            return 0.0
        
        # 计算匹配比例
        matches = sum(1 for w in words if w in content_lower)
        return matches / len(words)
    
    def _rrf_fusion(
        self,
        ranked_lists: list[list[RetrievalResult]],
        limit: int,
    ) -> list[RetrievalResult]:
        """Reciprocal Rank Fusion.
        
        融合多个排序列表：
        RRF(d) = Σ 1 / (k + rank(d))
        """
        k = self.config.rrf_k
        
        # 计算 RRF 分数
        scores: dict[str, float] = {}
        entries: dict[str, RetrievalResult] = {}
        
        for ranked_list in ranked_lists:
            for rank, result in enumerate(ranked_list):
                entry_id = result.entry.id
                
                # 累加 RRF 分数
                rrf_score = 1.0 / (k + rank + 1)
                scores[entry_id] = scores.get(entry_id, 0) + rrf_score
                
                # 保存最高分的结果
                if entry_id not in entries or result.score > entries[entry_id].score:
                    entries[entry_id] = result
        
        # 按 RRF 分数排序
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        # 构建结果
        results = []
        for entry_id in sorted_ids[:limit]:
            result = entries[entry_id]
            # 更新分数为 RRF 分数
            results.append(RetrievalResult(
                entry=result.entry,
                score=scores[entry_id],
                match_type="hybrid",
            ))
        
        return results


class ContextAwareRetriever(HybridRetriever):
    """上下文感知检索器.
    
    在混合检索的基础上，考虑：
    1. 当前工作记忆上下文
    2. 对话历史
    3. 用户偏好
    """
    
    def __init__(
        self,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        working: WorkingMemory,
        embedding_func: Optional[Callable] = None,
        config: Optional[RetrievalConfig] = None,
    ):
        super().__init__(episodic, semantic, working, embedding_func, config)
    
    async def retrieve_with_context(
        self,
        query: str,
        limit: Optional[int] = None,
        use_working_memory: bool = True,
        **kwargs,
    ) -> list[RetrievalResult]:
        """上下文感知检索.
        
        Args:
            query: 查询文本
            limit: 返回数量
            use_working_memory: 是否使用工作记忆增强查询
        """
        limit = limit or self.config.default_limit
        
        # 增强查询
        enhanced_query = query
        if use_working_memory and self.working:
            context = self.working.get_context()
            if context:
                # 使用最近的上下文增强查询
                recent_context = " ".join([
                    e.content[:50] for e in context[-3:]
                ])
                enhanced_query = f"{query} {recent_context}"
        
        # 执行检索
        results = await self.retrieve(enhanced_query, limit=limit * 2, **kwargs)
        
        # 重新排序：考虑与当前上下文的相关性
        if self.working:
            results = self._rerank_with_context(results, query)
        
        return results[:limit]
    
    def _rerank_with_context(
        self,
        results: list[RetrievalResult],
        query: str,
    ) -> list[RetrievalResult]:
        """基于上下文重新排序."""
        if not self.working:
            return results
        
        context_entries = self.working.get_context()
        if not context_entries:
            return results
        
        # 获取上下文中的关键实体/概念
        context_text = " ".join([e.content for e in context_entries[-5:]])
        context_words = set(re.findall(r'\w+', context_text.lower()))
        
        # 为每个结果计算上下文相关性增益
        for result in results:
            content_words = set(re.findall(r'\w+', result.entry.content.lower()))
            overlap = len(context_words & content_words)
            
            # 增加上下文相关性分数
            context_boost = min(0.3, overlap * 0.05)
            result.score += context_boost
        
        # 重新排序
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results
