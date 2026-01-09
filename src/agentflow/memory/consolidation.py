"""
记忆整合模块
============

基于以下论文的记忆整合机制：
- Generative Agents (2023): 反思机制
- RecallM (2023): 时间上下文理解和记忆整合
- SCM (Self-Controlled Memory, 2023): 自适应记忆管理

主要功能：
1. 记忆压缩：将多个相关记忆合并为摘要
2. 记忆遗忘：基于重要性和时间的自适应遗忘
3. 反思生成：从经历中提取高层次洞察
4. 知识提取：从对话中提取结构化知识
"""

import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import Any, Optional, Callable, Protocol
from dataclasses import dataclass

from .hierarchical import (
    MemoryEntry,
    MemoryType,
    EpisodicMemory,
    SemanticMemory,
)


class LLMInterface(Protocol):
    """LLM 接口协议."""
    
    async def generate(self, prompt: str) -> str:
        """生成文本."""
        ...


@dataclass
class ConsolidationConfig:
    """整合配置."""
    
    # 压缩阈值：当记忆数量超过此值时触发压缩
    compression_threshold: int = 100
    
    # 压缩后保留的比例
    compression_ratio: float = 0.5
    
    # 反思阈值：累计重要性超过此值时触发反思
    reflection_threshold: float = 10.0
    
    # 遗忘衰减率：每天的遗忘率
    forgetting_rate: float = 0.1
    
    # 最小重要性：低于此值的记忆可能被遗忘
    min_importance: float = 0.2
    
    # 知识提取的最小置信度
    knowledge_confidence: float = 0.7


class MemoryConsolidator:
    """记忆整合器.
    
    负责：
    1. 定期压缩旧记忆
    2. 生成反思和洞察
    3. 提取结构化知识
    4. 管理记忆遗忘
    """
    
    def __init__(
        self,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        llm: Optional[LLMInterface] = None,
        config: Optional[ConsolidationConfig] = None,
    ):
        self.episodic = episodic
        self.semantic = semantic
        self.llm = llm
        self.config = config or ConsolidationConfig()
        
        # 累计重要性（用于触发反思）
        self._accumulated_importance = 0.0
    
    async def consolidate(self) -> dict[str, Any]:
        """执行完整的记忆整合.
        
        Returns:
            整合报告
        """
        report = {
            "compressed_count": 0,
            "reflections_created": 0,
            "forgotten_count": 0,
            "knowledge_extracted": 0,
        }
        
        # 1. 压缩旧记忆
        compressed = await self.compress_old_memories()
        report["compressed_count"] = compressed
        
        # 2. 生成反思
        if self._accumulated_importance >= self.config.reflection_threshold:
            reflections = await self.generate_reflections()
            report["reflections_created"] = reflections
            self._accumulated_importance = 0
        
        # 3. 遗忘不重要的记忆
        forgotten = await self.forget_irrelevant()
        report["forgotten_count"] = forgotten
        
        return report
    
    async def compress_old_memories(self) -> int:
        """压缩旧记忆.
        
        将多个相关的旧记忆合并为摘要，减少存储空间。
        基于 RecallM 的时间整合思想。
        """
        if not self.llm:
            return 0
        
        # 检查是否需要压缩
        total_count = await self.episodic.count()
        if total_count < self.config.compression_threshold:
            return 0
        
        # 获取旧记忆（超过 7 天）
        old_cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        
        # 按天分组旧记忆
        recent = await self.episodic.get_recent(limit=total_count)
        old_memories = [
            m for m in recent
            if m.created_at < old_cutoff
        ]
        
        if len(old_memories) < 10:
            return 0
        
        # 按类型分组
        groups = {}
        for m in old_memories:
            key = m.memory_type.value
            if key not in groups:
                groups[key] = []
            groups[key].append(m)
        
        compressed_count = 0
        
        for memory_type, memories in groups.items():
            if len(memories) < 5:
                continue
            
            # 生成摘要
            try:
                summary = await self._summarize_memories(memories)
                
                # 创建摘要记忆
                summary_entry = MemoryEntry(
                    content=f"[摘要 - {memory_type}] {summary}",
                    memory_type=MemoryType.REFLECTION,
                    importance=max(m.importance for m in memories),
                    metadata={
                        "type": "compression",
                        "source_count": len(memories),
                        "period_start": min(m.created_at for m in memories).isoformat(),
                        "period_end": max(m.created_at for m in memories).isoformat(),
                    },
                )
                
                await self.episodic.add(summary_entry)
                
                # 删除原始记忆（保留高重要性的）
                for m in memories:
                    if m.importance < 0.7:
                        await self.episodic.delete(m.id)
                        compressed_count += 1
            
            except Exception as e:
                continue
        
        return compressed_count
    
    async def generate_reflections(self, force: bool = False) -> int:
        """生成反思.
        
        基于 Generative Agents 论文的反思机制：
        1. 识别最近记忆中的重要主题
        2. 生成高层次的洞察
        3. 存储为语义记忆
        
        Args:
            force: 强制生成反思，忽略阈值
        """
        if not self.llm:
            return 0
        
        if not force and self._accumulated_importance < self.config.reflection_threshold:
            return 0
        
        # 获取最近的重要记忆
        important_memories = await self.episodic.get_by_importance(
            limit=30,
            min_importance=0.5,
        )
        
        if len(important_memories) < 3:
            return 0
        
        # 生成反思问题
        questions = await self._generate_reflection_questions(important_memories)
        
        reflection_count = 0
        
        for question in questions[:3]:  # 最多 3 个反思
            try:
                # 检索相关记忆
                relevant = await self.episodic.search(question, limit=10)
                
                if len(relevant) < 2:
                    continue
                
                # 生成洞察
                insight = await self._generate_insight(
                    question,
                    [r.entry for r in relevant],
                )
                
                # 存储为语义记忆
                await self.semantic.add_reflection(
                    content=insight,
                    source_entries=[r.entry for r in relevant],
                    importance=0.8,
                )
                
                reflection_count += 1
            
            except Exception:
                continue
        
        return reflection_count
    
    async def forget_irrelevant(self) -> int:
        """遗忘不相关的记忆.
        
        基于 SCM (Self-Controlled Memory) 的自适应遗忘：
        1. 长时间未访问的记忆重要性降低
        2. 低于阈值的记忆被删除
        """
        # 获取所有记忆
        all_memories = await self.episodic.get_recent(limit=1000)
        
        forgotten = 0
        now = datetime.now(timezone.utc)
        
        for memory in all_memories:
            # 计算时间衰减
            days_since_access = 0
            if memory.last_accessed:
                days_since_access = (now - memory.last_accessed).days
            else:
                days_since_access = (now - memory.created_at).days
            
            # 应用遗忘衰减
            decay = pow(1 - self.config.forgetting_rate, days_since_access)
            new_importance = memory.importance * decay
            
            # 考虑访问次数的增益
            access_boost = min(0.3, memory.access_count * 0.02)
            new_importance = min(1.0, new_importance + access_boost)
            
            if new_importance < self.config.min_importance:
                # 删除不重要的记忆
                await self.episodic.delete(memory.id)
                forgotten += 1
            elif new_importance != memory.importance:
                # 更新重要性
                await self.episodic.update_importance(memory.id, new_importance)
        
        return forgotten
    
    async def extract_knowledge(
        self,
        memories: list[MemoryEntry],
    ) -> list[dict[str, Any]]:
        """从记忆中提取结构化知识.
        
        提取的知识类型：
        - 事实 (fact): 关于实体的客观信息
        - 偏好 (preference): 用户偏好和习惯
        - 关系 (relation): 实体之间的关系
        - 规则 (rule): 总结出的规律
        """
        if not self.llm:
            return []
        
        # 准备提取提示
        memory_text = "\n".join([
            f"- [{m.memory_type.value}] {m.content}"
            for m in memories[:20]
        ])
        
        prompt = f"""分析以下记忆，提取其中的结构化知识。

记忆内容：
{memory_text}

请提取以下类型的知识：
1. 事实 (fact): 关于人、事、物的客观信息
2. 偏好 (preference): 用户的偏好和习惯
3. 关系 (relation): 实体之间的关联
4. 规则 (rule): 可以总结出的规律

以 JSON 数组格式返回，每个知识项包含：
- type: 知识类型
- content: 知识内容
- confidence: 置信度 (0-1)
- entities: 涉及的实体列表

只返回 JSON，不要其他内容。
"""
        
        try:
            response = await self.llm.generate(prompt)
            
            # 解析 JSON
            # 尝试提取 JSON 部分
            json_start = response.find("[")
            json_end = response.rfind("]") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                knowledge = json.loads(json_str)
                
                # 过滤低置信度的知识
                return [
                    k for k in knowledge
                    if k.get("confidence", 0) >= self.config.knowledge_confidence
                ]
        
        except Exception:
            pass
        
        return []
    
    def track_importance(self, entry: MemoryEntry):
        """跟踪重要性累计."""
        self._accumulated_importance += entry.importance
    
    async def _summarize_memories(self, memories: list[MemoryEntry]) -> str:
        """生成记忆摘要."""
        memory_text = "\n".join([
            f"[{m.created_at.strftime('%Y-%m-%d')}] {m.content}"
            for m in memories
        ])
        
        prompt = f"""请将以下多条记忆整合为一个简洁的摘要，保留关键信息：

{memory_text}

要求：
1. 保留重要的事实和细节
2. 去除重复信息
3. 使用简洁的语言
4. 保持时间顺序的脉络

摘要："""
        
        return await self.llm.generate(prompt)
    
    async def _generate_reflection_questions(
        self,
        memories: list[MemoryEntry],
    ) -> list[str]:
        """生成反思问题."""
        memory_text = "\n".join([
            f"- {m.content[:100]}"
            for m in memories[:15]
        ])
        
        prompt = f"""基于以下记忆，生成 3 个深度反思问题，这些问题应该能够帮助总结重要的模式和洞察：

记忆：
{memory_text}

要求：
1. 问题应该关注模式、原因和意义
2. 问题应该能够整合多条记忆
3. 问题应该有深度，不是简单的事实回顾

只返回问题列表，每行一个问题："""
        
        response = await self.llm.generate(prompt)
        
        # 解析问题
        questions = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if line and not line.startswith("#"):
                # 移除可能的序号
                if line[0].isdigit() and line[1] in ".、":
                    line = line[2:].strip()
                questions.append(line)
        
        return questions
    
    async def _generate_insight(
        self,
        question: str,
        memories: list[MemoryEntry],
    ) -> str:
        """基于问题和相关记忆生成洞察."""
        memory_text = "\n".join([
            f"- {m.content}"
            for m in memories
        ])
        
        prompt = f"""基于以下问题和相关记忆，生成一个深度洞察：

问题：{question}

相关记忆：
{memory_text}

请提供：
1. 一个清晰的洞察或结论
2. 简要说明支持这个洞察的证据

洞察："""
        
        return await self.llm.generate(prompt)


class MemoryScheduler:
    """记忆整合调度器.
    
    定期执行记忆整合任务。
    """
    
    def __init__(
        self,
        consolidator: MemoryConsolidator,
        interval_hours: float = 24,
    ):
        self.consolidator = consolidator
        self.interval_hours = interval_hours
        self._task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """启动调度器."""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
    
    async def stop(self):
        """停止调度器."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
    
    async def _run_loop(self):
        """运行循环."""
        while self._running:
            try:
                await asyncio.sleep(self.interval_hours * 3600)
                await self.consolidator.consolidate()
            except asyncio.CancelledError:
                break
            except Exception:
                pass
