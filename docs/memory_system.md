# AgentFlow 记忆系统设计文档

## 概述

AgentFlow 的记忆系统基于最新的 AI Agent 研究论文设计，实现了类人的分层记忆架构。

## 参考论文

1. **MemGPT** (2023) - [Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560)
   - 分层记忆架构
   - 主动记忆管理
   - 工作记忆与长期记忆的协调

2. **Generative Agents** (2023) - [Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)
   - 记忆流 (Memory Stream)
   - 反思机制 (Reflection)
   - 重要性评分
   - 多维度检索函数

3. **RecallM** (2023) - [An Architecture for Temporal Context Understanding](https://arxiv.org/abs/2307.02738)
   - 时间上下文理解
   - 记忆整合
   - 知识提取

4. **CoALA** (2023) - [Cognitive Architectures for Language Agents](https://arxiv.org/abs/2309.02427)
   - 认知架构设计
   - 感知-记忆-行动循环
   - 工作记忆模型

5. **SCM** (2023) - [Self-Controlled Memory](https://arxiv.org/abs/2304.13343)
   - 自适应遗忘
   - 记忆重要性动态调整

## 架构设计

### 记忆层级

```
┌─────────────────────────────────────────────────────────────┐
│                    输入 (Perception)                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              感知缓冲 (Sensory Buffer)                       │
│  - 容量: 10 条                                               │
│  - 策略: FIFO (先进先出)                                      │
│  - 作用: 缓存最近的原始输入                                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              工作记忆 (Working Memory)                       │
│  ┌──────────────────────┐  ┌─────────────────────┐          │
│  │ 核心记忆 (Core)       │  │ 工作上下文 (Context)│          │
│  │ - persona (角色)      │  │ - 当前对话          │          │
│  │ - user_info (用户)    │  │ - 最近观察          │          │
│  │ - instructions (指令) │  │ - 临时工作区        │          │
│  └──────────────────────┘  └─────────────────────┘          │
│  - 对应 LLM 上下文窗口限制                                    │
│  - 支持主动压缩和淘汰                                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              情景记忆 (Episodic Memory)                      │
│  - 存储: SQLite + FTS5 全文搜索                              │
│  - 内容: 具体事件、对话、行动                                  │
│  - 属性: 时间戳、重要性、访问计数、关联ID                       │
│  - 检索: 时间、语义、关键词多维度                              │
└─────────────────────────────────────────────────────────────┘
                              │
                      ┌───────┴───────┐
                      │   反思生成     │
                      │ (Reflection)  │
                      └───────┬───────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              语义记忆 (Semantic Memory)                      │
│  - 存储: SQLite + FTS5                                      │
│  - 内容: 抽象知识、事实、规则                                  │
│  - 来源: 从情景记忆反思生成                                    │
│  - 特点: 不绑定具体时间                                       │
└─────────────────────────────────────────────────────────────┘
```

### 记忆条目结构

```python
class MemoryEntry:
    id: str                  # 唯一标识
    content: str             # 记忆内容
    memory_type: MemoryType  # 类型: observation/thought/action/reflection/...
    
    # 时间信息
    created_at: datetime     # 创建时间
    last_accessed: datetime  # 最后访问时间
    
    # 重要性和统计
    importance: float        # 重要性分数 (0-1)
    access_count: int        # 访问次数
    
    # 关联
    related_ids: list[str]   # 相关记忆ID
    source_id: str           # 来源记忆（用于反思）
    
    # 向量
    embedding: list[float]   # 嵌入向量（可选）
    
    # 去重
    content_hash: str        # 内容哈希
```

### 记忆类型

| 类型 | 说明 | 示例 |
|------|------|------|
| `observation` | 观察到的信息 | "用户说他喜欢Python" |
| `thought` | 推理和思考 | "用户可能需要学习框架" |
| `action` | 执行的行动 | "推荐了Django教程" |
| `reflection` | 反思生成的洞察 | "用户是有经验的开发者" |
| `plan` | 计划和意图 | "下一步应该询问项目需求" |
| `conversation` | 对话内容 | "用户: 你好" |

## 核心机制

### 1. 重要性评分

基于 Generative Agents 论文的重要性评估：

```python
def estimate_importance(content: str) -> float:
    score = 0.5  # 基础分
    
    # 长度因子
    if len(content) > 200: score += 0.1
    if len(content) > 500: score += 0.1
    
    # 关键词因子
    important_keywords = ["重要", "必须", "关键", "记住", ...]
    if any(kw in content for kw in important_keywords):
        score += 0.1
    
    # 情感强度
    if "!" in content or "！" in content: score += 0.05
    
    return min(1.0, score)
```

### 2. 检索评分函数

基于 Generative Agents 论文的三维检索：

```
score = α × recency + β × importance + γ × relevance
```

- **recency (时间衰减)**: `decay^hours_since_creation`
- **importance (重要性)**: 存储的重要性分数
- **relevance (相关性)**: 语义相似度或关键词匹配

### 3. 反思机制

基于 Generative Agents 的反思生成：

1. 累计重要性达到阈值时触发
2. 收集最近的重要记忆
3. 生成反思问题
4. 检索相关记忆
5. 生成高层次洞察
6. 存储为语义记忆

```python
async def reflect(memories: list[MemoryEntry]) -> str:
    # 1. 生成问题
    questions = generate_reflection_questions(memories)
    
    # 2. 对每个问题
    for question in questions:
        # 检索相关记忆
        relevant = search(question)
        # 生成洞察
        insight = generate_insight(question, relevant)
        # 存储
        semantic_memory.add(insight)
```

### 4. 记忆整合

基于 RecallM 和 SCM 的整合机制：

- **压缩**: 将旧记忆合并为摘要
- **遗忘**: 基于重要性和时间的自适应遗忘
- **知识提取**: 从记忆中提取结构化知识

## 检索策略

### 混合检索 (Hybrid Retrieval)

使用 RRF (Reciprocal Rank Fusion) 融合多种检索结果：

```python
RRF(d) = Σ 1 / (k + rank_i(d))
```

### 支持的策略

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| `semantic` | 语义相似度检索 | 概念相关查询 |
| `keyword` | 关键词匹配 | 精确查询 |
| `temporal` | 时间优先 | "昨天做了什么" |
| `importance` | 重要性优先 | 重要信息查询 |
| `hybrid` | 混合融合 | 通用查询 |

### 查询类型自动识别

```python
class QueryAnalyzer:
    def analyze(query: str) -> QueryType:
        # 时间查询: "昨天", "最近", "上周"
        # 个人查询: "我说过", "我的"
        # 过程查询: "如何", "怎么"
        # 默认: 语义查询
```

## 使用示例

### 基础使用

```python
from agentflow.memory import HierarchicalMemory, MemoryType

async with HierarchicalMemory(db_path="memory.db") as memory:
    # 记录观察
    await memory.observe(
        "用户是Python开发者",
        memory_type=MemoryType.OBSERVATION,
    )
    
    # 回忆
    results = await memory.recall("Python", limit=5)
    
    # 获取上下文
    context = await memory.get_context("编程建议")
```

### 配置检索

```python
from agentflow.memory import HybridRetriever, RetrievalConfig

config = RetrievalConfig(
    default_limit=10,
    alpha=1.0,  # 时间权重
    beta=1.0,   # 重要性权重
    gamma=1.5,  # 语义权重
)

retriever = HybridRetriever(episodic, semantic, config=config)
results = await retriever.retrieve("机器学习")
```

### 配置整合

```python
from agentflow.memory import MemoryConsolidator, ConsolidationConfig

config = ConsolidationConfig(
    compression_threshold=100,
    reflection_threshold=10.0,
    forgetting_rate=0.1,
)

consolidator = MemoryConsolidator(episodic, semantic, llm, config)
report = await consolidator.consolidate()
```

## 性能考虑

1. **SQLite + FTS5**: 轻量级且高效的全文搜索
2. **去重**: 基于内容哈希避免重复存储
3. **惰性初始化**: 数据库连接按需创建
4. **批量操作**: 支持批量检索和整合

## 扩展点

1. **嵌入函数**: 支持自定义嵌入模型
2. **重要性评估**: 支持 LLM 评估重要性
3. **反思生成**: 支持自定义反思 LLM
4. **向量数据库**: 可扩展到 ChromaDB/Pinecone

## 与旧版兼容

旧版接口仍然可用：

```python
from agentflow.memory import ShortTermMemory, LongTermMemory, DatabaseMemory
```

新版推荐使用：

```python
from agentflow.memory import HierarchicalMemory
```
