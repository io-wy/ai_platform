# AgentFlow

轻量级、高性能的 AI Agent 框架。

## ✨ 特性

### 核心能力
- 🧠 **多种推理模式** - ReAct、Chain-of-Thought、Tree-of-Thought、Reflexion、Plan & Execute
- 🔧 **丰富的工具生态** - 浏览器、终端、网络搜索、数据库、文件操作等
- 🔌 **多 LLM 后端** - OpenAI、vLLM、Ollama、Anthropic，完全兼容 OpenAI API
- 💾 **智能记忆系统** - 基于 2024 最新论文的记忆管理 (MemGPT, Generative Agents)

### 架构特点
- 🏗️ **Protocol-based 设计** - 使用协议而非继承，高度解耦
- ⚡ **Go 高性能网关** - 独立的 API/Agent Gateway，支持 gRPC 和 MCP 协议
- 📊 **全面监控** - Prometheus 指标、分布式追踪、告警系统
- 🔄 **MCP 协议支持** - 无缝对接 Model Context Protocol 服务器

## 📦 项目结构

```
llmapplication/
├── src/agentflow/         # Python Agent 框架
│   ├── core/              # 核心类型和协议
│   ├── memory/            # 记忆系统
│   ├── llm/               # LLM 提供者
│   ├── patterns/          # 推理模式
│   └── tools/             # 工具系统
├── gateway/               # Go 高性能网关
│   ├── apigateway/        # API 网关 (HTTP/gRPC/WebSocket)
│   ├── agentgateway/      # Agent 网关 (MCP 协议/监控)
│   └── proto/             # gRPC 定义
└── examples/              # 使用示例
```

## 🚀 快速开始

### 安装

```bash
# Python 框架
uv pip install -e .

# Go 网关 (可选)
cd gateway && make build
```

### 基础用法

```python
from agentflow import SimpleAgent
from agentflow.llm import OpenAIProvider
from agentflow.tools import tool

# 定义工具
@tool(description="计算数学表达式")
async def calculator(expression: str) -> str:
    return str(eval(expression))

# 创建 Agent
agent = (
    SimpleAgent("MathBot")
    .with_llm(OpenAIProvider(model="gpt-4o-mini"))
    .with_tools([calculator])
    .with_system_prompt("你是一个数学助手")
)

# 运行
result = await agent.run("计算 (15 + 27) * 3")
print(result.output)
```

### 使用记忆系统

```python
from agentflow.memory import Memory, SQLiteStore

# 创建带持久化的记忆
memory = Memory(store=SQLiteStore("memory.db"))

agent = (
    SimpleAgent("Assistant")
    .with_llm(llm)
    .with_memory(memory)
)

# Agent 会自动记住对话历史
await agent.run("我叫张三")
await agent.run("我叫什么名字？")  # 会记住之前的信息
```

### 使用 Go 网关

```bash
# 启动网关
cd gateway
make run-all

# 调用 API
curl -X POST http://localhost:8080/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'
```

## 📖 详细文档

### 配置详解

```python
from agentflow import Agent, AgentConfig, LLMConfig, ReasoningPattern
from agentflow.core.config import LLMProvider, MemoryConfig

config = AgentConfig(
    name="MyAgent",
    llm=LLMConfig(
        provider=LLMProvider.OPENAI,  # 或 VLLM, OLLAMA, ANTHROPIC
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=2000,
    ),
    memory=MemoryConfig(
        max_short_term_messages=50,
        enable_long_term=True,
        max_context_tokens=8000,
    ),
    pattern=ReasoningPattern.AUTO,  # 让模型自己选择推理模式
    system_prompt="你是一个专业的AI助手",
    max_iterations=10,
)

async with Agent(config=config) as agent:
    result = await agent.run("复杂任务...")
```

## 推理模式

### ReAct (推理 + 行动)

```python
agent = Agent.quick_start(pattern=ReasoningPattern.REACT)
```

适用于需要与工具交互的任务，交替进行思考和行动。

### Chain-of-Thought (思维链)

```python
agent = Agent.quick_start(pattern=ReasoningPattern.COT)
```

适用于需要逐步推理的问题，如数学计算、逻辑分析。

### Tree-of-Thought (思维树)

```python
agent = Agent.quick_start(pattern=ReasoningPattern.TOT)
```

适用于需要探索多个解决方案的问题，可以回溯和比较。

### Reflexion (反思)

```python
agent = Agent.quick_start(pattern=ReasoningPattern.REFLEXION)
```

适用于需要从失败中学习的任务，包含自我评估和改进。

### Plan & Execute (计划执行)

```python
agent = Agent.quick_start(pattern=ReasoningPattern.PLAN_EXECUTE)
```

适用于复杂的多步骤任务，先制定计划再逐步执行。

### Auto (自动选择)

```python
agent = Agent.quick_start(pattern=ReasoningPattern.AUTO)
```

让模型根据任务类型自动选择最合适的推理模式。

## 工具系统

### 使用内置工具

```python
from agentflow.tools import (
    FileReadTool, FileWriteTool,
    HTTPTool, BrowserTool,
    TerminalTool, PythonExecuteTool,
    WebSearchTool, DatabaseTool,
)

agent.register_tools([
    FileReadTool(),
    HTTPTool(),
    PythonExecuteTool(safe_mode=True),
])
```

### 创建自定义工具

#### 方式1: 使用装饰器

```python
from agentflow.tools import tool

@tool(name="calculator", description="计算数学表达式")
async def calculator(expression: str) -> str:
    return str(eval(expression))

agent.register_tool(calculator())
```

#### 方式2: 继承 BaseTool

```python
from agentflow.tools import BaseTool, ToolResult
from pydantic import BaseModel, Field

class SearchParams(BaseModel):
    query: str = Field(description="搜索关键词")
    limit: int = Field(default=10, description="结果数量")

class MySearchTool(BaseTool):
    name = "my_search"
    description = "自定义搜索工具"
    parameters = SearchParams
  
    async def execute(self, query: str, limit: int = 10) -> ToolResult:
        results = perform_search(query, limit)
        return ToolResult(success=True, output=results)

agent.register_tool(MySearchTool())
```

## 使用 vLLM

支持使用 vLLM 部署的本地模型或微调模型：

```python
from agentflow.core.config import LLMProvider

config = AgentConfig(
    llm=LLMConfig(
        provider=LLMProvider.VLLM,
        model="meta-llama/Llama-2-7b-chat-hf",
        api_base="http://localhost:8000/v1",
    ),
)
```

启动 vLLM 服务：

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --host 0.0.0.0 \
    --port 8000
```

## 使用 Ollama

```python
config = AgentConfig(
    llm=LLMConfig(
        provider=LLMProvider.OLLAMA,
        model="llama2",
        api_base="http://localhost:11434",
    ),
)
```

## 记忆系统

AgentFlow 提供了完整的记忆管理：

- **短期记忆**: 基于队列的最近对话记录
- **长期记忆**: 基于 ChromaDB 的语义检索
- **数据库记忆**: SQLite + FTS5 全文搜索支持
- **会话存储**: 支持多会话管理和历史检索
- **上下文管理**: 自动压缩和优化上下文窗口

```python
from agentflow.memory.database import DatabaseMemory, ConversationStore

# 使用数据库记忆
memory = DatabaseMemory("agent_memory.db")
await memory.add(MemoryEntry(content="重要信息", importance=0.9))
results = await memory.search("关键词")

# 会话存储
store = ConversationStore("conversations.db")
session = await store.create_session(user_id="user_1", agent_name="MyAgent")
await store.add_message(session.id, "user", "你好")
```

## vLLM 模块

独立的 vLLM 模块用于高吞吐量表单处理和结构化输出：

```python
from agentflow.vllm import (
    VLLMClient, VLLMConfig,
    FormProcessor, BatchProcessor,
    FormSchema, FormField, FieldType,
)

# 配置 vLLM 客户端
config = VLLMConfig(
    base_url="http://localhost:8000",
    model="Qwen/Qwen2.5-7B-Instruct",
)
client = VLLMClient(config)

# 表单处理
processor = FormProcessor(client)
invoice_data = await processor.extract_invoice(invoice_text)
receipt_data = await processor.extract_receipt(receipt_text)

# 批量处理
batch_processor = BatchProcessor(client, max_concurrent=4)
results = await batch_processor.process_batch(documents, schema)
```

## 配置文件

支持 JSON 和 YAML 配置文件：

```yaml
# agent_config.yaml
name: MyAgent
llm:
  provider: openai
  model: gpt-4o-mini
  temperature: 0.7
memory:
  max_short_term_messages: 50
  enable_long_term: true
pattern: auto
```

加载配置：

```python
config = AgentConfig.from_file("agent_config.yaml")
agent = Agent(config=config)
```

## 开发

### 运行测试

```bash
# 安装开发依赖
uv pip install -e ".[dev]"

# 运行测试
uv run pytest tests/ -v

# 带覆盖率的测试
uv run pytest tests/ -v --cov=agentflow --cov-report=term-missing

# 运行类型检查
uv run mypy src/agentflow

# 代码格式化
uv run ruff format src/ tests/
uv run ruff check src/ tests/ --fix
```

### 项目结构

```
src/agentflow/
├── core/           # 核心模块
│   ├── agent.py    # Agent 主类
│   ├── config.py   # 配置管理
│   └── message.py  # 消息类型
├── llm/            # LLM 客户端
│   ├── client.py   # 统一客户端接口
│   ├── config_loader.py  # 多场景 LLM 配置加载
│   └── providers/  # 各提供商实现
├── tools/          # 工具系统
│   ├── base.py     # 基础类和装饰器
│   ├── executor.py # 工具执行器
│   └── builtin/    # 内置工具
│       ├── browser.py    # 浏览器自动化
│       ├── terminal.py   # 终端命令执行
│       ├── http.py       # HTTP 请求
│       ├── file.py       # 文件操作
│       ├── code.py       # 代码执行
│       ├── database.py   # 数据库操作
│       ├── search.py     # 网络搜索
│       └── data.py       # JSON/文本处理
├── memory/         # 分层记忆系统
│   ├── hierarchical.py   # 分层记忆（基于论文）
│   ├── consolidation.py  # 记忆整合与反思
│   ├── retrieval.py      # 混合检索系统
│   ├── base.py           # 基础接口（兼容）
│   ├── short_term.py     # 短期记忆
│   ├── long_term.py      # 长期记忆 (ChromaDB)
│   ├── database.py       # 数据库记忆 (SQLite)
│   └── context.py        # 上下文管理
├── vllm/           # vLLM 高吞吐量模块
│   ├── schema.py     # 表单 Schema 定义
│   ├── client.py     # vLLM 客户端
│   └── processor.py  # 表单/批量处理器
└── patterns/       # 推理模式
    ├── react.py
    ├── cot.py
    ├── tot.py
    ├── reflexion.py
    ├── plan_execute.py
    └── auto.py

examples/           # 使用示例
├── chatbot.py        # 交互式聊天机器人
├── qa_system.py      # 知识问答系统
├── form_extraction.py # 表单数据提取
└── task_agent.py     # 多轮任务代理
```

## 记忆系统（论文驱动设计）

AgentFlow 的记忆系统基于最新研究论文设计：

**参考论文：**
- MemGPT (2023): 分层记忆架构、工作记忆管理
- Generative Agents (2023): 反思机制、重要性评估、多维检索
- RecallM (2023): 时间上下文理解、记忆整合
- CoALA (2023): 认知架构、感知-记忆-行动循环

**记忆层级：**

```
┌─────────────────────────────────────────────────────────┐
│                    感知缓冲 (Sensory Buffer)              │
│                    - 最近的原始输入                        │
│                    - 容量有限，FIFO                       │
├─────────────────────────────────────────────────────────┤
│                    工作记忆 (Working Memory)              │
│                    - 核心记忆（角色、用户信息）              │
│                    - 工作上下文（当前对话）                  │
│                    - 对应 LLM 上下文窗口                   │
├─────────────────────────────────────────────────────────┤
│                    情景记忆 (Episodic Memory)             │
│                    - 具体事件和经历                        │
│                    - SQLite + FTS5 持久化                 │
│                    - 支持时间和语义检索                     │
├─────────────────────────────────────────────────────────┤
│                    语义记忆 (Semantic Memory)             │
│                    - 反思生成的抽象知识                     │
│                    - 通用事实和规则                        │
│                    - 从情景记忆中提取                       │
└─────────────────────────────────────────────────────────┘
```

**使用示例：**

```python
from agentflow.memory import (
    HierarchicalMemory,
    MemoryType,
    HybridRetriever,
    MemoryConsolidator,
)

# 创建分层记忆系统
async with HierarchicalMemory(db_path="agent_memory.db") as memory:
    # 记录观察
    await memory.observe(
        "用户提到他们喜欢Python编程",
        memory_type=MemoryType.OBSERVATION,
    )
    
    # 记录行动
    await memory.observe(
        "推荐了几个Python学习资源",
        memory_type=MemoryType.ACTION,
    )
    
    # 回忆相关记忆
    results = await memory.recall("Python学习", limit=5)
    
    # 获取上下文（用于 LLM 提示）
    context = await memory.get_context("用户想学什么")
```

**检索策略：**

```python
from agentflow.memory import HybridRetriever, RetrievalStrategy

retriever = HybridRetriever(episodic, semantic)

# 自动选择策略
results = await retriever.retrieve("昨天我说了什么")

# 指定策略
results = await retriever.retrieve(
    "Python最佳实践",
    strategy=RetrievalStrategy.SEMANTIC,
)

# 时间范围查询
from datetime import datetime, timedelta
results = await retriever.retrieve(
    "会议",
    time_range=(datetime.now() - timedelta(days=7), datetime.now()),
)
```

**记忆整合：**

```python
from agentflow.memory import MemoryConsolidator, ConsolidationConfig

config = ConsolidationConfig(
    compression_threshold=100,  # 超过100条触发压缩
    reflection_threshold=10.0,  # 累计重要性触发反思
    forgetting_rate=0.1,        # 每天10%遗忘率
)

consolidator = MemoryConsolidator(episodic, semantic, llm, config)

# 执行整合（压缩、反思、遗忘）
report = await consolidator.consolidate()
```

## 示例

查看 `examples/` 目录获取更多示例：

- [chatbot.py](examples/chatbot.py) - 交互式聊天机器人（多角色、会话持久化）
- [qa_system.py](examples/qa_system.py) - 知识问答系统（文档索引、上下文检索）
- [form_extraction.py](examples/form_extraction.py) - 表单数据提取（发票、收据、名片）
- [task_agent.py](examples/task_agent.py) - 多轮任务代理（任务分解、多步执行）

### 快速运行示例

```bash
cd examples

# 交互式聊天
uv run python chatbot.py interactive

# 问答系统
uv run python qa_system.py

# 表单处理
uv run python form_extraction.py

# 任务代理
uv run python task_agent.py interactive
```

## License

MIT License
