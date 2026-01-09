"""
AgentFlow 示例集
================

本文件夹包含多个使用 AgentFlow 框架的真实场景示例。

LLM 配置解耦
------------

所有示例都支持通过 .env 文件配置 LLM，实现不同任务使用不同模型：

配置文件列表:
- .env.chatbot  - 对话机器人配置
- .env.qa       - 问答系统配置
- .env.form     - 表单处理配置（推荐 vLLM）
- .env.planner  - 任务规划器配置
- .env.executor - 任务执行器配置

配置文件格式:
```
LLM_PROVIDER=openai          # openai, vllm, ollama, anthropic
LLM_MODEL=gpt-4o-mini        # 模型名称
LLM_API_KEY=your_key         # API 密钥
LLM_API_BASE=https://...     # API 地址
LLM_TEMPERATURE=0.7          # 温度参数
LLM_MAX_TOKENS=2000          # 最大 token 数
```

示例列表
--------

1. chatbot.py - 交互式聊天机器人
   - 多种预设角色（助手、编程专家、创意写手、商业分析师）
   - 会话持久化存储
   - 历史记录查看和搜索
   - 运行方式：
     python chatbot.py [interactive|batch|resume] --env=.env.chatbot

2. qa_system.py - 知识问答系统
   - 文档索引和检索
   - 基于上下文的答案生成
   - 来源引用
   - 运行方式：
     python qa_system.py [interactive|custom|demo] --env=.env.qa

3. form_extraction.py - 表单数据提取
   - 发票信息提取
   - 收据数据解析
   - 名片信息提取
   - 自定义表单处理
   - 批量文档处理
   - 运行方式：
     python form_extraction.py --env=.env.form

4. task_agent.py - 多轮任务代理
   - 复杂任务分解
   - 多步骤执行（规划器和执行器可使用不同模型）
   - 状态追踪
   - 工具协调
   - 运行方式：
     python task_agent.py [interactive|demo] --planner=.env.planner --executor=.env.executor

快速开始
--------

1. 安装依赖：
   pip install -e .

2. 配置环境变量（复制并修改配置文件）：
   cp .env.chatbot.example .env.chatbot
   # 编辑 .env.chatbot 填入你的 API 密钥

3. 运行示例：
   cd examples
   python chatbot.py --env=../.env.chatbot

框架特性
--------

- Agent: 核心代理类，支持多种推理模式
- Memory: 数据库支持的记忆系统（SQLite + FTS5）
- Tools: 可扩展的工具系统
- vLLM: 高吞吐量推理模块
- LLMConfigLoader: 灵活的 LLM 配置加载器
"""

from .chatbot import ChatBot
from .qa_system import QASystem, KnowledgeBase
from .task_agent import TaskAgent, TaskPlanner, TaskExecutor

__all__ = [
    "ChatBot",
    "QASystem",
    "KnowledgeBase",
    "TaskAgent",
    "TaskPlanner",
    "TaskExecutor",
]
