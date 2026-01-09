"""
知识问答系统示例
================

展示如何使用 AgentFlow 创建一个基于文档的问答系统，支持：
- 文档索引
- 上下文检索
- 答案生成与引用
- 通过 .env 文件配置 LLM

配置文件: .env.qa
"""

import asyncio
import os
from typing import Optional
from dataclasses import dataclass

from agentflow import Agent, AgentConfig, LLMConfig, ReasoningPattern
from agentflow.memory.database import DatabaseMemory
from agentflow.tools import tool, BaseTool, ToolResult
from agentflow.llm.config_loader import LLMConfigLoader, load_llm_config


@dataclass
class Document:
    """文档对象."""
    id: str
    title: str
    content: str
    source: str = ""
    metadata: dict = None


class KnowledgeBase:
    """简单的知识库管理器."""
    
    def __init__(self, db_path: str = "knowledge.db"):
        self.db_path = db_path
        self.memory = DatabaseMemory(db_path, table_name="documents")
        self.documents: dict[str, Document] = {}
    
    async def initialize(self):
        """初始化知识库."""
        await self.memory._ensure_initialized()
    
    async def add_document(self, doc: Document):
        """添加文档到知识库."""
        from agentflow.memory.base import MemoryEntry
        
        # 分割成段落存储
        paragraphs = doc.content.split("\n\n")
        
        for i, para in enumerate(paragraphs):
            if para.strip():
                entry = MemoryEntry(
                    content=para.strip(),
                    role="document",
                    metadata={
                        "doc_id": doc.id,
                        "title": doc.title,
                        "source": doc.source,
                        "paragraph_index": i,
                    },
                    importance=0.8,
                )
                await self.memory.add(entry)
        
        self.documents[doc.id] = doc
        print(f"✓ 添加文档: {doc.title} ({len(paragraphs)} 段落)")
    
    async def search(self, query: str, limit: int = 5) -> list[dict]:
        """搜索相关内容."""
        entries = await self.memory.search(query, limit=limit)
        
        results = []
        for entry in entries:
            results.append({
                "content": entry.content,
                "doc_id": entry.metadata.get("doc_id"),
                "title": entry.metadata.get("title"),
                "source": entry.metadata.get("source"),
            })
        
        return results
    
    async def close(self):
        """关闭连接."""
        await self.memory.close()


class QASystem:
    """问答系统."""
    
    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        env_file: Optional[str] = None,
        knowledge_base: Optional[KnowledgeBase] = None,
    ):
        """初始化问答系统.
        
        Args:
            llm_config: LLM 配置对象（优先级最高）
            env_file: 环境配置文件路径（如 .env.qa）
            knowledge_base: 知识库实例
        """
        # 加载 LLM 配置
        if llm_config:
            self.llm_config = llm_config
        elif env_file:
            self.llm_config = LLMConfigLoader.from_env_file(env_file)
        else:
            self.llm_config = load_llm_config(task="qa")
        
        self.kb = knowledge_base
        self.agent: Agent = None
    
    async def initialize(self):
        """初始化问答系统."""
        config = AgentConfig(
            name="QA-System",
            llm=self.llm_config,
            pattern=ReasoningPattern.COT,
            system_prompt="""你是一个专业的知识问答助手。根据提供的上下文信息回答用户问题。

回答规则：
1. 只根据提供的上下文回答，不要编造信息
2. 如果上下文中没有相关信息，明确告知用户
3. 引用信息来源
4. 答案简洁、准确
5. 如果问题不清楚，请求澄清""",
        )
        
        self.agent = Agent(config=config)
    
    async def answer(self, question: str, context: Optional[str] = None) -> dict:
        """回答问题."""
        # 搜索相关知识
        relevant_docs = []
        if self.kb:
            relevant_docs = await self.kb.search(question, limit=5)
        
        # 构建上下文
        if relevant_docs:
            context_parts = ["相关知识：\n"]
            for i, doc in enumerate(relevant_docs, 1):
                context_parts.append(f"[{i}] 来源: {doc['title']}")
                context_parts.append(f"    {doc['content'][:500]}\n")
            context = "\n".join(context_parts)
        elif context:
            context = f"参考资料：\n{context}"
        else:
            context = "（无相关上下文）"
        
        # 构建提示
        prompt = f"""{context}

问题：{question}

请基于上述信息回答问题。如果引用了具体信息，请标注来源编号。"""
        
        # 获取答案
        result = await self.agent.run(prompt)
        
        return {
            "question": question,
            "answer": result.output,
            "sources": [doc["title"] for doc in relevant_docs],
            "success": result.success,
        }
    
    async def close(self):
        """关闭资源."""
        if self.agent:
            await self.agent.close()
        if self.kb:
            await self.kb.close()


# 示例文档
SAMPLE_DOCUMENTS = [
    Document(
        id="python_basics",
        title="Python 基础教程",
        source="Python官方文档",
        content="""Python 是一种易于学习又功能强大的编程语言。它提供了高效的高级数据结构，还能简单有效地面向对象编程。

Python 解释器易于扩展，可以使用 C 或 C++（或者其他可以通过 C 调用的语言）扩展新的功能和数据类型。

Python 的设计哲学强调代码的可读性和简洁的语法。相比 C++ 或 Java，Python 让开发者能够用更少的代码表达想法。

Python 支持多种编程范式，包括面向对象、命令式、函数式和过程式编程。它拥有动态类型系统和自动内存管理功能。

Python 的标准库非常广泛，提供了适用于各种任务的模块，包括网络编程、文本处理、数据库接口等。"""
    ),
    Document(
        id="python_types",
        title="Python 数据类型",
        source="Python官方文档",
        content="""Python 有几种内置数据类型：

数字类型：
- int：整数，如 1, 2, 3
- float：浮点数，如 1.0, 2.5
- complex：复数，如 1+2j

序列类型：
- str：字符串，如 "hello"
- list：列表，如 [1, 2, 3]
- tuple：元组，如 (1, 2, 3)

映射类型：
- dict：字典，如 {"key": "value"}

集合类型：
- set：集合，如 {1, 2, 3}
- frozenset：不可变集合

布尔类型：
- bool：True 或 False"""
    ),
    Document(
        id="python_functions",
        title="Python 函数定义",
        source="Python官方文档",
        content="""Python 使用 def 关键字定义函数：

```python
def greet(name):
    return f"Hello, {name}!"
```

函数可以有默认参数：
```python
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"
```

可以使用 *args 接收任意数量的位置参数：
```python
def sum_all(*args):
    return sum(args)
```

可以使用 **kwargs 接收任意数量的关键字参数：
```python
def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")
```

Lambda 表达式用于创建匿名函数：
```python
square = lambda x: x ** 2
```"""
    ),
]


async def demo_qa_system(env_file: Optional[str] = None):
    """演示问答系统.
    
    Args:
        env_file: 可选的环境配置文件路径
    """
    print("=" * 60)
    print("AgentFlow 知识问答系统")
    print("=" * 60)
    
    # 显示当前 LLM 配置
    llm_config = load_llm_config(env_file=env_file) if env_file else load_llm_config(task="qa")
    print(f"\n当前 LLM 配置:")
    print(f"  Provider: {llm_config.provider.value}")
    print(f"  Model: {llm_config.model}")
    print(f"  Temperature: {llm_config.temperature}")
    
    # 初始化知识库
    kb = KnowledgeBase()
    await kb.initialize()
    
    # 添加示例文档
    print("\n正在加载知识库...")
    for doc in SAMPLE_DOCUMENTS:
        await kb.add_document(doc)
    
    # 初始化问答系统
    qa = QASystem(llm_config=llm_config, knowledge_base=kb)
    await qa.initialize()
    
    # 测试问题
    questions = [
        "Python 是什么语言？有什么特点？",
        "Python 有哪些数据类型？",
        "如何在 Python 中定义函数？",
        "Python 支持面向对象编程吗？",
        "什么是 lambda 表达式？",
    ]
    
    print("\n" + "=" * 60)
    print("问答演示")
    print("=" * 60)
    
    for question in questions:
        print(f"\n问: {question}")
        result = await qa.answer(question)
        print(f"答: {result['answer']}")
        if result['sources']:
            print(f"来源: {', '.join(result['sources'])}")
        print("-" * 40)
    
    await qa.close()


async def interactive_qa(env_file: Optional[str] = None):
    """交互式问答.
    
    Args:
        env_file: 可选的环境配置文件路径
    """
    llm_config = load_llm_config(env_file=env_file) if env_file else load_llm_config(task="qa")
    
    kb = KnowledgeBase()
    await kb.initialize()
    
    # 加载文档
    for doc in SAMPLE_DOCUMENTS:
        await kb.add_document(doc)
    
    qa = QASystem(llm_config=llm_config, knowledge_base=kb)
    await qa.initialize()
    
    print("\n知识问答系统已就绪，输入问题或 'quit' 退出\n")
    
    while True:
        try:
            question = input("问: ").strip()
            if not question:
                continue
            if question.lower() == "quit":
                break
            
            result = await qa.answer(question)
            print(f"答: {result['answer']}\n")
            
        except KeyboardInterrupt:
            break
    
    await qa.close()


async def qa_with_custom_docs(env_file: Optional[str] = None):
    """使用自定义文档的问答.
    
    Args:
        env_file: 可选的环境配置文件路径
    """
    llm_config = load_llm_config(env_file=env_file) if env_file else load_llm_config(task="qa")
    
    kb = KnowledgeBase("custom_knowledge.db")
    await kb.initialize()
    
    # 添加自定义文档
    custom_doc = Document(
        id="company_policy",
        title="公司休假政策",
        source="员工手册2024",
        content="""年假政策：
        
工作满1年不满10年的，年休假5天；
工作满10年不满20年的，年休假10天；
工作满20年的，年休假15天。

病假政策：

员工因病需要治疗的，凭医院证明可请病假。病假期间工资按以下标准发放：
- 病假在2个月以内的，发放工资的60%
- 病假超过2个月的，发放工资的40%

请假流程：

1. 提前3天在系统中提交请假申请
2. 直接上级审批
3. 超过3天需要部门经理审批
4. 审批通过后方可休假"""
    )
    
    await kb.add_document(custom_doc)
    
    qa = QASystem(llm_config=llm_config, knowledge_base=kb)
    await qa.initialize()
    
    # 测试问答
    questions = [
        "年假有多少天？",
        "病假工资怎么算？",
        "请假需要走什么流程？",
    ]
    
    for q in questions:
        result = await qa.answer(q)
        print(f"Q: {q}")
        print(f"A: {result['answer']}\n")
    
    await qa.close()


if __name__ == "__main__":
    import sys
    
    # 解析命令行参数
    env_file = None
    mode = "demo"
    
    for arg in sys.argv[1:]:
        if arg.startswith("--env="):
            env_file = arg.split("=", 1)[1]
        elif arg in ("interactive", "custom", "demo"):
            mode = arg
    
    print(f"使用配置文件: {env_file or '.env.qa (默认)'}")
    
    if mode == "interactive":
        asyncio.run(interactive_qa(env_file))
    elif mode == "custom":
        asyncio.run(qa_with_custom_docs(env_file))
    else:
        asyncio.run(demo_qa_system(env_file))
