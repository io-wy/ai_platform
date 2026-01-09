"""
记忆系统使用示例
================

演示如何使用基于论文的分层记忆系统。

依赖：
    uv pip install aiosqlite
"""

import asyncio
from datetime import datetime, timezone, timedelta

from agentflow.memory import (
    # 分层记忆
    HierarchicalMemory,
    MemoryEntry,
    MemoryType,
    
    # 检索
    HybridRetriever,
    RetrievalStrategy,
    RetrievalConfig,
    
    # 整合
    MemoryConsolidator,
    ConsolidationConfig,
)


async def demo_basic_memory():
    """基础记忆操作演示."""
    print("\n" + "="*60)
    print("1. 基础记忆操作")
    print("="*60)
    
    async with HierarchicalMemory(db_path="demo_memory.db") as memory:
        # 清空之前的数据
        await memory.episodic.clear()
        
        # 记录观察
        print("\n添加观察...")
        entry1 = await memory.observe(
            "用户说他是一名Python开发者，有3年经验",
            memory_type=MemoryType.OBSERVATION,
            importance=0.8,
        )
        print(f"  -> 添加: {entry1.content[:50]}...")
        
        entry2 = await memory.observe(
            "用户对机器学习和深度学习很感兴趣",
            memory_type=MemoryType.OBSERVATION,
            importance=0.7,
        )
        print(f"  -> 添加: {entry2.content[:50]}...")
        
        # 记录行动
        entry3 = await memory.observe(
            "推荐了FastAPI框架用于构建API服务",
            memory_type=MemoryType.ACTION,
            importance=0.6,
        )
        print(f"  -> 添加: {entry3.content[:50]}...")
        
        # 记录思考
        entry4 = await memory.observe(
            "用户可能需要学习TensorFlow或PyTorch来开始机器学习",
            memory_type=MemoryType.THOUGHT,
            importance=0.65,
        )
        print(f"  -> 添加: {entry4.content[:50]}...")
        
        # 查询记忆数量
        count = await memory.episodic.count()
        print(f"\n当前记忆总数: {count}")
        
        # 回忆相关记忆
        print("\n检索与'机器学习'相关的记忆...")
        results = await memory.recall("机器学习", limit=5)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r.entry.memory_type.value}] {r.entry.content}")
            print(f"     分数: {r.score:.3f}")


async def demo_working_memory():
    """工作记忆演示."""
    print("\n" + "="*60)
    print("2. 工作记忆（上下文管理）")
    print("="*60)
    
    async with HierarchicalMemory(db_path="demo_memory.db") as memory:
        # 设置核心记忆
        memory.working.set_core("persona", "你是一个专业的AI编程助手，擅长Python和机器学习")
        memory.working.set_core("user_info", "用户：Python开发者，3年经验，对ML感兴趣")
        
        print("\n核心记忆已设置:")
        print(f"  角色: {memory.working.get_core('persona')}")
        print(f"  用户信息: {memory.working.get_core('user_info')}")
        
        # 模拟对话
        conversations = [
            "我想学习如何使用PyTorch",
            "有什么入门教程推荐吗",
            "我主要想做计算机视觉相关的项目",
        ]
        
        print("\n模拟对话:")
        for msg in conversations:
            entry = MemoryEntry(
                content=f"用户: {msg}",
                memory_type=MemoryType.CONVERSATION,
            )
            memory.working.add_to_context(entry)
            print(f"  用户: {msg}")
        
        # 生成上下文提示
        print("\n生成的上下文提示:")
        print("-" * 40)
        prompt = memory.working.to_prompt()
        print(prompt)
        print("-" * 40)


async def demo_retrieval_strategies():
    """检索策略演示."""
    print("\n" + "="*60)
    print("3. 多策略检索")
    print("="*60)
    
    async with HierarchicalMemory(db_path="demo_memory.db") as memory:
        # 添加更多测试数据
        test_data = [
            ("今天学习了Python装饰器的高级用法", MemoryType.OBSERVATION, 0.7),
            ("昨天完成了API接口的开发工作", MemoryType.ACTION, 0.8),
            ("计划下周开始学习Kubernetes", MemoryType.PLAN, 0.6),
            ("Python的GIL限制了多线程的性能", MemoryType.THOUGHT, 0.75),
            ("用户提到他们的项目需要处理大量并发请求", MemoryType.OBSERVATION, 0.85),
        ]
        
        print("\n添加测试数据...")
        for content, mem_type, importance in test_data:
            await memory.observe(content, memory_type=mem_type, importance=importance)
            print(f"  + {content[:40]}...")
        
        # 创建混合检索器
        retriever = HybridRetriever(
            memory.episodic,
            memory.semantic,
            config=RetrievalConfig(
                default_limit=5,
                alpha=1.0,  # 时间权重
                beta=1.0,   # 重要性权重
                gamma=1.5,  # 语义相关性权重
            ),
        )
        
        # 测试不同查询
        queries = [
            ("Python学习", None),  # 自动选择策略
            ("昨天做了什么", RetrievalStrategy.TEMPORAL),  # 时间策略
            ("并发和性能", RetrievalStrategy.KEYWORD),  # 关键词策略
        ]
        
        for query, strategy in queries:
            strategy_name = strategy.value if strategy else "auto"
            print(f"\n查询: '{query}' (策略: {strategy_name})")
            
            results = await retriever.retrieve(query, strategy=strategy)
            
            for i, r in enumerate(results[:3], 1):
                print(f"  {i}. {r.entry.content[:50]}...")
                print(f"     类型: {r.entry.memory_type.value}, 分数: {r.score:.3f}")


async def demo_memory_consolidation():
    """记忆整合演示."""
    print("\n" + "="*60)
    print("4. 记忆整合（反思与压缩）")
    print("="*60)
    
    # 注意：完整的整合需要 LLM 支持
    # 这里演示配置和基本流程
    
    print("\n记忆整合配置:")
    config = ConsolidationConfig(
        compression_threshold=100,
        compression_ratio=0.5,
        reflection_threshold=10.0,
        forgetting_rate=0.1,
        min_importance=0.2,
    )
    
    print(f"  压缩阈值: {config.compression_threshold} 条")
    print(f"  压缩比例: {config.compression_ratio}")
    print(f"  反思阈值: {config.reflection_threshold}")
    print(f"  遗忘率: {config.forgetting_rate}/天")
    print(f"  最小重要性: {config.min_importance}")
    
    async with HierarchicalMemory(db_path="demo_memory.db") as memory:
        # 不使用 LLM 的情况下只执行遗忘
        consolidator = MemoryConsolidator(
            memory.episodic,
            memory.semantic,
            llm=None,  # 无 LLM 时不会生成反思
            config=config,
        )
        
        print("\n执行遗忘检查（基于重要性和时间）...")
        forgotten = await consolidator.forget_irrelevant()
        print(f"  遗忘了 {forgotten} 条不重要的记忆")


async def demo_context_for_llm():
    """生成 LLM 上下文演示."""
    print("\n" + "="*60)
    print("5. 为 LLM 生成上下文")
    print("="*60)
    
    async with HierarchicalMemory(db_path="demo_memory.db") as memory:
        # 设置核心记忆
        memory.working.set_core(
            "persona",
            "你是一个专业的AI助手，帮助用户解决编程问题"
        )
        memory.working.set_core(
            "instructions",
            "优先使用Python示例，保持回答简洁"
        )
        
        # 获取综合上下文
        query = "如何处理高并发"
        context = await memory.get_context(query)
        
        print(f"\n查询: '{query}'")
        print("\n生成的上下文:")
        print("-" * 40)
        print(context)
        print("-" * 40)
        
        # 构建完整提示示例
        print("\n完整的 LLM 提示模板:")
        print("-" * 40)
        full_prompt = f"""
{context}

[用户问题]
{query}

请基于上述上下文和记忆回答用户问题。
"""
        print(full_prompt)


async def demo_importance_scoring():
    """重要性评分演示."""
    print("\n" + "="*60)
    print("6. 重要性评分机制")
    print("="*60)
    
    async with HierarchicalMemory(db_path="demo_memory.db") as memory:
        # 测试不同内容的自动重要性评估
        test_contents = [
            "今天天气不错",  # 低重要性
            "用户说这个功能非常重要！",  # 高重要性（关键词 + 感叹号）
            "需要记住这个关键的API密钥配置",  # 高重要性（关键词）
            "随便聊聊",  # 低重要性
            "这是一个关于项目架构的重要决定",  # 高重要性
        ]
        
        print("\n自动重要性评估:")
        for content in test_contents:
            importance = memory._estimate_importance(content)
            print(f"  {importance:.2f} | {content}")


async def main():
    """运行所有演示."""
    print("="*60)
    print("AgentFlow 记忆系统演示")
    print("基于 MemGPT、Generative Agents、RecallM 等论文设计")
    print("="*60)
    
    await demo_basic_memory()
    await demo_working_memory()
    await demo_retrieval_strategies()
    await demo_memory_consolidation()
    await demo_context_for_llm()
    await demo_importance_scoring()
    
    print("\n" + "="*60)
    print("演示完成！")
    print("="*60)
    
    # 清理演示数据库
    import os
    for f in ["demo_memory_episodic.db", "demo_memory_semantic.db"]:
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    asyncio.run(main())
