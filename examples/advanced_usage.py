"""
AgentFlow 高级用法示例
展示记忆系统、自动模式选择等高级功能
"""

import asyncio
from agentflow import Agent, AgentConfig, LLMConfig, ReasoningPattern
from agentflow.core.config import MemoryConfig
from agentflow.tools import tool, BaseTool, ToolResult


async def memory_management():
    """记忆管理示例"""
    config = AgentConfig(
        name="MemoryAgent",
        llm=LLMConfig(model="gpt-4o-mini"),
        memory=MemoryConfig(
            max_short_term_messages=50,  # 短期记忆最大消息数
            enable_long_term=True,       # 启用长期记忆
            max_context_tokens=8000,     # 上下文最大token数
            context_compression=True,    # 启用上下文压缩
        ),
    )
    
    async with Agent(config=config) as agent:
        # 模拟一系列对话
        conversations = [
            "我叫张三，是一名软件工程师",
            "我主要使用Python和Go语言",
            "我正在学习机器学习",
            "你还记得我的名字吗？",
            "我之前说过我会什么编程语言？",
        ]
        
        for msg in conversations:
            print(f"User: {msg}")
            response = await agent.chat(msg)
            print(f"Agent: {response}\n")
        
        # 查看记忆统计
        stats = await agent.get_memory_stats()
        print(f"记忆统计: {stats}")


async def auto_pattern_selection():
    """自动模式选择示例"""
    config = AgentConfig(
        name="AutoAgent",
        llm=LLMConfig(model="gpt-4o-mini"),
        pattern=ReasoningPattern.AUTO,  # 自动选择模式
        verbose=True,  # 显示详细日志
    )
    
    # 不同类型的任务会选择不同的推理模式
    tasks = [
        # 可能选择 CoT（需要逐步推理）
        "如果小明有5个苹果，给了小红2个，又买了3个，最后有多少个？",
        
        # 可能选择 ReAct（需要使用工具）
        "查找并总结最近的人工智能新闻",
        
        # 可能选择 Plan&Execute（复杂多步骤任务）
        "创建一个完整的项目计划：开发一个待办事项应用",
        
        # 可能选择 ToT（需要探索多个方案）
        "如何优化一个响应缓慢的API？列出所有可能的方案",
    ]
    
    async with Agent(config=config) as agent:
        for task in tasks:
            print(f"\n{'='*60}")
            print(f"任务: {task}")
            print('='*60)
            
            result = await agent.run(task)
            print(f"选择的模式: {result.metadata.get('selected_pattern', 'unknown')}")
            print(f"输出: {result.output[:300]}...")


async def custom_system_prompt():
    """自定义系统提示词"""
    config = AgentConfig(
        name="ExpertAgent",
        llm=LLMConfig(model="gpt-4o-mini"),
        system_prompt="""你是一位资深的软件架构师，具有以下特点：
1. 15年以上的软件开发经验
2. 精通分布式系统设计
3. 熟悉各种设计模式和最佳实践
4. 善于用简单的方式解释复杂概念

回答时请：
- 提供实用的建议
- 举具体的例子
- 考虑实际的权衡取舍
""",
    )
    
    async with Agent(config=config) as agent:
        result = await agent.run("如何设计一个高可用的微服务架构？")
        print(result.output)


async def streaming_response():
    """流式响应示例"""
    agent = Agent.quick_start(model="gpt-4o-mini")
    
    print("流式响应：")
    async for chunk in agent.run_stream("写一首关于编程的短诗"):
        print(chunk, end="", flush=True)
    print()
    
    await agent.close()


async def batch_processing():
    """批量处理示例"""
    agent = Agent.quick_start(model="gpt-4o-mini")
    
    questions = [
        "Python中的GIL是什么？",
        "什么是RESTful API？",
        "解释依赖注入的概念",
    ]
    
    # 并发处理多个问题
    tasks = [agent.run(q) for q in questions]
    results = await asyncio.gather(*tasks)
    
    for q, r in zip(questions, results):
        print(f"\nQ: {q}")
        print(f"A: {r.output[:200]}...")
    
    await agent.close()


async def error_handling():
    """错误处理示例"""
    agent = Agent.quick_start(
        model="gpt-4o-mini",
        pattern=ReasoningPattern.REFLEXION,  # 使用反思模式，可以从错误中学习
    )
    
    # 创建一个可能失败的工具
    @tool(name="risky_operation", description="一个可能失败的操作")
    async def risky_operation(should_fail: bool = False) -> str:
        if should_fail:
            raise ValueError("操作失败！")
        return "操作成功！"
    
    agent.register_tool(risky_operation())
    
    # 使用 Reflexion 模式处理可能的失败
    result = await agent.run("执行一个可能失败的操作，如果失败则尝试恢复")
    
    print(f"成功: {result.success}")
    print(f"输出: {result.output}")
    if result.error:
        print(f"错误: {result.error}")
    
    await agent.close()


async def context_window_management():
    """上下文窗口管理"""
    config = AgentConfig(
        llm=LLMConfig(
            model="gpt-4o-mini",
            max_tokens=1000,  # 限制生成token数
        ),
        memory=MemoryConfig(
            max_context_tokens=4000,  # 限制上下文窗口
            context_compression=True,  # 启用压缩
        ),
    )
    
    async with Agent(config=config) as agent:
        # 模拟长对话
        for i in range(20):
            response = await agent.chat(f"这是第{i+1}条消息。请记住这个数字。")
            if i % 5 == 4:
                # 每5条消息检查一次记忆
                stats = await agent.get_memory_stats()
                print(f"对话{i+1}: 短期记忆 {stats['short_term_entries']} 条")
        
        # 测试记忆保持
        response = await agent.chat("我刚才发送了多少条消息？")
        print(f"回答: {response}")


if __name__ == "__main__":
    print("=== 记忆管理 ===")
    asyncio.run(memory_management())
    
    print("\n=== 自动模式选择 ===")
    asyncio.run(auto_pattern_selection())
    
    print("\n=== 自定义系统提示词 ===")
    asyncio.run(custom_system_prompt())
    
    print("\n=== 错误处理 ===")
    asyncio.run(error_handling())
