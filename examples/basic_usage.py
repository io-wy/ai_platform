"""
AgentFlow 基础使用示例
展示如何创建和使用基本的 Agent
"""

import asyncio
from agentflow import Agent, AgentConfig, LLMConfig, ReasoningPattern


async def basic_usage():
    """基础用法示例"""
    # 方式1: 快速创建
    agent = Agent.quick_start(model="gpt-4o-mini")
    
    response = await agent.chat("你好！请介绍一下你自己。")
    print(f"Agent: {response}")
    
    await agent.close()


async def with_config():
    """使用配置创建 Agent"""
    config = AgentConfig(
        name="MyAssistant",
        llm=LLMConfig(
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=2000,
        ),
        pattern=ReasoningPattern.REACT,
        system_prompt="你是一个友好、专业的AI助手。",
        max_iterations=10,
        verbose=True,
    )
    
    async with Agent(config=config) as agent:
        result = await agent.run("分析一下今天的天气对户外活动的影响")
        
        print(f"成功: {result.success}")
        print(f"输出: {result.output}")
        print(f"迭代次数: {result.iterations}")


async def conversation():
    """多轮对话示例"""
    agent = Agent.quick_start(
        model="gpt-4o-mini",
        pattern=ReasoningPattern.COT,  # 使用思维链模式
    )
    
    messages = [
        "我想学习Python编程",
        "有什么好的学习资源推荐吗？",
        "对于初学者，你建议从哪个项目开始？",
    ]
    
    for msg in messages:
        print(f"User: {msg}")
        response = await agent.chat(msg)
        print(f"Agent: {response}\n")
    
    await agent.close()


async def with_different_patterns():
    """使用不同推理模式"""
    patterns = [
        (ReasoningPattern.COT, "思维链"),
        (ReasoningPattern.REACT, "ReAct"),
        (ReasoningPattern.TOT, "思维树"),
    ]
    
    question = "如果一个房间有3盏灯，外面有3个开关，你只能进入房间一次，如何确定每个开关对应哪盏灯？"
    
    for pattern, name in patterns:
        print(f"\n{'='*50}")
        print(f"使用 {name} 模式:")
        print('='*50)
        
        agent = Agent.quick_start(
            model="gpt-4o-mini",
            pattern=pattern,
        )
        
        result = await agent.run(question)
        print(result.output[:500] + "..." if len(result.output) > 500 else result.output)
        
        await agent.close()


if __name__ == "__main__":
    print("=== 基础用法 ===")
    asyncio.run(basic_usage())
    
    print("\n=== 配置方式 ===")
    asyncio.run(with_config())
    
    print("\n=== 多轮对话 ===")
    asyncio.run(conversation())
    
    print("\n=== 不同推理模式 ===")
    asyncio.run(with_different_patterns())
