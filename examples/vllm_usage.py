"""
AgentFlow 使用 vLLM 示例
展示如何配置和使用 vLLM 作为后端
"""

import asyncio
from agentflow import Agent, AgentConfig, LLMConfig, ReasoningPattern
from agentflow.core.config import LLMProvider


async def vllm_basic():
    """使用 vLLM 的基础示例"""
    # 配置 vLLM
    config = AgentConfig(
        name="vLLM-Agent",
        llm=LLMConfig(
            provider=LLMProvider.VLLM,
            model="meta-llama/Llama-2-7b-chat-hf",  # 或你微调的模型
            api_base="http://localhost:8000/v1",  # vLLM 服务地址
            temperature=0.7,
            max_tokens=2000,
        ),
        pattern=ReasoningPattern.AUTO,  # 让模型自己选择推理模式
    )
    
    async with Agent(config=config) as agent:
        result = await agent.run("解释什么是机器学习中的过拟合问题？")
        print(result.output)


async def vllm_with_custom_model():
    """使用微调后的自定义模型"""
    config = AgentConfig(
        name="CustomModel-Agent",
        llm=LLMConfig(
            provider=LLMProvider.VLLM,
            model="/path/to/your/finetuned/model",  # 本地微调模型路径
            api_base="http://localhost:8000/v1",
            temperature=0.5,
            # vLLM 特定参数
            extra_params={
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "stop": ["Human:", "User:"],
            },
        ),
        pattern=ReasoningPattern.REACT,
        system_prompt="你是一个专业的代码助手。",
    )
    
    async with Agent(config=config) as agent:
        result = await agent.run("写一个Python函数来实现快速排序算法")
        print(result.output)


async def ollama_example():
    """使用 Ollama 的示例（类似配置）"""
    config = AgentConfig(
        name="Ollama-Agent",
        llm=LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama2",  # 或其他 Ollama 支持的模型
            api_base="http://localhost:11434",
            temperature=0.7,
        ),
        pattern=ReasoningPattern.COT,
    )
    
    async with Agent(config=config) as agent:
        result = await agent.run("用简单的语言解释量子计算的基本原理")
        print(result.output)


async def multi_provider_comparison():
    """对比不同模型提供商的效果"""
    providers = [
        (LLMProvider.OPENAI, "gpt-4o-mini", None),
        (LLMProvider.VLLM, "meta-llama/Llama-2-7b-chat-hf", "http://localhost:8000/v1"),
        (LLMProvider.OLLAMA, "llama2", "http://localhost:11434"),
    ]
    
    question = "解释Python中的装饰器是什么，给出一个简单例子"
    
    for provider, model, api_base in providers:
        print(f"\n{'='*50}")
        print(f"提供商: {provider.value}, 模型: {model}")
        print('='*50)
        
        try:
            config = AgentConfig(
                llm=LLMConfig(
                    provider=provider,
                    model=model,
                    api_base=api_base,
                ),
            )
            
            async with Agent(config=config) as agent:
                result = await agent.run(question)
                # 显示前500个字符
                output = result.output[:500] + "..." if len(result.output) > 500 else result.output
                print(output)
                
        except Exception as e:
            print(f"错误: {e}")


if __name__ == "__main__":
    print("=== vLLM 基础示例 ===")
    # asyncio.run(vllm_basic())
    
    print("\n注意: 运行这些示例前，请确保：")
    print("1. vLLM 服务已启动: python -m vllm.entrypoints.openai.api_server --model <model_name>")
    print("2. 或 Ollama 已安装并运行: ollama serve")
    print("3. 修改示例中的模型名称和API地址")
    
    # 取消注释以运行示例:
    # asyncio.run(vllm_basic())
    # asyncio.run(vllm_with_custom_model())
    # asyncio.run(ollama_example())
    # asyncio.run(multi_provider_comparison())
