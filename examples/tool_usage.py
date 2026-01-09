"""
AgentFlow 工具使用示例
展示如何创建和使用工具
"""

import asyncio
from agentflow import Agent, AgentConfig, LLMConfig, ReasoningPattern
from agentflow.tools import (
    BaseTool,
    tool,
    ToolResult,
    FileReadTool,
    FileWriteTool,
    HTTPTool,
    PythonExecuteTool,
    TerminalTool,
)
from pydantic import BaseModel, Field


# ============ 使用装饰器创建工具 ============

@tool(name="calculator", description="执行基本数学计算")
async def calculator(expression: str) -> str:
    """
    计算数学表达式
    
    Args:
        expression: 要计算的数学表达式，如 "2 + 3 * 4"
    """
    try:
        # 安全的表达式计算
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return f"错误: 表达式包含不允许的字符"
        
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


@tool(name="word_counter", description="统计文本的字数")
async def word_counter(text: str) -> dict:
    """
    统计文本信息
    
    Args:
        text: 要统计的文本
    """
    return {
        "characters": len(text),
        "words": len(text.split()),
        "lines": len(text.splitlines()),
    }


# ============ 使用类创建工具 ============

class WeatherParams(BaseModel):
    """天气查询参数"""
    city: str = Field(description="城市名称")
    days: int = Field(default=1, description="预报天数", ge=1, le=7)


class WeatherTool(BaseTool):
    """天气查询工具"""
    
    name = "weather"
    description = "查询指定城市的天气预报"
    parameters = WeatherParams
    category = "info"
    
    async def execute(self, city: str, days: int = 1) -> ToolResult:
        # 这里模拟天气API调用
        weather_data = {
            "city": city,
            "forecast": [
                {"day": i + 1, "temp": 20 + i, "condition": "晴朗"}
                for i in range(days)
            ],
        }
        return ToolResult(success=True, output=weather_data)


class DatabaseQueryParams(BaseModel):
    """数据库查询参数"""
    table: str = Field(description="表名")
    query: str = Field(description="SQL查询条件")


class MockDatabaseTool(BaseTool):
    """模拟数据库查询工具"""
    
    name = "database_query"
    description = "查询数据库中的数据"
    parameters = DatabaseQueryParams
    category = "database"
    
    async def execute(self, table: str, query: str) -> ToolResult:
        # 模拟查询结果
        results = [
            {"id": 1, "name": "Alice", "age": 30},
            {"id": 2, "name": "Bob", "age": 25},
        ]
        return ToolResult(
            success=True,
            output={"table": table, "rows": results, "count": len(results)},
        )


async def custom_tools_example():
    """使用自定义工具"""
    agent = Agent.quick_start(
        model="gpt-4o-mini",
        pattern=ReasoningPattern.REACT,
    )
    
    # 注册工具
    agent.register_tools([
        calculator(),
        word_counter(),
        WeatherTool(),
        MockDatabaseTool(),
    ])
    
    print("可用工具:", agent.get_available_tools())
    
    # 使用工具执行任务
    tasks = [
        "计算 (15 + 27) * 3",
        "统计这段文字的字数：Python是一门优雅的编程语言，适合初学者学习",
        "查询北京未来3天的天气",
    ]
    
    for task in tasks:
        print(f"\n任务: {task}")
        result = await agent.run(task)
        print(f"结果: {result.output}")
    
    await agent.close()


async def builtin_tools_example():
    """使用内置工具"""
    agent = Agent.quick_start(
        model="gpt-4o-mini",
        pattern=ReasoningPattern.REACT,
    )
    
    # 注册内置工具
    agent.register_tools([
        FileReadTool(),
        HTTPTool(),
        PythonExecuteTool(safe_mode=True),
    ])
    
    # 执行Python代码任务
    result = await agent.run(
        "计算斐波那契数列的前10个数字，使用Python代码"
    )
    print(f"Python执行结果: {result.output}")
    
    await agent.close()


async def tool_with_context():
    """工具配合上下文使用"""
    agent = Agent.quick_start(
        model="gpt-4o-mini",
        pattern=ReasoningPattern.PLAN_EXECUTE,  # 使用计划执行模式
    )
    
    # 注册工具
    agent.register_tools([
        calculator(),
        WeatherTool(),
    ])
    
    # 复杂任务，需要多步骤执行
    task = """
    请完成以下任务：
    1. 查询上海的天气
    2. 如果温度超过25度，计算需要多少瓶水（假设每度需要0.1瓶水）
    3. 给出最终建议
    """
    
    result = await agent.run(task)
    print(result.output)
    
    await agent.close()


if __name__ == "__main__":
    print("=== 自定义工具示例 ===")
    asyncio.run(custom_tools_example())
    
    print("\n=== 内置工具示例 ===")
    asyncio.run(builtin_tools_example())
    
    print("\n=== 工具配合上下文 ===")
    asyncio.run(tool_with_context())
