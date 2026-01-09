"""
多轮任务代理示例
================

展示如何使用 AgentFlow 创建能够处理复杂多步骤任务的代理，支持：
- 任务分解
- 多步骤执行
- 状态追踪
- 错误恢复
- 工具协调
- 通过 .env 文件配置 LLM（规划器和执行器可使用不同模型）

配置文件: 
- .env.planner - 任务规划器配置
- .env.executor - 任务执行器配置
"""

import asyncio
import json
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any

from agentflow import Agent, AgentConfig, LLMConfig, ReasoningPattern
from agentflow.memory.database import DatabaseMemory, ConversationStore
from agentflow.tools import tool, BaseTool, ToolResult
from agentflow.llm.config_loader import LLMConfigLoader, load_llm_config


class TaskStatus(Enum):
    """任务状态."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class SubTask:
    """子任务."""
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    dependencies: list[str] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class Task:
    """主任务."""
    id: str
    goal: str
    subtasks: list[SubTask] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    context: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class TaskPlanner:
    """任务规划器 - 将复杂任务分解为子任务."""
    
    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        env_file: Optional[str] = None,
    ):
        """初始化任务规划器.
        
        Args:
            llm_config: LLM 配置对象（优先级最高）
            env_file: 环境配置文件路径（如 .env.planner）
        """
        # 加载 LLM 配置
        if llm_config:
            self.llm_config = llm_config
        elif env_file:
            self.llm_config = LLMConfigLoader.from_env_file(env_file)
        else:
            self.llm_config = load_llm_config(task="planner")
        
        self.agent: Agent = None
    
    async def initialize(self):
        """初始化规划器."""
        config = AgentConfig(
            name="TaskPlanner",
            llm=self.llm_config,
            pattern=ReasoningPattern.COT,
            system_prompt="""你是一个任务规划专家。你的职责是将复杂任务分解为可执行的子任务。

分解规则：
1. 每个子任务应该是独立可执行的
2. 明确子任务之间的依赖关系
3. 子任务描述要具体、可操作
4. 按照执行顺序排列
5. 考虑可能的并行执行

输出格式（JSON）：
{
    "subtasks": [
        {
            "id": "task_1",
            "description": "具体任务描述",
            "dependencies": []  // 依赖的任务ID列表
        }
    ]
}""",
        )
        self.agent = Agent(config=config)
    
    async def plan(self, goal: str, context: dict = None) -> list[SubTask]:
        """规划任务."""
        prompt = f"请将以下目标分解为具体的子任务：\n\n目标：{goal}"
        if context:
            prompt += f"\n\n上下文信息：{json.dumps(context, ensure_ascii=False)}"
        
        prompt += "\n\n请以 JSON 格式输出子任务列表。"
        
        result = await self.agent.run(prompt)
        
        # 解析结果
        try:
            # 提取 JSON
            output = result.output
            if "```json" in output:
                output = output.split("```json")[1].split("```")[0]
            elif "```" in output:
                output = output.split("```")[1].split("```")[0]
            
            data = json.loads(output)
            subtasks = []
            
            for item in data.get("subtasks", []):
                subtask = SubTask(
                    id=item["id"],
                    description=item["description"],
                    dependencies=item.get("dependencies", []),
                )
                subtasks.append(subtask)
            
            return subtasks
        except Exception as e:
            # 如果解析失败，返回单一任务
            return [SubTask(id="task_1", description=goal)]
    
    async def close(self):
        """关闭资源."""
        if self.agent:
            await self.agent.close()


class TaskExecutor:
    """任务执行器 - 执行具体子任务."""
    
    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        env_file: Optional[str] = None,
    ):
        """初始化任务执行器.
        
        Args:
            llm_config: LLM 配置对象（优先级最高）
            env_file: 环境配置文件路径（如 .env.executor）
        """
        # 加载 LLM 配置
        if llm_config:
            self.llm_config = llm_config
        elif env_file:
            self.llm_config = LLMConfigLoader.from_env_file(env_file)
        else:
            self.llm_config = load_llm_config(task="executor")
        
        self.agent: Agent = None
        self.tools: list[BaseTool] = []
    
    async def initialize(self, tools: list[BaseTool] = None):
        """初始化执行器."""
        self.tools = tools or []
        
        config = AgentConfig(
            name="TaskExecutor",
            llm=self.llm_config,
            pattern=ReasoningPattern.REACT,
            system_prompt="""你是一个任务执行专家。你的职责是执行具体的子任务并返回结果。

执行规则：
1. 仔细理解任务要求
2. 使用可用的工具完成任务
3. 如果无法完成，说明原因
4. 返回清晰的执行结果

可用工具将根据任务类型动态提供。""",
        )
        
        self.agent = Agent(config=config, tools=self.tools)
    
    async def execute(self, subtask: SubTask, context: dict = None) -> dict:
        """执行子任务."""
        prompt = f"请执行以下任务：\n\n任务：{subtask.description}"
        if context:
            prompt += f"\n\n相关上下文：{json.dumps(context, ensure_ascii=False)}"
        
        subtask.status = TaskStatus.IN_PROGRESS
        subtask.started_at = datetime.now()
        
        try:
            result = await self.agent.run(prompt)
            
            subtask.status = TaskStatus.COMPLETED
            subtask.completed_at = datetime.now()
            subtask.result = result.output
            
            return {
                "success": True,
                "output": result.output,
                "steps": len(result.steps) if result.steps else 0,
            }
        except Exception as e:
            subtask.status = TaskStatus.FAILED
            subtask.error = str(e)
            
            return {
                "success": False,
                "error": str(e),
            }
    
    async def close(self):
        """关闭资源."""
        if self.agent:
            await self.agent.close()


class TaskAgent:
    """多轮任务代理 - 协调规划和执行."""
    
    def __init__(
        self,
        planner_config: Optional[LLMConfig] = None,
        executor_config: Optional[LLMConfig] = None,
        planner_env_file: Optional[str] = None,
        executor_env_file: Optional[str] = None,
        db_path: str = "task_agent.db",
    ):
        """初始化任务代理.
        
        Args:
            planner_config: 规划器 LLM 配置
            executor_config: 执行器 LLM 配置
            planner_env_file: 规划器环境配置文件（如 .env.planner）
            executor_env_file: 执行器环境配置文件（如 .env.executor）
            db_path: 数据库路径
        """
        self.planner_config = planner_config
        self.executor_config = executor_config
        self.planner_env_file = planner_env_file
        self.executor_env_file = executor_env_file
        self.db_path = db_path
        
        self.planner: TaskPlanner = None
        self.executor: TaskExecutor = None
        self.memory: DatabaseMemory = None
        self.conversation: ConversationStore = None
        
        self.current_task: Task = None
        self.task_history: list[Task] = []
    
    async def initialize(self, tools: list[BaseTool] = None):
        """初始化代理."""
        # 初始化规划器（使用规划器专用配置）
        self.planner = TaskPlanner(
            llm_config=self.planner_config,
            env_file=self.planner_env_file,
        )
        await self.planner.initialize()
        
        # 初始化执行器（使用执行器专用配置）
        self.executor = TaskExecutor(
            llm_config=self.executor_config,
            env_file=self.executor_env_file,
        )
        await self.executor.initialize(tools)
        
        # 显示配置信息
        print(f"✓ 规划器: {self.planner.llm_config.model} ({self.planner.llm_config.provider.value})")
        print(f"✓ 执行器: {self.executor.llm_config.model} ({self.executor.llm_config.provider.value})")
        
        # 初始化存储
        self.memory = DatabaseMemory(self.db_path)
        await self.memory._ensure_initialized()
        
        self.conversation = ConversationStore(self.db_path)
        await self.conversation.initialize()
        
        print("✓ 任务代理初始化完成")
    
    async def run_task(self, goal: str, context: dict = None) -> Task:
        """运行完整任务流程."""
        print(f"\n{'=' * 60}")
        print(f"任务目标: {goal}")
        print("=" * 60)
        
        # 创建任务
        task = Task(
            id=f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            goal=goal,
            context=context or {},
        )
        self.current_task = task
        
        # 1. 任务规划
        print("\n[阶段 1] 任务规划...")
        task.subtasks = await self.planner.plan(goal, context)
        
        print(f"已分解为 {len(task.subtasks)} 个子任务：")
        for i, st in enumerate(task.subtasks, 1):
            deps = f" (依赖: {', '.join(st.dependencies)})" if st.dependencies else ""
            print(f"  {i}. [{st.id}] {st.description}{deps}")
        
        # 2. 执行子任务
        print("\n[阶段 2] 执行子任务...")
        task.status = TaskStatus.IN_PROGRESS
        
        completed_tasks = set()
        execution_context = dict(context or {})
        
        while True:
            # 找到可执行的任务
            executable = self._get_executable_tasks(task.subtasks, completed_tasks)
            
            if not executable:
                # 检查是否全部完成
                if all(st.status == TaskStatus.COMPLETED for st in task.subtasks):
                    break
                # 检查是否有任务失败
                if any(st.status == TaskStatus.FAILED for st in task.subtasks):
                    task.status = TaskStatus.FAILED
                    break
                # 检查是否被阻塞
                task.status = TaskStatus.BLOCKED
                break
            
            # 顺序执行可执行任务
            for subtask in executable:
                print(f"\n  执行: [{subtask.id}] {subtask.description}")
                
                result = await self.executor.execute(subtask, execution_context)
                
                if result["success"]:
                    print(f"  ✓ 完成")
                    completed_tasks.add(subtask.id)
                    # 更新上下文
                    execution_context[f"result_{subtask.id}"] = subtask.result
                else:
                    print(f"  ✗ 失败: {result.get('error', 'Unknown error')}")
        
        # 3. 总结结果
        print("\n[阶段 3] 任务总结")
        
        success_count = sum(1 for st in task.subtasks if st.status == TaskStatus.COMPLETED)
        fail_count = sum(1 for st in task.subtasks if st.status == TaskStatus.FAILED)
        
        if fail_count == 0:
            task.status = TaskStatus.COMPLETED
            print(f"✓ 任务完成！({success_count}/{len(task.subtasks)} 子任务成功)")
        else:
            print(f"✗ 任务部分失败 ({success_count} 成功, {fail_count} 失败)")
        
        # 保存到历史
        self.task_history.append(task)
        
        # 保存到记忆
        from agentflow.memory.base import MemoryEntry
        entry = MemoryEntry(
            content=f"任务: {goal}\n状态: {task.status.value}\n结果数: {success_count}/{len(task.subtasks)}",
            role="system",
            metadata={
                "task_id": task.id,
                "status": task.status.value,
                "subtask_count": len(task.subtasks),
            },
            importance=0.9,
        )
        await self.memory.add(entry)
        
        return task
    
    def _get_executable_tasks(
        self,
        subtasks: list[SubTask],
        completed: set[str],
    ) -> list[SubTask]:
        """获取可执行的任务（依赖已完成）."""
        executable = []
        
        for st in subtasks:
            if st.status != TaskStatus.PENDING:
                continue
            
            # 检查依赖是否都已完成
            if all(dep in completed for dep in st.dependencies):
                executable.append(st)
        
        return executable
    
    async def close(self):
        """关闭资源."""
        if self.planner:
            await self.planner.close()
        if self.executor:
            await self.executor.close()
        if self.memory:
            await self.memory.close()


# 示例工具
@tool
def search_web(query: str) -> str:
    """搜索网络获取信息.
    
    Args:
        query: 搜索关键词
    
    Returns:
        搜索结果摘要
    """
    # 模拟搜索结果
    return f"搜索「{query}」的结果：找到相关信息若干条..."


@tool
def write_file(filename: str, content: str) -> str:
    """写入文件.
    
    Args:
        filename: 文件名
        content: 文件内容
    
    Returns:
        写入状态
    """
    # 模拟写入
    return f"已将内容写入文件 {filename}"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """发送邮件.
    
    Args:
        to: 收件人邮箱
        subject: 邮件主题
        body: 邮件正文
    
    Returns:
        发送状态
    """
    # 模拟发送
    return f"邮件已发送至 {to}，主题：{subject}"


@tool
def calculate(expression: str) -> str:
    """计算数学表达式.
    
    Args:
        expression: 数学表达式
    
    Returns:
        计算结果
    """
    try:
        result = eval(expression)
        return f"计算结果: {expression} = {result}"
    except:
        return f"无法计算: {expression}"


async def demo_simple_task(
    planner_env: Optional[str] = None,
    executor_env: Optional[str] = None,
):
    """演示简单任务执行.
    
    Args:
        planner_env: 规划器环境配置文件
        executor_env: 执行器环境配置文件
    """
    print("\n" + "=" * 60)
    print("示例 1: 简单任务")
    print("=" * 60)
    
    agent = TaskAgent(
        planner_env_file=planner_env,
        executor_env_file=executor_env,
    )
    await agent.initialize()
    
    task = await agent.run_task(
        goal="计算 (100 + 200) * 3 的结果，并说明计算过程",
    )
    
    print("\n最终结果:")
    for st in task.subtasks:
        if st.result:
            print(f"  {st.description}: {st.result[:200]}...")
    
    await agent.close()


async def demo_complex_task(
    planner_env: Optional[str] = None,
    executor_env: Optional[str] = None,
):
    """演示复杂多步骤任务.
    
    Args:
        planner_env: 规划器环境配置文件
        executor_env: 执行器环境配置文件
    """
    print("\n" + "=" * 60)
    print("示例 2: 复杂多步骤任务")
    print("=" * 60)
    
    agent = TaskAgent(
        planner_env_file=planner_env,
        executor_env_file=executor_env,
    )
    # 注册工具
    await agent.initialize(tools=[search_web, write_file, calculate])
    
    task = await agent.run_task(
        goal="调研 Python 异步编程的最佳实践，整理成文档，并计算文档字数",
        context={
            "output_format": "markdown",
            "max_length": 2000,
        },
    )
    
    print("\n任务执行报告:")
    print(f"  任务ID: {task.id}")
    print(f"  状态: {task.status.value}")
    print(f"  子任务数: {len(task.subtasks)}")
    
    for st in task.subtasks:
        status_icon = "✓" if st.status == TaskStatus.COMPLETED else "✗"
        print(f"  {status_icon} [{st.id}] {st.description}")
    
    await agent.close()


async def demo_report_generation(
    planner_env: Optional[str] = None,
    executor_env: Optional[str] = None,
):
    """演示报告生成任务.
    
    Args:
        planner_env: 规划器环境配置文件
        executor_env: 执行器环境配置文件
    """
    print("\n" + "=" * 60)
    print("示例 3: 报告生成任务")
    print("=" * 60)
    
    agent = TaskAgent(
        planner_env_file=planner_env,
        executor_env_file=executor_env,
    )
    await agent.initialize(tools=[search_web, write_file, calculate, send_email])
    
    task = await agent.run_task(
        goal="生成2024年1月的销售报告，包含数据分析和趋势预测，并发送给团队",
        context={
            "sales_data": {
                "january": [12000, 15000, 18000, 14000],
                "products": ["A", "B", "C", "D"],
            },
            "team_email": "team@example.com",
        },
    )
    
    await agent.close()


async def demo_interactive_task(
    planner_env: Optional[str] = None,
    executor_env: Optional[str] = None,
):
    """交互式任务执行.
    
    Args:
        planner_env: 规划器环境配置文件
        executor_env: 执行器环境配置文件
    """
    print("\n" + "=" * 60)
    print("交互式任务代理")
    print("=" * 60)
    
    agent = TaskAgent(
        planner_env_file=planner_env,
        executor_env_file=executor_env,
    )
    await agent.initialize(tools=[search_web, write_file, calculate])
    
    print("\n输入任务目标，或输入 'quit' 退出\n")
    
    while True:
        try:
            goal = input("任务> ").strip()
            if not goal:
                continue
            if goal.lower() == "quit":
                break
            
            task = await agent.run_task(goal)
            
            print(f"\n任务 {task.id} {task.status.value}")
            
        except KeyboardInterrupt:
            break
    
    await agent.close()


async def main(
    planner_env: Optional[str] = None,
    executor_env: Optional[str] = None,
):
    """主函数.
    
    Args:
        planner_env: 规划器环境配置文件
        executor_env: 执行器环境配置文件
    """
    await demo_simple_task(planner_env, executor_env)
    await demo_complex_task(planner_env, executor_env)
    await demo_report_generation(planner_env, executor_env)
    
    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)
    print("\n提示: 运行交互模式:")
    print("  python task_agent.py interactive")
    print("  python task_agent.py interactive --planner=.env.planner --executor=.env.executor")


if __name__ == "__main__":
    import sys
    
    # 解析命令行参数
    planner_env = None
    executor_env = None
    mode = "demo"
    
    for arg in sys.argv[1:]:
        if arg.startswith("--planner="):
            planner_env = arg.split("=", 1)[1]
        elif arg.startswith("--executor="):
            executor_env = arg.split("=", 1)[1]
        elif arg in ("interactive", "demo"):
            mode = arg
    
    print(f"规划器配置: {planner_env or '.env.planner (默认)'}")
    print(f"执行器配置: {executor_env or '.env.executor (默认)'}")
    
    if mode == "interactive":
        asyncio.run(demo_interactive_task(planner_env, executor_env))
    else:
        asyncio.run(main(planner_env, executor_env))
