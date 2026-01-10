"""
轻量级 Agent 实现
==================

极简设计，支持：
- 多种推理模式
- 工具调用
- 记忆管理
- 事件钩子
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from .types import (
    AgentConfig,
    AgentResult,
    EventEmitter,
    LLMConfig,
    LLMProvider,
    MemoryProvider,
    Message,
    Pattern,
    Role,
    Tool,
    ToolCall,
    ToolRegistry,
)


class SimpleAgent:
    """极简 Agent.
    
    Example:
        ```python
        from agentflow.core import SimpleAgent
        from agentflow.llm import OpenAIProvider
        
        llm = OpenAIProvider(model="gpt-4o-mini")
        agent = SimpleAgent(llm=llm)
        
        result = await agent.run("Hello!")
        print(result.output)
        ```
    """
    
    def __init__(
        self,
        llm: LLMProvider,
        config: Optional[AgentConfig] = None,
        memory: Optional[MemoryProvider] = None,
        tools: Optional[list[Tool]] = None,
    ):
        self.llm = llm
        self.config = config or AgentConfig()
        self.memory = memory
        
        self.registry = ToolRegistry()
        if tools:
            for t in tools:
                self.registry.register(t)
        
        self.events = EventEmitter()
        self._messages: list[Message] = []
        
        # 初始化系统提示
        if self.config.system_prompt:
            self._messages.append(Message.system(self.config.system_prompt))
    
    def add_tool(self, tool: Tool) -> SimpleAgent:
        """添加工具 (链式调用)."""
        self.registry.register(tool)
        return self
    
    def set_memory(self, memory: MemoryProvider) -> SimpleAgent:
        """设置记忆 (链式调用)."""
        self.memory = memory
        return self
    
    async def run(
        self,
        task: str,
        **kwargs: Any,
    ) -> AgentResult:
        """执行任务.
        
        根据配置的模式选择执行策略。
        """
        await self.events.emit("task:start", {"task": task})
        
        # 添加用户消息
        self._messages.append(Message.user(task))
        
        # 记忆
        if self.memory:
            await self.memory.remember(task, role="user")
        
        try:
            # 根据模式执行
            if self.config.pattern == Pattern.SIMPLE:
                result = await self._simple_run(task, **kwargs)
            elif self.config.pattern == Pattern.REACT:
                result = await self._react_run(task, **kwargs)
            elif self.config.pattern == Pattern.COT:
                result = await self._cot_run(task, **kwargs)
            else:
                result = await self._simple_run(task, **kwargs)
            
            await self.events.emit("task:complete", {"result": result})
            return result
            
        except Exception as e:
            error_result = AgentResult(
                output="",
                success=False,
                error=str(e),
            )
            await self.events.emit("task:error", {"error": e})
            return error_result
    
    async def chat(self, message: str) -> str:
        """简单对话接口."""
        result = await self.run(message)
        return result.output
    
    async def _simple_run(self, task: str, **kwargs) -> AgentResult:
        """简单模式 - 直接 LLM 调用."""
        # 获取记忆上下文
        context = ""
        if self.memory:
            context = await self.memory.context(task)
        
        # 准备消息
        messages = list(self._messages)
        if context:
            messages.insert(1, Message.system(f"[Context]\n{context}"))
        
        # 调用 LLM
        response = await self.llm.complete(
            messages,
            tools=self.registry.to_openai_tools() if len(self.registry) > 0 else None,
        )
        
        # 处理工具调用
        if response.tool_calls:
            return await self._handle_tool_calls(messages, response.tool_calls)
        
        # 记忆响应
        if self.memory:
            await self.memory.remember(response.content, role="assistant")
        
        self._messages.append(Message.assistant(response.content))
        
        return AgentResult(output=response.content)
    
    async def _react_run(self, task: str, **kwargs) -> AgentResult:
        """ReAct 模式 - 思考-行动循环."""
        steps = []
        
        react_prompt = f"""You are using the ReAct framework. For each step:
1. Thought: Analyze what you need to do
2. Action: Choose a tool to use (or "Final Answer" if done)
3. Observation: See the result

Task: {task}

Available tools: {', '.join(t.name for t in self.registry)}

Begin!"""
        
        messages = list(self._messages)
        messages.append(Message.user(react_prompt))
        
        for i in range(self.config.max_iterations):
            response = await self.llm.complete(
                messages,
                tools=self.registry.to_openai_tools() if len(self.registry) > 0 else None,
            )
            
            steps.append({"type": "thought", "content": response.content})
            
            # 检查是否完成
            if "final answer" in response.content.lower() or not response.tool_calls:
                if self.memory:
                    await self.memory.remember(response.content, role="assistant")
                return AgentResult(output=response.content, steps=steps)
            
            # 执行工具
            for tc in response.tool_calls:
                tool = self.registry.get(tc.name)
                if tool:
                    try:
                        result = await tool.execute(**tc.arguments)
                        steps.append({
                            "type": "action",
                            "tool": tc.name,
                            "args": tc.arguments,
                            "result": str(result),
                        })
                        messages.append(Message.tool(str(result), tc.id, tc.name))
                    except Exception as e:
                        messages.append(Message.tool(f"Error: {e}", tc.id, tc.name))
            
            messages.append(Message.assistant(response.content))
        
        return AgentResult(
            output="Max iterations reached",
            success=False,
            steps=steps,
        )
    
    async def _cot_run(self, task: str, **kwargs) -> AgentResult:
        """Chain-of-Thought 模式."""
        cot_prompt = f"""Let's think step by step.

Task: {task}

Please reason through this step by step, showing your thought process clearly."""
        
        messages = list(self._messages)
        messages.append(Message.user(cot_prompt))
        
        response = await self.llm.complete(messages)
        
        if self.memory:
            await self.memory.remember(response.content, role="assistant")
        
        return AgentResult(
            output=response.content,
            steps=[{"type": "reasoning", "content": response.content}],
        )
    
    async def _handle_tool_calls(
        self,
        messages: list[Message],
        tool_calls: list[ToolCall],
    ) -> AgentResult:
        """处理工具调用."""
        steps = []
        
        for tc in tool_calls:
            tool = self.registry.get(tc.name)
            if not tool:
                messages.append(Message.tool(
                    f"Tool '{tc.name}' not found",
                    tc.id,
                    tc.name,
                ))
                continue
            
            try:
                result = await tool.execute(**tc.arguments)
                steps.append({
                    "type": "tool_call",
                    "tool": tc.name,
                    "args": tc.arguments,
                    "result": str(result),
                })
                messages.append(Message.tool(str(result), tc.id, tc.name))
            except Exception as e:
                messages.append(Message.tool(f"Error: {e}", tc.id, tc.name))
        
        # 继续生成
        response = await self.llm.complete(messages)
        
        if self.memory:
            await self.memory.remember(response.content, role="assistant")
        
        return AgentResult(output=response.content, steps=steps)
    
    def reset(self) -> None:
        """重置对话历史."""
        self._messages = []
        if self.config.system_prompt:
            self._messages.append(Message.system(self.config.system_prompt))
    
    async def close(self) -> None:
        """关闭资源."""
        pass
    
    async def __aenter__(self) -> SimpleAgent:
        return self
    
    async def __aexit__(self, *args) -> None:
        await self.close()
