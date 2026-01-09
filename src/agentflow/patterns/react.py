"""ReAct (Reasoning + Acting) pattern implementation."""

from typing import Any, Optional

from agentflow.core.message import Message, MessageRole, Conversation, ToolCall
from agentflow.llm.client import LLMClient
from agentflow.patterns.base import BasePattern, PatternResult
from agentflow.tools.executor import ToolExecutor


class ReActPattern(BasePattern):
    """ReAct (Reasoning + Acting) pattern.
    
    The ReAct pattern interleaves reasoning (thinking) with acting (tool use)
    in a loop until the task is completed or max iterations reached.
    
    Loop:
    1. Think: Reason about the current state and what to do next
    2. Act: Execute a tool if needed
    3. Observe: Process the tool result
    4. Repeat or conclude
    
    Reference: https://arxiv.org/abs/2210.03629
    
    Example:
        ```python
        pattern = ReActPattern(
            llm=LLMClient(),
            tool_executor=executor,
            max_iterations=10,
        )
        
        result = await pattern.run("What is the weather in Tokyo?")
        print(result.output)
        ```
    """
    
    name = "react"
    description = "Reasoning + Acting pattern for interleaved thinking and tool use"
    
    SYSTEM_PROMPT = '''You are an AI assistant that solves tasks using the ReAct (Reasoning + Acting) approach.

For each step, you should:
1. **Thought**: Reason about the current situation and what you need to do next.
2. **Action**: If you need to use a tool, call it. If you have enough information to answer, provide the final answer.

Guidelines:
- Think step by step before acting
- Use tools when you need external information or capabilities
- After getting tool results, reflect on what you learned
- Provide a clear final answer when the task is complete
- If a tool fails, try an alternative approach

When you have the final answer, respond directly without calling any more tools.'''

    def __init__(
        self,
        llm: LLMClient,
        tool_executor: Optional[ToolExecutor] = None,
        max_iterations: int = 10,
        verbose: bool = False,
        include_thought_in_response: bool = False,
    ):
        super().__init__(llm, tool_executor, max_iterations, verbose)
        self.include_thought_in_response = include_thought_in_response
    
    def get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT
    
    async def run(
        self,
        task: str,
        context: Optional[Conversation] = None,
        **kwargs: Any,
    ) -> PatternResult:
        """Execute ReAct pattern."""
        conversation = context or Conversation()
        
        # Add system prompt if not present
        if not any(m.role == MessageRole.SYSTEM for m in conversation.messages):
            conversation.add_system(self.get_system_prompt())
        
        # Add the task
        conversation.add_user(task)
        
        iterations = 0
        tool_calls_made = 0
        all_messages: list[Message] = []
        
        # Get tool schemas
        tools = None
        if self.tool_executor:
            tools = self.tool_executor.get_tool_schemas()
        
        while iterations < self.max_iterations:
            iterations += 1
            self.log(f"Iteration {iterations}/{self.max_iterations}")
            
            # Get LLM response
            try:
                response = await self.llm.chat(
                    messages=conversation.messages,
                    tools=tools,
                )
            except Exception as e:
                return PatternResult(
                    success=False,
                    output="",
                    iterations=iterations,
                    messages=all_messages,
                    tool_calls_made=tool_calls_made,
                    error=str(e),
                )
            
            assistant_message = response.message
            conversation.add(assistant_message)
            all_messages.append(assistant_message)
            
            self.log(f"Assistant response: {assistant_message.content[:200] if assistant_message.content else 'No content'}")
            
            # Check if we have tool calls
            if assistant_message.tool_calls and self.tool_executor:
                self.log(f"Executing {len(assistant_message.tool_calls)} tool calls")
                
                # Execute tools
                tool_results = await self.tool_executor.execute_tool_calls(
                    assistant_message.tool_calls
                )
                tool_calls_made += len(tool_results)
                
                # Add tool results to conversation
                for result in tool_results:
                    tool_message = Message.tool(result)
                    conversation.add(tool_message)
                    all_messages.append(tool_message)
                    
                    self.log(f"Tool {result.name}: {result.content[:200] if result.content else 'No output'}")
                
                # Continue loop to get next response
                continue
            
            # No tool calls - this is the final answer
            if assistant_message.content:
                return PatternResult(
                    success=True,
                    output=assistant_message.content,
                    iterations=iterations,
                    messages=all_messages,
                    tool_calls_made=tool_calls_made,
                )
            
            # No content and no tool calls - something went wrong
            return PatternResult(
                success=False,
                output="",
                iterations=iterations,
                messages=all_messages,
                tool_calls_made=tool_calls_made,
                error="Empty response from model",
            )
        
        # Max iterations reached
        last_content = ""
        for msg in reversed(all_messages):
            if msg.role == MessageRole.ASSISTANT and msg.content:
                last_content = msg.content
                break
        
        return PatternResult(
            success=False,
            output=last_content,
            iterations=iterations,
            messages=all_messages,
            tool_calls_made=tool_calls_made,
            error=f"Max iterations ({self.max_iterations}) reached",
        )
