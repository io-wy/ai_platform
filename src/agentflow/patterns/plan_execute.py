"""Plan and Execute pattern implementation."""

from typing import Any, Optional

from agentflow.core.message import Message, MessageRole, Conversation
from agentflow.llm.client import LLMClient
from agentflow.patterns.base import BasePattern, PatternResult
from agentflow.patterns.react import ReActPattern
from agentflow.tools.executor import ToolExecutor


class PlanAndExecutePattern(BasePattern):
    """Plan and Execute pattern.
    
    This pattern separates planning from execution:
    1. Create a high-level plan with steps
    2. Execute each step (potentially using ReAct)
    3. Optionally re-plan based on execution results
    
    Suitable for:
    - Complex multi-step tasks
    - Tasks requiring coordination
    - Long-running operations
    
    Example:
        ```python
        pattern = PlanAndExecutePattern(
            llm=LLMClient(),
            tool_executor=executor,
        )
        
        result = await pattern.run(
            "Research the latest AI news and write a summary report"
        )
        ```
    """
    
    name = "plan_and_execute"
    description = "Plan first, then execute step by step"
    
    PLANNER_PROMPT = '''You are a planning assistant. Create a step-by-step plan to accomplish the given task.

Task: {task}

Create a numbered plan with clear, actionable steps. Each step should be specific and achievable.
Consider what information you need and what actions to take.

Plan:'''

    REPLANNER_PROMPT = '''Review the progress and update the plan if needed.

Original task: {task}

Original plan:
{original_plan}

Completed steps:
{completed}

Current step result:
{current_result}

Remaining steps:
{remaining}

Should the plan be updated? If yes, provide the new remaining steps. If no, respond with "CONTINUE".

Response:'''

    EXECUTOR_PROMPT = '''Execute the following step of the plan.

Overall task: {task}

Current step: {step}

Previous results:
{previous}

Execute this step and provide the result:'''

    def __init__(
        self,
        llm: LLMClient,
        tool_executor: Optional[ToolExecutor] = None,
        max_iterations: int = 20,
        verbose: bool = False,
        use_react_for_steps: bool = True,
        allow_replanning: bool = True,
    ):
        super().__init__(llm, tool_executor, max_iterations, verbose)
        self.use_react_for_steps = use_react_for_steps
        self.allow_replanning = allow_replanning
        
        # Create ReAct pattern for step execution
        if use_react_for_steps and tool_executor:
            self.react = ReActPattern(
                llm=llm,
                tool_executor=tool_executor,
                max_iterations=5,  # Limit iterations per step
                verbose=verbose,
            )
        else:
            self.react = None
    
    async def run(
        self,
        task: str,
        context: Optional[Conversation] = None,
        **kwargs: Any,
    ) -> PatternResult:
        """Execute Plan and Execute pattern."""
        iterations = 0
        tool_calls_made = 0
        all_messages: list[Message] = []
        
        # Phase 1: Planning
        self.log("Phase 1: Creating plan")
        plan = await self._create_plan(task)
        
        if not plan:
            return PatternResult(
                success=False,
                output="",
                iterations=1,
                messages=[],
                error="Failed to create plan",
            )
        
        self.log(f"Plan created with {len(plan)} steps")
        original_plan = plan.copy()
        
        # Phase 2: Execution
        self.log("Phase 2: Executing plan")
        results: list[dict[str, Any]] = []
        
        step_index = 0
        while step_index < len(plan) and iterations < self.max_iterations:
            iterations += 1
            step = plan[step_index]
            
            self.log(f"Executing step {step_index + 1}/{len(plan)}: {step[:50]}...")
            
            # Execute the step
            step_result = await self._execute_step(task, step, results)
            
            results.append({
                "step": step,
                "result": step_result.output,
                "success": step_result.success,
            })
            all_messages.extend(step_result.messages)
            tool_calls_made += step_result.tool_calls_made
            
            self.log(f"Step result: {'Success' if step_result.success else 'Failed'}")
            
            # Optional replanning
            if self.allow_replanning and step_index < len(plan) - 1:
                remaining = plan[step_index + 1:]
                new_remaining = await self._replan(
                    task,
                    original_plan,
                    results,
                    step_result.output,
                    remaining,
                )
                
                if new_remaining and new_remaining != remaining:
                    self.log("Plan updated based on results")
                    plan = [r["step"] for r in results] + new_remaining
            
            step_index += 1
        
        # Phase 3: Synthesize results
        self.log("Phase 3: Synthesizing results")
        final_output = await self._synthesize(task, results)
        
        return PatternResult(
            success=all(r["success"] for r in results),
            output=final_output,
            iterations=iterations,
            messages=all_messages,
            tool_calls_made=tool_calls_made,
            metadata={
                "plan": original_plan,
                "steps_completed": len(results),
                "step_results": results,
            },
        )
    
    async def _create_plan(self, task: str) -> list[str]:
        """Create a plan for the task."""
        prompt = self.PLANNER_PROMPT.format(task=task)
        
        try:
            response = await self.llm.chat(
                messages=[Message.user(prompt)],
                temperature=0.3,
            )
            
            content = response.message.content or ""
            
            # Parse numbered steps
            lines = content.strip().split("\n")
            steps = []
            for line in lines:
                line = line.strip()
                # Remove numbering
                if line and line[0].isdigit():
                    # Remove "1.", "1)", "1:" etc.
                    import re
                    step = re.sub(r"^\d+[\.\)\:]\s*", "", line)
                    if step:
                        steps.append(step)
                elif line.startswith("-") or line.startswith("*"):
                    step = line[1:].strip()
                    if step:
                        steps.append(step)
            
            return steps if steps else [task]  # Fallback to single step
        
        except Exception as e:
            self.log(f"Planning failed: {e}", level="error")
            return []
    
    async def _execute_step(
        self,
        task: str,
        step: str,
        previous_results: list[dict[str, Any]],
    ) -> PatternResult:
        """Execute a single step of the plan."""
        # Format previous results
        previous = "\n".join(
            f"Step {i+1}: {r['step']}\nResult: {r['result'][:200]}..."
            for i, r in enumerate(previous_results)
        ) or "No previous steps."
        
        if self.react:
            # Use ReAct for step execution
            step_prompt = f"""Overall task: {task}

Current step to execute: {step}

Previous step results:
{previous}

Execute this step using available tools if needed."""
            
            return await self.react.run(step_prompt)
        
        else:
            # Simple LLM execution
            prompt = self.EXECUTOR_PROMPT.format(
                task=task,
                step=step,
                previous=previous,
            )
            
            try:
                response = await self.llm.chat(
                    messages=[Message.user(prompt)],
                )
                
                return PatternResult(
                    success=True,
                    output=response.message.content or "",
                    iterations=1,
                    messages=[response.message],
                )
            
            except Exception as e:
                return PatternResult(
                    success=False,
                    output="",
                    iterations=1,
                    messages=[],
                    error=str(e),
                )
    
    async def _replan(
        self,
        task: str,
        original_plan: list[str],
        completed: list[dict[str, Any]],
        current_result: str,
        remaining: list[str],
    ) -> Optional[list[str]]:
        """Re-evaluate and potentially update the plan."""
        prompt = self.REPLANNER_PROMPT.format(
            task=task,
            original_plan="\n".join(f"{i+1}. {s}" for i, s in enumerate(original_plan)),
            completed="\n".join(f"- {r['step']}: {r['result'][:100]}..." for r in completed),
            current_result=current_result[:300],
            remaining="\n".join(f"- {s}" for s in remaining),
        )
        
        try:
            response = await self.llm.chat(
                messages=[Message.user(prompt)],
                temperature=0.3,
            )
            
            content = response.message.content or ""
            
            if "CONTINUE" in content.upper():
                return remaining
            
            # Parse new steps
            lines = content.strip().split("\n")
            new_steps = []
            for line in lines:
                line = line.strip()
                if line and not line.upper().startswith("CONTINUE"):
                    import re
                    step = re.sub(r"^[\d\-\*\.]+\s*", "", line)
                    if step:
                        new_steps.append(step)
            
            return new_steps if new_steps else remaining
        
        except Exception:
            return remaining
    
    async def _synthesize(self, task: str, results: list[dict[str, Any]]) -> str:
        """Synthesize final output from step results."""
        results_text = "\n\n".join(
            f"**Step {i+1}: {r['step']}**\n{r['result']}"
            for i, r in enumerate(results)
        )
        
        prompt = f"""Synthesize the following step results into a final response for the task.

Task: {task}

Step results:
{results_text}

Provide a clear, comprehensive final response:"""
        
        try:
            response = await self.llm.chat(
                messages=[Message.user(prompt)],
            )
            return response.message.content or results_text
        
        except Exception:
            return results_text
