"""Auto pattern - lets the model decide which pattern to use."""

from typing import Any, Optional

from agentflow.core.message import Message, MessageRole, Conversation
from agentflow.llm.client import LLMClient
from agentflow.patterns.base import BasePattern, PatternResult
from agentflow.patterns.react import ReActPattern
from agentflow.patterns.cot import ChainOfThoughtPattern
from agentflow.patterns.tot import TreeOfThoughtPattern
from agentflow.patterns.reflexion import ReflexionPattern
from agentflow.patterns.plan_execute import PlanAndExecutePattern
from agentflow.tools.executor import ToolExecutor


class AutoPattern(BasePattern):
    """Auto pattern that selects the best reasoning approach.
    
    This meta-pattern analyzes the task and automatically selects
    the most appropriate reasoning pattern:
    
    - ReAct: For tasks requiring tool use and external information
    - CoT: For reasoning and calculation tasks
    - ToT: For complex exploration and creative tasks
    - Reflexion: For tasks requiring iteration and learning
    - Plan & Execute: For multi-step procedural tasks
    
    Example:
        ```python
        pattern = AutoPattern(
            llm=LLMClient(),
            tool_executor=executor,
        )
        
        # Pattern is automatically selected
        result = await pattern.run("Calculate 15 * 23 step by step")  # Uses CoT
        result = await pattern.run("Search for Python tutorials")  # Uses ReAct
        ```
    """
    
    name = "auto"
    description = "Automatically selects the best reasoning pattern for the task"
    
    SELECTOR_PROMPT = '''Analyze the following task and determine the best reasoning approach.

Task: {task}

Available patterns:
1. **react** - For tasks requiring tools/actions (web search, file operations, API calls)
2. **cot** - For reasoning, math, or logic problems that need step-by-step thinking
3. **tot** - For complex creative or exploration tasks with multiple possible paths
4. **reflexion** - For tasks requiring iteration and learning from mistakes
5. **plan_execute** - For multi-step procedural tasks that benefit from planning

Consider:
- Does the task need external tools or information? → react
- Is it a reasoning/calculation problem? → cot
- Does it need exploring multiple solutions? → tot
- Might it need multiple attempts to get right? → reflexion
- Is it a complex multi-step procedure? → plan_execute

Respond with just the pattern name (react, cot, tot, reflexion, or plan_execute):'''

    def __init__(
        self,
        llm: LLMClient,
        tool_executor: Optional[ToolExecutor] = None,
        max_iterations: int = 10,
        verbose: bool = False,
        default_pattern: str = "react",
    ):
        super().__init__(llm, tool_executor, max_iterations, verbose)
        self.default_pattern = default_pattern
        
        # Initialize all patterns
        self._patterns: dict[str, BasePattern] = {
            "react": ReActPattern(llm, tool_executor, max_iterations, verbose),
            "cot": ChainOfThoughtPattern(llm, tool_executor, max_iterations, verbose),
            "tot": TreeOfThoughtPattern(llm, tool_executor, max_iterations, verbose),
            "reflexion": ReflexionPattern(llm, tool_executor, max_iterations, verbose),
            "plan_execute": PlanAndExecutePattern(llm, tool_executor, max_iterations, verbose),
        }
    
    async def select_pattern(self, task: str) -> str:
        """Select the best pattern for the task."""
        prompt = self.SELECTOR_PROMPT.format(task=task)
        
        try:
            response = await self.llm.chat(
                messages=[Message.user(prompt)],
                temperature=0.1,  # Low temperature for consistent selection
            )
            
            content = response.message.content or ""
            pattern_name = content.strip().lower()
            
            # Normalize pattern names
            pattern_name = pattern_name.replace("_", "").replace("-", "").replace(" ", "")
            
            # Map to valid pattern names
            pattern_map = {
                "react": "react",
                "cot": "cot",
                "chainofthought": "cot",
                "tot": "tot",
                "treeofthought": "tot",
                "reflexion": "reflexion",
                "planexecute": "plan_execute",
                "planandexecute": "plan_execute",
            }
            
            selected = pattern_map.get(pattern_name, self.default_pattern)
            self.log(f"Selected pattern: {selected}")
            return selected
        
        except Exception as e:
            self.log(f"Pattern selection failed: {e}, using default", level="warning")
            return self.default_pattern
    
    async def run(
        self,
        task: str,
        context: Optional[Conversation] = None,
        force_pattern: Optional[str] = None,
        **kwargs: Any,
    ) -> PatternResult:
        """Execute the auto-selected pattern."""
        # Select pattern
        if force_pattern and force_pattern in self._patterns:
            pattern_name = force_pattern
        else:
            pattern_name = await self.select_pattern(task)
        
        pattern = self._patterns.get(pattern_name)
        
        if pattern is None:
            return PatternResult(
                success=False,
                output="",
                error=f"Unknown pattern: {pattern_name}",
            )
        
        self.log(f"Running with {pattern_name} pattern")
        
        # Execute selected pattern
        result = await pattern.run(task, context, **kwargs)
        
        # Add pattern selection to metadata
        result.metadata["selected_pattern"] = pattern_name
        
        return result
    
    def get_available_patterns(self) -> list[str]:
        """Get list of available pattern names."""
        return list(self._patterns.keys())
    
    def add_pattern(self, name: str, pattern: BasePattern) -> None:
        """Add a custom pattern."""
        self._patterns[name] = pattern
