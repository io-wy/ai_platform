"""Reflexion pattern implementation."""

from typing import Any, Optional

from agentflow.core.message import Message, MessageRole, Conversation
from agentflow.llm.client import LLMClient
from agentflow.patterns.base import BasePattern, PatternResult
from agentflow.tools.executor import ToolExecutor


class ReflexionPattern(BasePattern):
    """Reflexion pattern for self-reflection and learning.
    
    The Reflexion pattern uses verbal reinforcement to help the agent
    learn from its mistakes by:
    1. Attempting the task
    2. Evaluating the result
    3. Generating self-reflection
    4. Using reflection to improve on next attempt
    
    Reference: https://arxiv.org/abs/2303.11366
    
    Example:
        ```python
        pattern = ReflexionPattern(
            llm=LLMClient(),
            tool_executor=executor,
            max_trials=3,
        )
        
        result = await pattern.run(
            task="Write a function to calculate fibonacci numbers",
            evaluator=lambda x: "correct" if test_fibonacci(x) else "incorrect"
        )
        ```
    """
    
    name = "reflexion"
    description = "Reflexion pattern for learning from mistakes through self-reflection"
    
    ACTOR_PROMPT = '''You are an AI assistant that learns from experience.

{reflections}

Solve the following task:
{task}

Previous attempts and feedback:
{history}

Think carefully and provide your best solution:'''

    EVALUATOR_PROMPT = '''Evaluate the following solution for the task.

Task: {task}

Solution:
{solution}

Provide feedback:
1. Is this correct? (YES/NO)
2. If not, what specifically is wrong?
3. What would make it better?

Feedback:'''

    REFLECTOR_PROMPT = '''Based on your attempt and the feedback, generate a reflection that will help you do better next time.

Task: {task}

Your attempt:
{attempt}

Feedback received:
{feedback}

Write a brief reflection (2-3 sentences) about what went wrong and how to improve:'''

    def __init__(
        self,
        llm: LLMClient,
        tool_executor: Optional[ToolExecutor] = None,
        max_iterations: int = 10,
        verbose: bool = False,
        max_trials: int = 3,
        use_external_evaluator: bool = False,
    ):
        super().__init__(llm, tool_executor, max_iterations, verbose)
        self.max_trials = max_trials
        self.use_external_evaluator = use_external_evaluator
    
    async def run(
        self,
        task: str,
        context: Optional[Conversation] = None,
        evaluator: Optional[Any] = None,
        success_criteria: Optional[str] = None,
        **kwargs: Any,
    ) -> PatternResult:
        """Execute Reflexion pattern."""
        reflections: list[str] = []
        history: list[dict[str, str]] = []
        iterations = 0
        best_attempt: Optional[str] = None
        best_score = 0.0
        
        for trial in range(self.max_trials):
            iterations += 1
            self.log(f"Trial {trial + 1}/{self.max_trials}")
            
            # Build reflection context
            reflection_text = ""
            if reflections:
                reflection_text = "Previous reflections:\n" + "\n".join(
                    f"- {r}" for r in reflections
                )
            
            history_text = ""
            if history:
                history_text = "\n".join(
                    f"Attempt {i+1}: {h['attempt'][:200]}...\nFeedback: {h['feedback']}"
                    for i, h in enumerate(history)
                )
            
            # Generate attempt
            actor_prompt = self.ACTOR_PROMPT.format(
                reflections=reflection_text,
                task=task,
                history=history_text or "No previous attempts.",
            )
            
            try:
                response = await self.llm.chat(
                    messages=[Message.user(actor_prompt)],
                )
                attempt = response.message.content or ""
            except Exception as e:
                self.log(f"Actor failed: {e}", level="error")
                continue
            
            self.log(f"Generated attempt: {attempt[:200]}...")
            
            # Evaluate attempt
            if evaluator and callable(evaluator):
                # Use external evaluator
                feedback = evaluator(attempt)
                is_success = "correct" in feedback.lower() or "success" in feedback.lower()
            else:
                # Use LLM as evaluator
                feedback, is_success = await self._evaluate(task, attempt, success_criteria)
            
            self.log(f"Feedback: {feedback[:200]}...")
            
            # Track best attempt
            score = 1.0 if is_success else 0.5
            if score > best_score:
                best_score = score
                best_attempt = attempt
            
            if is_success:
                return PatternResult(
                    success=True,
                    output=attempt,
                    iterations=iterations,
                    messages=[Message.assistant(attempt)],
                    metadata={
                        "trials": trial + 1,
                        "reflections": reflections,
                    },
                )
            
            # Generate reflection
            reflection = await self._reflect(task, attempt, feedback)
            reflections.append(reflection)
            history.append({"attempt": attempt, "feedback": feedback})
            
            self.log(f"Reflection: {reflection}")
        
        # Return best attempt
        return PatternResult(
            success=False,
            output=best_attempt or "",
            iterations=iterations,
            messages=[Message.assistant(best_attempt or "")],
            metadata={
                "trials": self.max_trials,
                "reflections": reflections,
            },
            error=f"Failed to find correct solution in {self.max_trials} trials",
        )
    
    async def _evaluate(
        self,
        task: str,
        solution: str,
        success_criteria: Optional[str],
    ) -> tuple[str, bool]:
        """Evaluate a solution using the LLM."""
        eval_prompt = self.EVALUATOR_PROMPT.format(
            task=task + (f"\n\nSuccess criteria: {success_criteria}" if success_criteria else ""),
            solution=solution,
        )
        
        try:
            response = await self.llm.chat(
                messages=[Message.user(eval_prompt)],
                temperature=0.3,
            )
            
            feedback = response.message.content or ""
            is_success = "YES" in feedback.upper().split("\n")[0]
            
            return feedback, is_success
        
        except Exception as e:
            return f"Evaluation failed: {e}", False
    
    async def _reflect(self, task: str, attempt: str, feedback: str) -> str:
        """Generate a reflection based on the attempt and feedback."""
        reflect_prompt = self.REFLECTOR_PROMPT.format(
            task=task,
            attempt=attempt,
            feedback=feedback,
        )
        
        try:
            response = await self.llm.chat(
                messages=[Message.user(reflect_prompt)],
                temperature=0.5,
            )
            
            return response.message.content or "No reflection generated."
        
        except Exception as e:
            return f"Reflection failed: {e}"
