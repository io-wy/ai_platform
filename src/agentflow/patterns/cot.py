"""Chain of Thought (CoT) pattern implementation."""

from typing import Any, Optional

from agentflow.core.message import Message, MessageRole, Conversation
from agentflow.llm.client import LLMClient
from agentflow.patterns.base import BasePattern, PatternResult
from agentflow.tools.executor import ToolExecutor


class ChainOfThoughtPattern(BasePattern):
    """Chain of Thought (CoT) pattern.
    
    The CoT pattern encourages the model to break down complex reasoning
    into explicit steps before arriving at an answer.
    
    Variants:
    - Zero-shot CoT: Simply prompt with "Let's think step by step"
    - Few-shot CoT: Provide examples of step-by-step reasoning
    
    Reference: https://arxiv.org/abs/2201.11903
    
    Example:
        ```python
        pattern = ChainOfThoughtPattern(llm=LLMClient())
        
        result = await pattern.run(
            "If a train travels 120 miles in 2 hours, what is its average speed?"
        )
        print(result.output)
        ```
    """
    
    name = "cot"
    description = "Chain of Thought pattern for step-by-step reasoning"
    
    SYSTEM_PROMPT = '''You are an AI assistant that solves problems using Chain of Thought reasoning.

When given a problem:
1. Break it down into smaller, manageable steps
2. Work through each step explicitly, showing your reasoning
3. Build upon previous steps to reach the final answer
4. Clearly state your final answer at the end

Format your response as:
**Step 1:** [Your first reasoning step]
**Step 2:** [Your second reasoning step]
...
**Final Answer:** [Your conclusive answer]

Think carefully and show all your work.'''

    ZERO_SHOT_PROMPT = "\n\nLet's think step by step."
    
    def __init__(
        self,
        llm: LLMClient,
        tool_executor: Optional[ToolExecutor] = None,
        max_iterations: int = 1,  # CoT typically single-shot
        verbose: bool = False,
        few_shot_examples: Optional[list[dict[str, str]]] = None,
        use_zero_shot: bool = True,
    ):
        super().__init__(llm, tool_executor, max_iterations, verbose)
        self.few_shot_examples = few_shot_examples
        self.use_zero_shot = use_zero_shot
    
    def get_system_prompt(self) -> str:
        prompt = self.SYSTEM_PROMPT
        
        # Add few-shot examples if provided
        if self.few_shot_examples:
            prompt += "\n\nHere are some examples of good step-by-step reasoning:\n"
            for i, example in enumerate(self.few_shot_examples, 1):
                prompt += f"\n**Example {i}:**\n"
                prompt += f"Question: {example['question']}\n"
                prompt += f"Reasoning: {example['reasoning']}\n"
                prompt += f"Answer: {example['answer']}\n"
        
        return prompt
    
    async def run(
        self,
        task: str,
        context: Optional[Conversation] = None,
        **kwargs: Any,
    ) -> PatternResult:
        """Execute Chain of Thought pattern."""
        conversation = context or Conversation()
        
        # Add system prompt
        if not any(m.role == MessageRole.SYSTEM for m in conversation.messages):
            conversation.add_system(self.get_system_prompt())
        
        # Add the task with optional zero-shot prompt
        task_prompt = task
        if self.use_zero_shot and not self.few_shot_examples:
            task_prompt += self.ZERO_SHOT_PROMPT
        
        conversation.add_user(task_prompt)
        
        self.log(f"Running CoT on task: {task[:100]}...")
        
        try:
            response = await self.llm.chat(
                messages=conversation.messages,
            )
            
            assistant_message = response.message
            conversation.add(assistant_message)
            
            self.log(f"CoT response received, length: {len(assistant_message.content or '')}")
            
            return PatternResult(
                success=True,
                output=assistant_message.content or "",
                iterations=1,
                messages=[assistant_message],
                tool_calls_made=0,
                metadata={
                    "usage": response.usage,
                    "finish_reason": response.finish_reason,
                },
            )
        
        except Exception as e:
            return PatternResult(
                success=False,
                output="",
                iterations=1,
                messages=[],
                tool_calls_made=0,
                error=str(e),
            )


class SelfConsistencyCoT(ChainOfThoughtPattern):
    """Self-Consistency Chain of Thought.
    
    Generates multiple reasoning paths and selects the most consistent answer.
    
    Reference: https://arxiv.org/abs/2203.11171
    """
    
    name = "cot_self_consistency"
    description = "Self-Consistency CoT with multiple reasoning paths"
    
    def __init__(
        self,
        llm: LLMClient,
        tool_executor: Optional[ToolExecutor] = None,
        max_iterations: int = 1,
        verbose: bool = False,
        num_paths: int = 5,
        temperature: float = 0.7,
    ):
        super().__init__(llm, tool_executor, max_iterations, verbose)
        self.num_paths = num_paths
        self.temperature = temperature
    
    async def run(
        self,
        task: str,
        context: Optional[Conversation] = None,
        **kwargs: Any,
    ) -> PatternResult:
        """Execute Self-Consistency CoT."""
        conversation = context or Conversation()
        
        if not any(m.role == MessageRole.SYSTEM for m in conversation.messages):
            conversation.add_system(self.get_system_prompt())
        
        conversation.add_user(task + self.ZERO_SHOT_PROMPT)
        
        self.log(f"Generating {self.num_paths} reasoning paths...")
        
        # Generate multiple reasoning paths
        paths: list[str] = []
        answers: list[str] = []
        
        for i in range(self.num_paths):
            try:
                response = await self.llm.chat(
                    messages=conversation.messages,
                    temperature=self.temperature,
                )
                
                content = response.message.content or ""
                paths.append(content)
                
                # Extract answer (simple heuristic - look for "Final Answer" or last line)
                answer = self._extract_answer(content)
                answers.append(answer)
                
                self.log(f"Path {i+1} answer: {answer[:100]}...")
            
            except Exception as e:
                self.log(f"Path {i+1} failed: {e}", level="error")
        
        if not answers:
            return PatternResult(
                success=False,
                output="",
                iterations=self.num_paths,
                messages=[],
                error="All reasoning paths failed",
            )
        
        # Find most common answer (majority voting)
        from collections import Counter
        answer_counts = Counter(answers)
        most_common_answer = answer_counts.most_common(1)[0][0]
        
        # Find the best path with this answer
        best_path = next(
            (p for p, a in zip(paths, answers) if a == most_common_answer),
            paths[0]
        )
        
        return PatternResult(
            success=True,
            output=best_path,
            iterations=len(paths),
            messages=[Message.assistant(best_path)],
            metadata={
                "num_paths": len(paths),
                "answer_distribution": dict(answer_counts),
                "consensus_answer": most_common_answer,
            },
        )
    
    def _extract_answer(self, content: str) -> str:
        """Extract the final answer from CoT output."""
        # Look for explicit final answer markers
        markers = ["final answer:", "answer:", "therefore:", "so the answer is"]
        content_lower = content.lower()
        
        for marker in markers:
            if marker in content_lower:
                idx = content_lower.rfind(marker)
                answer = content[idx + len(marker):].strip()
                # Take first line or sentence
                answer = answer.split("\n")[0].strip()
                if answer:
                    return answer
        
        # Fallback: return last non-empty line
        lines = [l.strip() for l in content.split("\n") if l.strip()]
        return lines[-1] if lines else content
