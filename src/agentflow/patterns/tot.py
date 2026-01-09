"""Tree of Thought (ToT) pattern implementation."""

import asyncio
from typing import Any, Optional

from agentflow.core.message import Message, MessageRole, Conversation
from agentflow.llm.client import LLMClient
from agentflow.patterns.base import BasePattern, PatternResult
from agentflow.tools.executor import ToolExecutor


class ThoughtNode:
    """A node in the thought tree."""
    
    def __init__(
        self,
        thought: str,
        parent: Optional["ThoughtNode"] = None,
        score: float = 0.0,
    ):
        self.thought = thought
        self.parent = parent
        self.children: list["ThoughtNode"] = []
        self.score = score
        self.is_terminal = False
    
    def get_path(self) -> list[str]:
        """Get the path from root to this node."""
        path = []
        node: Optional[ThoughtNode] = self
        while node is not None:
            path.append(node.thought)
            node = node.parent
        return list(reversed(path))


class TreeOfThoughtPattern(BasePattern):
    """Tree of Thought (ToT) pattern.
    
    The ToT pattern explores multiple reasoning paths in a tree structure,
    evaluating and pruning branches to find the best solution.
    
    Suitable for:
    - Complex problem-solving
    - Creative tasks
    - Planning problems
    - Game playing
    
    Reference: https://arxiv.org/abs/2305.10601
    
    Example:
        ```python
        pattern = TreeOfThoughtPattern(
            llm=LLMClient(),
            breadth=3,
            depth=4,
        )
        
        result = await pattern.run(
            "Solve this puzzle: ..."
        )
        ```
    """
    
    name = "tot"
    description = "Tree of Thought pattern for exploring multiple reasoning paths"
    
    THOUGHT_GENERATOR_PROMPT = '''Given the problem and current reasoning path, generate {num_thoughts} different next steps to explore.
Each step should be a distinct approach or continuation.

Problem: {problem}

Current path:
{path}

Generate {num_thoughts} different possible next steps, each on a new line:'''

    EVALUATOR_PROMPT = '''Evaluate how promising this reasoning path is for solving the problem.
Rate from 0.0 to 1.0, where:
- 0.0-0.3: Unlikely to lead to solution
- 0.4-0.6: Might lead to solution  
- 0.7-1.0: Likely to lead to solution

Problem: {problem}

Reasoning path:
{path}

Respond with just a number between 0.0 and 1.0:'''

    SOLUTION_CHECK_PROMPT = '''Does this reasoning path provide a complete solution to the problem?

Problem: {problem}

Reasoning path:
{path}

Respond with "YES" if this is a complete solution, or "NO" if more reasoning is needed:'''

    def __init__(
        self,
        llm: LLMClient,
        tool_executor: Optional[ToolExecutor] = None,
        max_iterations: int = 50,
        verbose: bool = False,
        breadth: int = 3,  # Number of thoughts to generate at each step
        depth: int = 5,  # Maximum depth of the tree
        beam_width: int = 3,  # Number of best paths to keep
        threshold: float = 0.3,  # Minimum score to continue exploration
    ):
        super().__init__(llm, tool_executor, max_iterations, verbose)
        self.breadth = breadth
        self.depth = depth
        self.beam_width = beam_width
        self.threshold = threshold
    
    async def run(
        self,
        task: str,
        context: Optional[Conversation] = None,
        **kwargs: Any,
    ) -> PatternResult:
        """Execute Tree of Thought pattern using beam search."""
        self.log(f"Starting ToT with breadth={self.breadth}, depth={self.depth}")
        
        # Initialize root
        root = ThoughtNode(thought=f"Problem: {task}")
        
        # Current beam (best paths)
        beam: list[ThoughtNode] = [root]
        iterations = 0
        best_solution: Optional[ThoughtNode] = None
        
        for depth_level in range(self.depth):
            if iterations >= self.max_iterations:
                break
            
            self.log(f"Exploring depth {depth_level + 1}/{self.depth}")
            
            # Generate children for all nodes in beam
            all_candidates: list[ThoughtNode] = []
            
            for node in beam:
                iterations += 1
                if iterations >= self.max_iterations:
                    break
                
                # Check if this is already a solution
                if await self._is_solution(task, node):
                    node.is_terminal = True
                    if best_solution is None or node.score > best_solution.score:
                        best_solution = node
                    continue
                
                # Generate next thoughts
                thoughts = await self._generate_thoughts(task, node)
                
                # Create child nodes and evaluate
                for thought in thoughts:
                    iterations += 1
                    if iterations >= self.max_iterations:
                        break
                    
                    child = ThoughtNode(thought=thought, parent=node)
                    score = await self._evaluate_path(task, child)
                    child.score = score
                    node.children.append(child)
                    
                    if score >= self.threshold:
                        all_candidates.append(child)
            
            if not all_candidates:
                self.log("No promising candidates found, stopping")
                break
            
            # Select top-k candidates for next beam
            all_candidates.sort(key=lambda n: n.score, reverse=True)
            beam = all_candidates[:self.beam_width]
            
            self.log(f"Selected {len(beam)} candidates for next level, best score: {beam[0].score:.2f}")
        
        # Get best solution
        if best_solution is None:
            # Use the highest-scoring leaf
            best_solution = max(beam, key=lambda n: n.score) if beam else root
        
        # Format output
        path = best_solution.get_path()
        output = "\n\n".join([f"**Step {i}:** {step}" for i, step in enumerate(path)])
        
        return PatternResult(
            success=best_solution.is_terminal or best_solution.score > 0.5,
            output=output,
            iterations=iterations,
            messages=[Message.assistant(output)],
            metadata={
                "depth_reached": len(path),
                "final_score": best_solution.score,
                "is_complete_solution": best_solution.is_terminal,
            },
        )
    
    async def _generate_thoughts(self, problem: str, node: ThoughtNode) -> list[str]:
        """Generate next possible thoughts from current node."""
        path = node.get_path()
        path_str = "\n".join([f"{i+1}. {step}" for i, step in enumerate(path)])
        
        prompt = self.THOUGHT_GENERATOR_PROMPT.format(
            num_thoughts=self.breadth,
            problem=problem,
            path=path_str,
        )
        
        try:
            response = await self.llm.chat(
                messages=[Message.user(prompt)],
                temperature=0.8,  # Higher temperature for diversity
            )
            
            content = response.message.content or ""
            thoughts = [t.strip() for t in content.strip().split("\n") if t.strip()]
            return thoughts[:self.breadth]
        
        except Exception as e:
            self.log(f"Failed to generate thoughts: {e}", level="error")
            return []
    
    async def _evaluate_path(self, problem: str, node: ThoughtNode) -> float:
        """Evaluate how promising a reasoning path is."""
        path = node.get_path()
        path_str = "\n".join([f"{i+1}. {step}" for i, step in enumerate(path)])
        
        prompt = self.EVALUATOR_PROMPT.format(
            problem=problem,
            path=path_str,
        )
        
        try:
            response = await self.llm.chat(
                messages=[Message.user(prompt)],
                temperature=0.1,  # Low temperature for consistent evaluation
            )
            
            content = response.message.content or "0.5"
            # Extract number from response
            import re
            match = re.search(r"([0-9]*\.?[0-9]+)", content)
            if match:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
            return 0.5
        
        except Exception as e:
            self.log(f"Failed to evaluate path: {e}", level="error")
            return 0.5
    
    async def _is_solution(self, problem: str, node: ThoughtNode) -> bool:
        """Check if the current path is a complete solution."""
        path = node.get_path()
        if len(path) < 2:  # Need at least some reasoning
            return False
        
        path_str = "\n".join([f"{i+1}. {step}" for i, step in enumerate(path)])
        
        prompt = self.SOLUTION_CHECK_PROMPT.format(
            problem=problem,
            path=path_str,
        )
        
        try:
            response = await self.llm.chat(
                messages=[Message.user(prompt)],
                temperature=0.1,
            )
            
            content = response.message.content or ""
            return "YES" in content.upper()
        
        except Exception as e:
            self.log(f"Failed to check solution: {e}", level="error")
            return False
