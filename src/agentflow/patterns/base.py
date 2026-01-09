"""Base pattern interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from agentflow.core.message import Message, Conversation
from agentflow.llm.client import LLMClient
from agentflow.tools.executor import ToolExecutor


@dataclass
class PatternResult:
    """Result from pattern execution."""
    
    success: bool
    output: str
    iterations: int = 0
    messages: list[Message] = field(default_factory=list)
    tool_calls_made: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class BasePattern(ABC):
    """Base class for reasoning patterns.
    
    Patterns define how the agent reasons and acts to accomplish tasks.
    Different patterns are suited for different types of tasks:
    
    - ReAct: General-purpose reasoning + acting
    - CoT: Chain of Thought for complex reasoning
    - ToT: Tree of Thought for exploration tasks
    - Reflexion: Self-reflection for learning
    - PlanAndExecute: Plan then execute for structured tasks
    """
    
    name: str = "base"
    description: str = "Base reasoning pattern"
    
    def __init__(
        self,
        llm: LLMClient,
        tool_executor: Optional[ToolExecutor] = None,
        max_iterations: int = 10,
        verbose: bool = False,
    ):
        self.llm = llm
        self.tool_executor = tool_executor
        self.max_iterations = max_iterations
        self.verbose = verbose
    
    @abstractmethod
    async def run(
        self,
        task: str,
        context: Optional[Conversation] = None,
        **kwargs: Any,
    ) -> PatternResult:
        """Execute the pattern for a given task.
        
        Args:
            task: The task or query to process.
            context: Optional conversation context.
            **kwargs: Additional pattern-specific parameters.
            
        Returns:
            PatternResult containing the execution result.
        """
        pass
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this pattern."""
        return f"You are a helpful AI assistant using the {self.name} reasoning pattern."
    
    def log(self, message: str, level: str = "info") -> None:
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            import structlog
            logger = structlog.get_logger()
            getattr(logger, level)(message, pattern=self.name)
