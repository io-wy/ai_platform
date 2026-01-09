"""Core Agent implementation."""

from typing import Any, Optional, Type

import structlog

from agentflow.core.config import AgentConfig, ReasoningPattern, LLMConfig
from agentflow.core.message import Message, Conversation
from agentflow.llm.client import LLMClient
from agentflow.memory.context import ContextManager
from agentflow.memory.short_term import ShortTermMemory
from agentflow.memory.long_term import LongTermMemory
from agentflow.patterns.base import BasePattern, PatternResult
from agentflow.patterns.react import ReActPattern
from agentflow.patterns.cot import ChainOfThoughtPattern
from agentflow.patterns.tot import TreeOfThoughtPattern
from agentflow.patterns.reflexion import ReflexionPattern
from agentflow.patterns.plan_execute import PlanAndExecutePattern
from agentflow.patterns.auto import AutoPattern
from agentflow.tools.base import BaseTool, ToolRegistry
from agentflow.tools.executor import ToolExecutor

logger = structlog.get_logger()


class Agent:
    """Main Agent class for AgentFlow.
    
    The Agent orchestrates LLM interactions, tool usage, memory management,
    and reasoning patterns to accomplish tasks.
    
    Example:
        ```python
        from agentflow import Agent, AgentConfig
        from agentflow.tools.builtin import BrowserTool, TerminalTool
        
        # Create agent with configuration
        config = AgentConfig(
            name="MyAgent",
            pattern=ReasoningPattern.AUTO,
            llm=LLMConfig(model="gpt-4o-mini"),
        )
        
        agent = Agent(config)
        
        # Register tools
        agent.register_tool(BrowserTool())
        agent.register_tool(TerminalTool())
        
        # Run a task
        result = await agent.run("Search for the latest Python news")
        print(result.output)
        ```
    """
    
    # Pattern mapping
    PATTERN_CLASSES: dict[ReasoningPattern, Type[BasePattern]] = {
        ReasoningPattern.REACT: ReActPattern,
        ReasoningPattern.COT: ChainOfThoughtPattern,
        ReasoningPattern.TOT: TreeOfThoughtPattern,
        ReasoningPattern.REFLEXION: ReflexionPattern,
        ReasoningPattern.PLAN_AND_EXECUTE: PlanAndExecutePattern,
        ReasoningPattern.AUTO: AutoPattern,
    }
    
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        llm: Optional[LLMClient] = None,
        tools: Optional[list[BaseTool]] = None,
    ):
        """Initialize the Agent.
        
        Args:
            config: Agent configuration. Uses defaults if not provided.
            llm: LLM client. Created from config if not provided.
            tools: Initial list of tools to register.
        """
        self.config = config or AgentConfig()
        
        # Initialize LLM client
        self.llm = llm or LLMClient(config=self.config.llm)
        
        # Initialize tool registry and executor
        self.tool_registry = ToolRegistry()
        self.tool_executor = ToolExecutor(
            registry=self.tool_registry,
            config=self.config.tools,
        )
        
        # Register provided tools
        if tools:
            for tool in tools:
                self.register_tool(tool)
        
        # Initialize memory
        self._init_memory()
        
        # Initialize pattern
        self.pattern: Optional[BasePattern] = None
        self._init_pattern()
        
        # Conversation state
        self.conversation = Conversation()
        if self.config.system_prompt:
            self.conversation.add_system(self.config.system_prompt)
        
        logger.info(
            "Agent initialized",
            name=self.config.name,
            pattern=self.config.pattern.value,
            model=self.config.llm.model,
        )
    
    def _init_memory(self) -> None:
        """Initialize memory systems."""
        memory_config = self.config.memory
        
        # Short-term memory
        self.short_term_memory = ShortTermMemory(
            max_entries=memory_config.max_short_term_messages
        )
        
        # Long-term memory (optional)
        self.long_term_memory: Optional[LongTermMemory] = None
        if memory_config.enable_long_term:
            self.long_term_memory = LongTermMemory(
                embedding_func=self.llm.embed,
                embedding_model=memory_config.embedding_model,
                persist_directory=memory_config.memory_path if memory_config.persist_memory else None,
            )
        
        # Context manager
        self.context_manager = ContextManager(
            max_tokens=memory_config.max_context_tokens,
            model=self.config.llm.model,
            short_term=self.short_term_memory,
            long_term=self.long_term_memory,
            system_prompt=self.config.system_prompt,
        )
    
    def _init_pattern(self) -> None:
        """Initialize the reasoning pattern."""
        pattern_class = self.PATTERN_CLASSES.get(self.config.pattern)
        
        if pattern_class:
            self.pattern = pattern_class(
                llm=self.llm,
                tool_executor=self.tool_executor,
                max_iterations=self.config.max_iterations,
                verbose=self.config.verbose,
            )
    
    def register_tool(self, tool: BaseTool) -> None:
        """Register a tool with the agent.
        
        Args:
            tool: The tool instance to register.
        """
        self.tool_registry.register(tool)
        logger.debug("Tool registered", tool=tool.name, category=tool.category)
    
    def register_tools(self, tools: list[BaseTool]) -> None:
        """Register multiple tools.
        
        Args:
            tools: List of tool instances to register.
        """
        for tool in tools:
            self.register_tool(tool)
    
    def unregister_tool(self, name: str) -> None:
        """Unregister a tool by name.
        
        Args:
            name: Name of the tool to unregister.
        """
        self.tool_registry.unregister(name)
        logger.debug("Tool unregistered", tool=name)
    
    async def run(
        self,
        task: str,
        context: Optional[Conversation] = None,
        **kwargs: Any,
    ) -> PatternResult:
        """Run the agent on a task.
        
        Args:
            task: The task or query to process.
            context: Optional conversation context.
            **kwargs: Additional pattern-specific parameters.
            
        Returns:
            PatternResult containing the execution result.
        """
        logger.info("Running task", task=task[:100], pattern=self.config.pattern.value)
        
        # Add task to memory
        await self.context_manager.add_message(Message.user(task))
        
        # Get context with relevant memories
        if context is None:
            messages = await self.context_manager.get_context(
                query=task,
                include_relevant=self.long_term_memory is not None,
            )
            context = Conversation(messages=messages)
        
        # Run the pattern
        if self.pattern is None:
            raise RuntimeError("No pattern configured")
        
        result = await self.pattern.run(task, context, **kwargs)
        
        # Store result in memory
        if result.output:
            await self.context_manager.add_message(
                Message.assistant(result.output)
            )
        
        logger.info(
            "Task completed",
            success=result.success,
            iterations=result.iterations,
            tool_calls=result.tool_calls_made,
        )
        
        return result
    
    async def chat(self, message: str, **kwargs: Any) -> str:
        """Simple chat interface.
        
        Args:
            message: User message.
            **kwargs: Additional parameters.
            
        Returns:
            Agent's response string.
        """
        result = await self.run(message, **kwargs)
        return result.output
    
    async def run_with_tools(
        self,
        task: str,
        tools: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> PatternResult:
        """Run a task with specific tools enabled.
        
        Args:
            task: The task to process.
            tools: List of tool names to enable. If None, uses all registered tools.
            **kwargs: Additional parameters.
            
        Returns:
            PatternResult containing the execution result.
        """
        # Temporarily limit tools if specified
        original_enabled = self.config.tools.enabled_tools
        
        if tools:
            self.config.tools.enabled_tools = tools
        
        try:
            return await self.run(task, **kwargs)
        finally:
            self.config.tools.enabled_tools = original_enabled
    
    def set_pattern(self, pattern: ReasoningPattern) -> None:
        """Change the reasoning pattern.
        
        Args:
            pattern: The new pattern to use.
        """
        self.config.pattern = pattern
        self._init_pattern()
        logger.info("Pattern changed", pattern=pattern.value)
    
    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt.
        
        Args:
            prompt: The new system prompt.
        """
        self.config.system_prompt = prompt
        self.context_manager.system_prompt = prompt
        
        # Update conversation
        self.conversation.messages = [
            m for m in self.conversation.messages
            if m.role != Message.system
        ]
        if prompt:
            self.conversation.messages.insert(0, Message.system(prompt))
    
    async def clear_memory(self) -> None:
        """Clear all memory."""
        await self.context_manager.clear(keep_system=True)
        self.conversation.clear(keep_system=True)
        logger.info("Memory cleared")
    
    async def get_memory_stats(self) -> dict[str, Any]:
        """Get memory statistics.
        
        Returns:
            Dictionary with memory statistics.
        """
        return await self.context_manager.get_stats()
    
    def get_available_tools(self) -> list[str]:
        """Get list of available tool names.
        
        Returns:
            List of registered tool names.
        """
        return self.tool_registry.get_names()
    
    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Get OpenAI-compatible schemas for all tools.
        
        Returns:
            List of tool schemas.
        """
        return self.tool_executor.get_tool_schemas()
    
    async def close(self) -> None:
        """Close the agent and release resources."""
        await self.llm.close()
        logger.info("Agent closed", name=self.config.name)
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    @classmethod
    def from_config_file(cls, path: str) -> "Agent":
        """Create an agent from a configuration file.
        
        Args:
            path: Path to the configuration file (YAML or JSON).
            
        Returns:
            Configured Agent instance.
        """
        config = AgentConfig.from_file(path)
        return cls(config=config)
    
    @classmethod
    def quick_start(
        cls,
        model: str = "gpt-4o-mini",
        pattern: ReasoningPattern = ReasoningPattern.AUTO,
        tools: Optional[list[BaseTool]] = None,
    ) -> "Agent":
        """Quick start with minimal configuration.
        
        Args:
            model: Model name to use.
            pattern: Reasoning pattern.
            tools: Optional list of tools.
            
        Returns:
            Configured Agent instance.
        """
        config = AgentConfig(
            llm=LLMConfig(model=model),
            pattern=pattern,
        )
        return cls(config=config, tools=tools)
