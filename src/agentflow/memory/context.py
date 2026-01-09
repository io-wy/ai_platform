"""Context management and optimization."""

from typing import Any, Optional

import tiktoken

from agentflow.core.message import Message, MessageRole, Conversation
from agentflow.memory.base import BaseMemory, MemoryEntry
from agentflow.memory.short_term import ShortTermMemory
from agentflow.memory.long_term import LongTermMemory


class ContextManager:
    """Manages context window and memory for LLM conversations.
    
    Features:
    - Token counting and context window management
    - Automatic context compression when approaching limits
    - Integration with short-term and long-term memory
    - Retrieval-augmented generation (RAG) support
    
    Example:
        ```python
        context_mgr = ContextManager(
            max_tokens=8000,
            short_term=ShortTermMemory(max_entries=20),
            long_term=LongTermMemory(embedding_func=embed_func),
        )
        
        # Add messages
        await context_mgr.add_message(Message.user("What is Python?"))
        await context_mgr.add_message(Message.assistant("Python is a programming language..."))
        
        # Get optimized context for LLM
        messages = await context_mgr.get_context(
            query="Tell me more about Python",
            include_relevant=True,
        )
        ```
    """
    
    def __init__(
        self,
        max_tokens: int = 8000,
        model: str = "gpt-4o-mini",
        short_term: Optional[ShortTermMemory] = None,
        long_term: Optional[LongTermMemory] = None,
        system_prompt: Optional[str] = None,
        reserved_tokens: int = 1000,  # Reserve for response
    ):
        self.max_tokens = max_tokens
        self.model = model
        self.short_term = short_term or ShortTermMemory(max_entries=50)
        self.long_term = long_term
        self.system_prompt = system_prompt
        self.reserved_tokens = reserved_tokens
        
        # Token counting
        try:
            self._tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self._tokenizer.encode(text))
    
    def count_message_tokens(self, message: Message) -> int:
        """Count tokens in a message (including role overhead)."""
        # Approximate token overhead per message
        tokens = 4  # Role and formatting overhead
        if message.content:
            tokens += self.count_tokens(message.content)
        if message.name:
            tokens += self.count_tokens(message.name) + 1
        if message.tool_calls:
            for tc in message.tool_calls:
                tokens += self.count_tokens(tc.name)
                tokens += self.count_tokens(str(tc.arguments))
        return tokens
    
    async def add_message(self, message: Message) -> None:
        """Add a message to memory."""
        entry = MemoryEntry(
            content=message.content or "",
            role=message.role.value,
            metadata={
                "message_id": message.id,
                "has_tool_calls": bool(message.tool_calls),
            },
        )
        
        # Add to short-term memory
        await self.short_term.add(entry)
        
        # Optionally add to long-term memory
        if self.long_term and message.role in (MessageRole.USER, MessageRole.ASSISTANT):
            # Only store substantial content
            if len(entry.content) > 50:
                await self.long_term.add(entry)
    
    async def get_context(
        self,
        query: Optional[str] = None,
        include_system: bool = True,
        include_relevant: bool = True,
        max_relevant: int = 3,
        min_similarity: float = 0.7,
    ) -> list[Message]:
        """Get optimized context for LLM.
        
        Args:
            query: Current query for relevance search.
            include_system: Whether to include system prompt.
            include_relevant: Whether to include relevant memories from long-term.
            max_relevant: Maximum relevant memories to include.
            min_similarity: Minimum similarity for relevant memories.
            
        Returns:
            List of messages optimized for the context window.
        """
        messages: list[Message] = []
        available_tokens = self.max_tokens - self.reserved_tokens
        used_tokens = 0
        
        # System prompt
        if include_system and self.system_prompt:
            system_msg = Message.system(self.system_prompt)
            system_tokens = self.count_message_tokens(system_msg)
            if system_tokens <= available_tokens:
                messages.append(system_msg)
                used_tokens += system_tokens
        
        # Get relevant memories from long-term (RAG)
        relevant_memories: list[MemoryEntry] = []
        if include_relevant and self.long_term and query:
            relevant_memories = await self.long_term.search(
                query=query,
                limit=max_relevant,
                min_similarity=min_similarity,
            )
        
        # Add relevant memories as context
        if relevant_memories:
            context_parts = []
            for mem in relevant_memories:
                context_parts.append(f"[Previous context - {mem.role}]: {mem.content}")
            
            context_text = "\n".join(context_parts)
            context_tokens = self.count_tokens(context_text)
            
            if used_tokens + context_tokens <= available_tokens:
                context_msg = Message.system(
                    f"Relevant context from previous conversations:\n{context_text}"
                )
                messages.append(context_msg)
                used_tokens += context_tokens
        
        # Get recent messages from short-term memory
        recent_entries = await self.short_term.get_recent(limit=100)
        
        # Calculate tokens for recent messages and include as many as possible
        recent_messages = []
        for entry in reversed(recent_entries):  # Start from oldest
            msg = Message(
                role=MessageRole(entry.role),
                content=entry.content,
            )
            msg_tokens = self.count_message_tokens(msg)
            
            if used_tokens + msg_tokens <= available_tokens:
                recent_messages.append(msg)
                used_tokens += msg_tokens
            else:
                # Context full, stop adding
                break
        
        # Add recent messages in correct order
        messages.extend(recent_messages)
        
        return messages
    
    async def compress_context(
        self,
        messages: list[Message],
        target_tokens: int,
        llm_func: Optional[Any] = None,
    ) -> list[Message]:
        """Compress context to fit within target tokens.
        
        Strategies:
        1. Remove older messages
        2. Summarize conversation history
        3. Remove tool results (keep only outcomes)
        
        Args:
            messages: Messages to compress.
            target_tokens: Target token count.
            llm_func: Optional LLM function for summarization.
            
        Returns:
            Compressed message list.
        """
        current_tokens = sum(self.count_message_tokens(m) for m in messages)
        
        if current_tokens <= target_tokens:
            return messages
        
        # Strategy 1: Remove older non-system messages
        compressed = []
        system_messages = [m for m in messages if m.role == MessageRole.SYSTEM]
        other_messages = [m for m in messages if m.role != MessageRole.SYSTEM]
        
        compressed.extend(system_messages)
        used_tokens = sum(self.count_message_tokens(m) for m in system_messages)
        
        # Keep messages from the end
        for msg in reversed(other_messages):
            msg_tokens = self.count_message_tokens(msg)
            if used_tokens + msg_tokens <= target_tokens:
                compressed.insert(len(system_messages), msg)
                used_tokens += msg_tokens
        
        # Strategy 2: If still over and we have an LLM, summarize
        if used_tokens > target_tokens and llm_func is not None:
            # Summarize the middle section
            # This would require calling the LLM to create a summary
            pass
        
        return compressed
    
    async def clear(self, keep_system: bool = True) -> None:
        """Clear all context."""
        await self.short_term.clear()
        if self.long_term:
            await self.long_term.clear()
    
    async def get_stats(self) -> dict[str, Any]:
        """Get context statistics."""
        short_term_count = await self.short_term.count()
        long_term_count = await self.long_term.count() if self.long_term else 0
        
        return {
            "short_term_entries": short_term_count,
            "long_term_entries": long_term_count,
            "max_tokens": self.max_tokens,
            "reserved_tokens": self.reserved_tokens,
            "model": self.model,
        }
