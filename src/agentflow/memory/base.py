"""Base memory interface."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class MemoryEntry(BaseModel):
    """A single memory entry."""
    
    id: str = Field(default_factory=lambda: uuid4().hex)
    content: str
    role: str = "user"  # user, assistant, system, tool
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    # For vector search
    embedding: Optional[list[float]] = None
    
    # Importance score (for memory pruning)
    importance: float = Field(default=1.0, ge=0.0, le=1.0)
    
    # Number of times this memory was retrieved
    access_count: int = 0
    last_accessed: Optional[datetime] = None


class BaseMemory(ABC):
    """Abstract base class for memory implementations."""
    
    @abstractmethod
    async def add(self, entry: MemoryEntry) -> None:
        """Add a memory entry."""
        pass
    
    @abstractmethod
    async def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get a memory entry by ID."""
        pass
    
    @abstractmethod
    async def search(
        self,
        query: str,
        limit: int = 5,
        **kwargs: Any,
    ) -> list[MemoryEntry]:
        """Search for relevant memories."""
        pass
    
    @abstractmethod
    async def get_recent(self, limit: int = 10) -> list[MemoryEntry]:
        """Get the most recent memories."""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all memories."""
        pass
    
    @abstractmethod
    async def count(self) -> int:
        """Get total number of memories."""
        pass
    
    async def add_message(
        self,
        content: str,
        role: str = "user",
        metadata: Optional[dict[str, Any]] = None,
        importance: float = 1.0,
    ) -> MemoryEntry:
        """Convenience method to add a message as memory."""
        entry = MemoryEntry(
            content=content,
            role=role,
            metadata=metadata or {},
            importance=importance,
        )
        await self.add(entry)
        return entry
    
    async def to_messages(self, limit: Optional[int] = None) -> list[dict[str, Any]]:
        """Convert recent memories to message format for LLM."""
        entries = await self.get_recent(limit=limit or 100)
        return [
            {
                "role": entry.role,
                "content": entry.content,
            }
            for entry in entries
        ]
