"""Short-term memory implementation."""

from collections import deque
from datetime import datetime
from typing import Any, Optional

from agentflow.memory.base import BaseMemory, MemoryEntry


class ShortTermMemory(BaseMemory):
    """Short-term memory using a fixed-size deque.
    
    This is a simple in-memory implementation that keeps the most recent
    N entries. Ideal for maintaining conversation context within a session.
    
    Example:
        ```python
        memory = ShortTermMemory(max_entries=20)
        
        await memory.add_message("Hello!", role="user")
        await memory.add_message("Hi there!", role="assistant")
        
        # Get recent messages
        recent = await memory.get_recent(limit=10)
        
        # Search by content (simple keyword matching)
        results = await memory.search("hello")
        ```
    """
    
    def __init__(self, max_entries: int = 50):
        self.max_entries = max_entries
        self._entries: deque[MemoryEntry] = deque(maxlen=max_entries)
        self._index: dict[str, MemoryEntry] = {}
    
    async def add(self, entry: MemoryEntry) -> None:
        """Add a memory entry."""
        # If at capacity, remove oldest from index
        if len(self._entries) >= self.max_entries:
            oldest = self._entries[0]
            self._index.pop(oldest.id, None)
        
        self._entries.append(entry)
        self._index[entry.id] = entry
    
    async def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get a memory entry by ID."""
        return self._index.get(entry_id)
    
    async def search(
        self,
        query: str,
        limit: int = 5,
        **kwargs: Any,
    ) -> list[MemoryEntry]:
        """Search for relevant memories using simple keyword matching."""
        query_lower = query.lower()
        
        # Simple relevance scoring based on keyword occurrence
        scored_entries = []
        for entry in self._entries:
            content_lower = entry.content.lower()
            score = 0
            
            # Count keyword matches
            for word in query_lower.split():
                if word in content_lower:
                    score += content_lower.count(word)
            
            if score > 0:
                scored_entries.append((score, entry))
        
        # Sort by score descending
        scored_entries.sort(key=lambda x: x[0], reverse=True)
        
        # Return top entries
        results = [entry for _, entry in scored_entries[:limit]]
        
        # Update access stats
        for entry in results:
            entry.access_count += 1
            entry.last_accessed = datetime.utcnow()
        
        return results
    
    async def get_recent(self, limit: int = 10) -> list[MemoryEntry]:
        """Get the most recent memories."""
        entries = list(self._entries)
        return entries[-limit:]
    
    async def clear(self) -> None:
        """Clear all memories."""
        self._entries.clear()
        self._index.clear()
    
    async def count(self) -> int:
        """Get total number of memories."""
        return len(self._entries)
    
    def get_all(self) -> list[MemoryEntry]:
        """Get all entries (sync method for convenience)."""
        return list(self._entries)
    
    async def remove(self, entry_id: str) -> bool:
        """Remove a specific entry."""
        entry = self._index.pop(entry_id, None)
        if entry:
            # Rebuild deque without the entry
            self._entries = deque(
                [e for e in self._entries if e.id != entry_id],
                maxlen=self.max_entries,
            )
            return True
        return False
    
    async def get_by_role(self, role: str) -> list[MemoryEntry]:
        """Get all entries with a specific role."""
        return [e for e in self._entries if e.role == role]
    
    async def summarize_stats(self) -> dict[str, Any]:
        """Get statistics about the memory."""
        entries = list(self._entries)
        if not entries:
            return {"count": 0}
        
        roles = {}
        for e in entries:
            roles[e.role] = roles.get(e.role, 0) + 1
        
        return {
            "count": len(entries),
            "max_entries": self.max_entries,
            "roles": roles,
            "oldest": entries[0].timestamp.isoformat() if entries else None,
            "newest": entries[-1].timestamp.isoformat() if entries else None,
        }
