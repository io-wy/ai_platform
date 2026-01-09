"""Tests for memory system."""

import pytest
from datetime import datetime

from agentflow.memory.base import MemoryEntry
from agentflow.memory.short_term import ShortTermMemory


class TestMemoryEntry:
    """Tests for MemoryEntry."""
    
    def test_creation(self):
        """Test creating a memory entry."""
        entry = MemoryEntry(content="Test content", role="user")
        
        assert entry.content == "Test content"
        assert entry.role == "user"
        assert entry.importance == 1.0
        assert entry.access_count == 0
    
    def test_with_metadata(self):
        """Test entry with metadata."""
        entry = MemoryEntry(
            content="Test",
            role="assistant",
            metadata={"key": "value"},
            importance=0.8,
        )
        
        assert entry.metadata == {"key": "value"}
        assert entry.importance == 0.8


class TestShortTermMemory:
    """Tests for ShortTermMemory."""
    
    @pytest.mark.asyncio
    async def test_add_and_get(self):
        """Test adding and retrieving entries."""
        memory = ShortTermMemory(max_entries=10)
        
        entry = MemoryEntry(content="Hello", role="user")
        await memory.add(entry)
        
        retrieved = await memory.get(entry.id)
        assert retrieved is not None
        assert retrieved.content == "Hello"
    
    @pytest.mark.asyncio
    async def test_add_message_convenience(self):
        """Test add_message convenience method."""
        memory = ShortTermMemory()
        
        entry = await memory.add_message("Hello!", role="user")
        
        assert entry.content == "Hello!"
        assert entry.role == "user"
        assert await memory.count() == 1
    
    @pytest.mark.asyncio
    async def test_max_entries(self):
        """Test that max entries is enforced."""
        memory = ShortTermMemory(max_entries=3)
        
        for i in range(5):
            await memory.add_message(f"Message {i}")
        
        assert await memory.count() == 3
        
        # Should have the last 3 messages
        recent = await memory.get_recent(5)
        assert len(recent) == 3
        assert recent[0].content == "Message 2"
        assert recent[2].content == "Message 4"
    
    @pytest.mark.asyncio
    async def test_search(self):
        """Test searching memories."""
        memory = ShortTermMemory()
        
        await memory.add_message("The weather is sunny today")
        await memory.add_message("I like programming in Python")
        await memory.add_message("The sun is bright")
        
        results = await memory.search("sun")
        
        assert len(results) >= 1
        # Should find entries with "sun" or "sunny"
        assert any("sun" in r.content.lower() for r in results)
    
    @pytest.mark.asyncio
    async def test_get_recent(self):
        """Test getting recent entries."""
        memory = ShortTermMemory()
        
        for i in range(5):
            await memory.add_message(f"Message {i}")
        
        recent = await memory.get_recent(3)
        
        assert len(recent) == 3
        assert recent[0].content == "Message 2"
        assert recent[2].content == "Message 4"
    
    @pytest.mark.asyncio
    async def test_clear(self):
        """Test clearing memory."""
        memory = ShortTermMemory()
        
        await memory.add_message("Test 1")
        await memory.add_message("Test 2")
        
        assert await memory.count() == 2
        
        await memory.clear()
        
        assert await memory.count() == 0
    
    @pytest.mark.asyncio
    async def test_get_by_role(self):
        """Test getting entries by role."""
        memory = ShortTermMemory()
        
        await memory.add_message("User message 1", role="user")
        await memory.add_message("Assistant response", role="assistant")
        await memory.add_message("User message 2", role="user")
        
        user_messages = await memory.get_by_role("user")
        
        assert len(user_messages) == 2
        assert all(m.role == "user" for m in user_messages)
    
    @pytest.mark.asyncio
    async def test_remove(self):
        """Test removing specific entry."""
        memory = ShortTermMemory()
        
        entry = await memory.add_message("To be removed")
        await memory.add_message("To keep")
        
        assert await memory.count() == 2
        
        removed = await memory.remove(entry.id)
        
        assert removed is True
        assert await memory.count() == 1
        assert await memory.get(entry.id) is None
    
    @pytest.mark.asyncio
    async def test_summarize_stats(self):
        """Test getting memory statistics."""
        memory = ShortTermMemory(max_entries=50)
        
        await memory.add_message("User 1", role="user")
        await memory.add_message("Assistant 1", role="assistant")
        await memory.add_message("User 2", role="user")
        
        stats = await memory.summarize_stats()
        
        assert stats["count"] == 3
        assert stats["max_entries"] == 50
        assert stats["roles"]["user"] == 2
        assert stats["roles"]["assistant"] == 1
