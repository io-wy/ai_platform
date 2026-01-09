"""Database-backed memory storage."""

import json
import asyncio
from datetime import datetime, timezone
from typing import Any, Optional
from pathlib import Path

from agentflow.memory.base import BaseMemory, MemoryEntry


class DatabaseMemory(BaseMemory):
    """SQLite-backed persistent memory storage.
    
    Features:
    - Persistent storage across sessions
    - Full-text search support
    - Efficient querying
    - Transaction support
    """
    
    def __init__(
        self,
        db_path: str = "agentflow_memory.db",
        table_name: str = "memories",
    ):
        self.db_path = db_path
        self.table_name = table_name
        self._conn = None
        self._initialized = False
    
    async def _ensure_initialized(self):
        """Ensure database is initialized."""
        if self._initialized:
            return
        
        try:
            import aiosqlite
        except ImportError:
            raise ImportError("aiosqlite is required for DatabaseMemory. Install with: pip install aiosqlite")
        
        self._conn = await aiosqlite.connect(self.db_path)
        
        # Create table with full-text search
        await self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                role TEXT,
                timestamp TEXT NOT NULL,
                importance REAL DEFAULT 1.0,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT,
                metadata TEXT,
                embedding TEXT
            )
        """)
        
        # Create indexes
        await self._conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self.table_name}_role 
            ON {self.table_name}(role)
        """)
        await self._conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self.table_name}_timestamp 
            ON {self.table_name}(timestamp)
        """)
        await self._conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self.table_name}_importance 
            ON {self.table_name}(importance DESC)
        """)
        
        # Create FTS table for full-text search
        await self._conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {self.table_name}_fts 
            USING fts5(id, content, role)
        """)
        
        await self._conn.commit()
        self._initialized = True
    
    async def add(self, entry: MemoryEntry) -> MemoryEntry:
        """Add an entry to memory."""
        await self._ensure_initialized()
        
        metadata_json = json.dumps(entry.metadata) if entry.metadata else None
        embedding_json = json.dumps(entry.embedding) if entry.embedding else None
        
        await self._conn.execute(f"""
            INSERT OR REPLACE INTO {self.table_name}
            (id, content, role, timestamp, importance, access_count, last_accessed, metadata, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.id,
            entry.content,
            entry.role,
            entry.timestamp.isoformat(),
            entry.importance,
            entry.access_count,
            entry.last_accessed.isoformat() if entry.last_accessed else None,
            metadata_json,
            embedding_json,
        ))
        
        # Add to FTS index
        await self._conn.execute(f"""
            INSERT OR REPLACE INTO {self.table_name}_fts (id, content, role)
            VALUES (?, ?, ?)
        """, (entry.id, entry.content, entry.role))
        
        await self._conn.commit()
        return entry
    
    async def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get an entry by ID."""
        await self._ensure_initialized()
        
        cursor = await self._conn.execute(f"""
            SELECT id, content, role, timestamp, importance, access_count, last_accessed, metadata, embedding
            FROM {self.table_name}
            WHERE id = ?
        """, (entry_id,))
        
        row = await cursor.fetchone()
        if not row:
            return None
        
        return self._row_to_entry(row)
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        role: Optional[str] = None,
        min_importance: float = 0.0,
        **kwargs: Any,
    ) -> list[MemoryEntry]:
        """Search memories using full-text search."""
        await self._ensure_initialized()
        
        # Use FTS for search
        sql = f"""
            SELECT m.id, m.content, m.role, m.timestamp, m.importance, 
                   m.access_count, m.last_accessed, m.metadata, m.embedding
            FROM {self.table_name} m
            JOIN {self.table_name}_fts fts ON m.id = fts.id
            WHERE {self.table_name}_fts MATCH ?
            AND m.importance >= ?
        """
        params = [query, min_importance]
        
        if role:
            sql += " AND m.role = ?"
            params.append(role)
        
        sql += f" ORDER BY rank LIMIT {limit}"
        
        cursor = await self._conn.execute(sql, params)
        rows = await cursor.fetchall()
        
        entries = [self._row_to_entry(row) for row in rows]
        
        # Update access count
        for entry in entries:
            await self._update_access(entry.id)
        
        return entries
    
    async def get_recent(self, limit: int = 10, role: Optional[str] = None) -> list[MemoryEntry]:
        """Get recent entries."""
        await self._ensure_initialized()
        
        sql = f"""
            SELECT id, content, role, timestamp, importance, access_count, last_accessed, metadata, embedding
            FROM {self.table_name}
        """
        params = []
        
        if role:
            sql += " WHERE role = ?"
            params.append(role)
        
        sql += f" ORDER BY timestamp DESC LIMIT {limit}"
        
        cursor = await self._conn.execute(sql, params)
        rows = await cursor.fetchall()
        
        return [self._row_to_entry(row) for row in rows]
    
    async def get_by_role(self, role: str, limit: int = 100) -> list[MemoryEntry]:
        """Get entries by role."""
        await self._ensure_initialized()
        
        cursor = await self._conn.execute(f"""
            SELECT id, content, role, timestamp, importance, access_count, last_accessed, metadata, embedding
            FROM {self.table_name}
            WHERE role = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (role, limit))
        
        rows = await cursor.fetchall()
        return [self._row_to_entry(row) for row in rows]
    
    async def remove(self, entry_id: str) -> bool:
        """Remove an entry."""
        await self._ensure_initialized()
        
        cursor = await self._conn.execute(f"""
            DELETE FROM {self.table_name} WHERE id = ?
        """, (entry_id,))
        
        await self._conn.execute(f"""
            DELETE FROM {self.table_name}_fts WHERE id = ?
        """, (entry_id,))
        
        await self._conn.commit()
        return cursor.rowcount > 0
    
    async def clear(self) -> None:
        """Clear all entries."""
        await self._ensure_initialized()
        
        await self._conn.execute(f"DELETE FROM {self.table_name}")
        await self._conn.execute(f"DELETE FROM {self.table_name}_fts")
        await self._conn.commit()
    
    async def count(self) -> int:
        """Count total entries."""
        await self._ensure_initialized()
        
        cursor = await self._conn.execute(f"SELECT COUNT(*) FROM {self.table_name}")
        row = await cursor.fetchone()
        return row[0] if row else 0
    
    async def get_important(self, limit: int = 10, min_importance: float = 0.5) -> list[MemoryEntry]:
        """Get important memories."""
        await self._ensure_initialized()
        
        cursor = await self._conn.execute(f"""
            SELECT id, content, role, timestamp, importance, access_count, last_accessed, metadata, embedding
            FROM {self.table_name}
            WHERE importance >= ?
            ORDER BY importance DESC, access_count DESC
            LIMIT ?
        """, (min_importance, limit))
        
        rows = await cursor.fetchall()
        return [self._row_to_entry(row) for row in rows]
    
    async def update_importance(self, entry_id: str, importance: float) -> bool:
        """Update entry importance."""
        await self._ensure_initialized()
        
        cursor = await self._conn.execute(f"""
            UPDATE {self.table_name}
            SET importance = ?
            WHERE id = ?
        """, (importance, entry_id))
        
        await self._conn.commit()
        return cursor.rowcount > 0
    
    async def _update_access(self, entry_id: str):
        """Update access count and time."""
        now = datetime.now(timezone.utc).isoformat()
        await self._conn.execute(f"""
            UPDATE {self.table_name}
            SET access_count = access_count + 1, last_accessed = ?
            WHERE id = ?
        """, (now, entry_id))
        await self._conn.commit()
    
    def _row_to_entry(self, row: tuple) -> MemoryEntry:
        """Convert database row to MemoryEntry."""
        return MemoryEntry(
            id=row[0],
            content=row[1],
            role=row[2],
            timestamp=datetime.fromisoformat(row[3]),
            importance=row[4],
            access_count=row[5],
            last_accessed=datetime.fromisoformat(row[6]) if row[6] else None,
            metadata=json.loads(row[7]) if row[7] else {},
            embedding=json.loads(row[8]) if row[8] else None,
        )
    
    async def close(self):
        """Close the database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None
            self._initialized = False
    
    async def __aenter__(self):
        await self._ensure_initialized()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class ConversationStore:
    """Store and manage conversation history in database.
    
    Features:
    - Session management
    - Conversation retrieval
    - Message search
    """
    
    def __init__(self, db_path: str = "agentflow_conversations.db"):
        self.db_path = db_path
        self._conn = None
        self._initialized = False
    
    async def _ensure_initialized(self):
        """Initialize database."""
        if self._initialized:
            return
        
        import aiosqlite
        self._conn = await aiosqlite.connect(self.db_path)
        
        # Sessions table
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                name TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT
            )
        """)
        
        # Messages table
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        
        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_session 
            ON messages(session_id)
        """)
        
        await self._conn.commit()
        self._initialized = True
    
    async def create_session(
        self,
        session_id: Optional[str] = None,
        name: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """Create a new conversation session."""
        await self._ensure_initialized()
        
        import uuid
        session_id = session_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        
        await self._conn.execute("""
            INSERT INTO sessions (id, name, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (
            session_id,
            name or f"Session {session_id[:8]}",
            now,
            now,
            json.dumps(metadata) if metadata else None,
        ))
        
        await self._conn.commit()
        return session_id
    
    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """Add a message to a session."""
        await self._ensure_initialized()
        
        import uuid
        message_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        
        await self._conn.execute("""
            INSERT INTO messages (id, session_id, role, content, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            message_id,
            session_id,
            role,
            content,
            now,
            json.dumps(metadata) if metadata else None,
        ))
        
        # Update session timestamp
        await self._conn.execute("""
            UPDATE sessions SET updated_at = ? WHERE id = ?
        """, (now, session_id))
        
        await self._conn.commit()
        return message_id
    
    async def get_session_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Get all messages in a session."""
        await self._ensure_initialized()
        
        sql = """
            SELECT id, role, content, timestamp, metadata
            FROM messages
            WHERE session_id = ?
            ORDER BY timestamp ASC
        """
        if limit:
            sql += f" LIMIT {limit}"
        
        cursor = await self._conn.execute(sql, (session_id,))
        rows = await cursor.fetchall()
        
        return [
            {
                "id": row[0],
                "role": row[1],
                "content": row[2],
                "timestamp": row[3],
                "metadata": json.loads(row[4]) if row[4] else {},
            }
            for row in rows
        ]
    
    async def get_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent sessions."""
        await self._ensure_initialized()
        
        cursor = await self._conn.execute("""
            SELECT id, name, created_at, updated_at, metadata
            FROM sessions
            ORDER BY updated_at DESC
            LIMIT ?
        """, (limit,))
        
        rows = await cursor.fetchall()
        return [
            {
                "id": row[0],
                "name": row[1],
                "created_at": row[2],
                "updated_at": row[3],
                "metadata": json.loads(row[4]) if row[4] else {},
            }
            for row in rows
        ]
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and its messages."""
        await self._ensure_initialized()
        
        await self._conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        cursor = await self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        await self._conn.commit()
        
        return cursor.rowcount > 0
    
    async def search_messages(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search messages across sessions."""
        await self._ensure_initialized()
        
        sql = """
            SELECT m.id, m.session_id, m.role, m.content, m.timestamp, m.metadata, s.name
            FROM messages m
            JOIN sessions s ON m.session_id = s.id
            WHERE m.content LIKE ?
        """
        params = [f"%{query}%"]
        
        if session_id:
            sql += " AND m.session_id = ?"
            params.append(session_id)
        
        sql += f" ORDER BY m.timestamp DESC LIMIT {limit}"
        
        cursor = await self._conn.execute(sql, params)
        rows = await cursor.fetchall()
        
        return [
            {
                "id": row[0],
                "session_id": row[1],
                "role": row[2],
                "content": row[3],
                "timestamp": row[4],
                "metadata": json.loads(row[5]) if row[5] else {},
                "session_name": row[6],
            }
            for row in rows
        ]
    
    async def close(self):
        """Close database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None
            self._initialized = False
    
    async def __aenter__(self):
        await self._ensure_initialized()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
