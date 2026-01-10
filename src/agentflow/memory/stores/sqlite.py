"""
SQLite 存储后端
===============

轻量级持久化存储，使用 SQLite + FTS5 全文搜索。
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from .core import (
    MemoryItem,
    MemKind,
    MemRole,
    SearchResult,
)


class SQLiteStore:
    """SQLite 存储后端.
    
    特性:
    - FTS5 全文搜索
    - 异步友好 (使用同步 SQLite，但操作快速)
    - 自动迁移
    
    Example:
        ```python
        store = SQLiteStore("memory.db")
        await store.init()
        
        item = MemoryItem(content="Hello")
        await store.put(item)
        
        results = await store.search("hello")
        ```
    """
    
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS memories (
        id TEXT PRIMARY KEY,
        content TEXT NOT NULL,
        kind TEXT NOT NULL,
        role TEXT NOT NULL,
        created_at REAL NOT NULL,
        accessed_at REAL DEFAULT 0,
        importance REAL DEFAULT 0.5,
        access_count INTEGER DEFAULT 0,
        embedding TEXT,
        parent_id TEXT,
        tags TEXT,
        meta TEXT,
        content_hash TEXT
    );
    
    CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);
    CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);
    CREATE INDEX IF NOT EXISTS idx_memories_hash ON memories(content_hash);
    
    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
        id, content, tags,
        tokenize='unicode61'
    );
    """
    
    def __init__(self, path: str = "memory.db"):
        self._path = Path(path)
        self._conn: Optional[sqlite3.Connection] = None
    
    async def init(self) -> None:
        """初始化数据库."""
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(self.SCHEMA)
        self._conn.commit()
    
    async def close(self) -> None:
        """关闭连接."""
        if self._conn:
            self._conn.close()
            self._conn = None
    
    async def put(self, item: MemoryItem) -> str:
        """存储记忆项."""
        if not self._conn:
            await self.init()
        
        # 检查去重
        cursor = self._conn.execute(
            "SELECT id FROM memories WHERE content_hash = ?",
            (item.hash,)
        )
        existing = cursor.fetchone()
        if existing:
            # 更新访问计数
            self._conn.execute(
                "UPDATE memories SET access_count = access_count + 1 WHERE id = ?",
                (existing["id"],)
            )
            self._conn.commit()
            return existing["id"]
        
        # 插入主表
        self._conn.execute("""
            INSERT INTO memories 
            (id, content, kind, role, created_at, accessed_at, importance, 
             access_count, embedding, parent_id, tags, meta, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            item.id,
            item.content,
            item.kind.value,
            item.role.value,
            item.created_at,
            item.accessed_at,
            item.importance,
            item.access_count,
            json.dumps(item.embedding) if item.embedding else None,
            item.parent_id,
            ",".join(item.tags),
            json.dumps(item.meta),
            item.hash,
        ))
        
        # 插入 FTS
        self._conn.execute("""
            INSERT INTO memories_fts (id, content, tags)
            VALUES (?, ?, ?)
        """, (item.id, item.content, ",".join(item.tags)))
        
        self._conn.commit()
        return item.id
    
    async def get(self, id: str) -> Optional[MemoryItem]:
        """按 ID 获取."""
        if not self._conn:
            await self.init()
        
        cursor = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?",
            (id,)
        )
        row = cursor.fetchone()
        
        if not row:
            return None
        
        # 更新访问
        import time
        self._conn.execute("""
            UPDATE memories 
            SET accessed_at = ?, access_count = access_count + 1 
            WHERE id = ?
        """, (time.time(), id))
        self._conn.commit()
        
        return self._row_to_item(row)
    
    async def delete(self, id: str) -> bool:
        """删除记忆."""
        if not self._conn:
            await self.init()
        
        cursor = self._conn.execute(
            "DELETE FROM memories WHERE id = ?",
            (id,)
        )
        self._conn.execute(
            "DELETE FROM memories_fts WHERE id = ?",
            (id,)
        )
        self._conn.commit()
        return cursor.rowcount > 0
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        **filters: Any,
    ) -> list[SearchResult]:
        """FTS 全文搜索."""
        if not self._conn:
            await self.init()
        
        # 构建 SQL
        sql = """
            SELECT m.*, bm25(memories_fts) as fts_score
            FROM memories m
            JOIN memories_fts f ON m.id = f.id
            WHERE memories_fts MATCH ?
        """
        params: list[Any] = [query]
        
        if "kind" in filters:
            sql += " AND m.kind = ?"
            params.append(filters["kind"].value if hasattr(filters["kind"], "value") else filters["kind"])
        
        if "role" in filters:
            sql += " AND m.role = ?"
            params.append(filters["role"].value if hasattr(filters["role"], "value") else filters["role"])
        
        sql += f" ORDER BY fts_score LIMIT {limit}"
        
        try:
            cursor = self._conn.execute(sql, params)
            rows = cursor.fetchall()
        except sqlite3.OperationalError:
            # FTS 查询语法错误，回退到 LIKE
            return await self._search_like(query, limit, filters)
        
        results = []
        for row in rows:
            item = self._row_to_item(row)
            fts_score = abs(row["fts_score"]) / 10  # 归一化
            score = item.score(relevance=min(1.0, fts_score))
            results.append(SearchResult(item=item, score=score, match_type="fts"))
        
        return results
    
    async def _search_like(
        self,
        query: str,
        limit: int,
        filters: dict,
    ) -> list[SearchResult]:
        """LIKE 回退搜索."""
        sql = "SELECT * FROM memories WHERE content LIKE ?"
        params: list[Any] = [f"%{query}%"]
        
        if "kind" in filters:
            sql += " AND kind = ?"
            params.append(filters["kind"].value if hasattr(filters["kind"], "value") else filters["kind"])
        
        sql += f" ORDER BY importance DESC, created_at DESC LIMIT {limit}"
        
        cursor = self._conn.execute(sql, params)
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            item = self._row_to_item(row)
            score = item.score(relevance=0.5)
            results.append(SearchResult(item=item, score=score, match_type="like"))
        
        return results
    
    async def recent(self, limit: int = 20) -> list[MemoryItem]:
        """获取最近记忆."""
        if not self._conn:
            await self.init()
        
        cursor = self._conn.execute(
            "SELECT * FROM memories ORDER BY created_at DESC LIMIT ?",
            (limit,)
        )
        return [self._row_to_item(row) for row in cursor.fetchall()]
    
    async def clear(self) -> int:
        """清空所有记忆."""
        if not self._conn:
            await self.init()
        
        cursor = self._conn.execute("SELECT COUNT(*) FROM memories")
        count = cursor.fetchone()[0]
        
        self._conn.execute("DELETE FROM memories")
        self._conn.execute("DELETE FROM memories_fts")
        self._conn.commit()
        
        return count
    
    async def count(self) -> int:
        """统计记忆数量."""
        if not self._conn:
            await self.init()
        
        cursor = self._conn.execute("SELECT COUNT(*) FROM memories")
        return cursor.fetchone()[0]
    
    def _row_to_item(self, row: sqlite3.Row) -> MemoryItem:
        """转换数据库行为 MemoryItem."""
        return MemoryItem(
            id=row["id"],
            content=row["content"],
            kind=MemKind(row["kind"]),
            role=MemRole(row["role"]),
            created_at=row["created_at"],
            accessed_at=row["accessed_at"] or 0,
            importance=row["importance"],
            access_count=row["access_count"],
            embedding=json.loads(row["embedding"]) if row["embedding"] else None,
            parent_id=row["parent_id"],
            tags=tuple(row["tags"].split(",")) if row["tags"] else (),
            meta=json.loads(row["meta"]) if row["meta"] else {},
        )
    
    async def __aenter__(self):
        await self.init()
        return self
    
    async def __aexit__(self, *args):
        await self.close()
