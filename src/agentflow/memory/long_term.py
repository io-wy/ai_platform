"""Long-term memory with vector search."""

from datetime import datetime
from typing import Any, Optional

from agentflow.memory.base import BaseMemory, MemoryEntry


class LongTermMemory(BaseMemory):
    """Long-term memory using vector embeddings for semantic search.
    
    Uses ChromaDB for vector storage and retrieval. Supports:
    - Semantic similarity search
    - Metadata filtering
    - Persistent storage
    
    Example:
        ```python
        from agentflow.llm import LLMClient
        
        client = LLMClient()
        memory = LongTermMemory(
            embedding_func=client.embed,
            persist_directory="./memory_db"
        )
        
        await memory.add_message("The capital of France is Paris.", role="assistant")
        
        # Semantic search
        results = await memory.search("What city is France's capital?")
        ```
    """
    
    def __init__(
        self,
        embedding_func: Optional[Any] = None,
        collection_name: str = "agentflow_memory",
        persist_directory: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
    ):
        self.embedding_func = embedding_func
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        
        self._client = None
        self._collection = None
    
    async def _ensure_collection(self):
        """Ensure ChromaDB collection is initialized."""
        if self._collection is None:
            try:
                import chromadb
                from chromadb.config import Settings
            except ImportError:
                raise ImportError(
                    "chromadb is required for LongTermMemory. "
                    "Install it with: pip install chromadb"
                )
            
            if self.persist_directory:
                self._client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=Settings(anonymized_telemetry=False),
                )
            else:
                self._client = chromadb.Client(
                    settings=Settings(anonymized_telemetry=False),
                )
            
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
    
    async def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text."""
        if self.embedding_func is None:
            raise ValueError("embedding_func is required for long-term memory")
        
        embeddings = await self.embedding_func([text], model=self.embedding_model)
        return embeddings[0]
    
    async def add(self, entry: MemoryEntry) -> None:
        """Add a memory entry with embedding."""
        await self._ensure_collection()
        
        # Generate embedding if not provided
        if entry.embedding is None:
            entry.embedding = await self._get_embedding(entry.content)
        
        # Prepare metadata
        metadata = {
            "role": entry.role,
            "timestamp": entry.timestamp.isoformat(),
            "importance": entry.importance,
            **{k: str(v) for k, v in entry.metadata.items()},
        }
        
        self._collection.add(
            ids=[entry.id],
            embeddings=[entry.embedding],
            documents=[entry.content],
            metadatas=[metadata],
        )
    
    async def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get a memory entry by ID."""
        await self._ensure_collection()
        
        result = self._collection.get(
            ids=[entry_id],
            include=["documents", "metadatas", "embeddings"],
        )
        
        if not result["ids"]:
            return None
        
        metadata = result["metadatas"][0]
        return MemoryEntry(
            id=entry_id,
            content=result["documents"][0],
            role=metadata.get("role", "user"),
            timestamp=datetime.fromisoformat(metadata.get("timestamp", datetime.utcnow().isoformat())),
            importance=float(metadata.get("importance", 1.0)),
            embedding=result["embeddings"][0] if result["embeddings"] else None,
            metadata={k: v for k, v in metadata.items() if k not in ("role", "timestamp", "importance")},
        )
    
    async def search(
        self,
        query: str,
        limit: int = 5,
        min_similarity: float = 0.5,
        role_filter: Optional[str] = None,
        **kwargs: Any,
    ) -> list[MemoryEntry]:
        """Search for relevant memories using semantic similarity."""
        await self._ensure_collection()
        
        # Get query embedding
        query_embedding = await self._get_embedding(query)
        
        # Build where clause
        where = None
        if role_filter:
            where = {"role": role_filter}
        
        # Query ChromaDB
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where,
            include=["documents", "metadatas", "embeddings", "distances"],
        )
        
        entries = []
        for i, entry_id in enumerate(results["ids"][0]):
            # Convert distance to similarity (ChromaDB returns distances)
            distance = results["distances"][0][i]
            similarity = 1 - distance  # For cosine distance
            
            if similarity < min_similarity:
                continue
            
            metadata = results["metadatas"][0][i]
            entry = MemoryEntry(
                id=entry_id,
                content=results["documents"][0][i],
                role=metadata.get("role", "user"),
                timestamp=datetime.fromisoformat(metadata.get("timestamp", datetime.utcnow().isoformat())),
                importance=float(metadata.get("importance", 1.0)),
                embedding=results["embeddings"][0][i] if results["embeddings"] else None,
                metadata={
                    "similarity": similarity,
                    **{k: v for k, v in metadata.items() if k not in ("role", "timestamp", "importance")},
                },
            )
            
            # Update access stats (note: ChromaDB doesn't store this, track locally if needed)
            entry.access_count += 1
            entry.last_accessed = datetime.utcnow()
            
            entries.append(entry)
        
        return entries
    
    async def get_recent(self, limit: int = 10) -> list[MemoryEntry]:
        """Get the most recent memories."""
        await self._ensure_collection()
        
        # ChromaDB doesn't have good sorting support, so we get all and sort
        # For large collections, consider using a database alongside
        result = self._collection.get(
            include=["documents", "metadatas", "embeddings"],
        )
        
        if not result["ids"]:
            return []
        
        entries = []
        for i, entry_id in enumerate(result["ids"]):
            metadata = result["metadatas"][i]
            entry = MemoryEntry(
                id=entry_id,
                content=result["documents"][i],
                role=metadata.get("role", "user"),
                timestamp=datetime.fromisoformat(metadata.get("timestamp", datetime.utcnow().isoformat())),
                importance=float(metadata.get("importance", 1.0)),
                embedding=result["embeddings"][i] if result["embeddings"] else None,
            )
            entries.append(entry)
        
        # Sort by timestamp descending
        entries.sort(key=lambda e: e.timestamp, reverse=True)
        
        return entries[:limit]
    
    async def clear(self) -> None:
        """Clear all memories."""
        await self._ensure_collection()
        
        # Delete and recreate collection
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    
    async def count(self) -> int:
        """Get total number of memories."""
        await self._ensure_collection()
        return self._collection.count()
    
    async def delete(self, entry_id: str) -> bool:
        """Delete a specific memory."""
        await self._ensure_collection()
        
        try:
            self._collection.delete(ids=[entry_id])
            return True
        except Exception:
            return False
    
    async def update_importance(self, entry_id: str, importance: float) -> bool:
        """Update the importance score of a memory."""
        await self._ensure_collection()
        
        try:
            self._collection.update(
                ids=[entry_id],
                metadatas=[{"importance": importance}],
            )
            return True
        except Exception:
            return False
