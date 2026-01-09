"""Database query tool."""

from typing import Any, Optional

from pydantic import BaseModel, Field

from agentflow.tools.base import BaseTool, ToolResult


class DatabaseParameters(BaseModel):
    """Parameters for database operations."""
    
    action: str = Field(
        description="Action: 'query' (SELECT), 'execute' (INSERT/UPDATE/DELETE), 'schema' (get schema)"
    )
    sql: Optional[str] = Field(default=None, description="SQL query to execute")
    table: Optional[str] = Field(default=None, description="Table name for schema action")
    params: Optional[dict[str, Any]] = Field(default=None, description="Query parameters")


class DatabaseTool(BaseTool):
    """Database query tool supporting SQLAlchemy.
    
    Supports:
    - SQLite
    - PostgreSQL
    - MySQL
    - Any SQLAlchemy-compatible database
    """
    
    name = "database"
    description = "Execute SQL queries on a database. Supports SELECT queries, modifications, and schema inspection."
    parameters = DatabaseParameters
    category = "data"
    is_dangerous = True
    
    def __init__(
        self,
        database_url: str = "sqlite+aiosqlite:///agentflow.db",
        readonly: bool = False,
        max_results: int = 100,
        **config: Any,
    ):
        super().__init__(**config)
        self.database_url = database_url
        self.readonly = readonly
        self.max_results = max_results
        self._engine = None
    
    async def _get_engine(self):
        """Get or create the database engine."""
        if self._engine is None:
            from sqlalchemy.ext.asyncio import create_async_engine
            self._engine = create_async_engine(self.database_url, echo=False)
        return self._engine
    
    async def execute(
        self,
        action: str,
        sql: Optional[str] = None,
        table: Optional[str] = None,
        params: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute database action."""
        try:
            engine = await self._get_engine()
            
            if action == "query":
                return await self._execute_query(engine, sql, params)
            elif action == "execute":
                if self.readonly:
                    return ToolResult(
                        success=False,
                        output=None,
                        error="Database is in readonly mode",
                    )
                return await self._execute_modification(engine, sql, params)
            elif action == "schema":
                return await self._get_schema(engine, table)
            else:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Unknown action: {action}",
                )
        
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
    
    async def _execute_query(
        self,
        engine,
        sql: Optional[str],
        params: Optional[dict[str, Any]],
    ) -> ToolResult:
        """Execute a SELECT query."""
        if not sql:
            return ToolResult(success=False, output=None, error="SQL is required for query action")
        
        # Security check: only allow SELECT
        sql_upper = sql.strip().upper()
        if not sql_upper.startswith("SELECT"):
            return ToolResult(
                success=False,
                output=None,
                error="Only SELECT queries are allowed in query action",
            )
        
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import AsyncConnection
        
        async with engine.connect() as conn:
            conn: AsyncConnection
            result = await conn.execute(text(sql), params or {})
            rows = result.fetchmany(self.max_results)
            columns = result.keys()
            
            # Convert to list of dicts
            data = [dict(zip(columns, row)) for row in rows]
            
            return ToolResult(
                success=True,
                output=data,
                metadata={
                    "columns": list(columns),
                    "row_count": len(data),
                    "truncated": len(rows) >= self.max_results,
                },
            )
    
    async def _execute_modification(
        self,
        engine,
        sql: Optional[str],
        params: Optional[dict[str, Any]],
    ) -> ToolResult:
        """Execute INSERT/UPDATE/DELETE."""
        if not sql:
            return ToolResult(success=False, output=None, error="SQL is required for execute action")
        
        from sqlalchemy import text
        
        async with engine.begin() as conn:
            result = await conn.execute(text(sql), params or {})
            return ToolResult(
                success=True,
                output=f"Query executed successfully. Rows affected: {result.rowcount}",
                metadata={"rowcount": result.rowcount},
            )
    
    async def _get_schema(self, engine, table: Optional[str]) -> ToolResult:
        """Get database schema."""
        from sqlalchemy import inspect
        from sqlalchemy.ext.asyncio import AsyncConnection
        
        async with engine.connect() as conn:
            conn: AsyncConnection
            
            def get_schema(sync_conn):
                inspector = inspect(sync_conn)
                tables = inspector.get_table_names()
                
                if table:
                    if table not in tables:
                        return {"error": f"Table '{table}' not found"}
                    
                    columns = inspector.get_columns(table)
                    pk = inspector.get_pk_constraint(table)
                    fks = inspector.get_foreign_keys(table)
                    indexes = inspector.get_indexes(table)
                    
                    return {
                        "table": table,
                        "columns": [
                            {
                                "name": col["name"],
                                "type": str(col["type"]),
                                "nullable": col.get("nullable", True),
                                "default": str(col.get("default")) if col.get("default") else None,
                            }
                            for col in columns
                        ],
                        "primary_key": pk.get("constrained_columns", []),
                        "foreign_keys": fks,
                        "indexes": indexes,
                    }
                else:
                    schema = {}
                    for t in tables:
                        columns = inspector.get_columns(t)
                        schema[t] = [col["name"] for col in columns]
                    return {"tables": schema}
            
            result = await conn.run_sync(get_schema)
            
            if "error" in result:
                return ToolResult(success=False, output=None, error=result["error"])
            
            return ToolResult(success=True, output=result)
    
    async def close(self):
        """Close the database connection."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
