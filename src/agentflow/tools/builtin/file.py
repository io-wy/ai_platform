"""File operation tools."""

import os
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from agentflow.tools.base import BaseTool, ToolResult


class FileReadParameters(BaseModel):
    """Parameters for reading files."""
    
    path: str = Field(description="Path to the file to read")
    encoding: str = Field(default="utf-8", description="File encoding")
    start_line: Optional[int] = Field(default=None, description="Start line (1-indexed)")
    end_line: Optional[int] = Field(default=None, description="End line (1-indexed)")


class FileReadTool(BaseTool):
    """Read file contents."""
    
    name = "read_file"
    description = "Read the contents of a file. Can read entire file or specific line ranges."
    parameters = FileReadParameters
    category = "file"
    
    def __init__(
        self,
        allowed_paths: Optional[list[str]] = None,
        max_size: int = 1000000,  # 1MB
        **config: Any,
    ):
        super().__init__(**config)
        self.allowed_paths = allowed_paths
        self.max_size = max_size
    
    def _is_path_allowed(self, path: str) -> bool:
        """Check if path is allowed."""
        if self.allowed_paths is None:
            return True
        
        path = os.path.abspath(path)
        for allowed in self.allowed_paths:
            allowed = os.path.abspath(allowed)
            if path.startswith(allowed):
                return True
        return False
    
    async def execute(
        self,
        path: str,
        encoding: str = "utf-8",
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Read file contents."""
        try:
            path = os.path.abspath(path)
            
            if not self._is_path_allowed(path):
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Path not allowed: {path}",
                )
            
            if not os.path.exists(path):
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"File not found: {path}",
                )
            
            file_size = os.path.getsize(path)
            if file_size > self.max_size:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"File too large: {file_size} bytes (max: {self.max_size})",
                )
            
            with open(path, "r", encoding=encoding) as f:
                if start_line is not None or end_line is not None:
                    lines = f.readlines()
                    start = (start_line or 1) - 1
                    end = end_line or len(lines)
                    content = "".join(lines[start:end])
                else:
                    content = f.read()
            
            return ToolResult(
                success=True,
                output=content,
                metadata={
                    "path": path,
                    "size": len(content),
                    "encoding": encoding,
                },
            )
        
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class FileWriteParameters(BaseModel):
    """Parameters for writing files."""
    
    path: str = Field(description="Path to the file to write")
    content: str = Field(description="Content to write")
    encoding: str = Field(default="utf-8", description="File encoding")
    mode: str = Field(default="write", description="Mode: 'write' (overwrite) or 'append'")
    create_dirs: bool = Field(default=True, description="Create parent directories if needed")


class FileWriteTool(BaseTool):
    """Write content to files."""
    
    name = "write_file"
    description = "Write content to a file. Can overwrite or append."
    parameters = FileWriteParameters
    category = "file"
    is_dangerous = True
    
    def __init__(
        self,
        allowed_paths: Optional[list[str]] = None,
        **config: Any,
    ):
        super().__init__(**config)
        self.allowed_paths = allowed_paths
    
    def _is_path_allowed(self, path: str) -> bool:
        """Check if path is allowed."""
        if self.allowed_paths is None:
            return True
        
        path = os.path.abspath(path)
        for allowed in self.allowed_paths:
            allowed = os.path.abspath(allowed)
            if path.startswith(allowed):
                return True
        return False
    
    async def execute(
        self,
        path: str,
        content: str,
        encoding: str = "utf-8",
        mode: str = "write",
        create_dirs: bool = True,
        **kwargs: Any,
    ) -> ToolResult:
        """Write to file."""
        try:
            path = os.path.abspath(path)
            
            if not self._is_path_allowed(path):
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Path not allowed: {path}",
                )
            
            if create_dirs:
                os.makedirs(os.path.dirname(path), exist_ok=True)
            
            file_mode = "w" if mode == "write" else "a"
            
            with open(path, file_mode, encoding=encoding) as f:
                f.write(content)
            
            return ToolResult(
                success=True,
                output=f"Successfully wrote {len(content)} characters to {path}",
                metadata={
                    "path": path,
                    "size": len(content),
                    "mode": mode,
                },
            )
        
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class ListDirectoryParameters(BaseModel):
    """Parameters for listing directory contents."""
    
    path: str = Field(description="Path to the directory")
    pattern: Optional[str] = Field(default=None, description="Glob pattern to filter files")
    recursive: bool = Field(default=False, description="Whether to list recursively")
    include_hidden: bool = Field(default=False, description="Include hidden files")


class ListDirectoryTool(BaseTool):
    """List directory contents."""
    
    name = "list_directory"
    description = "List files and directories in a given path."
    parameters = ListDirectoryParameters
    category = "file"
    
    def __init__(
        self,
        allowed_paths: Optional[list[str]] = None,
        max_results: int = 1000,
        **config: Any,
    ):
        super().__init__(**config)
        self.allowed_paths = allowed_paths
        self.max_results = max_results
    
    def _is_path_allowed(self, path: str) -> bool:
        """Check if path is allowed."""
        if self.allowed_paths is None:
            return True
        
        path = os.path.abspath(path)
        for allowed in self.allowed_paths:
            allowed = os.path.abspath(allowed)
            if path.startswith(allowed):
                return True
        return False
    
    async def execute(
        self,
        path: str,
        pattern: Optional[str] = None,
        recursive: bool = False,
        include_hidden: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        """List directory contents."""
        try:
            path = os.path.abspath(path)
            
            if not self._is_path_allowed(path):
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Path not allowed: {path}",
                )
            
            if not os.path.isdir(path):
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Not a directory: {path}",
                )
            
            path_obj = Path(path)
            entries = []
            
            if recursive:
                glob_pattern = pattern or "**/*"
                items = path_obj.glob(glob_pattern)
            elif pattern:
                items = path_obj.glob(pattern)
            else:
                items = path_obj.iterdir()
            
            for item in items:
                if not include_hidden and item.name.startswith("."):
                    continue
                
                entry = {
                    "name": item.name,
                    "path": str(item),
                    "type": "directory" if item.is_dir() else "file",
                }
                
                if item.is_file():
                    entry["size"] = item.stat().st_size
                
                entries.append(entry)
                
                if len(entries) >= self.max_results:
                    break
            
            return ToolResult(
                success=True,
                output=entries,
                metadata={
                    "path": path,
                    "count": len(entries),
                    "truncated": len(entries) >= self.max_results,
                },
            )
        
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
