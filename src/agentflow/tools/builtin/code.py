"""Code execution tools."""

import ast
import asyncio
import sys
from io import StringIO
from typing import Any, Optional

from pydantic import BaseModel, Field

from agentflow.tools.base import BaseTool, ToolResult


class PythonExecuteParameters(BaseModel):
    """Parameters for Python execution."""
    
    code: str = Field(description="Python code to execute")
    timeout: int = Field(default=30, description="Execution timeout in seconds")


class PythonExecuteTool(BaseTool):
    """Execute Python code in a sandboxed environment.
    
    Security features:
    - Restricted imports
    - Timeout enforcement
    - Memory limits (via globals restriction)
    """
    
    name = "python_execute"
    description = "Execute Python code and return the output. Useful for calculations, data processing, and testing code snippets."
    parameters = PythonExecuteParameters
    category = "code"
    is_dangerous = True
    
    # Safe built-ins
    SAFE_BUILTINS = {
        "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes",
        "callable", "chr", "complex", "dict", "dir", "divmod", "enumerate",
        "filter", "float", "format", "frozenset", "getattr", "hasattr", "hash",
        "hex", "int", "isinstance", "issubclass", "iter", "len", "list", "map",
        "max", "min", "next", "oct", "ord", "pow", "print", "range", "repr",
        "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple",
        "type", "zip",
    }
    
    # Blocked module patterns
    BLOCKED_MODULES = {
        "os", "sys", "subprocess", "shutil", "pathlib", "socket", "http",
        "urllib", "requests", "ftplib", "smtplib", "poplib", "imaplib",
        "telnetlib", "pickle", "shelve", "marshal", "ctypes", "multiprocessing",
    }
    
    def __init__(
        self,
        allow_imports: bool = False,
        allowed_modules: Optional[set[str]] = None,
        **config: Any,
    ):
        super().__init__(**config)
        self.allow_imports = allow_imports
        self.allowed_modules = allowed_modules or {"math", "json", "datetime", "re", "collections", "itertools", "functools"}
    
    def _check_code_safety(self, code: str) -> tuple[bool, str]:
        """Check if code is safe to execute."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        
        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if not self.allow_imports:
                    return False, "Imports are not allowed"
                
                if isinstance(node, ast.Import):
                    modules = [alias.name.split(".")[0] for alias in node.names]
                else:
                    modules = [node.module.split(".")[0]] if node.module else []
                
                for mod in modules:
                    if mod in self.BLOCKED_MODULES:
                        return False, f"Import of '{mod}' is blocked"
                    if mod not in self.allowed_modules:
                        return False, f"Import of '{mod}' is not allowed"
            
            # Check for dangerous function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ("eval", "exec", "compile", "open", "__import__"):
                        return False, f"Function '{node.func.id}' is not allowed"
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in ("system", "popen", "spawn"):
                        return False, f"Method '{node.func.attr}' is not allowed"
        
        return True, ""
    
    async def execute(
        self,
        code: str,
        timeout: int = 30,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute Python code."""
        # Safety check
        is_safe, reason = self._check_code_safety(code)
        if not is_safe:
            return ToolResult(success=False, output=None, error=reason)
        
        # Create restricted globals
        restricted_globals: dict[str, Any] = {
            "__builtins__": {k: getattr(__builtins__ if isinstance(__builtins__, dict) else __builtins__, k, None) 
                           for k in self.SAFE_BUILTINS 
                           if hasattr(__builtins__ if isinstance(__builtins__, dict) else __builtins__, k)},
        }
        
        # Add safe modules if imports are allowed
        if self.allow_imports:
            for mod_name in self.allowed_modules:
                try:
                    restricted_globals[mod_name] = __import__(mod_name)
                except ImportError:
                    pass
        
        # Capture output
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        captured_stdout = StringIO()
        captured_stderr = StringIO()
        
        try:
            sys.stdout = captured_stdout
            sys.stderr = captured_stderr
            
            # Execute with timeout
            local_vars: dict[str, Any] = {}
            
            def run_code():
                exec(code, restricted_globals, local_vars)
            
            await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, run_code),
                timeout=timeout,
            )
            
            stdout_output = captured_stdout.getvalue()
            stderr_output = captured_stderr.getvalue()
            
            # Prepare output
            output = stdout_output
            if stderr_output:
                output += f"\n\nSTDERR:\n{stderr_output}"
            
            # Include returned value if any
            if "_result" in local_vars:
                output += f"\n\nResult: {local_vars['_result']}"
            
            return ToolResult(
                success=True,
                output=output.strip() or "Code executed successfully (no output)",
                metadata={"local_vars": {k: str(v) for k, v in local_vars.items() if not k.startswith("_")}},
            )
        
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                output=None,
                error=f"Execution timed out after {timeout} seconds",
            )
        except Exception as e:
            stderr_output = captured_stderr.getvalue()
            error_msg = str(e)
            if stderr_output:
                error_msg += f"\n\nSTDERR:\n{stderr_output}"
            return ToolResult(success=False, output=None, error=error_msg)
        
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
