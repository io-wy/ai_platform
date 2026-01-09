"""Terminal command execution tool."""

import asyncio
import os
import shlex
from typing import Any, Optional

from pydantic import BaseModel, Field

from agentflow.tools.base import BaseTool, ToolResult


class TerminalParameters(BaseModel):
    """Parameters for terminal tool."""
    
    command: str = Field(description="Shell command to execute")
    working_dir: Optional[str] = Field(default=None, description="Working directory for command execution")
    timeout: Optional[int] = Field(default=60, description="Timeout in seconds")
    shell: bool = Field(default=True, description="Whether to use shell execution")


class TerminalTool(BaseTool):
    """Execute shell commands in the terminal.
    
    Security features:
    - Command blocking for dangerous operations
    - Timeout enforcement
    - Working directory restriction (optional)
    """
    
    name = "terminal"
    description = "Execute shell commands in the terminal. Use for system operations, file manipulation, running scripts, etc."
    parameters = TerminalParameters
    category = "system"
    requires_confirmation = True
    is_dangerous = True
    
    # Default blocked patterns
    DEFAULT_BLOCKED = [
        "rm -rf /",
        "rm -rf /*",
        "sudo rm -rf",
        "mkfs",
        "dd if=",
        ":(){:|:&};:",
        "> /dev/sd",
        "chmod -R 777 /",
        "chown -R",
    ]
    
    def __init__(
        self,
        allowed_commands: Optional[list[str]] = None,
        blocked_commands: Optional[list[str]] = None,
        default_working_dir: Optional[str] = None,
        **config: Any,
    ):
        super().__init__(**config)
        self.allowed_commands = allowed_commands  # If set, only these commands are allowed
        self.blocked_commands = blocked_commands or self.DEFAULT_BLOCKED
        self.default_working_dir = default_working_dir
    
    def _is_command_allowed(self, command: str) -> tuple[bool, str]:
        """Check if command is allowed to execute."""
        command_lower = command.lower()
        
        # Check blocked patterns
        for blocked in self.blocked_commands:
            if blocked.lower() in command_lower:
                return False, f"Command blocked: matches pattern '{blocked}'"
        
        # Check allowed list if set
        if self.allowed_commands is not None:
            # Extract the base command
            try:
                parts = shlex.split(command)
                base_cmd = parts[0] if parts else ""
            except ValueError:
                base_cmd = command.split()[0] if command.split() else ""
            
            if base_cmd not in self.allowed_commands:
                return False, f"Command '{base_cmd}' is not in the allowed list"
        
        return True, ""
    
    async def execute(
        self,
        command: str,
        working_dir: Optional[str] = None,
        timeout: int = 60,
        shell: bool = True,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute a shell command."""
        # Security check
        allowed, reason = self._is_command_allowed(command)
        if not allowed:
            return ToolResult(success=False, output=None, error=reason)
        
        # Determine working directory
        cwd = working_dir or self.default_working_dir or os.getcwd()
        
        try:
            if shell:
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                )
            else:
                args = shlex.split(command)
                process = await asyncio.create_subprocess_exec(
                    *args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Command timed out after {timeout} seconds",
                )
            
            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")
            
            # Limit output size
            max_output = 50000
            if len(stdout_str) > max_output:
                stdout_str = stdout_str[:max_output] + "\n... (output truncated)"
            if len(stderr_str) > max_output:
                stderr_str = stderr_str[:max_output] + "\n... (output truncated)"
            
            output = stdout_str
            if stderr_str:
                output += f"\n\nSTDERR:\n{stderr_str}"
            
            return ToolResult(
                success=process.returncode == 0,
                output=output,
                error=stderr_str if process.returncode != 0 else None,
                metadata={
                    "return_code": process.returncode,
                    "working_dir": cwd,
                },
            )
        
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
