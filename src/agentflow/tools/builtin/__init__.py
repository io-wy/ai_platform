"""Built-in tools for AgentFlow."""

from agentflow.tools.builtin.browser import BrowserTool
from agentflow.tools.builtin.terminal import TerminalTool
from agentflow.tools.builtin.search import WebSearchTool
from agentflow.tools.builtin.database import DatabaseTool
from agentflow.tools.builtin.http import HTTPTool
from agentflow.tools.builtin.file import FileReadTool, FileWriteTool, ListDirectoryTool
from agentflow.tools.builtin.code import PythonExecuteTool
from agentflow.tools.builtin.data import JSONTool, TextExtractTool

__all__ = [
    "BrowserTool",
    "TerminalTool",
    "WebSearchTool",
    "DatabaseTool",
    "HTTPTool",
    "FileReadTool",
    "FileWriteTool",
    "ListDirectoryTool",
    "PythonExecuteTool",
    "JSONTool",
    "TextExtractTool",
]


def get_default_tools():
    """Get a list of default tool instances."""
    return [
        BrowserTool(),
        TerminalTool(),
        WebSearchTool(),
        HTTPTool(),
        FileReadTool(),
        FileWriteTool(),
        ListDirectoryTool(),
        PythonExecuteTool(),
    ]
