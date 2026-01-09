"""Message types for AgentFlow."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FUNCTION = "function"  # Legacy support


class ToolCall(BaseModel):
    """Represents a tool call from the model."""
    
    id: str = Field(default_factory=lambda: f"call_{uuid4().hex[:8]}")
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    
    @classmethod
    def from_openai_format(cls, tool_call: dict[str, Any]) -> "ToolCall":
        """Create from OpenAI tool call format."""
        import json
        
        args = tool_call.get("function", {}).get("arguments", "{}")
        if isinstance(args, str):
            args = json.loads(args)
        
        return cls(
            id=tool_call.get("id", f"call_{uuid4().hex[:8]}"),
            name=tool_call.get("function", {}).get("name", ""),
            arguments=args,
        )


class ToolResult(BaseModel):
    """Result from a tool execution."""
    
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """A message in the conversation."""
    
    id: str = Field(default_factory=lambda: uuid4().hex)
    role: MessageRole
    content: Optional[str] = None
    name: Optional[str] = None
    
    # For tool calls
    tool_calls: Optional[list[ToolCall]] = None
    tool_call_id: Optional[str] = None
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    # Token usage
    tokens: Optional[int] = None
    
    @classmethod
    def system(cls, content: str, **kwargs: Any) -> "Message":
        """Create a system message."""
        return cls(role=MessageRole.SYSTEM, content=content, **kwargs)
    
    @classmethod
    def user(cls, content: str, **kwargs: Any) -> "Message":
        """Create a user message."""
        return cls(role=MessageRole.USER, content=content, **kwargs)
    
    @classmethod
    def assistant(
        cls,
        content: Optional[str] = None,
        tool_calls: Optional[list[ToolCall]] = None,
        **kwargs: Any,
    ) -> "Message":
        """Create an assistant message."""
        return cls(
            role=MessageRole.ASSISTANT,
            content=content,
            tool_calls=tool_calls,
            **kwargs,
        )
    
    @classmethod
    def tool(cls, result: ToolResult, **kwargs: Any) -> "Message":
        """Create a tool result message."""
        return cls(
            role=MessageRole.TOOL,
            content=result.content,
            tool_call_id=result.tool_call_id,
            name=result.name,
            metadata={"is_error": result.is_error, **result.metadata},
            **kwargs,
        )
    
    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI message format."""
        msg: dict[str, Any] = {
            "role": self.role.value,
        }
        
        if self.content is not None:
            msg["content"] = self.content
        
        if self.name is not None:
            msg["name"] = self.name
        
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": str(tc.arguments) if not isinstance(tc.arguments, str) 
                                     else tc.arguments,
                    },
                }
                for tc in self.tool_calls
            ]
        
        if self.tool_call_id is not None:
            msg["tool_call_id"] = self.tool_call_id
        
        return msg
    
    @classmethod
    def from_openai_format(cls, data: dict[str, Any]) -> "Message":
        """Create from OpenAI message format."""
        tool_calls = None
        if "tool_calls" in data and data["tool_calls"]:
            tool_calls = [
                ToolCall.from_openai_format(tc) for tc in data["tool_calls"]
            ]
        
        return cls(
            role=MessageRole(data["role"]),
            content=data.get("content"),
            name=data.get("name"),
            tool_calls=tool_calls,
            tool_call_id=data.get("tool_call_id"),
        )


class Conversation(BaseModel):
    """A conversation consisting of multiple messages."""
    
    id: str = Field(default_factory=lambda: uuid4().hex)
    messages: list[Message] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    def add(self, message: Message) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)
    
    def add_system(self, content: str, **kwargs: Any) -> Message:
        """Add a system message."""
        msg = Message.system(content, **kwargs)
        self.add(msg)
        return msg
    
    def add_user(self, content: str, **kwargs: Any) -> Message:
        """Add a user message."""
        msg = Message.user(content, **kwargs)
        self.add(msg)
        return msg
    
    def add_assistant(
        self,
        content: Optional[str] = None,
        tool_calls: Optional[list[ToolCall]] = None,
        **kwargs: Any,
    ) -> Message:
        """Add an assistant message."""
        msg = Message.assistant(content, tool_calls, **kwargs)
        self.add(msg)
        return msg
    
    def add_tool_result(self, result: ToolResult, **kwargs: Any) -> Message:
        """Add a tool result message."""
        msg = Message.tool(result, **kwargs)
        self.add(msg)
        return msg
    
    def to_openai_format(self) -> list[dict[str, Any]]:
        """Convert all messages to OpenAI format."""
        return [msg.to_openai_format() for msg in self.messages]
    
    def get_last_message(self) -> Optional[Message]:
        """Get the last message."""
        return self.messages[-1] if self.messages else None
    
    def get_messages_by_role(self, role: MessageRole) -> list[Message]:
        """Get all messages with a specific role."""
        return [msg for msg in self.messages if msg.role == role]
    
    def total_tokens(self) -> int:
        """Get total tokens used in the conversation."""
        return sum(msg.tokens or 0 for msg in self.messages)
    
    def truncate(self, max_messages: int, keep_system: bool = True) -> None:
        """Truncate conversation to max_messages, optionally keeping system messages."""
        if len(self.messages) <= max_messages:
            return
        
        if keep_system:
            system_messages = [m for m in self.messages if m.role == MessageRole.SYSTEM]
            other_messages = [m for m in self.messages if m.role != MessageRole.SYSTEM]
            remaining_slots = max_messages - len(system_messages)
            self.messages = system_messages + other_messages[-remaining_slots:]
        else:
            self.messages = self.messages[-max_messages:]
    
    def clear(self, keep_system: bool = True) -> None:
        """Clear all messages, optionally keeping system messages."""
        if keep_system:
            self.messages = [m for m in self.messages if m.role == MessageRole.SYSTEM]
        else:
            self.messages = []
