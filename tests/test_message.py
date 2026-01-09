"""Tests for message module."""

import pytest
from datetime import datetime

from agentflow.core.message import (
    Message,
    MessageRole,
    ToolCall,
    ToolResult,
    Conversation,
)


class TestToolCall:
    """Tests for ToolCall."""
    
    def test_creation(self):
        """Test ToolCall creation."""
        tc = ToolCall(name="test_tool", arguments={"arg1": "value1"})
        
        assert tc.name == "test_tool"
        assert tc.arguments == {"arg1": "value1"}
        assert tc.id.startswith("call_")
    
    def test_from_openai_format(self):
        """Test creating from OpenAI format."""
        openai_tc = {
            "id": "call_123",
            "function": {
                "name": "search",
                "arguments": '{"query": "test"}',
            },
        }
        
        tc = ToolCall.from_openai_format(openai_tc)
        
        assert tc.id == "call_123"
        assert tc.name == "search"
        assert tc.arguments == {"query": "test"}


class TestMessage:
    """Tests for Message."""
    
    def test_system_message(self):
        """Test creating system message."""
        msg = Message.system("You are a helpful assistant.")
        
        assert msg.role == MessageRole.SYSTEM
        assert msg.content == "You are a helpful assistant."
    
    def test_user_message(self):
        """Test creating user message."""
        msg = Message.user("Hello!")
        
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello!"
    
    def test_assistant_message(self):
        """Test creating assistant message."""
        msg = Message.assistant("Hi there!")
        
        assert msg.role == MessageRole.ASSISTANT
        assert msg.content == "Hi there!"
    
    def test_assistant_with_tool_calls(self):
        """Test assistant message with tool calls."""
        tool_calls = [
            ToolCall(name="search", arguments={"query": "test"}),
        ]
        msg = Message.assistant(content=None, tool_calls=tool_calls)
        
        assert msg.role == MessageRole.ASSISTANT
        assert msg.content is None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "search"
    
    def test_tool_message(self):
        """Test creating tool result message."""
        result = ToolResult(
            tool_call_id="call_123",
            name="search",
            content="Search results...",
        )
        msg = Message.tool(result)
        
        assert msg.role == MessageRole.TOOL
        assert msg.content == "Search results..."
        assert msg.tool_call_id == "call_123"
        assert msg.name == "search"
    
    def test_to_openai_format(self):
        """Test converting to OpenAI format."""
        msg = Message.user("Hello!")
        openai_msg = msg.to_openai_format()
        
        assert openai_msg["role"] == "user"
        assert openai_msg["content"] == "Hello!"
    
    def test_from_openai_format(self):
        """Test creating from OpenAI format."""
        openai_msg = {
            "role": "assistant",
            "content": "Hello!",
        }
        msg = Message.from_openai_format(openai_msg)
        
        assert msg.role == MessageRole.ASSISTANT
        assert msg.content == "Hello!"


class TestConversation:
    """Tests for Conversation."""
    
    def test_add_messages(self):
        """Test adding messages to conversation."""
        conv = Conversation()
        
        conv.add_system("System prompt")
        conv.add_user("User message")
        conv.add_assistant("Assistant response")
        
        assert len(conv.messages) == 3
        assert conv.messages[0].role == MessageRole.SYSTEM
        assert conv.messages[1].role == MessageRole.USER
        assert conv.messages[2].role == MessageRole.ASSISTANT
    
    def test_get_last_message(self):
        """Test getting last message."""
        conv = Conversation()
        
        assert conv.get_last_message() is None
        
        conv.add_user("Hello")
        assert conv.get_last_message().content == "Hello"
    
    def test_get_messages_by_role(self):
        """Test filtering messages by role."""
        conv = Conversation()
        conv.add_user("User 1")
        conv.add_assistant("Assistant 1")
        conv.add_user("User 2")
        
        user_messages = conv.get_messages_by_role(MessageRole.USER)
        
        assert len(user_messages) == 2
        assert all(m.role == MessageRole.USER for m in user_messages)
    
    def test_to_openai_format(self):
        """Test converting conversation to OpenAI format."""
        conv = Conversation()
        conv.add_user("Hello")
        conv.add_assistant("Hi!")
        
        openai_msgs = conv.to_openai_format()
        
        assert len(openai_msgs) == 2
        assert openai_msgs[0]["role"] == "user"
        assert openai_msgs[1]["role"] == "assistant"
    
    def test_truncate(self):
        """Test truncating conversation."""
        conv = Conversation()
        conv.add_system("System")
        for i in range(10):
            conv.add_user(f"Message {i}")
        
        conv.truncate(5, keep_system=True)
        
        assert len(conv.messages) == 5
        assert conv.messages[0].role == MessageRole.SYSTEM
    
    def test_clear(self):
        """Test clearing conversation."""
        conv = Conversation()
        conv.add_system("System")
        conv.add_user("User")
        conv.add_assistant("Assistant")
        
        conv.clear(keep_system=True)
        
        assert len(conv.messages) == 1
        assert conv.messages[0].role == MessageRole.SYSTEM
        
        conv.clear(keep_system=False)
        
        assert len(conv.messages) == 0
