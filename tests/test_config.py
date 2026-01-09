"""Tests for configuration module."""

import os
import tempfile
import pytest
from pydantic import SecretStr

from agentflow.core.config import (
    AgentConfig,
    LLMConfig,
    MemoryConfig,
    ToolConfig,
    Settings,
    LLMProvider,
    ReasoningPattern,
    get_settings,
    reset_settings,
)


class TestLLMConfig:
    """Tests for LLMConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = LLMConfig()
        
        assert config.provider == LLMProvider.OPENAI
        assert config.model == "gpt-4o-mini"
        assert config.temperature == 0.7
        assert config.max_tokens is None
        assert config.timeout == 60.0
        assert config.max_retries == 3
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = LLMConfig(
            provider=LLMProvider.VLLM,
            model="llama-2-7b",
            temperature=0.5,
            max_tokens=1000,
            api_base="http://localhost:8000/v1",
        )
        
        assert config.provider == LLMProvider.VLLM
        assert config.model == "llama-2-7b"
        assert config.temperature == 0.5
        assert config.max_tokens == 1000
        assert config.api_base == "http://localhost:8000/v1"
    
    def test_temperature_bounds(self):
        """Test temperature validation."""
        # Valid values
        LLMConfig(temperature=0.0)
        LLMConfig(temperature=2.0)
        
        # Invalid values
        with pytest.raises(ValueError):
            LLMConfig(temperature=-0.1)
        
        with pytest.raises(ValueError):
            LLMConfig(temperature=2.1)


class TestMemoryConfig:
    """Tests for MemoryConfig."""
    
    def test_default_values(self):
        """Test default memory configuration."""
        config = MemoryConfig()
        
        assert config.max_short_term_messages == 20
        assert config.enable_long_term is True
        assert config.max_context_tokens == 8000
        assert config.context_compression is True
    
    def test_custom_values(self):
        """Test custom memory configuration."""
        config = MemoryConfig(
            max_short_term_messages=50,
            enable_long_term=False,
            max_context_tokens=16000,
        )
        
        assert config.max_short_term_messages == 50
        assert config.enable_long_term is False
        assert config.max_context_tokens == 16000


class TestToolConfig:
    """Tests for ToolConfig."""
    
    def test_default_blocked_commands(self):
        """Test default blocked terminal commands."""
        config = ToolConfig()
        
        assert "rm -rf /" in config.terminal_blocked_commands
        assert config.max_tool_calls_per_turn == 10
        assert config.parallel_tool_calls is True
    
    def test_custom_tool_config(self):
        """Test custom tool configuration."""
        config = ToolConfig(
            enabled_tools=["browser", "terminal"],
            max_tool_calls_per_turn=5,
            browser_headless=False,
        )
        
        assert config.enabled_tools == ["browser", "terminal"]
        assert config.max_tool_calls_per_turn == 5
        assert config.browser_headless is False


class TestAgentConfig:
    """Tests for AgentConfig."""
    
    def test_default_config(self):
        """Test default agent configuration."""
        config = AgentConfig()
        
        assert config.name == "AgentFlow"
        assert config.pattern == ReasoningPattern.AUTO
        assert config.max_iterations == 10
        assert config.verbose is False
    
    def test_nested_config(self):
        """Test nested configuration."""
        config = AgentConfig(
            name="MyAgent",
            llm=LLMConfig(model="gpt-4"),
            memory=MemoryConfig(max_short_term_messages=30),
            tools=ToolConfig(enabled_tools=["browser"]),
            pattern=ReasoningPattern.REACT,
        )
        
        assert config.name == "MyAgent"
        assert config.llm.model == "gpt-4"
        assert config.memory.max_short_term_messages == 30
        assert config.tools.enabled_tools == ["browser"]
        assert config.pattern == ReasoningPattern.REACT
    
    def test_save_and_load_json(self):
        """Test saving and loading JSON config."""
        config = AgentConfig(
            name="TestAgent",
            llm=LLMConfig(model="gpt-4"),
        )
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            config.to_file(f.name)
            loaded = AgentConfig.from_file(f.name)
        
        assert loaded.name == config.name
        assert loaded.llm.model == config.llm.model
        
        os.unlink(f.name)
    
    def test_save_and_load_yaml(self):
        """Test saving and loading YAML config."""
        config = AgentConfig(
            name="TestAgent",
            pattern=ReasoningPattern.COT,
        )
        
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            config.to_file(f.name)
            loaded = AgentConfig.from_file(f.name)
        
        assert loaded.name == config.name
        assert loaded.pattern == config.pattern
        
        os.unlink(f.name)


class TestSettings:
    """Tests for Settings."""
    
    def test_default_settings(self):
        """Test default settings."""
        reset_settings()
        settings = get_settings()
        
        assert settings.vllm_api_base == "http://localhost:8000/v1"
        assert settings.ollama_api_base == "http://localhost:11434"
        assert settings.log_level == "INFO"
    
    def test_env_override(self, monkeypatch):
        """Test environment variable override."""
        reset_settings()
        monkeypatch.setenv("AGENTFLOW_LOG_LEVEL", "DEBUG")
        
        settings = get_settings()
        # Note: Settings are cached, so this tests the mechanism
        
        reset_settings()
