"""Configuration management for AgentFlow."""

from enum import Enum
from typing import Any, Optional
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    VLLM = "vllm"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    CUSTOM = "custom"


class ReasoningPattern(str, Enum):
    """Available reasoning patterns."""
    REACT = "react"  # Reasoning + Acting
    COT = "cot"  # Chain of Thought
    TOT = "tot"  # Tree of Thought
    REFLEXION = "reflexion"  # Self-reflection
    PLAN_AND_EXECUTE = "plan_and_execute"
    AUTO = "auto"  # Let the model decide


class Settings(BaseSettings):
    """Global settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="AGENTFLOW_",
        extra="ignore",
    )
    
    # OpenAI settings
    openai_api_key: Optional[SecretStr] = Field(default=None, alias="OPENAI_API_KEY")
    openai_api_base: Optional[str] = Field(default=None, alias="OPENAI_API_BASE")
    openai_org_id: Optional[str] = Field(default=None, alias="OPENAI_ORG_ID")
    
    # Azure OpenAI settings
    azure_openai_api_key: Optional[SecretStr] = Field(default=None, alias="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: Optional[str] = Field(default=None, alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_version: str = Field(default="2024-02-01", alias="AZURE_OPENAI_API_VERSION")
    
    # vLLM settings
    vllm_api_base: str = Field(default="http://localhost:8000/v1", alias="VLLM_API_BASE")
    vllm_api_key: Optional[SecretStr] = Field(default=None, alias="VLLM_API_KEY")
    
    # Ollama settings
    ollama_api_base: str = Field(default="http://localhost:11434", alias="OLLAMA_API_BASE")
    
    # Anthropic settings
    anthropic_api_key: Optional[SecretStr] = Field(default=None, alias="ANTHROPIC_API_KEY")
    
    # Database settings
    database_url: str = Field(default="sqlite+aiosqlite:///agentflow.db", alias="DATABASE_URL")
    redis_url: Optional[str] = Field(default=None, alias="REDIS_URL")
    
    # Vector store settings
    chromadb_path: str = Field(default="./chromadb", alias="CHROMADB_PATH")
    
    # Search settings
    serpapi_key: Optional[SecretStr] = Field(default=None, alias="SERPAPI_KEY")
    google_api_key: Optional[SecretStr] = Field(default=None, alias="GOOGLE_API_KEY")
    google_cse_id: Optional[str] = Field(default=None, alias="GOOGLE_CSE_ID")
    
    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(default="json", alias="LOG_FORMAT")
    
    # General settings
    max_retries: int = Field(default=3, alias="MAX_RETRIES")
    timeout: float = Field(default=60.0, alias="TIMEOUT")


class LLMConfig(BaseModel):
    """Configuration for LLM client."""
    
    provider: LLMProvider = LLMProvider.OPENAI
    model: str = "gpt-4o-mini"
    api_key: Optional[SecretStr] = None
    api_base: Optional[str] = None
    
    # Generation parameters
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    stop: Optional[list[str]] = None
    
    # Additional parameters
    timeout: float = Field(default=60.0, ge=1.0)
    max_retries: int = Field(default=3, ge=0)
    
    # For Azure
    azure_deployment: Optional[str] = None
    api_version: Optional[str] = None
    
    # Extra parameters for custom providers
    extra_params: dict[str, Any] = Field(default_factory=dict)


class MemoryConfig(BaseModel):
    """Configuration for memory system."""
    
    # Short-term memory
    max_short_term_messages: int = Field(default=20, ge=1)
    
    # Long-term memory (vector store)
    enable_long_term: bool = True
    embedding_model: str = "text-embedding-3-small"
    max_retrieval_results: int = Field(default=5, ge=1)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    
    # Context window management
    max_context_tokens: int = Field(default=8000, ge=100)
    context_compression: bool = True
    
    # Persistence
    persist_memory: bool = True
    memory_path: str = "./memory"


class ToolConfig(BaseModel):
    """Configuration for tool system."""
    
    # Enabled tools
    enabled_tools: list[str] = Field(default_factory=list)
    
    # Tool execution
    max_tool_calls_per_turn: int = Field(default=10, ge=1)
    tool_timeout: float = Field(default=30.0, ge=1.0)
    parallel_tool_calls: bool = True
    
    # Browser tool settings
    browser_headless: bool = True
    browser_timeout: float = Field(default=30.0, ge=1.0)
    
    # Terminal tool settings
    terminal_allowed_commands: Optional[list[str]] = None
    terminal_blocked_commands: list[str] = Field(
        default_factory=lambda: ["rm -rf /", "sudo rm", "mkfs", ":(){:|:&};:"]
    )
    terminal_working_dir: Optional[str] = None
    
    # Database tool settings
    database_readonly: bool = False
    max_query_results: int = Field(default=100, ge=1)


class AgentConfig(BaseModel):
    """Main configuration for Agent."""
    
    name: str = "AgentFlow"
    description: str = "A flexible LLM agent"
    
    # LLM configuration
    llm: LLMConfig = Field(default_factory=LLMConfig)
    
    # Reasoning pattern
    pattern: ReasoningPattern = ReasoningPattern.AUTO
    max_iterations: int = Field(default=10, ge=1)
    
    # Memory configuration
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    
    # Tool configuration
    tools: ToolConfig = Field(default_factory=ToolConfig)
    
    # System prompt
    system_prompt: Optional[str] = None
    
    # Logging and debugging
    verbose: bool = False
    debug: bool = False
    
    @classmethod
    def from_file(cls, path: str | Path) -> "AgentConfig":
        """Load configuration from a file."""
        import json
        import yaml
        
        path = Path(path)
        content = path.read_text()
        
        if path.suffix in (".yaml", ".yml"):
            data = yaml.safe_load(content)
        elif path.suffix == ".json":
            data = json.loads(content)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")
        
        return cls.model_validate(data)
    
    def to_file(self, path: str | Path) -> None:
        """Save configuration to a file."""
        import json
        import yaml
        
        path = Path(path)
        data = self.model_dump(mode="json")
        
        if path.suffix in (".yaml", ".yml"):
            content = yaml.dump(data, default_flow_style=False)
        elif path.suffix == ".json":
            content = json.dumps(data, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")
        
        path.write_text(content)


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset the global settings instance."""
    global _settings
    _settings = None
