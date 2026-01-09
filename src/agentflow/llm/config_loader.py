"""
LLM 配置加载器
==============

支持从不同的 .env 文件加载 LLM 配置，实现不同任务使用不同模型的解耦。

使用方式:
    # 加载特定环境文件
    config = LLMConfigLoader.from_env_file(".env.chatbot")
    
    # 加载预定义任务配置
    config = LLMConfigLoader.for_task("chatbot")
    
    # 从环境变量前缀加载
    config = LLMConfigLoader.from_prefix("CHATBOT")
"""

import os
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass

from pydantic import SecretStr

from agentflow.core.config import LLMConfig, LLMProvider


@dataclass
class TaskLLMConfig:
    """任务专用 LLM 配置."""
    
    # 基础配置
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    
    # 生成参数
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    
    # 其他
    timeout: float = 60.0
    max_retries: int = 3
    
    def to_llm_config(self) -> LLMConfig:
        """转换为 LLMConfig 对象."""
        return LLMConfig(
            provider=LLMProvider(self.provider),
            model=self.model,
            api_key=SecretStr(self.api_key) if self.api_key else None,
            api_base=self.api_base,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )


class LLMConfigLoader:
    """LLM 配置加载器."""
    
    # 预定义的任务配置文件映射
    TASK_ENV_FILES = {
        "chatbot": ".env.chatbot",
        "qa": ".env.qa",
        "form": ".env.form",
        "task": ".env.task",
        "planner": ".env.planner",
        "executor": ".env.executor",
        "default": ".env",
    }
    
    # 环境变量键映射
    ENV_KEYS = {
        "provider": "LLM_PROVIDER",
        "model": "LLM_MODEL",
        "api_key": "LLM_API_KEY",
        "api_base": "LLM_API_BASE",
        "temperature": "LLM_TEMPERATURE",
        "max_tokens": "LLM_MAX_TOKENS",
        "top_p": "LLM_TOP_P",
        "timeout": "LLM_TIMEOUT",
        "max_retries": "LLM_MAX_RETRIES",
    }
    
    @classmethod
    def from_env_file(
        cls,
        env_file: Union[str, Path],
        prefix: str = "",
        base_path: Optional[Path] = None,
    ) -> LLMConfig:
        """从指定的 .env 文件加载配置.
        
        Args:
            env_file: 环境文件路径（如 .env.chatbot）
            prefix: 环境变量前缀（如 CHATBOT_）
            base_path: 基础路径，默认为当前工作目录
            
        Returns:
            LLMConfig 实例
        """
        if base_path is None:
            base_path = Path.cwd()
        
        env_path = base_path / env_file
        
        # 加载 .env 文件
        env_vars = cls._load_env_file(env_path)
        
        # 解析配置
        return cls._parse_config(env_vars, prefix)
    
    @classmethod
    def for_task(
        cls,
        task_name: str,
        base_path: Optional[Path] = None,
    ) -> LLMConfig:
        """加载预定义任务的配置.
        
        Args:
            task_name: 任务名称 (chatbot, qa, form, task, planner, executor)
            base_path: 基础路径
            
        Returns:
            LLMConfig 实例
        """
        env_file = cls.TASK_ENV_FILES.get(task_name, cls.TASK_ENV_FILES["default"])
        prefix = f"{task_name.upper()}_" if task_name != "default" else ""
        
        return cls.from_env_file(env_file, prefix, base_path)
    
    @classmethod
    def from_prefix(
        cls,
        prefix: str,
        env_file: str = ".env",
        base_path: Optional[Path] = None,
    ) -> LLMConfig:
        """从带前缀的环境变量加载配置.
        
        支持在单个 .env 文件中定义多个配置：
        
        CHATBOT_LLM_MODEL=gpt-4o-mini
        CHATBOT_LLM_TEMPERATURE=0.7
        
        QA_LLM_MODEL=gpt-4o
        QA_LLM_TEMPERATURE=0.3
        
        Args:
            prefix: 环境变量前缀（不含尾部下划线）
            env_file: 环境文件
            base_path: 基础路径
            
        Returns:
            LLMConfig 实例
        """
        prefix = prefix.upper()
        if not prefix.endswith("_"):
            prefix = f"{prefix}_"
        
        return cls.from_env_file(env_file, prefix, base_path)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> LLMConfig:
        """从字典创建配置.
        
        Args:
            config_dict: 配置字典
            
        Returns:
            LLMConfig 实例
        """
        task_config = TaskLLMConfig(
            provider=config_dict.get("provider", "openai"),
            model=config_dict.get("model", "gpt-4o-mini"),
            api_key=config_dict.get("api_key"),
            api_base=config_dict.get("api_base"),
            temperature=float(config_dict.get("temperature", 0.7)),
            max_tokens=int(config_dict["max_tokens"]) if config_dict.get("max_tokens") else None,
            top_p=float(config_dict.get("top_p", 1.0)),
            timeout=float(config_dict.get("timeout", 60.0)),
            max_retries=int(config_dict.get("max_retries", 3)),
        )
        return task_config.to_llm_config()
    
    @classmethod
    def _load_env_file(cls, env_path: Path) -> dict:
        """加载 .env 文件内容."""
        env_vars = {}
        
        if not env_path.exists():
            # 如果文件不存在，返回当前环境变量
            return dict(os.environ)
        
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                
                # 跳过空行和注释
                if not line or line.startswith("#"):
                    continue
                
                # 解析 KEY=VALUE
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # 移除引号
                    if value and value[0] in ('"', "'") and value[-1] == value[0]:
                        value = value[1:-1]
                    
                    env_vars[key] = value
        
        # 合并当前环境变量（环境变量优先级更高）
        for key, value in os.environ.items():
            if key not in env_vars:
                env_vars[key] = value
        
        return env_vars
    
    @classmethod
    def _parse_config(cls, env_vars: dict, prefix: str = "") -> LLMConfig:
        """解析环境变量为配置."""
        def get_value(key: str) -> Optional[str]:
            """获取环境变量值，支持多种命名方式."""
            env_key = cls.ENV_KEYS.get(key, f"LLM_{key.upper()}")
            
            # 尝试带前缀的键
            if prefix:
                prefixed_key = f"{prefix}{env_key}"
                if prefixed_key in env_vars:
                    return env_vars[prefixed_key]
            
            # 尝试不带前缀的键
            if env_key in env_vars:
                return env_vars[env_key]
            
            # 尝试 OPENAI 兼容键
            openai_mappings = {
                "api_key": "OPENAI_API_KEY",
                "api_base": "OPENAI_API_BASE",
            }
            if key in openai_mappings:
                compat_key = openai_mappings[key]
                if prefix:
                    prefixed_compat = f"{prefix}{compat_key}"
                    if prefixed_compat in env_vars:
                        return env_vars[prefixed_compat]
                if compat_key in env_vars:
                    return env_vars[compat_key]
            
            return None
        
        # 构建配置
        provider_str = get_value("provider") or "openai"
        
        task_config = TaskLLMConfig(
            provider=provider_str,
            model=get_value("model") or "gpt-4o-mini",
            api_key=get_value("api_key"),
            api_base=get_value("api_base"),
            temperature=float(get_value("temperature") or 0.7),
            max_tokens=int(get_value("max_tokens")) if get_value("max_tokens") else None,
            top_p=float(get_value("top_p") or 1.0),
            timeout=float(get_value("timeout") or 60.0),
            max_retries=int(get_value("max_retries") or 3),
        )
        
        return task_config.to_llm_config()
    
    @classmethod
    def create_env_template(
        cls,
        task_name: str,
        output_path: Optional[Path] = None,
    ) -> str:
        """创建环境配置模板.
        
        Args:
            task_name: 任务名称
            output_path: 输出路径（可选，不提供则返回字符串）
            
        Returns:
            模板内容字符串
        """
        prefix = f"{task_name.upper()}_" if task_name else ""
        
        template = f"""# {task_name.title() if task_name else 'Default'} LLM Configuration
# ============================================

# LLM Provider: openai, azure_openai, vllm, ollama, anthropic, custom
{prefix}LLM_PROVIDER=openai

# Model name
{prefix}LLM_MODEL=gpt-4o-mini

# API Configuration
{prefix}LLM_API_KEY=your_api_key_here
{prefix}LLM_API_BASE=https://api.openai.com/v1

# Generation Parameters
{prefix}LLM_TEMPERATURE=0.7
{prefix}LLM_MAX_TOKENS=2000
{prefix}LLM_TOP_P=1.0

# Request Settings
{prefix}LLM_TIMEOUT=60.0
{prefix}LLM_MAX_RETRIES=3
"""
        
        if output_path:
            output_path.write_text(template, encoding="utf-8")
        
        return template


# 便捷函数
def load_llm_config(
    env_file: Optional[str] = None,
    task: Optional[str] = None,
    prefix: Optional[str] = None,
    **overrides,
) -> LLMConfig:
    """加载 LLM 配置的便捷函数.
    
    优先级: overrides > env_file/task/prefix > 默认值
    
    Args:
        env_file: 环境文件路径
        task: 任务名称
        prefix: 环境变量前缀
        **overrides: 覆盖参数
        
    Returns:
        LLMConfig 实例
        
    Examples:
        # 从特定文件加载
        config = load_llm_config(env_file=".env.chatbot")
        
        # 按任务加载
        config = load_llm_config(task="qa")
        
        # 从前缀加载并覆盖
        config = load_llm_config(prefix="CHATBOT", temperature=0.5)
    """
    if env_file:
        config = LLMConfigLoader.from_env_file(env_file)
    elif task:
        config = LLMConfigLoader.for_task(task)
    elif prefix:
        config = LLMConfigLoader.from_prefix(prefix)
    else:
        config = LLMConfigLoader.from_env_file(".env")
    
    # 应用覆盖
    if overrides:
        config_dict = config.model_dump()
        for key, value in overrides.items():
            if value is not None and key in config_dict:
                config_dict[key] = value
        
        # 重新创建配置
        config = LLMConfig(**config_dict)
    
    return config
