"""Base tool classes and decorators."""

import inspect
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Type, TypeVar, get_type_hints

from pydantic import BaseModel, Field, create_model


@dataclass
class ToolResult:
    """Result from a tool execution."""
    
    success: bool
    output: Any
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_string(self) -> str:
        """Convert result to string for LLM consumption."""
        if self.success:
            if isinstance(self.output, str):
                return self.output
            return json.dumps(self.output, ensure_ascii=False, indent=2)
        return f"Error: {self.error}"


class BaseTool(ABC):
    """Base class for all tools.
    
    Tools can be created by subclassing BaseTool or using the @tool decorator.
    
    Example:
        ```python
        class CalculatorTool(BaseTool):
            name = "calculator"
            description = "Perform mathematical calculations"
            
            class Parameters(BaseModel):
                expression: str = Field(description="Math expression to evaluate")
            
            async def execute(self, expression: str) -> ToolResult:
                try:
                    result = eval(expression)
                    return ToolResult(success=True, output=result)
                except Exception as e:
                    return ToolResult(success=False, output=None, error=str(e))
        ```
    """
    
    name: str = ""
    description: str = ""
    parameters: Optional[Type[BaseModel]] = None
    
    # Tool metadata
    category: str = "general"
    requires_confirmation: bool = False
    is_dangerous: bool = False
    
    def __init__(self, **config: Any):
        """Initialize the tool with optional configuration."""
        self.config = config
    
    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with the given parameters.
        
        Args:
            **kwargs: Tool-specific parameters.
            
        Returns:
            ToolResult containing the execution result.
        """
        pass
    
    def get_schema(self) -> dict[str, Any]:
        """Get the OpenAI-compatible tool schema."""
        parameters_schema = {"type": "object", "properties": {}, "required": []}
        
        if self.parameters is not None:
            schema = self.parameters.model_json_schema()
            parameters_schema = {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", []),
            }
            
            # Handle definitions/refs
            if "$defs" in schema:
                parameters_schema["$defs"] = schema["$defs"]
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters_schema,
            },
        }
    
    async def validate_parameters(self, **kwargs: Any) -> dict[str, Any]:
        """Validate and parse parameters."""
        if self.parameters is not None:
            validated = self.parameters.model_validate(kwargs)
            return validated.model_dump()
        return kwargs
    
    async def __call__(self, **kwargs: Any) -> ToolResult:
        """Call the tool, validating parameters first."""
        try:
            validated_kwargs = await self.validate_parameters(**kwargs)
            return await self.execute(**validated_kwargs)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


T = TypeVar("T", bound=Callable[..., Any])


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    category: str = "general",
    requires_confirmation: bool = False,
    is_dangerous: bool = False,
) -> Callable[[T], Type[BaseTool]]:
    """Decorator to create a tool from a function.
    
    Example:
        ```python
        @tool(name="calculator", description="Evaluate math expressions")
        async def calculator(expression: str) -> str:
            '''
            Args:
                expression: The mathematical expression to evaluate.
            '''
            return str(eval(expression))
        ```
    """
    
    def decorator(func: T) -> Type[BaseTool]:
        # Get function metadata
        func_name = name or func.__name__
        func_description = description or func.__doc__ or ""
        
        # Get type hints for parameters
        hints = get_type_hints(func)
        sig = inspect.signature(func)
        
        # Build Pydantic model for parameters
        fields: dict[str, Any] = {}
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            
            param_type = hints.get(param_name, str)
            default = ... if param.default is inspect.Parameter.empty else param.default
            
            # Try to extract description from docstring
            param_desc = ""
            if func.__doc__:
                # Simple docstring parsing
                for line in func.__doc__.split("\n"):
                    if param_name in line and ":" in line:
                        param_desc = line.split(":", 1)[-1].strip()
                        break
            
            fields[param_name] = (param_type, Field(default=default, description=param_desc))
        
        # Create the parameters model
        ParametersModel = create_model(f"{func_name}Parameters", **fields)
        
        # Capture variables in closure for the class
        original_func = func
        tool_category = category
        tool_requires_confirmation = requires_confirmation
        tool_is_dangerous = is_dangerous
        
        # Create the tool class with execute implemented
        class FunctionTool(BaseTool):
            name = func_name
            description = func_description.split("\n")[0].strip()  # First line
            parameters = ParametersModel
            category = tool_category
            requires_confirmation = tool_requires_confirmation
            is_dangerous = tool_is_dangerous
            _func = original_func
            
            async def execute(self, **kwargs: Any) -> ToolResult:
                try:
                    if inspect.iscoroutinefunction(original_func):
                        result = await original_func(**kwargs)
                    else:
                        result = original_func(**kwargs)
                    return ToolResult(success=True, output=result)
                except Exception as e:
                    return ToolResult(success=False, output=None, error=str(e))
        
        return FunctionTool
    
    return decorator


class ToolRegistry:
    """Registry for managing tools.
    
    Example:
        ```python
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        registry.register(BrowserTool())
        
        # Get all tools
        tools = registry.get_all()
        
        # Get tools by category
        web_tools = registry.get_by_category("web")
        
        # Get tool schemas for LLM
        schemas = registry.get_schemas()
        ```
    """
    
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        self._categories: dict[str, list[str]] = {}
    
    def register(self, tool: BaseTool) -> None:
        """Register a tool."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        
        self._tools[tool.name] = tool
        
        # Track category
        if tool.category not in self._categories:
            self._categories[tool.category] = []
        self._categories[tool.category].append(tool.name)
    
    def register_class(self, tool_class: Type[BaseTool], **config: Any) -> None:
        """Register a tool class, instantiating it."""
        tool = tool_class(**config)
        self.register(tool)
    
    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' is not registered")
        
        tool = self._tools.pop(name)
        self._categories[tool.category].remove(name)
    
    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def get_all(self) -> list[BaseTool]:
        """Get all registered tools."""
        return list(self._tools.values())
    
    def get_by_category(self, category: str) -> list[BaseTool]:
        """Get all tools in a category."""
        names = self._categories.get(category, [])
        return [self._tools[name] for name in names]
    
    def get_names(self) -> list[str]:
        """Get all tool names."""
        return list(self._tools.keys())
    
    def get_categories(self) -> list[str]:
        """Get all categories."""
        return list(self._categories.keys())
    
    def get_schemas(self, names: Optional[list[str]] = None) -> list[dict[str, Any]]:
        """Get OpenAI-compatible schemas for tools.
        
        Args:
            names: Optional list of tool names to include. If None, includes all.
        """
        tools = self._tools.values()
        if names is not None:
            tools = [t for t in tools if t.name in names]
        
        return [tool.get_schema() for tool in tools]
    
    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools
    
    def __len__(self) -> int:
        """Get the number of registered tools."""
        return len(self._tools)
    
    def __iter__(self):
        """Iterate over tools."""
        return iter(self._tools.values())


# Global registry instance
_global_registry: Optional[ToolRegistry] = None


def get_global_registry() -> ToolRegistry:
    """Get the global tool registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def register_tool(tool: BaseTool) -> None:
    """Register a tool to the global registry."""
    get_global_registry().register(tool)


def register_tool_class(tool_class: Type[BaseTool], **config: Any) -> None:
    """Register a tool class to the global registry."""
    get_global_registry().register_class(tool_class, **config)
