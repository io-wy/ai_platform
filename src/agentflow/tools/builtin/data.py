"""JSON processing tool for structured data extraction and manipulation."""

import json
import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from agentflow.tools.base import BaseTool, ToolResult


class JSONParameters(BaseModel):
    """Parameters for JSON operations."""
    
    action: str = Field(
        description="Action: 'parse', 'extract', 'validate', 'transform', 'query'"
    )
    data: str = Field(description="JSON string or text containing JSON")
    schema: Optional[dict] = Field(default=None, description="JSON schema for validation")
    path: Optional[str] = Field(default=None, description="JSONPath expression for extraction")
    template: Optional[dict] = Field(default=None, description="Template for transformation")


class JSONTool(BaseTool):
    """Tool for JSON processing and manipulation.
    
    Supports:
    - Parsing JSON from text
    - Extracting data using JSONPath
    - Validating against schema
    - Transforming data
    """
    
    name = "json_tool"
    description = "Process and manipulate JSON data. Extract, validate, and transform structured data."
    parameters = JSONParameters
    category = "data"
    
    async def execute(
        self,
        action: str,
        data: str,
        schema: Optional[dict] = None,
        path: Optional[str] = None,
        template: Optional[dict] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute JSON operation."""
        try:
            if action == "parse":
                return await self._parse_json(data)
            elif action == "extract":
                return await self._extract_json(data, path)
            elif action == "validate":
                return await self._validate_json(data, schema)
            elif action == "transform":
                return await self._transform_json(data, template)
            elif action == "query":
                return await self._query_json(data, path)
            else:
                return ToolResult(success=False, output=None, error=f"Unknown action: {action}")
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
    
    async def _parse_json(self, data: str) -> ToolResult:
        """Parse JSON from text, extracting JSON even if embedded in other text."""
        # Try direct parse first
        try:
            parsed = json.loads(data)
            return ToolResult(success=True, output=parsed)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from text
        json_patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Objects
            r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]',  # Arrays
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, data, re.DOTALL)
            for match in matches:
                try:
                    parsed = json.loads(match)
                    return ToolResult(
                        success=True, 
                        output=parsed,
                        metadata={"extracted": True}
                    )
                except json.JSONDecodeError:
                    continue
        
        return ToolResult(success=False, output=None, error="No valid JSON found in text")
    
    async def _extract_json(self, data: str, path: Optional[str]) -> ToolResult:
        """Extract data using simple path notation."""
        parsed = json.loads(data) if isinstance(data, str) else data
        
        if not path:
            return ToolResult(success=True, output=parsed)
        
        # Simple path parsing (e.g., "data.items[0].name")
        current = parsed
        parts = re.split(r'\.|\[|\]', path)
        parts = [p for p in parts if p]
        
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list):
                try:
                    idx = int(part)
                    current = current[idx]
                except (ValueError, IndexError):
                    return ToolResult(success=False, output=None, error=f"Invalid index: {part}")
            else:
                return ToolResult(success=False, output=None, error=f"Cannot access '{part}' on {type(current)}")
            
            if current is None:
                break
        
        return ToolResult(success=True, output=current)
    
    async def _validate_json(self, data: str, schema: Optional[dict]) -> ToolResult:
        """Validate JSON against a schema."""
        if not schema:
            return ToolResult(success=False, output=None, error="Schema is required for validation")
        
        parsed = json.loads(data) if isinstance(data, str) else data
        
        # Simple validation without jsonschema dependency
        errors = []
        
        def validate_value(value, schema_part, path="root"):
            if "type" in schema_part:
                expected_type = schema_part["type"]
                type_map = {
                    "string": str,
                    "number": (int, float),
                    "integer": int,
                    "boolean": bool,
                    "array": list,
                    "object": dict,
                    "null": type(None),
                }
                expected = type_map.get(expected_type)
                if expected and not isinstance(value, expected):
                    errors.append(f"{path}: expected {expected_type}, got {type(value).__name__}")
            
            if isinstance(value, dict) and "properties" in schema_part:
                for key, prop_schema in schema_part["properties"].items():
                    if key in value:
                        validate_value(value[key], prop_schema, f"{path}.{key}")
                    elif schema_part.get("required") and key in schema_part.get("required", []):
                        errors.append(f"{path}: missing required property '{key}'")
            
            if isinstance(value, list) and "items" in schema_part:
                for i, item in enumerate(value):
                    validate_value(item, schema_part["items"], f"{path}[{i}]")
        
        validate_value(parsed, schema)
        
        if errors:
            return ToolResult(success=False, output=None, error="\n".join(errors))
        
        return ToolResult(success=True, output={"valid": True, "data": parsed})
    
    async def _transform_json(self, data: str, template: Optional[dict]) -> ToolResult:
        """Transform JSON using a template."""
        if not template:
            return ToolResult(success=False, output=None, error="Template is required for transformation")
        
        parsed = json.loads(data) if isinstance(data, str) else data
        
        def apply_template(template_part, source):
            if isinstance(template_part, str) and template_part.startswith("$"):
                path = template_part[1:]
                current = source
                for part in path.split("."):
                    if isinstance(current, dict):
                        current = current.get(part)
                    elif isinstance(current, list) and part.isdigit():
                        current = current[int(part)]
                    else:
                        current = None
                return current
            elif isinstance(template_part, dict):
                return {k: apply_template(v, source) for k, v in template_part.items()}
            elif isinstance(template_part, list):
                return [apply_template(item, source) for item in template_part]
            else:
                return template_part
        
        result = apply_template(template, parsed)
        return ToolResult(success=True, output=result)
    
    async def _query_json(self, data: str, path: Optional[str]) -> ToolResult:
        """Query JSON with filtering."""
        parsed = json.loads(data) if isinstance(data, str) else data
        
        if not path:
            return ToolResult(success=True, output=parsed)
        
        # Support simple queries like "items[?price>100]"
        match = re.match(r'(\w+)\[\?(\w+)([<>=!]+)(.+)\]', path)
        if match:
            field, key, op, value = match.groups()
            items = parsed.get(field, [])
            
            # Type conversion for comparison
            try:
                value = float(value) if '.' in value else int(value)
            except ValueError:
                value = value.strip('"\'')
            
            ops = {
                '==': lambda a, b: a == b,
                '!=': lambda a, b: a != b,
                '>': lambda a, b: a > b,
                '<': lambda a, b: a < b,
                '>=': lambda a, b: a >= b,
                '<=': lambda a, b: a <= b,
            }
            
            op_func = ops.get(op)
            if op_func:
                filtered = [item for item in items if op_func(item.get(key), value)]
                return ToolResult(success=True, output=filtered)
        
        # Fall back to simple extraction
        return await self._extract_json(data, path)


class TextExtractParameters(BaseModel):
    """Parameters for text extraction."""
    
    text: str = Field(description="Text to extract from")
    pattern: str = Field(description="Regex pattern or extraction type")
    extract_type: str = Field(
        default="regex",
        description="Type: 'regex', 'emails', 'urls', 'phones', 'numbers', 'dates'"
    )


class TextExtractTool(BaseTool):
    """Extract structured data from text."""
    
    name = "text_extract"
    description = "Extract structured data from unstructured text. Find emails, URLs, phone numbers, dates, or use custom regex."
    parameters = TextExtractParameters
    category = "data"
    
    # Pre-built patterns
    PATTERNS = {
        "emails": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "urls": r'https?://[^\s<>"{}|\\^`\[\]]+',
        "phones": r'[\+]?[(]?[0-9]{1,3}[)]?[-\s\.]?[0-9]{1,4}[-\s\.]?[0-9]{1,4}[-\s\.]?[0-9]{1,9}',
        "numbers": r'-?\d+\.?\d*',
        "dates": r'\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
    }
    
    async def execute(
        self,
        text: str,
        pattern: str = "",
        extract_type: str = "regex",
        **kwargs: Any,
    ) -> ToolResult:
        """Extract data from text."""
        try:
            if extract_type == "regex":
                if not pattern:
                    return ToolResult(success=False, output=None, error="Pattern is required for regex extraction")
                regex_pattern = pattern
            else:
                regex_pattern = self.PATTERNS.get(extract_type)
                if not regex_pattern:
                    return ToolResult(
                        success=False, 
                        output=None, 
                        error=f"Unknown extract_type: {extract_type}. Use: {list(self.PATTERNS.keys())}"
                    )
            
            matches = re.findall(regex_pattern, text, re.IGNORECASE)
            unique_matches = list(dict.fromkeys(matches))  # Preserve order, remove duplicates
            
            return ToolResult(
                success=True,
                output=unique_matches,
                metadata={
                    "total_matches": len(matches),
                    "unique_matches": len(unique_matches),
                    "pattern": regex_pattern,
                }
            )
        
        except re.error as e:
            return ToolResult(success=False, output=None, error=f"Invalid regex pattern: {e}")
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
