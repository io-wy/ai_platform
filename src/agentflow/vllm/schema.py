"""Schema definitions for vLLM structured output."""

from typing import Any, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field


class FieldType(str, Enum):
    """Supported field types for form extraction."""
    TEXT = "text"
    NUMBER = "number"
    DATE = "date"
    EMAIL = "email"
    PHONE = "phone"
    ADDRESS = "address"
    BOOLEAN = "boolean"
    ENUM = "enum"
    LIST = "list"
    NESTED = "nested"


class FormField(BaseModel):
    """Definition of a form field to extract."""
    
    name: str = Field(description="Field name/key")
    field_type: FieldType = Field(default=FieldType.TEXT, description="Type of the field")
    description: str = Field(default="", description="Description for the LLM")
    required: bool = Field(default=True, description="Whether field is required")
    options: Optional[list[str]] = Field(default=None, description="Options for enum type")
    default: Optional[Any] = Field(default=None, description="Default value if not found")
    validation_pattern: Optional[str] = Field(default=None, description="Regex pattern for validation")
    nested_schema: Optional["FormSchema"] = Field(default=None, description="Schema for nested fields")
    
    def to_prompt_description(self) -> str:
        """Generate a description for prompting."""
        desc = f"- {self.name}"
        if self.description:
            desc += f": {self.description}"
        desc += f" (type: {self.field_type.value}"
        if self.required:
            desc += ", required"
        else:
            desc += ", optional"
        if self.options:
            desc += f", options: {self.options}"
        desc += ")"
        return desc


class FormSchema(BaseModel):
    """Schema defining structure of data to extract."""
    
    name: str = Field(default="form", description="Schema name")
    description: str = Field(default="", description="Schema description")
    fields: list[FormField] = Field(default_factory=list, description="Fields to extract")
    
    def to_json_schema(self) -> dict[str, Any]:
        """Convert to JSON schema for structured output."""
        properties = {}
        required = []
        
        type_map = {
            FieldType.TEXT: "string",
            FieldType.NUMBER: "number",
            FieldType.DATE: "string",
            FieldType.EMAIL: "string",
            FieldType.PHONE: "string",
            FieldType.ADDRESS: "string",
            FieldType.BOOLEAN: "boolean",
            FieldType.ENUM: "string",
            FieldType.LIST: "array",
            FieldType.NESTED: "object",
        }
        
        for field in self.fields:
            field_schema: dict[str, Any] = {
                "type": type_map.get(field.field_type, "string"),
            }
            
            if field.description:
                field_schema["description"] = field.description
            
            if field.field_type == FieldType.ENUM and field.options:
                field_schema["enum"] = field.options
            
            if field.field_type == FieldType.LIST:
                field_schema["items"] = {"type": "string"}
            
            if field.field_type == FieldType.NESTED and field.nested_schema:
                field_schema = field.nested_schema.to_json_schema()
            
            properties[field.name] = field_schema
            
            if field.required:
                required.append(field.name)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }
    
    def to_prompt(self) -> str:
        """Generate extraction prompt."""
        lines = [
            f"Extract the following information from the text:",
            "",
        ]
        for field in self.fields:
            lines.append(field.to_prompt_description())
        
        lines.append("")
        lines.append("Return the extracted data as a JSON object.")
        
        return "\n".join(lines)


class ExtractionResult(BaseModel):
    """Result of form/data extraction."""
    
    success: bool = Field(description="Whether extraction succeeded")
    data: Optional[dict[str, Any]] = Field(default=None, description="Extracted data")
    raw_response: Optional[str] = Field(default=None, description="Raw model response")
    confidence: float = Field(default=1.0, description="Confidence score 0-1")
    missing_fields: list[str] = Field(default_factory=list, description="Fields that couldn't be extracted")
    validation_errors: list[str] = Field(default_factory=list, description="Validation errors")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class BatchItem(BaseModel):
    """Single item in a batch processing request."""
    
    id: str = Field(description="Unique identifier for the item")
    text: str = Field(description="Text to process")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class BatchResult(BaseModel):
    """Result of batch processing."""
    
    id: str = Field(description="Item identifier")
    result: ExtractionResult = Field(description="Extraction result")
    processing_time_ms: float = Field(default=0, description="Processing time in milliseconds")


class ClassificationSchema(BaseModel):
    """Schema for text classification."""
    
    categories: list[str] = Field(description="Available categories")
    multi_label: bool = Field(default=False, description="Allow multiple labels")
    description: str = Field(default="", description="Task description")
    
    def to_prompt(self) -> str:
        """Generate classification prompt."""
        if self.multi_label:
            return f"""Classify the following text into one or more of these categories: {self.categories}

{self.description if self.description else ''}

Return a JSON object with:
- "categories": list of matching categories
- "confidence": confidence score for each (0-1)"""
        else:
            return f"""Classify the following text into exactly one of these categories: {self.categories}

{self.description if self.description else ''}

Return a JSON object with:
- "category": the selected category
- "confidence": confidence score (0-1)"""


class QASchema(BaseModel):
    """Schema for question answering."""
    
    context_required: bool = Field(default=True, description="Whether context is required")
    answer_format: str = Field(default="text", description="Format: 'text', 'json', 'list'")
    max_length: Optional[int] = Field(default=None, description="Maximum answer length")
    
    def to_prompt(self, question: str, context: Optional[str] = None) -> str:
        """Generate QA prompt."""
        prompt_parts = []
        
        if context:
            prompt_parts.append(f"Context:\n{context}\n")
        
        prompt_parts.append(f"Question: {question}")
        
        if self.answer_format == "json":
            prompt_parts.append("\nProvide your answer as a JSON object with 'answer' and 'confidence' fields.")
        elif self.answer_format == "list":
            prompt_parts.append("\nProvide your answer as a JSON array of relevant points.")
        
        return "\n".join(prompt_parts)
