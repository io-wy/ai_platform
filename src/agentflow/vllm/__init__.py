"""
vLLM Module - Specialized for form processing, batch inference and structured output tasks.
"""

from agentflow.vllm.client import VLLMClient
from agentflow.vllm.processor import FormProcessor, BatchProcessor
from agentflow.vllm.schema import FormField, FormSchema, ExtractionResult

__all__ = [
    "VLLMClient",
    "FormProcessor",
    "BatchProcessor",
    "FormField",
    "FormSchema",
    "ExtractionResult",
]
