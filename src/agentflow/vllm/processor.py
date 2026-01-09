"""Batch and form processors for vLLM."""

import asyncio
import json
import time
from typing import Any, AsyncIterator, Optional, Callable
from dataclasses import dataclass, field

from agentflow.vllm.client import VLLMClient, VLLMConfig
from agentflow.vllm.schema import (
    FormSchema, FormField, FieldType,
    ExtractionResult, BatchItem, BatchResult,
    ClassificationSchema,
)


@dataclass
class ProcessingStats:
    """Statistics for batch processing."""
    total_items: int = 0
    successful: int = 0
    failed: int = 0
    total_time_ms: float = 0
    avg_time_ms: float = 0
    items_per_second: float = 0


class FormProcessor:
    """Process forms and extract structured data.
    
    Supports:
    - Form field extraction
    - Invoice processing
    - Receipt parsing
    - Document data extraction
    """
    
    def __init__(
        self,
        client: VLLMClient,
        default_schema: Optional[FormSchema] = None,
    ):
        self.client = client
        self.default_schema = default_schema
    
    @classmethod
    def create(
        cls,
        model: str,
        api_base: str = "http://localhost:8000/v1",
        **config_kwargs: Any,
    ) -> "FormProcessor":
        """Create a FormProcessor with a new client."""
        config = VLLMConfig(model=model, api_base=api_base, **config_kwargs)
        client = VLLMClient(config)
        return cls(client)
    
    async def extract(
        self,
        text: str,
        schema: Optional[FormSchema] = None,
        **kwargs: Any,
    ) -> ExtractionResult:
        """Extract data from text using schema."""
        use_schema = schema or self.default_schema
        if not use_schema:
            raise ValueError("Schema is required for extraction")
        
        return await self.client.extract(text, use_schema, **kwargs)
    
    async def extract_invoice(self, text: str, **kwargs: Any) -> ExtractionResult:
        """Extract invoice data."""
        schema = FormSchema(
            name="invoice",
            description="Invoice data extraction",
            fields=[
                FormField(name="invoice_number", field_type=FieldType.TEXT, description="Invoice number/ID"),
                FormField(name="date", field_type=FieldType.DATE, description="Invoice date"),
                FormField(name="due_date", field_type=FieldType.DATE, description="Payment due date", required=False),
                FormField(name="vendor_name", field_type=FieldType.TEXT, description="Vendor/seller name"),
                FormField(name="vendor_address", field_type=FieldType.ADDRESS, description="Vendor address", required=False),
                FormField(name="customer_name", field_type=FieldType.TEXT, description="Customer/buyer name"),
                FormField(name="customer_address", field_type=FieldType.ADDRESS, description="Customer address", required=False),
                FormField(name="subtotal", field_type=FieldType.NUMBER, description="Subtotal amount"),
                FormField(name="tax", field_type=FieldType.NUMBER, description="Tax amount", required=False),
                FormField(name="total", field_type=FieldType.NUMBER, description="Total amount"),
                FormField(name="currency", field_type=FieldType.TEXT, description="Currency code", required=False),
                FormField(
                    name="line_items", 
                    field_type=FieldType.LIST, 
                    description="List of items (description, quantity, unit_price, amount)",
                    required=False,
                ),
            ],
        )
        return await self.extract(text, schema, **kwargs)
    
    async def extract_receipt(self, text: str, **kwargs: Any) -> ExtractionResult:
        """Extract receipt data."""
        schema = FormSchema(
            name="receipt",
            description="Receipt data extraction",
            fields=[
                FormField(name="merchant_name", field_type=FieldType.TEXT, description="Store/merchant name"),
                FormField(name="merchant_address", field_type=FieldType.ADDRESS, description="Store address", required=False),
                FormField(name="date", field_type=FieldType.DATE, description="Transaction date"),
                FormField(name="time", field_type=FieldType.TEXT, description="Transaction time", required=False),
                FormField(name="items", field_type=FieldType.LIST, description="Purchased items with prices"),
                FormField(name="subtotal", field_type=FieldType.NUMBER, description="Subtotal"),
                FormField(name="tax", field_type=FieldType.NUMBER, description="Tax amount", required=False),
                FormField(name="total", field_type=FieldType.NUMBER, description="Total amount"),
                FormField(name="payment_method", field_type=FieldType.TEXT, description="Payment method", required=False),
            ],
        )
        return await self.extract(text, schema, **kwargs)
    
    async def extract_contact(self, text: str, **kwargs: Any) -> ExtractionResult:
        """Extract contact information."""
        schema = FormSchema(
            name="contact",
            description="Contact information extraction",
            fields=[
                FormField(name="name", field_type=FieldType.TEXT, description="Full name"),
                FormField(name="email", field_type=FieldType.EMAIL, description="Email address", required=False),
                FormField(name="phone", field_type=FieldType.PHONE, description="Phone number", required=False),
                FormField(name="company", field_type=FieldType.TEXT, description="Company name", required=False),
                FormField(name="title", field_type=FieldType.TEXT, description="Job title", required=False),
                FormField(name="address", field_type=FieldType.ADDRESS, description="Address", required=False),
                FormField(name="website", field_type=FieldType.TEXT, description="Website URL", required=False),
            ],
        )
        return await self.extract(text, schema, **kwargs)
    
    async def extract_custom(
        self,
        text: str,
        fields: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ExtractionResult:
        """Extract with custom field definitions.
        
        Args:
            text: Text to process
            fields: List of field definitions, e.g.:
                [{"name": "product_name", "type": "text", "description": "Product name"}]
        """
        form_fields = []
        for f in fields:
            field_type = FieldType(f.get("type", "text"))
            form_fields.append(FormField(
                name=f["name"],
                field_type=field_type,
                description=f.get("description", ""),
                required=f.get("required", True),
                options=f.get("options"),
            ))
        
        schema = FormSchema(name="custom", fields=form_fields)
        return await self.extract(text, schema, **kwargs)
    
    async def close(self):
        """Close the client."""
        await self.client.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class BatchProcessor:
    """Process multiple items in batch for high throughput.
    
    Features:
    - Concurrent processing
    - Progress tracking
    - Error handling with retries
    - Streaming results
    """
    
    def __init__(
        self,
        client: VLLMClient,
        max_concurrency: int = 10,
        retry_count: int = 2,
    ):
        self.client = client
        self.max_concurrency = max_concurrency
        self.retry_count = retry_count
        self._semaphore: Optional[asyncio.Semaphore] = None
    
    @classmethod
    def create(
        cls,
        model: str,
        api_base: str = "http://localhost:8000/v1",
        max_concurrency: int = 10,
        **config_kwargs: Any,
    ) -> "BatchProcessor":
        """Create a BatchProcessor with a new client."""
        config = VLLMConfig(model=model, api_base=api_base, **config_kwargs)
        client = VLLMClient(config)
        return cls(client, max_concurrency)
    
    async def _process_item(
        self,
        item: BatchItem,
        schema: FormSchema,
        **kwargs: Any,
    ) -> BatchResult:
        """Process a single item."""
        start_time = time.time()
        
        for attempt in range(self.retry_count + 1):
            try:
                async with self._semaphore:
                    result = await self.client.extract(item.text, schema, **kwargs)
                
                processing_time = (time.time() - start_time) * 1000
                return BatchResult(
                    id=item.id,
                    result=result,
                    processing_time_ms=processing_time,
                )
            
            except Exception as e:
                if attempt == self.retry_count:
                    return BatchResult(
                        id=item.id,
                        result=ExtractionResult(
                            success=False,
                            validation_errors=[f"Failed after {self.retry_count + 1} attempts: {str(e)}"],
                        ),
                        processing_time_ms=(time.time() - start_time) * 1000,
                    )
                await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
    
    async def process_batch(
        self,
        items: list[BatchItem],
        schema: FormSchema,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs: Any,
    ) -> tuple[list[BatchResult], ProcessingStats]:
        """Process a batch of items.
        
        Args:
            items: List of items to process
            schema: Extraction schema
            progress_callback: Optional callback(completed, total)
        
        Returns:
            Tuple of (results, stats)
        """
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        start_time = time.time()
        
        async def process_with_progress(item: BatchItem, idx: int) -> BatchResult:
            result = await self._process_item(item, schema, **kwargs)
            if progress_callback:
                progress_callback(idx + 1, len(items))
            return result
        
        tasks = [
            process_with_progress(item, idx)
            for idx, item in enumerate(items)
        ]
        
        results = await asyncio.gather(*tasks)
        
        total_time = (time.time() - start_time) * 1000
        successful = sum(1 for r in results if r.result.success)
        
        stats = ProcessingStats(
            total_items=len(items),
            successful=successful,
            failed=len(items) - successful,
            total_time_ms=total_time,
            avg_time_ms=total_time / len(items) if items else 0,
            items_per_second=len(items) / (total_time / 1000) if total_time > 0 else 0,
        )
        
        return results, stats
    
    async def process_stream(
        self,
        items: list[BatchItem],
        schema: FormSchema,
        **kwargs: Any,
    ) -> AsyncIterator[BatchResult]:
        """Process items and yield results as they complete."""
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        
        async def process_item(item: BatchItem) -> BatchResult:
            return await self._process_item(item, schema, **kwargs)
        
        tasks = {
            asyncio.create_task(process_item(item)): item.id
            for item in items
        }
        
        while tasks:
            done, pending = await asyncio.wait(
                tasks.keys(),
                return_when=asyncio.FIRST_COMPLETED,
            )
            
            for task in done:
                result = task.result()
                del tasks[task]
                yield result
    
    async def classify_batch(
        self,
        texts: list[str],
        categories: list[str],
        multi_label: bool = False,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Classify multiple texts.
        
        Args:
            texts: List of texts to classify
            categories: Available categories
        """
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        
        async def classify_one(text: str) -> dict[str, Any]:
            async with self._semaphore:
                return await self.client.classify(text, categories, multi_label, **kwargs)
        
        tasks = [classify_one(text) for text in texts]
        return await asyncio.gather(*tasks)
    
    async def qa_batch(
        self,
        questions: list[str],
        contexts: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Answer multiple questions.
        
        Args:
            questions: List of questions
            contexts: Optional list of contexts (one per question)
        """
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        
        async def answer_one(question: str, context: Optional[str]) -> dict[str, Any]:
            async with self._semaphore:
                return await self.client.answer_question(question, context, **kwargs)
        
        contexts = contexts or [None] * len(questions)
        tasks = [answer_one(q, c) for q, c in zip(questions, contexts)]
        return await asyncio.gather(*tasks)
    
    async def close(self):
        """Close the client."""
        await self.client.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
