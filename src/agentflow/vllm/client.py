"""vLLM client for high-throughput inference."""

import json
import asyncio
from typing import Any, AsyncIterator, Optional, Union

import httpx
from pydantic import BaseModel, Field

from agentflow.vllm.schema import FormSchema, ExtractionResult


class VLLMConfig(BaseModel):
    """Configuration for vLLM client."""
    
    api_base: str = Field(default="http://localhost:8000/v1", description="vLLM server URL")
    model: str = Field(description="Model name or path")
    api_key: Optional[str] = Field(default=None, description="API key if required")
    timeout: float = Field(default=120.0, description="Request timeout")
    max_retries: int = Field(default=3, description="Maximum retries")
    
    # Generation parameters
    temperature: float = Field(default=0.1, description="Sampling temperature")
    max_tokens: int = Field(default=2048, description="Maximum tokens to generate")
    top_p: float = Field(default=0.95, description="Top-p sampling")
    top_k: int = Field(default=50, description="Top-k sampling")
    repetition_penalty: float = Field(default=1.0, description="Repetition penalty")
    
    # vLLM specific
    use_beam_search: bool = Field(default=False, description="Use beam search")
    best_of: int = Field(default=1, description="Number of sequences to generate")
    presence_penalty: float = Field(default=0.0, description="Presence penalty")
    frequency_penalty: float = Field(default=0.0, description="Frequency penalty")


class VLLMClient:
    """Client for vLLM server with OpenAI-compatible API.
    
    Optimized for:
    - High throughput batch processing
    - Structured output generation
    - Form data extraction
    """
    
    def __init__(self, config: VLLMConfig):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            
            self._client = httpx.AsyncClient(
                base_url=self.config.api_base,
                headers=headers,
                timeout=self.config.timeout,
            )
        return self._client
    
    async def close(self):
        """Close the client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        schema: Optional[FormSchema] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text completion.
        
        Args:
            prompt: Input prompt
            system_prompt: System prompt
            schema: Optional schema for structured output
            **kwargs: Override generation parameters
        """
        client = await self._get_client()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add schema instructions if provided
        if schema:
            prompt = f"{schema.to_prompt()}\n\nText to process:\n{prompt}"
        
        messages.append({"role": "user", "content": prompt})
        
        request_data = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "presence_penalty": kwargs.get("presence_penalty", self.config.presence_penalty),
            "frequency_penalty": kwargs.get("frequency_penalty", self.config.frequency_penalty),
        }
        
        # Add response format for structured output
        if schema:
            request_data["response_format"] = {
                "type": "json_object",
            }
        
        response = await client.post("/chat/completions", json=request_data)
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream text generation."""
        client = await self._get_client()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        request_data = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "stream": True,
        }
        
        async with client.stream("POST", "/chat/completions", json=request_data) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
                    except json.JSONDecodeError:
                        continue
    
    async def extract(
        self,
        text: str,
        schema: FormSchema,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> ExtractionResult:
        """Extract structured data from text using schema.
        
        Args:
            text: Text to extract from
            schema: Extraction schema
            system_prompt: Optional custom system prompt
            **kwargs: Generation parameters
        """
        default_system = f"""You are a precise data extraction assistant. 
Extract the requested information from the given text and return it as valid JSON.
If a required field cannot be found, use null.
Be accurate and only extract information that is explicitly stated."""

        try:
            response = await self.generate(
                prompt=text,
                system_prompt=system_prompt or default_system,
                schema=schema,
                **kwargs,
            )
            
            # Parse JSON response
            try:
                data = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    return ExtractionResult(
                        success=False,
                        raw_response=response,
                        validation_errors=["Failed to parse JSON response"],
                    )
            
            # Validate extracted data
            missing = []
            for field in schema.fields:
                if field.required and (field.name not in data or data[field.name] is None):
                    missing.append(field.name)
            
            return ExtractionResult(
                success=len(missing) == 0,
                data=data,
                raw_response=response,
                missing_fields=missing,
                confidence=1.0 if not missing else 0.5,
            )
        
        except Exception as e:
            return ExtractionResult(
                success=False,
                validation_errors=[str(e)],
            )
    
    async def classify(
        self,
        text: str,
        categories: list[str],
        multi_label: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Classify text into categories.
        
        Args:
            text: Text to classify
            categories: List of possible categories
            multi_label: Allow multiple labels
        """
        if multi_label:
            prompt = f"""Classify the following text into one or more of these categories: {categories}

Text: {text}

Return a JSON object with "categories" (list of matching categories) and "confidence" (list of scores)."""
        else:
            prompt = f"""Classify the following text into exactly one of these categories: {categories}

Text: {text}

Return a JSON object with "category" (string) and "confidence" (float 0-1)."""
        
        response = await self.generate(prompt, **kwargs)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"category": categories[0], "confidence": 0.0, "error": "Parse failed"}
    
    async def answer_question(
        self,
        question: str,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Answer a question, optionally with context.
        
        Args:
            question: Question to answer
            context: Optional context/document
        """
        if context:
            prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Provide a JSON response with "answer" and "confidence" (0-1) fields."""
        else:
            prompt = f"""Answer the following question.

Question: {question}

Provide a JSON response with "answer" and "confidence" (0-1) fields."""
        
        response = await self.generate(prompt, **kwargs)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"answer": response, "confidence": 0.5}
    
    async def summarize(
        self,
        text: str,
        max_length: int = 200,
        style: str = "concise",
        **kwargs: Any,
    ) -> str:
        """Summarize text.
        
        Args:
            text: Text to summarize
            max_length: Target summary length
            style: Summary style (concise, detailed, bullet_points)
        """
        style_instructions = {
            "concise": "Provide a brief, concise summary.",
            "detailed": "Provide a comprehensive summary covering all main points.",
            "bullet_points": "Provide a summary as bullet points.",
        }
        
        prompt = f"""{style_instructions.get(style, style_instructions['concise'])}
Keep the summary under {max_length} words.

Text to summarize:
{text}"""
        
        return await self.generate(prompt, **kwargs)
    
    async def health_check(self) -> bool:
        """Check if vLLM server is healthy."""
        try:
            client = await self._get_client()
            response = await client.get("/health")
            return response.status_code == 200
        except Exception:
            return False
    
    async def get_models(self) -> list[str]:
        """Get list of available models."""
        try:
            client = await self._get_client()
            response = await client.get("/models")
            data = response.json()
            return [m["id"] for m in data.get("data", [])]
        except Exception:
            return []
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
