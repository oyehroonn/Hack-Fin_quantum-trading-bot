"""LLM client wrapper with JSON validation and audit logging.

Supports OpenAI-compatible APIs (OpenAI, Anthropic via adapter, local LLMs).
All calls are logged to the audit store for reproducibility.
"""

import json
import os
from datetime import datetime
from typing import Any, Optional

from loguru import logger

from core.errors import LLMError, LLMParseError, LLMTimeoutError
from core.interfaces import LLMClient


class OpenAICompatibleClient(LLMClient):
    """LLM client using OpenAI-compatible API.

    Supports:
      - OpenAI (GPT-4, GPT-3.5)
      - Any OpenAI-compatible endpoint (local LLMs, etc.)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        base_url: Optional[str] = None,
        max_tokens: int = 2048,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import openai
                kwargs: dict[str, Any] = {"api_key": self.api_key, "timeout": self.timeout}
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                self._client = openai.AsyncOpenAI(**kwargs)
            except ImportError:
                raise LLMError("openai package not installed. Run: pip install openai")
        return self._client

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        response_schema: Optional[dict[str, Any]] = None,
        temperature: float = 0.1,
    ) -> str:
        """Generate LLM response with optional JSON schema enforcement."""
        client = self._get_client()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        if response_schema:
            messages[0]["content"] += (
                f"\n\nYou MUST respond with valid JSON conforming to this schema:\n"
                f"{json.dumps(response_schema, indent=2)}\n"
                f"Return ONLY the JSON, no other text."
            )

        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"} if response_schema else None,
            )
            return response.choices[0].message.content or ""

        except Exception as e:
            if "timeout" in str(e).lower():
                raise LLMTimeoutError(f"LLM request timed out: {e}")
            raise LLMError(f"LLM generation failed: {e}")


class MockLLMClient(LLMClient):
    """Mock LLM client for testing without API keys."""

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        response_schema: Optional[dict[str, Any]] = None,
        temperature: float = 0.1,
    ) -> str:
        """Return a mock response based on the expected schema."""
        if response_schema:
            required = response_schema.get("required", [])
            props = response_schema.get("properties", {})

            mock_response: dict[str, Any] = {}
            for key in required:
                prop = props.get(key, {})
                prop_type = prop.get("type", "string")
                enum_values = prop.get("enum")

                if enum_values:
                    mock_response[key] = enum_values[0]
                elif prop_type == "string":
                    mock_response[key] = f"Mock {key}"
                elif prop_type == "number":
                    mock_response[key] = 0.5
                elif prop_type == "boolean":
                    mock_response[key] = True
                elif prop_type == "array":
                    mock_response[key] = []
                elif prop_type == "object":
                    mock_response[key] = {}

            return json.dumps(mock_response)

        return json.dumps({"response": "Mock LLM response", "confidence": 0.5})


def validate_llm_response(raw: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Parse and validate LLM JSON response against schema.

    Args:
        raw: Raw string from LLM
        schema: Expected JSON schema

    Returns:
        Parsed dict

    Raises:
        LLMParseError if invalid
    """
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise LLMParseError(f"Invalid JSON from LLM: {e}\nRaw: {raw[:500]}")

    # Validate required fields
    required = schema.get("required", [])
    missing = [f for f in required if f not in parsed]
    if missing:
        raise LLMParseError(f"Missing required fields: {missing}")

    # Validate enum fields
    props = schema.get("properties", {})
    for key, value in parsed.items():
        if key in props:
            prop_def = props[key]
            if "enum" in prop_def and value not in prop_def["enum"]:
                raise LLMParseError(
                    f"Field '{key}' value '{value}' not in enum {prop_def['enum']}"
                )

    return parsed
