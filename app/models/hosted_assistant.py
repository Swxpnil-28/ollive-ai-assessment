"""
Hosted Assistant using Google Gemini API (new google-genai SDK).
"""
from __future__ import annotations

import time
from typing import Any, Generator, Optional

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.models.base_assistant import BaseAssistant, GenerationResult, Message, ModelType
from app.utils.config import get_config
from app.utils.logger import get_logger

logger = get_logger(__name__)
config = get_config()

GEMINI_PRICING = {
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro": {"input": 3.50, "output": 10.50},
    "gemini-2.5-flash": {"input": 0.10, "output": 0.40},
}

def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = GEMINI_PRICING.get(model, {"input": 0.075, "output": 0.30})
    return (
        input_tokens * pricing["input"] / 1_000_000
        + output_tokens * pricing["output"] / 1_000_000
    )

def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class HostedAssistant(BaseAssistant):

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            model_name=config.hosted_model_name,
            model_type=ModelType.HOSTED,
            **kwargs,
        )
        self._client = None

    def initialize(self) -> None:
        if self._initialized:
            return

        if not config.gemini_configured:
            raise ValueError(
                "GEMINI_API_KEY not set. Please add it to your .env file. "
                "Get a free key at aistudio.google.com"
            )

        from google import genai
        self._client = genai.Client(api_key=config.gemini_api_key)
        self._initialized = True
        logger.info("gemini_assistant_initialized", model=self.model_name)

    def _build_contents(self, messages: list[Message]) -> list[dict]:
        """Convert messages to Gemini contents format."""
        contents = []
        for msg in messages:
            if msg.role == "system":
                continue
            role = "user" if msg.role == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg.content}]})
        return contents

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _call_api(self, contents: list, stream: bool) -> Any:
        from google.genai import types
        system_msg = self.system_prompt

        if stream:
            return self._client.models.generate_content_stream(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_msg,
                    temperature=self.temperature,
                    max_output_tokens=self.max_new_tokens,
                    top_p=config.top_p,
                ),
            )
        return self._client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_msg,
                temperature=self.temperature,
                max_output_tokens=self.max_new_tokens,
                top_p=config.top_p,
            ),
        )

    def _generate(
        self,
        messages: list[Message],
        stream: bool = False,
    ) -> GenerationResult | Generator[str, None, None]:
        if stream:
            return self._generate_stream(messages)
        return self._generate_sync(messages)

    def _generate_sync(self, messages: list[Message]) -> GenerationResult:
        contents = self._build_contents(messages)

        start = time.perf_counter()
        response = self._call_api(contents, stream=False)
        latency_ms = (time.perf_counter() - start) * 1000

        text = response.text or ""

        try:
            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = response.usage_metadata.candidates_token_count
        except Exception:
            input_tokens = _estimate_tokens(" ".join(m.content for m in messages))
            output_tokens = _estimate_tokens(text)

        cost = _estimate_cost(self.model_name, input_tokens, output_tokens)

        logger.info(
            "gemini_response",
            model=self.model_name,
            latency_ms=round(latency_ms, 1),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=round(cost, 6),
        )

        return GenerationResult(
            text=text,
            model_type=ModelType.HOSTED,
            model_name=self.model_name,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            metadata={"estimated_cost_usd": cost},
        )

    def _generate_stream(self, messages: list[Message]) -> Generator[str, None, None]:
        contents = self._build_contents(messages)
        response = self._call_api(contents, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text

    def is_available(self) -> bool:
        return self._initialized and self._client is not None

    def get_info(self) -> dict[str, Any]:
        info = super().get_info()
        info.update({
            "api_provider": "Google Gemini",
            "pricing": GEMINI_PRICING.get(self.model_name, {}),
        })
        return info
