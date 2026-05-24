"""
Base assistant abstraction.
All concrete assistants (OSS, Hosted) implement this interface.
This ensures identical behavior, same prompts, same evaluation surface.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generator, Optional
from enum import Enum

from app.utils.logger import get_logger

logger = get_logger(__name__)


class ModelType(str, Enum):
    OSS = "oss"
    HOSTED = "hosted"


@dataclass
class Message:
    """A single conversation turn."""
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class GenerationResult:
    """Result from a single generation call."""
    text: str
    model_type: ModelType
    model_name: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    was_filtered: bool = False
    filter_reason: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> float:
        if self.latency_ms <= 0:
            return 0.0
        return (self.output_tokens / self.latency_ms) * 1000


SYSTEM_PROMPT = """You are Ollive Assistant, a helpful, harmless, and honest AI assistant.

You provide clear, accurate, and thoughtful responses. You:
- Answer questions helpfully and concisely
- Admit when you don't know something
- Refuse to help with harmful, illegal, or unethical requests
- Do not make up facts or hallucinate information
- Maintain a friendly, professional tone

If asked to do something harmful, illegal, or unethical, politely decline and explain why."""


class BaseAssistant(ABC):
    """
    Abstract base class for all assistant implementations.

    Architecture decision: Using ABC + dataclasses over LangChain abstractions
    to keep dependencies minimal and give us full control over the inference loop.
    This is important for evaluation — we need precise latency, token counts, and
    the ability to intercept at every layer.
    """

    def __init__(
        self,
        model_name: str,
        model_type: ModelType,
        system_prompt: str = SYSTEM_PROMPT,
        max_history_turns: int = 10,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
    ) -> None:
        self.model_name = model_name
        self.model_type = model_type
        self.system_prompt = system_prompt
        self.max_history_turns = max_history_turns
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self._initialized = False

        logger.info(
            "assistant_created",
            model_name=model_name,
            model_type=model_type.value,
        )

    @abstractmethod
    def initialize(self) -> None:
        """Load model weights, authenticate API, etc."""
        ...

    @abstractmethod
    def _generate(
        self,
        messages: list[Message],
        stream: bool = False,
    ) -> GenerationResult | Generator[str, None, GenerationResult]:
        """
        Core generation method. Returns either a full result or a streaming generator.
        Subclasses implement this — NOT chat().
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is ready (model loaded / API reachable)."""
        ...

    def chat(
        self,
        user_message: str,
        history: list[Message],
        stream: bool = False,
    ) -> GenerationResult | Generator[str, None, None]:
        """
        Public chat interface. Handles:
        - history windowing
        - system prompt injection
        - pre/post hooks for guardrails and observability
        """
        if not self._initialized:
            self.initialize()

        # Build windowed message list
        messages = self._build_messages(user_message, history)

        logger.info(
            "chat_request",
            model=self.model_name,
            history_turns=len(history),
            user_message_len=len(user_message),
            stream=stream,
        )

        return self._generate(messages, stream=stream)

    def _build_messages(
        self, user_message: str, history: list[Message]
    ) -> list[Message]:
        """
        Build the message list with:
        1. System prompt
        2. Windowed history (last N turns)
        3. Current user message
        """
        messages: list[Message] = [
            Message(role="system", content=self.system_prompt)
        ]

        # Window history to last N complete turns (user+assistant pairs)
        windowed = history[-(self.max_history_turns * 2):]
        messages.extend(windowed)

        messages.append(Message(role="user", content=user_message))
        return messages

    def get_info(self) -> dict[str, Any]:
        """Return model metadata for UI display."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type.value,
            "initialized": self._initialized,
            "max_history_turns": self.max_history_turns,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
        }
