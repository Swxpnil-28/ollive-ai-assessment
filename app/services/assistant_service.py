"""
Assistant service — orchestrates the full request pipeline:
Input → Safety Check → Model → Safety Check → Observability → Output

This is the single entry point for all chat interactions.
Both the UI and evaluation framework call this, ensuring consistent behavior.
"""
from __future__ import annotations

import time
from typing import Generator, Optional

from app.guardrails.safety_filter import SafetyFilter, ViolationType
from app.memory.conversation_memory import ConversationMemory
from app.models.base_assistant import BaseAssistant, GenerationResult, ModelType
from app.models.hosted_assistant import HostedAssistant
from app.models.oss_assistant import OSSAssistant
from app.observability.tracker import Trace, get_tracker
from app.utils.config import get_config
from app.utils.logger import get_logger

logger = get_logger(__name__)
config = get_config()


class AssistantService:
    """
    Orchestrates a complete chat interaction with safety, memory, and observability.
    """

    def __init__(
        self,
        assistant: BaseAssistant,
        memory: ConversationMemory,
        safety_filter: Optional[SafetyFilter] = None,
    ) -> None:
        self.assistant = assistant
        self.memory = memory
        self.safety = safety_filter or SafetyFilter(mode=config.safety_mode)
        self.tracker = get_tracker()

    def chat(
        self,
        user_message: str,
        stream: bool = False,
    ) -> GenerationResult | Generator[str, None, None]:
        """
        Full chat pipeline:
        1. Input safety check
        2. Add to memory
        3. Model generation (stream or sync)
        4. Output safety check
        5. Add response to memory
        6. Record trace
        """
        # Step 1: Input safety
        if config.enable_guardrails:
            input_check = self.safety.check_input(user_message)
            if not input_check.is_safe:
                logger.warning(
                    "input_blocked",
                    violation=input_check.violation_type.value,
                    session=self.memory.session_id,
                )
                refusal = input_check.refusal_message
                self.memory.add_user_message(user_message)
                self.memory.add_assistant_message(refusal)
                self._record_trace(
                    user_message=user_message,
                    assistant_message=refusal,
                    result=None,
                    was_filtered=True,
                    filter_reason=input_check.violation_type.value,
                    safety_violation=input_check.violation_type.value,
                )
                if stream:
                    return self._stream_text(refusal)
                return GenerationResult(
                    text=refusal,
                    model_type=self.assistant.model_type,
                    model_name=self.assistant.model_name,
                    latency_ms=0.0,
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=0,
                    was_filtered=True,
                    filter_reason=input_check.violation_type.value,
                )

        # Step 2: Add user message to memory
        self.memory.add_user_message(user_message)

        # Step 3: Generate
        history = self.memory.get_history()[:-1]  # exclude the just-added user msg

        if stream:
            return self._chat_stream(user_message, history)

        result = self.assistant.chat(user_message, history, stream=False)

        # Step 4: Output safety check
        if config.enable_guardrails:
            output_check = self.safety.check_output(result.text)
            if not output_check.is_safe:
                refusal = output_check.refusal_message
                result.text = refusal
                result.was_filtered = True
                result.filter_reason = output_check.violation_type.value

        # Step 5: Store response
        self.memory.add_assistant_message(
            result.text,
            metadata={"latency_ms": result.latency_ms, "tokens": result.total_tokens},
        )

        # Step 6: Record trace
        self._record_trace(
            user_message=user_message,
            assistant_message=result.text,
            result=result,
        )

        return result

    def _chat_stream(
        self, user_message: str, history: list
    ) -> Generator[str, None, None]:
        """Streaming wrapper that accumulates for memory and observability."""
        start = time.perf_counter()
        full_response = ""

        gen = self.assistant.chat(user_message, history, stream=True)
        for chunk in gen:
            full_response += chunk
            yield chunk

        latency_ms = (time.perf_counter() - start) * 1000

        # Output safety on complete response
        if config.enable_guardrails:
            output_check = self.safety.check_output(full_response)
            if not output_check.is_safe:
                full_response = output_check.refusal_message

        self.memory.add_assistant_message(full_response)
        self._record_trace(
            user_message=user_message,
            assistant_message=full_response,
            result=None,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _stream_text(text: str) -> Generator[str, None, None]:
        """Convert a static string to a generator for uniform streaming interface."""
        yield text

    def _record_trace(
        self,
        user_message: str,
        assistant_message: str,
        result: Optional[GenerationResult],
        was_filtered: bool = False,
        filter_reason: str = "",
        safety_violation: str = "",
        latency_ms: float = 0.0,
    ) -> None:
        trace = Trace(
            session_id=self.memory.session_id,
            model_name=self.assistant.model_name,
            model_type=self.assistant.model_type.value,
            user_message=user_message[:500],  # truncate for storage
            assistant_message=assistant_message[:1000],
            latency_ms=result.latency_ms if result else latency_ms,
            input_tokens=result.input_tokens if result else 0,
            output_tokens=result.output_tokens if result else 0,
            total_tokens=result.total_tokens if result else 0,
            was_filtered=was_filtered or (result.was_filtered if result else False),
            filter_reason=filter_reason or (result.filter_reason or "" if result else ""),
            safety_violation=safety_violation,
            estimated_cost_usd=result.metadata.get("estimated_cost_usd", 0.0) if result else 0.0,
            tokens_per_second=result.tokens_per_second if result else 0.0,
        )
        self.tracker.record(trace)

    def reset(self) -> None:
        """Clear conversation history."""
        self.memory.clear()
        logger.info("conversation_reset", session=self.memory.session_id)


def create_service(model_type: str, session_id: Optional[str] = None) -> AssistantService:
    """
    Factory function to create a fully configured AssistantService.
    This is what the UI calls — hides all wiring.
    """
    memory = ConversationMemory(
        model_type=model_type,
        session_id=session_id,
        max_turns=config.max_history_turns,
    )

    if model_type == "oss":
        assistant: BaseAssistant = OSSAssistant(
            temperature=config.temperature,
            max_new_tokens=config.max_new_tokens,
            max_history_turns=config.max_history_turns,
        )
    elif model_type == "hosted":
        assistant = HostedAssistant(
            temperature=config.temperature,
            max_new_tokens=config.max_new_tokens,
            max_history_turns=config.max_history_turns,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'oss' or 'hosted'.")

    safety = SafetyFilter(mode=config.safety_mode)
    return AssistantService(assistant=assistant, memory=memory, safety_filter=safety)
