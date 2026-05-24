"""
OSS Assistant using Qwen2.5-0.5B-Instruct via HuggingFace Transformers.

Architecture decisions:
- Uses AutoModelForCausalLM for compatibility across model families
- 4-bit quantization via bitsandbytes for low VRAM (runs on ~2GB)
- CPU fallback for free-tier deployment (HF Spaces zero GPU)
- Streaming via TextIteratorStreamer in a thread
- Careful OOM handling with graceful degradation
"""
from __future__ import annotations

import gc
import os
import threading
import time
from typing import Any, Generator, Optional

import psutil
try:
    import torch
except ImportError:
    torch = None  # type: ignore

from app.models.base_assistant import BaseAssistant, GenerationResult, Message, ModelType
from app.utils.config import get_config
from app.utils.logger import get_logger

logger = get_logger(__name__)
config = get_config()


def _get_optimal_device() -> str:
    """Select best available device."""
    if config.oss_device != "auto":
        return config.oss_device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _estimate_tokens(text: str) -> int:
    """Fast token count estimation: ~4 chars per token."""
    return max(1, len(text) // 4)


class OSSAssistant(BaseAssistant):
    """
    Local open-source assistant.
    Supports 4-bit / 8-bit quantization, CPU fallback, and streaming.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            model_name=config.oss_model_name,
            model_type=ModelType.OSS,
            **kwargs,
        )
        self.device = _get_optimal_device()
        self.quantization = config.oss_quantization
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None

    def initialize(self) -> None:
        """Load model with quantization. Handles OOM gracefully."""
        if self._initialized:
            return

        logger.info(
            "loading_oss_model",
            model=self.model_name,
            device=self.device,
            quantization=self.quantization,
        )

        try:
            self._load_model()
            self._initialized = True
            logger.info("oss_model_loaded", model=self.model_name, device=self.device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("oom_fallback_cpu", error=str(e))
                self._cleanup()
                self.device = "cpu"
                self.quantization = "none"
                self._load_model()
                self._initialized = True
            else:
                raise

    def _load_model(self) -> None:
        """Internal model loading with quantization config."""
        # Lazy imports to avoid import-time torch load
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if self.quantization == "4bit" and self.device != "cpu":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = "auto"
        elif self.quantization == "8bit" and self.device != "cpu":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = "auto"
        else:
            # CPU or no quantization
            if self.device == "cpu":
                model_kwargs["torch_dtype"] = torch.float32
            else:
                model_kwargs["torch_dtype"] = torch.bfloat16
                model_kwargs["device_map"] = self.device

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **model_kwargs
        )

        if self.device == "cpu" and self.quantization == "none":
            self._model = self._model.to("cpu")

        self._model.eval()

    def _format_messages_for_model(self, messages: list[Message]) -> str:
        """
        Apply Qwen2.5 chat template. Falls back to manual formatting if
        tokenizer doesn't have apply_chat_template.
        """
        msg_dicts = [m.to_dict() for m in messages]

        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                return self._tokenizer.apply_chat_template(
                    msg_dicts,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        # Manual fallback for models without chat template
        formatted = ""
        for msg in messages:
            if msg.role == "system":
                formatted += f"<|system|>\n{msg.content}\n"
            elif msg.role == "user":
                formatted += f"<|user|>\n{msg.content}\n"
            elif msg.role == "assistant":
                formatted += f"<|assistant|>\n{msg.content}\n"
        formatted += "<|assistant|>\n"
        return formatted

    def _generate(
        self,
        messages: list[Message],
        stream: bool = False,
    ) -> GenerationResult | Generator[str, None, None]:
        """Generate response. Supports streaming via TextIteratorStreamer."""
        if stream:
            return self._generate_stream(messages)
        return self._generate_sync(messages)

    def _generate_sync(self, messages: list[Message]) -> GenerationResult:
        """Synchronous (non-streaming) generation."""
        from transformers import GenerationConfig

        prompt = self._format_messages_for_model(messages)
        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        )

        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_token_count = inputs["input_ids"].shape[1]

        start = time.perf_counter()
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=config.top_p,
                do_sample=self.temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        latency_ms = (time.perf_counter() - start) * 1000

        # Decode only new tokens
        new_tokens = output_ids[0][input_token_count:]
        output_text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        output_token_count = len(new_tokens)

        return GenerationResult(
            text=output_text,
            model_type=ModelType.OSS,
            model_name=self.model_name,
            latency_ms=latency_ms,
            input_tokens=input_token_count,
            output_tokens=output_token_count,
            total_tokens=input_token_count + output_token_count,
            metadata={
                "device": str(device),
                "quantization": self.quantization,
                "memory_mb": self._get_memory_usage(),
            },
        )

    def _generate_stream(self, messages: list[Message]) -> Generator[str, None, None]:
        """
        Streaming generation using TextIteratorStreamer in a background thread.
        Yields text tokens as they're generated.
        """
        from transformers import TextIteratorStreamer

        prompt = self._format_messages_for_model(messages)
        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        )
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = {
            **inputs,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": config.top_p,
            "do_sample": self.temperature > 0,
            "pad_token_id": self._tokenizer.eos_token_id,
            "streamer": streamer,
        }

        thread = threading.Thread(
            target=self._model.generate,
            kwargs=gen_kwargs,
            daemon=True,
        )
        thread.start()

        for token in streamer:
            if token:
                yield token

        thread.join()

    def is_available(self) -> bool:
        return self._initialized and self._model is not None

    def _cleanup(self) -> None:
        """Free GPU/CPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_memory_usage(self) -> float:
        """Return current process memory in MB."""
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / (1024 * 1024)

    def get_info(self) -> dict[str, Any]:
        info = super().get_info()
        info.update({
            "device": self.device,
            "quantization": self.quantization,
            "memory_mb": self._get_memory_usage(),
        })
        return info
