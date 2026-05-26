"""
Centralized configuration management using Pydantic Settings.
Single source of truth for all environment variables and defaults.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    """Application-wide configuration with env var support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Environment ---
    app_env: Literal["development", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # --- OSS Model ---
    oss_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    oss_device: str = "cpu"
    oss_quantization: Literal["4bit", "8bit", "none"] = "none"

    # --- Hosted Model (Gemini) ---
    gemini_api_key: str = Field(default="", description="Google Gemini API key")
    hosted_model_name: str = "gemini-2.5-flash"

    # --- Generation ---
    max_history_turns: int = 10
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    # --- Safety ---
    safety_mode: Literal["strict", "moderate", "off"] = "strict"
    enable_guardrails: bool = True

    # --- Observability ---
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # --- Evaluation ---
    eval_judge_model: str = "llama-3.3-70b-versatile"
    groq_judge_key: str = Field(default="", description="Groq API key for LLM judge")
    eval_output_dir: str = "reports/"

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        return v

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @property
    def gemini_configured(self) -> bool:
        return bool(self.gemini_api_key)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Cached config singleton."""
    return AppConfig()
