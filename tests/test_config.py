"""Tests for configuration management."""
import pytest
from app.utils.config import AppConfig


def test_default_config():
    cfg = AppConfig()
    assert cfg.max_history_turns == 10
    assert cfg.safety_mode in ("strict", "moderate", "off")
    assert 0.0 <= cfg.temperature <= 2.0


def test_temperature_validation():
    with pytest.raises(Exception):
        AppConfig(temperature=3.0)


def test_langfuse_disabled_without_keys():
    cfg = AppConfig(langfuse_public_key="", langfuse_secret_key="")
    assert not cfg.langfuse_enabled


def test_groq_configured():
    cfg = AppConfig(groq_api_key="test_key")
    assert cfg.groq_configured
