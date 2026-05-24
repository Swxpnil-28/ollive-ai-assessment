"""Tests for conversation memory."""
import pytest
from app.memory.conversation_memory import ConversationMemory
from app.models.base_assistant import Message


def test_add_and_retrieve():
    mem = ConversationMemory(model_type="test", persist=False)
    mem.add_user_message("Hello")
    mem.add_assistant_message("Hi there!")
    history = mem.get_history()
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[1].role == "assistant"


def test_clear():
    mem = ConversationMemory(model_type="test", persist=False)
    mem.add_user_message("Hello")
    mem.clear()
    assert mem.turn_count == 0


def test_windowing():
    mem = ConversationMemory(model_type="test", persist=False, max_turns=2)
    for i in range(5):
        mem.add_user_message(f"Message {i}")
        mem.add_assistant_message(f"Response {i}")
    history = mem.get_history()
    # Should only return last 2 turns = 4 messages
    assert len(history) <= 4


def test_display_history():
    mem = ConversationMemory(model_type="test", persist=False)
    mem.add_user_message("Q1")
    mem.add_assistant_message("A1")
    mem.add_user_message("Q2")
    mem.add_assistant_message("A2")
    pairs = mem.get_display_history()
    assert len(pairs) == 2
    assert pairs[0] == ("Q1", "A1")
    assert pairs[1] == ("Q2", "A2")
