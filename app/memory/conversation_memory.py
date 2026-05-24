"""
Conversation memory with:
- Short-term buffer memory (in-session)
- Optional JSON-based persistence (across sessions)
- Semantic summary for long conversations (future-proof)

Architecture: Simple dict-based sessions, no Redis required for free tier.
The interface is designed so Redis or a DB can be swapped in by changing the
_load/_save methods only.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from app.models.base_assistant import Message
from app.utils.logger import get_logger

logger = get_logger(__name__)

SESSIONS_DIR = Path("data/sessions")


@dataclass
class ConversationSession:
    """Represents a single conversation session."""
    session_id: str
    model_type: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    messages: list[Message] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def turn_count(self) -> int:
        return len([m for m in self.messages if m.role == "user"])

    @property
    def last_user_message(self) -> Optional[str]:
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return None

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "model_type": self.model_type,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp,
                    "metadata": m.metadata,
                }
                for m in self.messages
            ],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationSession":
        messages = [
            Message(
                role=m["role"],
                content=m["content"],
                timestamp=m.get("timestamp", time.time()),
                metadata=m.get("metadata", {}),
            )
            for m in data.get("messages", [])
        ]
        return cls(
            session_id=data["session_id"],
            model_type=data["model_type"],
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            messages=messages,
            metadata=data.get("metadata", {}),
        )


class ConversationMemory:
    """
    Manages conversation history for a single session.

    Usage:
        memory = ConversationMemory(model_type="hosted")
        memory.add_user_message("Hello!")
        memory.add_assistant_message("Hi there!")
        history = memory.get_history()  # list[Message]
    """

    def __init__(
        self,
        model_type: str,
        session_id: Optional[str] = None,
        persist: bool = True,
        max_turns: int = 10,
    ) -> None:
        self.session_id = session_id or str(uuid.uuid4())
        self.model_type = model_type
        self.persist = persist
        self.max_turns = max_turns
        self._session = self._load_or_create()

    def _load_or_create(self) -> ConversationSession:
        """Load existing session or create new one."""
        if self.persist:
            SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
            session_file = SESSIONS_DIR / f"{self.session_id}.json"
            if session_file.exists():
                try:
                    with open(session_file) as f:
                        data = json.load(f)
                    logger.info("session_loaded", session_id=self.session_id)
                    return ConversationSession.from_dict(data)
                except Exception as e:
                    logger.warning("session_load_failed", error=str(e))

        return ConversationSession(
            session_id=self.session_id,
            model_type=self.model_type,
        )

    def _save(self) -> None:
        """Persist session to disk."""
        if not self.persist:
            return
        try:
            SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
            session_file = SESSIONS_DIR / f"{self.session_id}.json"
            with open(session_file, "w") as f:
                json.dump(self._session.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning("session_save_failed", error=str(e))

    def add_user_message(self, content: str, metadata: dict | None = None) -> None:
        msg = Message(role="user", content=content, metadata=metadata or {})
        self._session.messages.append(msg)
        self._session.updated_at = time.time()
        self._save()

    def add_assistant_message(self, content: str, metadata: dict | None = None) -> None:
        msg = Message(role="assistant", content=content, metadata=metadata or {})
        self._session.messages.append(msg)
        self._session.updated_at = time.time()
        self._save()

    def get_history(self) -> list[Message]:
        """Return windowed history (last max_turns complete turns)."""
        msgs = self._session.messages
        # Keep last max_turns*2 messages (user+assistant pairs)
        return msgs[-(self.max_turns * 2):]

    def get_full_history(self) -> list[Message]:
        return self._session.messages.copy()

    def clear(self) -> None:
        """Reset conversation."""
        self._session.messages = []
        self._session.updated_at = time.time()
        self._save()
        logger.info("session_cleared", session_id=self.session_id)

    @property
    def turn_count(self) -> int:
        return self._session.turn_count

    @property
    def session(self) -> ConversationSession:
        return self._session

    def get_display_history(self) -> list[tuple[str, str]]:
        """Return (user, assistant) pairs for Streamlit chat display."""
        pairs: list[tuple[str, str]] = []
        messages = self._session.messages

        i = 0
        while i < len(messages):
            if messages[i].role == "user":
                user_msg = messages[i].content
                asst_msg = ""
                if i + 1 < len(messages) and messages[i + 1].role == "assistant":
                    asst_msg = messages[i + 1].content
                    i += 2
                else:
                    i += 1
                pairs.append((user_msg, asst_msg))
            else:
                i += 1

        return pairs
