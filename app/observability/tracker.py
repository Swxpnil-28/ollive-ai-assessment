"""
Lightweight observability layer.

Strategy:
- Local JSON log file (always works, free tier, zero deps)
- Optional Langfuse integration (free cloud tracing)
- Metrics accumulation for dashboard display

This is designed so the app works perfectly with zero configuration,
but adding a LANGFUSE_PUBLIC_KEY unlocks full cloud tracing.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from app.utils.config import get_config
from app.utils.logger import get_logger

logger = get_logger(__name__)
config = get_config()

TRACES_FILE = Path("data/traces.jsonl")


@dataclass
class Trace:
    """Single request trace record."""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    model_name: str = ""
    model_type: str = ""
    user_message: str = ""
    assistant_message: str = ""
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    was_filtered: bool = False
    filter_reason: str = ""
    safety_violation: str = ""
    estimated_cost_usd: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)
    tokens_per_second: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class ObservabilityTracker:
    """
    Tracks all interactions for metrics and debugging.

    Local mode: appends to JSONL file
    Langfuse mode: additionally sends to Langfuse cloud
    """

    def __init__(self) -> None:
        self._langfuse_client = None
        self._traces: list[Trace] = []
        self._setup_langfuse()
        self._load_existing_traces()

    def _setup_langfuse(self) -> None:
        """Initialize Langfuse if credentials are configured."""
        if not config.langfuse_enabled:
            logger.info("langfuse_disabled", reason="no credentials configured")
            return
        try:
            from langfuse import Langfuse
            self._langfuse_client = Langfuse(
                public_key=config.langfuse_public_key,
                secret_key=config.langfuse_secret_key,
                host=config.langfuse_host,
            )
            logger.info("langfuse_initialized")
        except ImportError:
            logger.warning("langfuse_not_installed")
        except Exception as e:
            logger.warning("langfuse_init_failed", error=str(e))

    def _load_existing_traces(self) -> None:
        """Load existing traces from disk for dashboard display."""
        if not TRACES_FILE.exists():
            return
        try:
            with open(TRACES_FILE) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        # Convert dict back to Trace
                        self._traces.append(Trace(**{
                            k: v for k, v in data.items()
                            if k in Trace.__dataclass_fields__
                        }))
            logger.info("traces_loaded", count=len(self._traces))
        except Exception as e:
            logger.warning("traces_load_failed", error=str(e))

    def record(self, trace: Trace) -> None:
        """Record a trace to local storage and optionally Langfuse."""
        self._traces.append(trace)
        self._write_local(trace)
        self._send_langfuse(trace)

    def _write_local(self, trace: Trace) -> None:
        """Append trace to local JSONL file."""
        try:
            TRACES_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(TRACES_FILE, "a") as f:
                f.write(json.dumps(trace.to_dict()) + "\n")
        except Exception as e:
            logger.warning("trace_write_failed", error=str(e))

    def _send_langfuse(self, trace: Trace) -> None:
        """Send trace to Langfuse if available."""
        if not self._langfuse_client:
            return
        try:
            lf_trace = self._langfuse_client.trace(
                id=trace.trace_id,
                name="assistant_response",
                session_id=trace.session_id,
                input=trace.user_message,
                output=trace.assistant_message,
                metadata={
                    "model_name": trace.model_name,
                    "model_type": trace.model_type,
                    "latency_ms": trace.latency_ms,
                    "tokens": trace.total_tokens,
                    "was_filtered": trace.was_filtered,
                    "cost": trace.estimated_cost_usd,
                },
            )
        except Exception as e:
            logger.warning("langfuse_send_failed", error=str(e))

    # ─── Metrics ────────────────────────────────────────────────────────────────

    def get_summary_stats(self) -> dict:
        """Aggregate metrics for dashboard display."""
        if not self._traces:
            return {
                "total_requests": 0,
                "oss_requests": 0,
                "hosted_requests": 0,
                "avg_latency_oss_ms": 0,
                "avg_latency_hosted_ms": 0,
                "total_tokens": 0,
                "safety_violations": 0,
                "estimated_cost_usd": 0,
            }

        oss = [t for t in self._traces if t.model_type == "oss"]
        hosted = [t for t in self._traces if t.model_type == "hosted"]
        violations = [t for t in self._traces if t.was_filtered]

        return {
            "total_requests": len(self._traces),
            "oss_requests": len(oss),
            "hosted_requests": len(hosted),
            "avg_latency_oss_ms": round(
                sum(t.latency_ms for t in oss) / max(len(oss), 1), 1
            ),
            "avg_latency_hosted_ms": round(
                sum(t.latency_ms for t in hosted) / max(len(hosted), 1), 1
            ),
            "total_tokens": sum(t.total_tokens for t in self._traces),
            "safety_violations": len(violations),
            "estimated_cost_usd": round(
                sum(t.estimated_cost_usd for t in hosted), 6
            ),
        }

    def get_recent_traces(self, n: int = 20) -> list[Trace]:
        return sorted(self._traces, key=lambda t: t.timestamp, reverse=True)[:n]

    @property
    def traces(self) -> list[Trace]:
        return self._traces.copy()


# Global singleton
_tracker: Optional[ObservabilityTracker] = None


def get_tracker() -> ObservabilityTracker:
    global _tracker
    if _tracker is None:
        _tracker = ObservabilityTracker()
    return _tracker
