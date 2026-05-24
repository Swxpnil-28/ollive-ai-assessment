"""
Reusable Streamlit UI components.
"""
from __future__ import annotations

import streamlit as st
from typing import Optional


def render_message(role: str, content: str) -> None:
    """Render a single chat message with avatar."""
    with st.chat_message(role):
        st.markdown(content)


def render_metrics_row(metrics: dict) -> None:
    """Render a row of metric cards."""
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        with col:
            st.metric(label, value)


def render_model_badge(model_type: str, model_name: str) -> None:
    """Render a colored badge for model identification."""
    if model_type == "oss":
        color = "#10b981"
        icon = "🌿"
        label = "OSS Local"
    else:
        color = "#6366f1"
        icon = "⚡"
        label = "Groq Hosted"

    st.markdown(
        f"""<div style="
            display: inline-block;
            padding: 4px 12px;
            background: {color}22;
            border: 1px solid {color};
            border-radius: 20px;
            color: {color};
            font-size: 0.85em;
            font-weight: 600;
            margin-bottom: 8px;
        ">{icon} {label}: <code style="color:{color}">{model_name.split('/')[-1]}</code></div>""",
        unsafe_allow_html=True,
    )


def render_safety_badge(was_filtered: bool, violation: str = "") -> None:
    """Show safety status indicator."""
    if was_filtered:
        st.warning(f"🛡️ Safety filter triggered: `{violation}`")
    else:
        st.success("✅ Safety check passed")


def render_generation_stats(latency_ms: float, tokens: int, cost: float = 0.0) -> None:
    """Show generation stats below a response."""
    parts = [
        f"⏱ {latency_ms:.0f}ms",
        f"🔤 {tokens} tokens",
    ]
    if cost > 0:
        parts.append(f"💰 ${cost:.5f}")
    st.caption(" · ".join(parts))
