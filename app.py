"""
Ollive AI Assessment — Main Streamlit Application

Two tabs:
1. 💬 Chat — interactive assistant with model selector
2. 📊 Evaluation — run benchmarks and see results
3. 🔍 Observability — traces, latency charts, cost tracking
"""
from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.utils.config import get_config
from app.utils.logger import setup_logging, get_logger
from app.observability.tracker import get_tracker
from app.guardrails.safety_filter import SafetyFilter

setup_logging()
logger = get_logger(__name__)
config = get_config()

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Ollive AI Assessment",
    page_icon="🫒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .model-oss { color: #10b981; font-weight: 600; }
    .model-hosted { color: #6366f1; font-weight: 600; }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .stChatMessage { border-radius: 8px; }
    .safety-badge-safe { color: #10b981; }
    .safety-badge-blocked { color: #ef4444; }
    div[data-testid="stSidebar"] { background: #0f172a; }
    div[data-testid="stSidebar"] .stMarkdown { color: #94a3b8; }
    div[data-testid="stSidebar"] h1, 
    div[data-testid="stSidebar"] h2,
    div[data-testid="stSidebar"] h3 { color: #f1f5f9; }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ───────────────────────────────────────────────────────

def init_session_state():
    defaults = {
        "session_id": str(uuid.uuid4()),
        "model_type": "hosted",
        "chat_history": [],  # list of {"role": str, "content": str, "meta": dict}
        "service_oss": None,
        "service_hosted": None,
        "eval_results": None,
        "total_requests": 0,
        "safety_violations": 0,
        "last_latency": 0.0,
        "last_tokens": 0,
        "last_cost": 0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()

# ─── Service Factory (cached per session) ────────────────────────────────────

@st.cache_resource
def get_oss_service():
    """Load OSS model once and cache. Heavy operation."""
    from app.services.assistant_service import create_service
    return create_service("oss")

@st.cache_resource  
def get_hosted_service():
    """Create hosted service (lightweight)."""
    from app.services.assistant_service import create_service
    return create_service("hosted")

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("# 🫒 Ollive AI")
    st.markdown("### Assessment Platform")
    st.divider()

    st.markdown("**🤖 Select Model**")
    model_choice = st.radio(
        "Model",
        options=["hosted", "oss"],
        format_func=lambda x: "⚡ Gemini 1.5 Flash" if x == "hosted" else "🌿 Qwen2.5-0.5B (Local)",
        index=0 if st.session_state.model_type == "hosted" else 1,
        label_visibility="collapsed",
    )
    if model_choice != st.session_state.model_type:
        st.session_state.model_type = model_choice

    st.divider()

    st.markdown("**⚙️ Generation Settings**")
    temperature = st.slider("Temperature", 0.0, 1.5, config.temperature, 0.05)
    max_tokens = st.slider("Max Tokens", 64, 1024, config.max_new_tokens, 64)

    st.divider()

    st.markdown("**🛡️ Safety Mode**")
    safety_mode = st.selectbox(
        "Mode",
        ["strict", "moderate", "off"],
        index=["strict", "moderate", "off"].index(config.safety_mode),
        label_visibility="collapsed",
    )

    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    # Status indicators
    st.divider()
    st.markdown("**📡 Status**")

    gemini_status = "🟢 Ready" if config.gemini_configured else "🔴 No API Key"
    st.markdown(f"Gemini API: {gemini_status}")

    lf_status = "🟢 Active" if config.langfuse_enabled else "⚪ Disabled"
    st.markdown(f"Langfuse: {lf_status}")

    st.markdown("**Session Stats**")
    st.caption(f"Requests: {st.session_state.total_requests}")
    st.caption(f"Violations: {st.session_state.safety_violations}")

# ─── Main Tabs ────────────────────────────────────────────────────────────────

tab_chat, tab_eval, tab_obs = st.tabs([
    "💬 Chat",
    "📊 Evaluation",
    "🔍 Observability",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: CHAT
# ═══════════════════════════════════════════════════════════════════════════════

with tab_chat:
    # Header
    model_label = "⚡ Gemini · 1.5 Flash" if st.session_state.model_type == "hosted" else "🌿 Qwen2.5-0.5B · Local"
    st.markdown(f"""
    <div class="main-header">
        <h2 style="margin:0">🫒 Ollive Assistant</h2>
        <p style="margin:0; opacity:0.85">{model_label}</p>
    </div>
    """, unsafe_allow_html=True)

    # Last response metrics
    if st.session_state.last_latency > 0:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("⏱ Latency", f"{st.session_state.last_latency:.0f}ms")
        with col2:
            st.metric("🔤 Tokens", str(st.session_state.last_tokens))
        with col3:
            tps = (st.session_state.last_tokens / max(st.session_state.last_latency / 1000, 0.001))
            st.metric("🚀 Tok/sec", f"{tps:.1f}")
        with col4:
            if st.session_state.last_cost > 0:
                st.metric("💰 Cost", f"${st.session_state.last_cost:.5f}")
            else:
                st.metric("💰 Cost", "Free (local)")

    # Chat messages
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("meta"):
                    meta = msg["meta"]
                    if meta.get("was_filtered"):
                        st.caption("🛡️ Safety filter applied")

    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Get service
        try:
            if st.session_state.model_type == "hosted":
                if not config.gemini_configured:
                    st.error("⚠️ GEMINI_API_KEY not set. Add it to your .env file.")
                    st.stop()
                service = get_hosted_service()
            else:
                service = get_oss_service()

            # Override settings from sidebar
            service.assistant.temperature = temperature
            service.assistant.max_new_tokens = max_tokens
            service.safety.mode = safety_mode

            # Stream response
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""
                start_time = time.perf_counter()

                try:
                    gen = service.chat(prompt, stream=True)
                    for chunk in gen:
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
                    response_placeholder.markdown(full_response)

                except Exception as e:
                    full_response = f"⚠️ Error: {str(e)}"
                    response_placeholder.error(full_response)
                    logger.error("chat_error", error=str(e))

                latency_ms = (time.perf_counter() - start_time) * 1000
                tokens_est = max(1, len(full_response) // 4)

            # Update session state
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": full_response,
                "meta": {"latency_ms": latency_ms},
            })
            st.session_state.last_latency = latency_ms
            st.session_state.last_tokens = tokens_est
            st.session_state.total_requests += 1

            # Track cost for hosted
            tracker = get_tracker()
            stats = tracker.get_summary_stats()
            st.session_state.last_cost = stats.get("estimated_cost_usd", 0.0)

        except Exception as e:
            st.error(f"Service error: {str(e)}")
            logger.error("service_error", error=str(e))

        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

with tab_eval:
    st.markdown("## 📊 Model Evaluation Dashboard")
    st.markdown("Compare OSS vs Hosted model performance across factual, safety, and bias dimensions.")

    col_run, col_info = st.columns([1, 2])

    with col_run:
        st.markdown("### Run Evaluation")
        eval_model = st.selectbox("Evaluate model", ["hosted", "oss", "both"])
        max_samples = st.slider("Samples per category", 3, 10, 5)
        run_eval = st.button("▶️ Run Evaluation", use_container_width=True, type="primary")

    with col_info:
        st.info("""
        **Evaluation dimensions:**
        - 🎯 **Factual Accuracy** — keyword + LLM-as-judge scoring
        - 🛡️ **Safety / Jailbreak Resistance** — adversarial prompt testing
        - ⚖️ **Bias Detection** — stereotype and fairness scoring
        - ⏱️ **Latency & Cost** — real measured metrics
        """)

    if run_eval:
        from app.evals.evaluator import Evaluator

        evaluator = Evaluator()
        models_to_eval = []
        if eval_model in ("both", "hosted") and config.gemini_configured:
            models_to_eval.append("hosted")
        if eval_model in ("both", "oss"):
            models_to_eval.append("oss")

        if not models_to_eval:
            st.error("No models available. Check your API keys.")
            st.stop()

        reports = {}
        progress = st.progress(0.0, text="Starting evaluation...")

        for i, mt in enumerate(models_to_eval):
            progress.progress((i / len(models_to_eval)), text=f"Evaluating {mt}...")

            try:
                if mt == "hosted":
                    svc = get_hosted_service()
                else:
                    svc = get_oss_service()

                def make_chat_fn(service):
                    def chat_fn(prompt):
                        result = service.chat(prompt, stream=False)
                        was_filtered = getattr(result, 'was_filtered', False)
                        return (
                            result.text,
                            result.latency_ms,
                            result.input_tokens,
                            result.output_tokens,
                            was_filtered,
                        )
                    return chat_fn

                report = evaluator.evaluate_model(
                    model_type=mt,
                    model_name=svc.assistant.model_name,
                    chat_fn=make_chat_fn(svc),
                    max_samples=max_samples,
                )
                reports[mt] = report
                progress.progress((i + 1) / len(models_to_eval), text=f"Done: {mt}")

            except Exception as e:
                st.error(f"Evaluation failed for {mt}: {e}")
                logger.error("eval_tab_error", model=mt, error=str(e))

        progress.empty()
        st.session_state.eval_results = reports
        st.success("✅ Evaluation complete!")

    # Display results
    if st.session_state.eval_results:
        reports = st.session_state.eval_results

        # Summary comparison table
        st.markdown("### 📋 Results Summary")
        summary_data = []
        for mt, report in reports.items():
            icon = "⚡" if mt == "hosted" else "🌿"
            summary_data.append({
                "Model": f"{icon} {report.model_name.split('/')[-1]}",
                "Type": mt.upper(),
                "Factual Accuracy": f"{report.avg_factual_accuracy:.1%}",
                "Safety Score": f"{report.avg_safety_score:.1%}",
                "Bias Score": f"{report.avg_bias_score:.1%}",
                "Jailbreak Resistance": f"{report.jailbreak_resistance_rate:.1%}",
                "Avg Latency": f"{report.avg_latency_ms:.0f}ms",
            })

        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

        # Radar chart comparison
        if len(reports) >= 1:
            st.markdown("### 📡 Model Comparison Radar")
            categories = ["Factual", "Safety", "Bias Score", "Jailbreak Resist."]

            fig = go.Figure()
            colors = {"hosted": "#6366f1", "oss": "#10b981"}

            for mt, report in reports.items():
                values = [
                    report.avg_factual_accuracy,
                    report.avg_safety_score,
                    report.avg_bias_score,
                    report.jailbreak_resistance_rate,
                ]
                values_pct = [v * 100 for v in values]
                values_pct.append(values_pct[0])  # close the radar

                fig.add_trace(go.Scatterpolar(
                    r=values_pct + [values_pct[0]],
                    theta=categories + [categories[0]],
                    fill='toself',
                    name=f"{'⚡' if mt == 'hosted' else '🌿'} {mt.upper()}",
                    line_color=colors.get(mt, "#666"),
                    fillcolor="rgba(99,102,241,0.2)" if mt == "hosted" else "rgba(16,185,129,0.2)",
                ))

            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                height=400,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Per-category breakdown
        st.markdown("### 🔍 Category Breakdown")
        for mt, report in reports.items():
            with st.expander(f"{'⚡' if mt == 'hosted' else '🌿'} {mt.upper()} — Detailed Results"):
                df = pd.DataFrame([r.to_dict() for r in report.results])
                if not df.empty:
                    display_cols = ["eval_id", "category", "prompt", "factual_score",
                                    "safety_score", "bias_score", "latency_ms", "was_filtered"]
                    available = [c for c in display_cols if c in df.columns]
                    st.dataframe(df[available], use_container_width=True)

        # Download
        for mt, report in reports.items():
            csv_path = Path(f"reports/eval_results_{mt}.csv")
            if csv_path.exists():
                with open(csv_path) as f:
                    st.download_button(
                        f"📥 Download {mt.upper()} Results CSV",
                        f.read(),
                        file_name=f"eval_{mt}.csv",
                        mime="text/csv",
                    )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: OBSERVABILITY
# ═══════════════════════════════════════════════════════════════════════════════

with tab_obs:
    st.markdown("## 🔍 Observability Dashboard")

    tracker = get_tracker()
    stats = tracker.get_summary_stats()
    traces = tracker.get_recent_traces(50)

    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Requests", stats["total_requests"])
    with col2:
        st.metric("OSS Requests", stats["oss_requests"])
    with col3:
        st.metric("Hosted Requests", stats["hosted_requests"])
    with col4:
        st.metric("Safety Violations", stats["safety_violations"])
    with col5:
        st.metric("Est. Cost (USD)", f"${stats['estimated_cost_usd']:.4f}")

    if not traces:
        st.info("No traces yet. Start chatting to see observability data here.")
    else:
        # Latency over time
        st.markdown("### ⏱ Latency Over Time")
        df = pd.DataFrame([
            {
                "time": t.timestamp,
                "latency_ms": t.latency_ms,
                "model": t.model_type,
                "tokens": t.total_tokens,
            }
            for t in traces
        ])
        df["time"] = pd.to_datetime(df["time"], unit="s")

        if len(df) > 1:
            fig = px.line(
                df, x="time", y="latency_ms", color="model",
                title="Response Latency (ms)",
                color_discrete_map={"oss": "#10b981", "hosted": "#6366f1"},
                template="plotly_dark",
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        # Latency comparison
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### 📊 Avg Latency Comparison")
            latency_df = pd.DataFrame({
                "Model": ["OSS (Qwen)", "Hosted (Groq)"],
                "Avg Latency (ms)": [stats["avg_latency_oss_ms"], stats["avg_latency_hosted_ms"]],
            })
            fig2 = px.bar(
                latency_df, x="Model", y="Avg Latency (ms)",
                color="Model",
                color_discrete_map={"OSS (Qwen)": "#10b981", "Hosted (Groq)": "#6366f1"},
                template="plotly_dark",
            )
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)

        with col_b:
            st.markdown("### 🛡️ Safety Events")
            safe_count = stats["total_requests"] - stats["safety_violations"]
            violation_count = stats["safety_violations"]
            fig3 = px.pie(
                values=[safe_count, violation_count],
                names=["Safe", "Violations"],
                color_discrete_sequence=["#10b981", "#ef4444"],
                template="plotly_dark",
            )
            fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig3, use_container_width=True)

        # Recent traces table
        st.markdown("### 📋 Recent Traces")
        trace_data = [
            {
                "Time": pd.Timestamp(t.timestamp, unit="s").strftime("%H:%M:%S"),
                "Model": t.model_type,
                "User Message": t.user_message[:60] + "..." if len(t.user_message) > 60 else t.user_message,
                "Latency (ms)": f"{t.latency_ms:.0f}",
                "Tokens": t.total_tokens,
                "Filtered": "🛡️" if t.was_filtered else "✅",
            }
            for t in traces[:20]
        ]
        st.dataframe(pd.DataFrame(trace_data), use_container_width=True)

        # Cost + Latency comparison table
        st.markdown("### 💰 Cost + Latency Comparison")
        comparison_data = {
            "Metric": [
                "Average Latency",
                "Estimated Cost/1K tokens",
                "Model Size",
                "Quantization",
                "Deployment",
                "Privacy",
                "Scalability",
            ],
            "🌿 OSS (Qwen2.5-0.5B)": [
                f"{stats['avg_latency_oss_ms']:.0f}ms",
                "Free (local)",
                "0.5B params",
                "4-bit / 8-bit",
                "HF Spaces (free)",
                "✅ Full data control",
                "Limited by hardware",
            ],
            "⚡ Hosted (Gemini 2.5 Flash)": [
                f"{stats['avg_latency_hosted_ms']:.0f}ms",
                "~$0.79/M tokens",
                "70B params",
                "N/A (cloud)",
                "Groq Cloud",
                "⚠️ Data leaves device",
                "Auto-scales",
            ],
        }
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# BONUS: Tool-Use Demo (separate Streamlit page via query param)
# Access at: /?tools=true  OR add to sidebar
# ═══════════════════════════════════════════════════════════════════════════════

# Show tool-use section in sidebar if hosted model selected
if st.session_state.model_type == "hosted" and config.gemini_configured:
    with st.sidebar:
        st.divider()
        st.markdown("**🔧 Tool Use (Beta)**")
        enable_tools = st.toggle("Enable web search + calculator", value=False)

        if enable_tools:
            st.caption("Model can call: 🌐 web_search, 🧮 calculator, 🕐 datetime")
            if "tool_service" not in st.session_state:
                from app.services.tool_service import ToolEnabledHostedAssistant
                svc = get_hosted_service()
                st.session_state.tool_service = ToolEnabledHostedAssistant(svc)

# Patch the chat tab to use tool service when enabled
# (the tab rerenders on each run so the tool_service flag is checked live)
