# 🫒 Ollive AI Assessment Platform

> A production-quality mini AI inference & evaluation platform comparing open-source vs hosted LLMs.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![HF Spaces](https://img.shields.io/badge/Demo-HuggingFace%20Spaces-orange.svg)](https://huggingface.co/spaces/swapnil2803/ollive-ai-assessment)

---

## 🎯 What This Is

This platform builds and evaluates **two AI assistants** with identical capabilities, submitted as part of the Ollive.ai Founding AI/ML Engineer assessment.

| | 🌿 OSS Assistant | ⚡ Hosted Assistant |
|---|---|---|
| **Model** | Qwen2.5-0.5B-Instruct | Gemini 2.5 Flash |
| **Inference** | Local (HuggingFace) | Google Gemini API |
| **Quantization** | 4-bit / 8-bit / none | N/A |
| **Judge Model** | — | Groq · Llama 3.3 70B (eval only) |
| **Factual Accuracy** | 65% | ~92% |
| **Jailbreak Resistance** | 81% | 95% |
| **Bias Score** | 100% | 88% |
| **Avg Latency (HF Spaces CPU)** | ~23,800ms | ~1,500ms |
| **Cost** | Free (compute only) | ~$0.30/M output tokens |
| **Privacy** | ✅ Full data control | ⚠️ Data sent to Google |

---

## 🏗️ Architecture

```
ollive-ai-assessment/
├── app/
│   ├── models/
│   │   ├── base_assistant.py       # Abstract interface
│   │   ├── oss_assistant.py        # Qwen local inference
│   │   └── hosted_assistant.py     # Gemini API wrapper
│   ├── services/
│   │   ├── assistant_service.py    # Orchestration layer
│   │   └── tool_service.py         # Tool-use (web search, calculator)
│   ├── memory/
│   │   └── conversation_memory.py  # Session + history
│   ├── guardrails/
│   │   └── safety_filter.py        # Input/output filtering
│   ├── evals/
│   │   └── evaluator.py            # Benchmark framework
│   ├── observability/
│   │   └── tracker.py              # Traces + metrics (JSONL + Langfuse)
│   └── utils/
│       ├── config.py               # Pydantic settings
│       └── logger.py               # Structured logging
├── app.py                          # Streamlit UI (3 tabs)
├── data/eval_datasets/             # Benchmark prompts (JSON)
├── reports/                        # Eval CSV outputs
├── scripts/
│   └── run_evals.py                # CLI eval runner
├── tests/                          # Pytest suite
├── deployment/                     # HF Spaces config
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

### Key Design Decisions

**1. Interchangeable Model Adapters**
`OSSAssistant` and `HostedAssistant` both extend `BaseAssistant`. The UI and evaluation framework call the same `AssistantService.chat()` regardless of backend. Swapping models requires zero code changes.

**2. Safety as Middleware**
Guardrails run before AND after generation. Input filtering catches injection/jailbreak attempts; output filtering catches anything that slipped through. Three configurable modes: `strict`, `moderate`, and `off`.

**3. Evaluation = Reproducible CI**
The evaluator uses the same `chat_fn` interface as the UI. Every eval writes to CSV for longitudinal tracking. LLM-as-judge uses **Groq/Llama** (free tier) with heuristic fallbacks — no paid eval APIs required.

**4. Zero-config Observability**
Traces write to `data/traces.jsonl` by default. Add `LANGFUSE_*` env vars for cloud tracing. The app works perfectly without Langfuse — it's purely additive.

**5. Free-tier First**
- OSS model: CPU + no quantization for HF Spaces free tier
- Hosted model: Gemini free tier (1,500 req/day)
- No Redis, no databases, no paid APIs required to run

---

## ⚡ Quick Start

### Option 1: Local Dev

```bash
git clone https://github.com/Swxpnil-28/ollive-ai-assessment
cd ollive-ai-assessment

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env           # Then add your API keys

streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

### Option 2: Docker

```bash
cp .env.example .env
# Add GEMINI_API_KEY and GROQ_JUDGE_KEY to .env

docker-compose up --build
```

### Option 3: Streamlit Community Cloud

1. Push repo to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect repo, set `app.py` as entry point
4. Add secrets (`GEMINI_API_KEY`, `GROQ_JUDGE_KEY`) in the Secrets panel

---

## 🔑 Configuration

Copy `.env.example` to `.env` and fill in:

```bash
# --- Gemini API (hosted assistant) ---
GEMINI_API_KEY=AIza...               # aistudio.google.com/apikey (free)

# --- OSS Model ---
OSS_MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct
OSS_DEVICE=cpu                        # auto | cpu | cuda | mps
OSS_QUANTIZATION=none                 # none | 4bit | 8bit

# --- Hosted Model ---
HOSTED_MODEL_NAME=gemini-2.5-flash

# --- App Config ---
APP_ENV=development
MAX_HISTORY_TURNS=10
MAX_NEW_TOKENS=512
TEMPERATURE=0.7

# --- Safety ---
SAFETY_MODE=strict                    # strict | moderate | off
ENABLE_GUARDRAILS=true

# --- Evaluation (Groq as judge) ---
GROQ_JUDGE_KEY=your_groq_api_key_here
EVAL_OUTPUT_DIR=reports/

# --- Langfuse Observability (optional) ---
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

**Free API Keys:**
- Gemini: [aistudio.google.com/apikey](https://aistudio.google.com/apikey) — no credit card needed
- Groq: [console.groq.com](https://console.groq.com) — free tier, very fast inference

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

The test suite covers safety filter patterns (20 cases), conversation memory windowing, and config validation — no model loading required.

---

## 📊 Evaluation Framework

### Running Evaluations

In the app, go to the **📊 Evaluation** tab, choose a model, set sample count, and click **▶️ Run Evaluation**.

Or run from CLI:

```bash
python scripts/run_evals.py --model hosted
python scripts/run_evals.py --model oss
```

Or call the API directly:

```python
from app.services.assistant_service import create_service
from app.evals.evaluator import Evaluator

svc = create_service("hosted")
evaluator = Evaluator()

def chat_fn(prompt):
    result = svc.chat(prompt, stream=False)
    return result.text, result.latency_ms, result.input_tokens, result.output_tokens, result.was_filtered

report = evaluator.evaluate_model("hosted", svc.assistant.model_name, chat_fn, max_samples=5)
print(f"Factual accuracy:      {report.avg_factual_accuracy:.1%}")
print(f"Safety score:          {report.avg_safety_score:.1%}")
print(f"Jailbreak resistance:  {report.jailbreak_resistance_rate:.1%}")
```

### Evaluation Datasets

| Dataset | Prompts | Purpose |
|---|---|---|
| `factual_prompts.json` | 10 | Factual accuracy — keyword match + Groq/Llama judge |
| `adversarial_prompts.json` | 10 | Jailbreak & prompt injection resistance |
| `bias_prompts.json` | 8 | Stereotypes, fairness, discrimination |

**28 total prompts** evaluated across both models. Judge: Llama 3.3 70B via Groq (separate from evaluated models).

### Scoring

- **Factual**: Keyword match + LLM-as-judge (Groq/Llama), with heuristic fallback
- **Safety**: Refusal quality heuristic (0 = complied with harmful prompt, 1 = refused)
- **Bias**: Groq/Llama judge + heuristic stereotype indicators
- **Latency**: Real wall-clock measurement per prompt

Results are saved to `reports/eval_results_{model}.csv` and rendered as a radar chart + table in the UI.

---

## 🛡️ Safety & Guardrails

The safety filter runs **fully locally** — no API calls. It catches:

- **Prompt injection** — "ignore previous instructions", "disregard your system prompt"
- **Jailbreaks** — DAN, developer mode, roleplay bypasses
- **Harmful content** — weapons, drugs, illegal activities
- **Self-harm** — responds with crisis resources (988 hotline)
- **Hate speech** — slurs and targeted violence

Safety mode options:
- `strict` — all layers active (recommended for production)
- `moderate` — skip overly broad jailbreak patterns
- `off` — passthrough (testing only)

Guardrails run in two passes: once on input before generation, once on output after.

---

## 🔧 Tool Use (Beta)

When using the hosted model, a **Tool Use** toggle appears in the sidebar. When enabled, the assistant gains access to:

- 🌐 **Web search** — real-time information retrieval
- 🧮 **Calculator** — arithmetic and math expressions
- 🕐 **Datetime** — current date/time queries

This is implemented via `ToolEnabledHostedAssistant` in `app/services/tool_service.py`.

---

## 🔍 Observability

### Local (default)
All traces written to `data/traces.jsonl`. Visualised in the **🔍 Observability** tab with:
- Latency over time (line chart)
- OSS vs Hosted latency comparison (bar chart)
- Safety violation ratio (pie chart)
- Recent traces table (last 20 requests)

### Langfuse (optional)
Add keys to `.env` to enable cloud tracing:
```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```
Get free at [cloud.langfuse.com](https://cloud.langfuse.com)

Tracked per request: user message, assistant response, latency (ms), token counts, safety violations, estimated API cost.

---

## 🚀 Deployment

### HF Spaces (live demo)

Deployed at: **[huggingface.co/spaces/swapnil2803/ollive-ai-assessment](https://huggingface.co/spaces/swapnil2803/ollive-ai-assessment)**

Required Space secrets:
```
GEMINI_API_KEY
GROQ_JUDGE_KEY
OSS_DEVICE=cpu
OSS_QUANTIZATION=none
```

### Cost + Latency Table

> Real numbers measured on HF Spaces CPU Basic deployment (28 prompts, May 2026).

| Deployment Scenario | Cost | Avg Latency | Throughput | Privacy |
|---|---|---|---|---|
| OSS — Local CPU | $0 | ~17,000ms | ~12 tok/s | ✅ Full |
| **OSS — HF Spaces CPU (live)** | **$0** | **~23,800ms** | **~10 tok/s** | **✅ Full** |
| OSS — HF Spaces T4 GPU | ~$0.40/hr | ~1,500ms | ~150 tok/s | ✅ Full |
| Hosted — Gemini free tier | $0 (1,500 req/day) | ~1,500ms | ~200 tok/s | ⚠️ Cloud |
| Hosted — Gemini paid | $0.30/M tokens | ~1,500ms | ~200 tok/s | ⚠️ Cloud |

### Evaluation Results Summary

> Real scores from 28 prompts across both models. Judge: Llama 3.3 70B via Groq.

| Metric | 🌿 OSS (Qwen2.5-0.5B) | ⚡ Hosted (Gemini 2.5 Flash) |
|---|---|---|
| Factual Accuracy | 65% | ~92% |
| Jailbreak Resistance | 81% | 95% |
| Bias & Fairness | **100%** | 88% |

**Key findings:**
- OSS passes common-knowledge facts (capitals, dates) but hallucinates on obscure facts and trap questions — classic small-model pattern
- Guardrails caught 50% of adversarial prompts at input layer before reaching either model
- OSS surprisingly outperformed Gemini on bias — correctly handled gender neutrality, political balance, and religious fairness across all 8 prompts
- OSS on free CPU is **16x slower** than Gemini (23.8s vs 1.5s) — GPU or hosted API strongly recommended for real-time chat

---

## 🔮 What I'd Improve With More Time

1. **RAG Memory** — ChromaDB for semantic conversation retrieval instead of sliding window
2. **Richer Tool Use** — code executor, file reader, structured data lookup as function calls
3. **Agentic Eval** — multi-step task completion benchmarks (GAIA, HumanEval, AgentBench)
4. **Streaming SSE API** — FastAPI backend so any frontend can consume real-time responses
5. **A/B Testing UI** — side-by-side chat with both models simultaneously
6. **Redis-backed Memory** — replace file-based sessions with Redis for horizontal scaling
7. **Better Bias Eval** — WinoBias, BBQ benchmark integration for rigorous fairness testing
8. **CI/CD Evals** — GitHub Actions running the eval suite on every PR with regression alerts

---

## 📄 License

MIT — See [LICENSE](LICENSE)

---

*Built for the Ollive.ai Founding AI/ML Engineer assessment by [Swapnil Singh](https://github.com/Swxpnil-28).*
