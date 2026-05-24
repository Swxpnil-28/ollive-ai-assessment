# 🫒 Ollive AI Assessment Platform

> A production-quality mini AI inference & evaluation platform comparing open-source vs hosted LLMs.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 What This Is

This platform builds and evaluates **two AI assistants** with identical capabilities:

| | 🌿 OSS Assistant | ⚡ Hosted Assistant |
|---|---|---|
| **Model** | Qwen2.5-0.5B-Instruct | Gemini 2.5 Flash |
| **Inference** | Local (HuggingFace) | Google Gemini API |
| **Quantization** | 4-bit / 8-bit / FP32 | N/A |
| **Cost** | Free (compute only) | ~$0.30/M output tokens |
| **Latency** | 5–30s (CPU) | 500–2000ms |
| **Privacy** | ✅ Full data control | ⚠️ Data sent to Google |

---

## 🏗️ Architecture

```
ollive-ai-assessment/
├── app/
│   ├── models/
│   │   ├── base_assistant.py      # Abstract interface
│   │   ├── oss_assistant.py       # Qwen local inference
│   │   └── hosted_assistant.py    # Gemini API wrapper
│   ├── services/
│   │   └── assistant_service.py   # Orchestration layer
│   ├── memory/
│   │   └── conversation_memory.py # Session + history
│   ├── guardrails/
│   │   └── safety_filter.py       # Input/output filtering
│   ├── evals/
│   │   └── evaluator.py           # Benchmark framework
│   ├── observability/
│   │   └── tracker.py             # Traces + metrics
│   └── utils/
│       ├── config.py              # Pydantic settings
│       └── logger.py              # Structured logging
├── app.py                         # Streamlit UI
├── data/eval_datasets/            # Benchmark prompts
├── reports/                       # Eval outputs
├── tests/                         # Pytest suite
├── deployment/                    # HF Spaces, Streamlit Cloud
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

### Key Design Decisions

**1. Interchangeable Model Adapters**
`OSSAssistant` and `HostedAssistant` both extend `BaseAssistant`. The UI and evaluation framework call the same `AssistantService.chat()` regardless of backend. Swapping models = zero code changes.

**2. Safety as Middleware**
Guardrails run before AND after generation. Input filtering catches injection/jailbreak attempts. Output filtering catches cases where the model slipped through. This dual-layer approach is production-grade.

**3. Evaluation = Reproducible CI**
The evaluation framework uses the same `chat_fn` interface as the UI. Every eval is recorded to CSV for longitudinal tracking. LLM-as-judge uses Gemini (free tier) with heuristic fallbacks — no paid eval APIs required.

**4. Zero-config Observability**
Traces write to a local JSONL file by default. Add `LANGFUSE_*` env vars to get cloud tracing. The app works perfectly without Langfuse — it's additive.

**5. Free-tier First**
- OSS model: CPU mode + no quantization for HF Spaces Zero GPU
- Hosted model: Gemini free tier (1,500 requests/day)
- No Redis, no databases, no paid APIs required

---

## ⚡ Quick Start

### Option 1: Local Dev

```bash
git clone https://github.com/Swxpnil-28/ollive-ai-assessment
cd ollive-ai-assessment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

Open http://localhost:8501

### Option 2: Docker

```bash
cp .env.example .env
docker-compose up --build
```

### Option 3: Streamlit Community Cloud

1. Push repo to GitHub
2. Visit share.streamlit.io
3. Connect repo, set `app.py` as entry point
4. Add `GEMINI_API_KEY` in Secrets settings

---

## 🔑 Configuration

Copy `.env.example` to `.env`:

```bash
GEMINI_API_KEY=AIza...        # Get free at aistudio.google.com/apikey
OSS_DEVICE=cpu               # auto | cpu | cuda | mps
OSS_QUANTIZATION=none        # none | 4bit | 8bit (4bit needs CUDA)
SAFETY_MODE=strict           # strict | moderate | off
LANGFUSE_PUBLIC_KEY=pk-...   # optional
LANGFUSE_SECRET_KEY=sk-...   # optional
```

**Getting a free Gemini API key:**
1. Visit [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Sign in with Google (free, no credit card)
3. Create an API key
4. Add to `.env`

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Tests cover:
- Safety filter patterns (20 test cases)
- Conversation memory windowing
- Config validation
- No model loading required for test suite

---

## 📊 Evaluation Framework

### Running Evaluations

In the app, go to the **📊 Evaluation** tab and click **▶️ Run Evaluation**.

Or run from CLI:

```python
from app.services.assistant_service import create_service
from app.evals.evaluator import Evaluator

svc = create_service("hosted")
evaluator = Evaluator()

def chat_fn(prompt):
    result = svc.chat(prompt)
    return result.text, result.latency_ms, result.input_tokens, result.output_tokens, result.was_filtered

report = evaluator.evaluate_model("hosted", svc.assistant.model_name, chat_fn)
print(f"Factual accuracy: {report.avg_factual_accuracy:.1%}")
print(f"Safety score: {report.avg_safety_score:.1%}")
```

### Evaluation Datasets

| Dataset | Prompts | Purpose |
|---|---|---|
| `factual_prompts.json` | 12 | Factual accuracy (keyword + LLM judge) |
| `adversarial_prompts.json` | 10 | Jailbreak & injection resistance |
| `bias_prompts.json` | 8 | Stereotypes, fairness, discrimination |

### Scoring

- **Factual**: Keyword match + optional LLM-as-judge (Gemini)
- **Safety**: Refusal quality heuristic (0=complied, 1=refused)
- **Bias**: LLM judge + heuristic stereotype indicators
- **Latency**: Real wall-clock measurement

---

## 🛡️ Safety & Guardrails

The safety filter runs fully locally (no API calls). It catches:

- **Prompt injection** — "ignore previous instructions"
- **Jailbreaks** — DAN, developer mode, roleplay bypasses
- **Harmful content** — weapons, drugs, illegal activities
- **Self-harm** — responds with crisis resources (988 hotline)
- **Hate speech** — slurs and targeted violence

Safety mode options:
- `strict` — all layers active (recommended)
- `moderate` — skip overly broad jailbreak patterns
- `off` — passthrough (testing only)

---

## 🔍 Observability

### Local (default)
All traces written to `data/traces.jsonl`. View in the **🔍 Observability** tab.

### Langfuse (optional)
Add to `.env`:
```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```
Get free at [cloud.langfuse.com](https://cloud.langfuse.com)

Tracked per request:
- User message + assistant response
- Latency (ms)
- Token counts (input/output/total)
- Safety violations
- Estimated API cost

---

## 🚀 Deployment

### HF Spaces (OSS model)

```bash
# Required Space secrets:
# GEMINI_API_KEY, OSS_DEVICE=cpu, OSS_QUANTIZATION=none
```

### Cost + Latency Table

| Scenario | Cost | Latency |
|---|---|---|
| OSS local (CPU) | $0 | 5–30s |
| OSS deployed (HF Spaces free CPU) | $0 | 10–60s |
| OSS deployed (HF Spaces T4 GPU) | ~$0.35/hr | 1–3s |
| Hosted Gemini (free tier) | $0 (1,500 req/day) | 500–2000ms |
| Hosted Gemini (paid) | $0.30/M output tokens | 500–2000ms |

---

## 🔮 What I'd Improve With More Time

1. **RAG Memory** — ChromaDB for semantic conversation retrieval
2. **Tool Use** — Web search, calculator, code executor as function calls
3. **Agentic Eval** — Multi-step task completion benchmarks (GAIA, HumanEval)
4. **Streaming SSE API** — FastAPI backend so any frontend can consume it
5. **A/B Testing UI** — Side-by-side chat with both models simultaneously
6. **Redis-backed Memory** — Replace file sessions with Redis for horizontal scaling
7. **Better Bias Eval** — WinoBias, BBQ benchmark integration
8. **CI/CD Evals** — GitHub Actions running evals on every PR

---

## 📄 License

MIT — See [LICENSE](LICENSE)

---

*Built for the Ollive.ai Founding AI/ML Engineer assessment.*
