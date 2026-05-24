---
title: Ollive AI Assessment
emoji: 🫒
colorFrom: purple
colorTo: indigo
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
pinned: false
license: mit
---

# 🫒 Ollive AI Assessment Platform

A mini AI inference & evaluation platform comparing:
- **🌿 OSS**: Qwen2.5-0.5B-Instruct (local)
- **⚡ Hosted**: Llama 3.3 70B via Groq API

## Setup on HF Spaces

1. Fork this Space
2. Add secrets in Space Settings:
   - `GROQ_API_KEY` — Get free at console.groq.com
   - `OSS_DEVICE` = `cpu` (Spaces zero GPU tier)
   - `OSS_QUANTIZATION` = `none` (CPU mode)
