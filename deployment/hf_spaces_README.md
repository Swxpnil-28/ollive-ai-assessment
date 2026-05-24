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
- **⚡ Hosted**: Gemini 2.5 Flash

## Setup on HF Spaces

1. Fork this Space
2. Add secrets in Space Settings:
   - `GEMINI_API_KEY` — Get free at aistudio.google.com/apikey
   - `OSS_DEVICE` = `cpu` (Spaces zero GPU tier)
   - `OSS_QUANTIZATION` = `none` (CPU mode)
