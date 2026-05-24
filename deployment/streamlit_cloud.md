# Deploy to Streamlit Community Cloud

## Steps

1. Push repo to GitHub
2. Go to share.streamlit.io
3. Connect your GitHub repo
4. Set `app.py` as the main file
5. Add secrets in the Streamlit Cloud dashboard:

```toml
# .streamlit/secrets.toml (don't commit this)
GROQ_API_KEY = "your_groq_key"
OSS_DEVICE = "cpu"
OSS_QUANTIZATION = "none"
SAFETY_MODE = "strict"
```

## Notes
- Streamlit Cloud has limited RAM (~1GB). OSS model may be slow.
- Hosted (Groq) mode works great on Streamlit Cloud.
- Set `OSS_QUANTIZATION=none` and `OSS_DEVICE=cpu` for free tier.
