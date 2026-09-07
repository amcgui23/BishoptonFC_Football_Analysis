# Bishopton FC Video Analyst

Streamlit app for AI-assisted youth football match analysis using Gemini video understanding.

## Current long-video approach
- Gemini 3.8 Flash is the default model.
- Uses the Interactions API with background execution for long-running analysis.
- Uses agentic video processing for long-form matches.
- Automatically retries temporary 429/5xx/high-demand failures with exponential backoff.
- Automatically falls back from Gemini 3.8 Flash to Gemini 3.7 Flash if a background job reports high demand.
- Supports public YouTube URLs and local video uploads up to the app's configured upload limit.

## Streamlit Secrets
Add this to Streamlit Cloud App Settings > Secrets:

```toml
GEMINI_API_KEY = "AQ.your-key-here"
```

Keep the real key private and never commit it to GitHub.

### v5 YouTube fix
YouTube analysis uses the current documented Interactions API YouTube input shape without the `processing` field. Agentic processing remains enabled for uploaded/File API videos, where Google documents that option. Background execution and automatic retry/backoff remain enabled.
