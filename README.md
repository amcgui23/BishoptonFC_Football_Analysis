# Bishopton Football Analyst – YouTube Streaming Fix v6

Streamlit app for youth football match analysis using the Gemini Interactions API.

## v6 YouTube architecture

- Gemini 3.8 Flash by default
- YouTube URL is sent as a video input with `processing: "agentic"`
- Video input is placed before the text prompt, matching the current Gemini video guidance
- Uses `stream=True` for long YouTube matches instead of background polling
- Displays processing/agentic progress while Gemini explores the match
- Retries temporary/high-demand failures and can fall back to Gemini 3.7 Flash
- Uploaded videos continue to use the File API + agentic processing + background execution

Google's current video documentation recommends streaming or background execution for long/complex video requests, and specifically documents agentic video processing for Gemini 3.8 Flash.

## Streamlit Secrets

Add one line in Streamlit Community Cloud Secrets:

`GEMINI_API_KEY = "AQ.your-key-here"`

Do not commit API keys to GitHub.
