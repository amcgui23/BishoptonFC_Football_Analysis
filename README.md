# Bishopton FC Video Analyst

Streamlit app for AI-assisted youth football match analysis using Google's Gemini API.

## What this version does
- YouTube public video analysis
- Local video upload up to 3 GB through Streamlit (Gemini free-tier file limit may be lower)
- Gemini 3.7 Flash by default
- Agentic video processing for long-form football matches
- Background Gemini Interactions API jobs to avoid the normal synchronous request deadline
- Polling UI that shows the Gemini job status while the analysis runs
- Optional squad roster with shirt numbers
- Optional coach focus
- Match report, tactical observations, player ratings, statistics, key moments and coach report
- JSON export

## Gemini key
In Streamlit Community Cloud, open App Settings > Secrets and paste:

```toml
GEMINI_API_KEY = "AQ.your-key-here"
```

Use straight ASCII quotes. Do not commit the key to GitHub.

## Long-video design
The app uses Gemini's Interactions API with `background=True` and agentic video processing. This is important for full football matches because a normal synchronous request can hit a gateway deadline before Gemini finishes. Keep the Streamlit page open while the app polls the background interaction.

## YouTube requirements
The video must be public. Private and unlisted YouTube videos are not supported by Gemini's direct YouTube input.
