# Bishopton FC Video Analyst

AI-assisted football match review for Bishopton FC, designed to be used from an iPad browser through Streamlit Community Cloud.

## Features

- Analyse a public YouTube match link directly with Gemini.
- Or upload an MP4/MOV/M4V/AVI/WebM match video.
- Up to 3 GB upload interface; Gemini File API direct-file analysis remains limited to 2 GB per file.
- Optional squad roster with shirt numbers.
- Optional coach focus.
- Player ratings, strengths and development points.
- Team tactical observations and coach report.
- Key moments with timestamps where reliably observable.
- Conservative statistics: the model uses `null` where a number cannot be counted reliably.
- Downloadable raw JSON report.

## Gemini API key

Google AI Studio is transitioning from Standard API keys to Authorization (auth) keys. New AI Studio keys may begin with `AQ.`. The app accepts the current auth-key format and passes the configured key explicitly to the Google Gen AI SDK.

In Streamlit Community Cloud, add this secret:

```toml
GEMINI_API_KEY = "AQ.your_key_here"
```

Do not commit the key to GitHub.

## Deploy

1. Push these files to GitHub.
2. In Streamlit Community Cloud, open the app's Settings/Secrets.
3. Add `GEMINI_API_KEY` with the value from Google AI Studio.
4. Save/redeploy the app.

## Notes on video sources

YouTube analysis requires a public YouTube video URL supported by Gemini. Private/unlisted videos are not suitable for the direct YouTube input method.

For uploaded files, the app accepts up to 3 GB at the Streamlit layer, but it intentionally prevents direct Gemini File API analysis above 2 GB because Google's per-file limit is 2 GB.
