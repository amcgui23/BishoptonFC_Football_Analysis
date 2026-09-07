# Bishopton FC Football Video Analyst

An iPad-friendly Streamlit application for AI-assisted football match analysis using Google's Gemini video understanding.

## What it does

- Analyse a public YouTube match link directly from the app.
- Upload MP4/MOV/M4V/AVI/WebM match footage from an iPad or computer.
- Optionally provide a squad roster with shirt numbers.
- Ask the AI to focus on specific coaching themes.
- Produce a match report, tactical observations, player reviews, key moments and cautiously estimated statistics.
- Download the structured analysis as JSON.

## Deploy with Streamlit Community Cloud

1. Create a GitHub repository and upload this folder.
2. Go to Streamlit Community Cloud and connect GitHub.
3. Create an app using `app.py` as the entrypoint.
4. In the app's Advanced settings / Secrets, add:

```toml
GEMINI_API_KEY = "your-key-here"
```

5. Deploy.

The API key is deliberately not stored in the repository.

## Local use

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then set `GEMINI_API_KEY` in the environment or `.streamlit/secrets.toml`.

## Important limitations

AI video analysis can miss players, actions or exact counts, especially when footage is distant, obstructed, low resolution or filmed from one angle. The app therefore tells Gemini not to invent statistics or player identities.

Large match files can be expensive and slow to process. For a large full-match recording, the YouTube option is often the easiest route because Gemini can accept a public YouTube URL directly without the app downloading the video first.

The default model is `gemini-3.7-flash`; it can be changed in the sidebar or with `GEMINI_MODEL`.


## Large video uploads

This version accepts uploads up to **3 GB** in Streamlit, providing some headroom for large match recordings. However, the Gemini File API currently supports a maximum of **2 GB per individual file**. Videos larger than 2 GB are therefore accepted by the uploader but blocked from analysis with an explanatory message. For a 2 GB-or-less recording, the app uploads it to Gemini using the File API, which Google recommends for large/long videos.

## YouTube analysis

The app includes a **YouTube link** option. Paste a public YouTube video URL and Gemini analyses it directly. Google currently documents YouTube URL input for public videos; private and unlisted videos are not supported by this input method. The YouTube feature is currently in preview, and Google notes that rate limits/pricing may change.

The YouTube option uses the same football-analysis prompt and report format as uploaded videos, including player ratings, tactical observations, key moments, statistics where reliably observable, and a coach report.
