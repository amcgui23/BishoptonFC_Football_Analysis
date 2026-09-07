import json
import os
import re
import tempfile
import time
import requests
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(page_title="Bishopton FC Video Analyst", page_icon="⚽", layout="wide")

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
FALLBACK_MODELS = ["gemini-3.8-flash", "gemini-3.7-flash"]
MAX_CREATE_RETRIES = 5
MAX_POLL_RETRIES = 8
MAX_UPLOAD_MB = 3072
MAX_GEMINI_MB = 2048

PROMPT = r'''
You are a football analyst reviewing a youth football match video for a grassroots coaching team.
This is an U12/U13-style match, so be constructive, developmental and evidence-based. Do not invent
player identities, actions, statistics, scores or timestamps. If something cannot be reliably observed,
say "not reliably observable" rather than guessing.

IMPORTANT PLAYER IDENTIFICATION RULE:
The coach may provide a roster below. Only assign an observed player to a roster name when the shirt
number, appearance, position, or repeated visual evidence makes the identification reasonably reliable.
If not reliable, use a neutral label such as "Player #7" or "Blue #4" and mark confidence low.

Analyse the supplied video and return ONLY valid JSON matching this schema:
{
  "match": {
    "team_us": "",
    "team_opponent": "",
    "score_us": null,
    "score_opponent": null,
    "competition": "",
    "date": "",
    "summary": "",
    "tactical_observations": [],
    "key_moments": [{"time":"MM:SS","type":"goal|chance|save|defensive|turnover|build-up|other","description":""}],
    "confidence_notes": []
  },
  "team_performance": {
    "strengths": [],
    "development_points": [],
    "attacking": "",
    "defending": "",
    "transition": "",
    "set_pieces": "",
    "shape_and_spacing": ""
  },
  "players": [
    {
      "name": "",
      "shirt_number": "",
      "team": "us|opponent|unknown",
      "position": "",
      "rating": null,
      "confidence": "high|medium|low",
      "summary": "",
      "strengths": [],
      "development_points": [],
      "key_moments": [{"time":"MM:SS","description":""}]
    }
  ],
  "statistics": {
    "goals_us": null,
    "goals_opponent": null,
    "shots_us": null,
    "shots_opponent": null,
    "clear_chances_us": null,
    "clear_chances_opponent": null,
    "corners_us": null,
    "corners_opponent": null,
    "note": ""
  },
  "coach_report": ""
}

STATISTICS RULE: Only provide numerical statistics where the video gives sufficient evidence to count
reliably. Otherwise use null and explain why in statistics.note.

PLAYER RATING RULE: Rate observable performance only. Use 1-10, with most ratings around 5-7. Avoid
punitive language. For youth players, emphasise learning and repeatable behaviours over outcome.

TIMESTAMP RULE: Use timestamps from the video. If exact timing is uncertain, give the nearest reliable
timestamp and say so in confidence_notes.

COACH ROSTER:
{{ROSTER}}

COACH NOTES:
{{NOTES}}
'''


def get_api_key():
    """Return the Gemini key exactly as configured, with harmless copy/paste cleanup.

    Google is transitioning AI Studio keys from Standard keys to Authorization (AQ.)
    keys. The current google-genai SDK accepts an explicit API key, so we pass the
    cleaned value directly to genai.Client rather than relying on environment-variable
    precedence. This also avoids accidentally selecting an old GOOGLE_API_KEY.
    """
    value = None
    try:
        value = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass
    if not value:
        value = os.getenv("GEMINI_API_KEY")
    if not value:
        value = os.getenv("GOOGLE_API_KEY")
    if not value:
        return None

    # Make the app tolerant of a key copied with whitespace or accidental wrapping quotes.
    value = str(value).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'\"', "'"}:
        value = value[1:-1].strip()
    return value


def validate_api_key_format(api_key: str):
    """Give a useful message without exposing the secret."""
    if not api_key:
        raise RuntimeError(
            "No Gemini API key configured. In Streamlit Secrets, add GEMINI_API_KEY = \"AQ....\"."
        )

    # AI Studio now creates Authorization keys beginning with AQ. Standard keys may
    # still exist if explicitly restricted. Do not reject either format locally; let
    # Google's API determine whether the key is active/linked.
    if api_key.startswith("AQ."):
        return "auth"
    if api_key.startswith("AIza"):
        return "standard"
    # Keep this permissive because Google can change key prefixes without notice.
    return "other"


def extract_json(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def build_prompt(roster: str, notes: str):
    return PROMPT.replace("{{ROSTER}}", roster or "No roster supplied.").replace(
        "{{NOTES}}", notes or "No additional notes supplied."
    )


def _interaction_headers(api_key: str):
    return {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
        "Api-Revision": "2026-05-20",
    }


def _extract_interaction_output(data):
    steps = data.get("steps") or []
    texts = []
    for step in steps:
        if step.get("type") != "model_output":
            continue
        for content in step.get("content") or []:
            if content.get("type") == "text" and content.get("text"):
                texts.append(content["text"])
    if texts:
        return "\n".join(texts)
    raise RuntimeError("Gemini completed the interaction but returned no model text.")


def _raise_google_error(response, action):
    try:
        payload = response.json()
    except Exception:
        payload = {"raw": response.text[:2000]}
    err = payload.get("error", {}) if isinstance(payload, dict) else {}
    code = err.get("code") or response.status_code
    status = err.get("status", "")
    message = err.get("message") or response.text[:2000]
    raise RuntimeError(f"Gemini {action} failed: HTTP {code} {status} — {message}")


def _is_retryable_http(status_code: int, message: str = ""):
    msg = (message or "").lower()
    return status_code in {429, 500, 502, 503, 504} or "high demand" in msg or "spikes in demand" in msg


def _backoff(attempt: int):
    # 2, 4, 8, 16, 30 seconds, capped to keep retries practical on Streamlit.
    return min(30, 2 ** attempt)


def _create_background_interaction(api_key, model_name, inputs):
    payload = {
        "model": model_name,
        "input": inputs,
        "background": True,
    }
    last_error = None
    for attempt in range(MAX_CREATE_RETRIES):
        try:
            response = requests.post(
                "https://generativelanguage.googleapis.com/v1beta/interactions",
                headers=_interaction_headers(api_key),
                json=payload,
                timeout=60,
            )
            if response.ok:
                data = response.json()
                if not data.get("id"):
                    raise RuntimeError(f"Gemini returned an unexpected interaction response: {data}")
                return data
            try:
                err_payload = response.json()
            except Exception:
                err_payload = {"raw": response.text[:2000]}
            err = err_payload.get("error", {}) if isinstance(err_payload, dict) else {}
            message = err.get("message") or response.text[:2000]
            last_error = RuntimeError(
                f"Gemini interaction creation failed: HTTP {err.get('code') or response.status_code} "
                f"{err.get('status','')} — {message}"
            )
            if not _is_retryable_http(response.status_code, message) or attempt == MAX_CREATE_RETRIES - 1:
                raise last_error
            wait = _backoff(attempt)
            st.info(f"Gemini is busy ({model_name}). Retrying automatically in {wait}s…")
            time.sleep(wait)
        except requests.RequestException as exc:
            last_error = exc
            if attempt == MAX_CREATE_RETRIES - 1:
                raise RuntimeError(f"Could not reach Gemini after {MAX_CREATE_RETRIES} attempts: {exc}") from exc
            wait = _backoff(attempt)
            st.info(f"Temporary connection problem. Retrying in {wait}s…")
            time.sleep(wait)
    raise last_error or RuntimeError("Gemini interaction creation failed.")


class GeminiHighDemandError(RuntimeError):
    pass


def _wait_for_background_interaction(api_key, interaction, label="Gemini analysis", max_wait_seconds=3600):
    """Poll a Gemini background interaction, retrying temporary/high-demand errors."""
    started = time.time()
    progress = st.progress(0, text=f"{label} started in Gemini…")
    interaction_id = interaction.get("id")
    if not interaction_id:
        raise RuntimeError("Gemini did not return a background interaction ID.")

    last_status = None
    poll_failures = 0
    while True:
        elapsed = int(time.time() - started)
        if elapsed > max_wait_seconds:
            raise TimeoutError(
                f"Gemini analysis is still running after {max_wait_seconds // 60} minutes. "
                f"The background job may continue on Google's servers (interaction {interaction_id})."
            )

        try:
            response = requests.get(
                f"https://generativelanguage.googleapis.com/v1beta/interactions/{interaction_id}",
                headers=_interaction_headers(api_key),
                timeout=60,
            )
            if not response.ok:
                try:
                    payload = response.json()
                except Exception:
                    payload = {}
                err = payload.get("error", {}) if isinstance(payload, dict) else {}
                message = err.get("message") or response.text[:2000]
                if _is_retryable_http(response.status_code, message) and poll_failures < MAX_POLL_RETRIES:
                    wait = _backoff(min(poll_failures, 4))
                    poll_failures += 1
                    progress.progress(
                        min(95, max(5, int((elapsed / max_wait_seconds) * 90))),
                        text=f"Gemini is busy while checking the analysis. Retrying in {wait}s…",
                    )
                    time.sleep(wait)
                    continue
                raise RuntimeError(
                    f"Gemini interaction polling failed: HTTP {err.get('code') or response.status_code} "
                    f"{err.get('status','')} — {message}"
                )
            poll_failures = 0
            current = response.json()
            status = current.get("status")
            if status != last_status:
                progress.progress(
                    min(95, max(5, int((elapsed / max_wait_seconds) * 90))),
                    text=f"{label} — Gemini status: {status or 'in progress'}…",
                )
                last_status = status

            if status == "completed":
                output_text = _extract_interaction_output(current)
                progress.progress(100, text="Analysis complete")
                return extract_json(output_text)

            if status in {"failed", "cancelled"}:
                error = current.get("error") or "no error details returned"
                error_text = json.dumps(error) if isinstance(error, dict) else str(error)
                if "high demand" in error_text.lower() or "spikes in demand" in error_text.lower():
                    raise GeminiHighDemandError(f"Gemini reported high demand: {error_text}")
                raise RuntimeError(f"Gemini background analysis {status}: {error_text}")

            time.sleep(5)
        except requests.RequestException as exc:
            if poll_failures >= MAX_POLL_RETRIES:
                raise RuntimeError(f"Gemini polling connection failed after retries: {exc}") from exc
            wait = _backoff(min(poll_failures, 4))
            poll_failures += 1
            progress.progress(
                min(95, max(5, int((elapsed / max_wait_seconds) * 90))),
                text=f"Temporary polling connection problem. Retrying in {wait}s…",
            )
            time.sleep(wait)


def _analyse_with_model_fallback(api_key, inputs_factory, preferred_model):
    models = []
    for model in [preferred_model, *FALLBACK_MODELS]:
        if model and model not in models:
            models.append(model)

    last_error = None
    for index, model in enumerate(models):
        try:
            st.caption(f"Gemini model: {model}")
            interaction = _create_background_interaction(api_key, model, inputs_factory())
            return _wait_for_background_interaction(api_key, interaction, label=f"{model} match analysis")
        except GeminiHighDemandError as exc:
            last_error = exc
            if index < len(models) - 1:
                st.warning(f"{model} is currently under heavy demand. Trying {models[index + 1]} automatically…")
                continue
            raise
    raise last_error or RuntimeError("No Gemini model was available for the analysis.")

def analyse_youtube(url: str, roster: str, notes: str, model_name: str):
    api_key = get_api_key()
    validate_api_key_format(api_key)
    prompt = build_prompt(roster, notes)
    try:
        return _analyse_with_model_fallback(
            api_key,
            lambda: [
                {
                    "type": "video",
                    "uri": url.strip(),
                    "processing": "agentic",
                },
                {"type": "text", "text": prompt},
            ],
            model_name,
        )
    except Exception as exc:
        raise RuntimeError(f"YouTube analysis failed: {exc}") from exc


def analyse_video(path: str, roster: str, notes: str, model_name: str):
    api_key = get_api_key()
    validate_api_key_format(api_key)

    client = genai.Client(api_key=api_key)
    uploaded = client.files.upload(file=path)

    progress = st.progress(0, text="Uploading video to Gemini…")
    while not uploaded.state or uploaded.state.name == "PROCESSING":
        time.sleep(3)
        uploaded = client.files.get(name=uploaded.name)
        progress.progress(15, text=f"Gemini is processing the video… ({uploaded.state.name})")

    if uploaded.state.name == "FAILED":
        raise RuntimeError("Gemini could not process the uploaded video.")

    prompt = build_prompt(roster, notes)
    progress.progress(35, text="Starting long-video analysis…")
    try:
        return _analyse_with_model_fallback(
            api_key,
            lambda: [
                {
                    "type": "video",
                    "uri": uploaded.uri,
                    "mime_type": uploaded.mime_type,
                    "processing": "agentic",
                },
                {"type": "text", "text": prompt},
            ],
            model_name,
        )
    except Exception as exc:
        raise RuntimeError(f"Uploaded video analysis failed: {exc}") from exc

def rating_label(r):
    if r is None:
        return "—"
    try:
        x = float(r)
        if x >= 8:
            return f"{x:.1f} ⭐"
        if x >= 7:
            return f"{x:.1f} 👍"
        if x >= 5:
            return f"{x:.1f}"
        return f"{x:.1f}"
    except Exception:
        return str(r)


def show_player(p):
    name = p.get("name", "Unknown player")
    team = p.get("team", "unknown")
    number = p.get("shirt_number", "")
    rating = rating_label(p.get("rating"))
    confidence = p.get("confidence", "low")
    st.markdown(f"### {name} {f'#{number}' if number else ''} — {rating}")
    st.caption(f"{team.title()} · {p.get('position') or 'Position not established'} · ID confidence: {confidence}")
    if p.get("summary"):
        st.write(p["summary"])
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Strengths**")
        for x in p.get("strengths", []):
            st.write(f"• {x}")
    with c2:
        st.markdown("**Development points**")
        for x in p.get("development_points", []):
            st.write(f"• {x}")
    moments = p.get("key_moments", [])
    if moments:
        with st.expander("Key moments"):
            for m in moments:
                st.write(f"**{m.get('time','?')}** — {m.get('description','')}")


st.title("⚽ Bishopton FC Video Analyst")
st.caption("AI-assisted match review for grassroots coaching — designed to work from an iPad browser.")

with st.sidebar:
    st.header("Match setup")
    model_name = st.text_input("Gemini model", value=MODEL_NAME, help="Default is Gemini 3.8 Flash. If it is temporarily busy, the app automatically retries and can fall back to Gemini 3.7 Flash.")
    roster = st.text_area(
        "Optional squad / roster",
        height=180,
        placeholder="Example:\n#1 Jack — GK\n#4 Lewis — CB\n#7 Charlie — CM\n#9 Sam — ST\n…",
        help="Adding shirt numbers makes player identification more reliable.",
    )
    notes = st.text_area(
        "Coach focus",
        height=120,
        placeholder="Example: assess build-up from the goalkeeper, pressing after losing the ball, and midfield spacing.",
    )
    st.info("For best results, provide shirt numbers and upload the clearest match footage available.")
    st.caption("Gemini API keys created in AI Studio may begin with AQ. — that is supported. Long-video analysis uses agentic processing with automatic retry/backoff and a 3.7 Flash fallback.")

st.subheader("Choose your video source")
source = st.radio(
    "Video source",
    ["YouTube link", "Upload video"],
    horizontal=True,
    label_visibility="collapsed",
)

if source == "YouTube link":
    youtube_url = st.text_input(
        "YouTube match link",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Use a public YouTube video. Private and unlisted videos cannot be analysed by Gemini's YouTube input.",
    )
    st.info("💡 For a large full-match recording, YouTube is often the easiest option because the video does not need to be uploaded through this app.")

    if youtube_url:
        is_youtube = bool(re.match(r"^https?://(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[^\s&]+", youtube_url.strip(), re.IGNORECASE))
        if not is_youtube:
            st.warning("That doesn't look like a standard YouTube video link. Paste a link such as https://www.youtube.com/watch?v=... .")

        if st.button("🔎 Analyse YouTube Match", type="primary", use_container_width=True, disabled=not is_youtube):
            try:
                with st.spinner("Sending the YouTube match to Gemini for analysis…"):
                    result = analyse_youtube(youtube_url, roster, notes, model_name)
                st.session_state["analysis"] = result
                st.session_state["analysis_source"] = youtube_url.strip()
                st.rerun()
            except Exception as exc:
                st.error(f"YouTube analysis failed: {exc}")
                st.caption("Check that the video is public, the URL is correct, and your Gemini API key/model has access to the current YouTube video feature.")
    else:
        st.markdown("**YouTube requirements:** public video, valid YouTube URL, and a Gemini API key in Streamlit Secrets.")

else:
    uploaded_file = st.file_uploader(
        "Upload your match video",
        type=["mp4", "mov", "m4v", "avi", "webm"],
        max_upload_size=MAX_UPLOAD_MB,
        help="Uploads up to 3 GB are accepted by the app. Gemini File API currently supports up to 2 GB per file, so files above 2 GB will be rejected with a clear message.",
    )

    if uploaded_file:
        st.video(uploaded_file)
        size_mb = uploaded_file.size / (1024 * 1024)
        st.caption(f"{uploaded_file.name} · {size_mb:.1f} MB")
        if size_mb > MAX_GEMINI_MB:
            st.warning(
                f"This video is {size_mb/1024:.2f} GB. The app accepts files up to 3 GB, "
                "but Gemini's File API currently limits an individual file to 2 GB. "
                "Please export/compress the video to 2 GB or less before analysing it."
            )

        if st.button("🔎 Analyse Match", type="primary", use_container_width=True, disabled=size_mb > MAX_GEMINI_MB):
            suffix = Path(uploaded_file.name).suffix.lower() or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                temp_path = tmp.name
            try:
                with st.spinner("Preparing the match analysis…"):
                    result = analyse_video(temp_path, roster, notes, model_name)
                st.session_state["analysis"] = result
                st.session_state["analysis_source"] = uploaded_file.name
                st.rerun()
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
            finally:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

result = st.session_state.get("analysis")
if result:
    st.divider()
    match = result.get("match", {})
    source_label = st.session_state.get("analysis_source")
    if source_label:
        st.caption(f"Video source: {source_label}")
    tabs = st.tabs(["📋 Match report", "👤 Players", "📊 Statistics", "🎯 Key moments", "🧠 Coach report", "🧾 Raw JSON"])

    with tabs[0]:
        us = match.get("team_us") or "Our team"
        opp = match.get("team_opponent") or "Opponent"
        s1 = match.get("score_us")
        s2 = match.get("score_opponent")
        score = f"{s1} – {s2}" if s1 is not None and s2 is not None else "Score not reliably established"
        st.header(f"{us}  {score}  {opp}")
        if match.get("competition") or match.get("date"):
            st.caption(" · ".join(x for x in [match.get("competition"), match.get("date")] if x))
        st.subheader("Match summary")
        st.write(match.get("summary") or "No summary returned.")
        tp = result.get("team_performance", {})
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("What went well")
            for x in tp.get("strengths", []): st.write(f"• {x}")
        with c2:
            st.subheader("Development priorities")
            for x in tp.get("development_points", []): st.write(f"• {x}")
        st.subheader("Tactical observations")
        for x in match.get("tactical_observations", []): st.write(f"• {x}")
        if match.get("confidence_notes"):
            st.warning("Analysis confidence notes: " + " ".join(match["confidence_notes"]))

    with tabs[1]:
        players = result.get("players", [])
        ours = [p for p in players if p.get("team") == "us"]
        opps = [p for p in players if p.get("team") == "opponent"]
        if ours:
            st.subheader("Our players")
            for p in ours: show_player(p)
        if opps:
            st.subheader("Opposition players")
            for p in opps: show_player(p)
        if not players: st.info("No individual players could be analysed reliably.")

    with tabs[2]:
        stats = result.get("statistics", {})
        rows = {k.replace("_", " ").title(): v for k, v in stats.items() if k != "note"}
        st.table(rows)
        if stats.get("note"): st.info(stats["note"])
        tp = result.get("team_performance", {})
        for label, key in [("Attacking", "attacking"), ("Defending", "defending"), ("Transitions", "transition"), ("Set pieces", "set_pieces"), ("Shape & spacing", "shape_and_spacing")]:
            st.subheader(label)
            st.write(tp.get(key) or "Not reliably observable")

    with tabs[3]:
        moments = match.get("key_moments", [])
        if moments:
            for m in moments:
                st.markdown(f"**{m.get('time','?')} — {m.get('type','moment').title()}**")
                st.write(m.get("description", ""))
        else:
            st.info("No key moments were returned.")

    with tabs[4]:
        st.write(result.get("coach_report") or "No coach report returned.")

    with tabs[5]:
        st.json(result)

    st.download_button(
        "⬇️ Download analysis as JSON",
        data=json.dumps(result, indent=2, ensure_ascii=False),
        file_name="bishopton_match_analysis.json",
        mime="application/json",
        use_container_width=True,
    )
else:
    st.markdown("### How to use it")
    st.write("1. Add your squad with shirt numbers in the sidebar.  2. Choose **YouTube link** or **Upload video**.  3. Add any coaching focus.  4. Tap the analysis button.")
    st.info("For large full-match recordings, uploading the match to YouTube and using the YouTube link can be easier than uploading the MP4 directly.")
    st.info("This is an AI-assisted analysis tool. It should support coaching decisions, not replace a coach's own review of the footage.")
