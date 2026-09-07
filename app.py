import json
import os
import re
import tempfile
import time
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(page_title="Bishopton FC Video Analyst", page_icon="⚽", layout="wide")

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
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


def analyse_youtube(url: str, roster: str, notes: str, model_name: str):
    api_key = get_api_key()
    validate_api_key_format(api_key)

    # Gemini's current Interactions API accepts public YouTube URLs directly,
    # so the match does not need to be downloaded through Streamlit first.
    client = genai.Client(api_key=api_key)
    prompt = build_prompt(roster, notes)
    response = client.interactions.create(
        model=model_name,
        input=[
            {"type": "text", "text": prompt},
            {"type": "video", "uri": url.strip()},
        ],
    )
    output_text = getattr(response, "output_text", None)
    if not output_text:
        raise RuntimeError("Gemini returned no analysis text for the YouTube video.")
    return extract_json(output_text)


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
    progress.progress(35, text="Analysing match footage…")

    response = client.models.generate_content(
        model=model_name,
        contents=[uploaded, types.Part.from_text(text=prompt)],
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
        ),
    )
    progress.progress(100, text="Analysis complete")
    return extract_json(response.text)


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
    model_name = st.text_input("Gemini model", value=MODEL_NAME, help="Default is Gemini 3.7 Flash. Change only if you know the model is available to your API key.")
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
    st.caption("Gemini API keys created in AI Studio may begin with AQ. — that is supported. Keep the key in Streamlit Secrets and never commit it to GitHub.")

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
