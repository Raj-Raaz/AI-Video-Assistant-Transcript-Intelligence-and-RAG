import logging
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Cloud secrets use the same names as the local .env file.
for name in ("GROQ_API_KEY", "MISTRAL_API_KEY", "COHERE_API_KEY",
             "SARVAM_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"):
    try:
        if name in st.secrets:
            os.environ[name] = str(st.secrets[name])
    except FileNotFoundError:
        break

from main import run_pipeline
from core.rag_engine import ask_question
from core.llm_utils import RateLimitError

st.set_page_config(page_title="AI Video Assistant", page_icon="🎬", layout="wide")
st.title("AI Video Assistant")
st.caption("Turn a video into a summary, useful notes, and answers.")

with st.sidebar:
    st.header("Your video")
    source_type = st.radio("Source", ["YouTube link", "Upload a file"])
    language = st.selectbox("Spoken language", ["en", "hi", "hinglish"],
                            format_func=lambda value: {"en": "English", "hi": "Hindi", "hinglish": "Hinglish"}[value])
    if source_type == "YouTube link":
        url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
        upload = None
    else:
        url = ""
        upload = st.file_uploader("Audio or video", type=["mp4", "mov", "mkv", "webm", "mp3", "wav", "m4a", "ogg"])
    analyze = st.button("Analyze video", type="primary", use_container_width=True)
    st.caption("Each analysis transcribes the audio again. Results stay available during this session.")
    if st.button("Clear results", use_container_width=True):
        for key in ("result", "messages", "source_label"):
            st.session_state.pop(key, None)

if analyze:
    error = None
    if source_type == "YouTube link":
        parsed = urlparse(url.strip())
        host = (parsed.hostname or "").lower()
        if parsed.scheme not in ("http", "https") or not (
            host in ("youtu.be", "youtube.com") or host.endswith(".youtube.com")
        ):
            error = "Enter a YouTube video link, not a website or GitHub repository."
    elif upload is None:
        error = "Choose an audio or video file first."
    required = ["GROQ_API_KEY", "COHERE_API_KEY"]
    if language == "hinglish":
        required.append("SARVAM_API_KEY")
    missing = [name for name in required if not os.getenv(name)]
    if missing and not error:
        error = "Missing configuration: " + ", ".join(missing)

    if error:
        st.error(error)
    else:
        for key in ("result", "messages", "source_label"):
            st.session_state.pop(key, None)
        with st.status("Analyzing your video...", expanded=True) as status:
            try:
                # Separate temporary files for each analysis; no transcript files.
                work_root = Path(__file__).parent / "downloads"
                work_root.mkdir(exist_ok=True)
                with tempfile.TemporaryDirectory(dir=work_root) as folder:
                    if upload is not None:
                        source = str(Path(folder) / ("upload" + Path(upload.name).suffix.lower()))
                        Path(source).write_bytes(upload.getbuffer())
                        label = upload.name
                    else:
                        source = url.strip()
                        label = source
                    result = run_pipeline(source, language, output_dir=folder, on_progress=st.write)
                st.session_state.result = result
                st.session_state.source_label = label
                st.session_state.messages = []
                status.update(label="Your results are ready", state="complete", expanded=False)
            except RateLimitError as exc:
                status.update(label="API limit reached", state="error")
                st.error(str(exc))
            except Exception:
                logging.exception("Video analysis failed")
                status.update(label="Analysis could not finish", state="error")
                st.error("Check that the video is accessible, Qdrant is running, and your API keys are valid. Details are in the server terminal.")

result = st.session_state.get("result")
if result is None:
    st.info("Add a YouTube link or upload a file, then select Analyze video.")
    st.stop()

st.subheader(result["title"])
st.caption("Source: " + st.session_state.source_label)
summary_tab, notes_tab, transcript_tab = st.tabs(["Summary", "Notes", "Full transcript"])
with summary_tab:
    st.markdown(result["summary"])
with notes_tab:
    for heading, field in [("Action items", "action_items"), ("Key decisions", "key_decisions"), ("Open questions", "questions")]:
        st.subheader(heading)
        st.markdown(result[field])
with transcript_tab:
    st.text_area("Transcript", result["transcript"], height=350, disabled=True)
    st.download_button("Download transcript", result["transcript"], file_name="transcript.txt", mime="text/plain")

st.divider()
st.subheader("Chat with your video")
st.caption("Ask a specific question about this video. Each question is answered using its transcript.")
for message in st.session_state.get("messages", []):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Finding an answer..."):
                answer = ask_question(result["rag_chain"], question)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except RateLimitError as exc:
            st.error(str(exc))
        except Exception:
            logging.exception("Video chat failed")
            st.error("Could not answer this question. Check your connection and try again.")
