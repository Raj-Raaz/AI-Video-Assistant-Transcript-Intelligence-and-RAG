# AI Video Assistant

## Run locally

Keep your existing API keys in `.env`. Start Qdrant and then the UI:

```powershell
docker compose up -d
python -m streamlit run app.py
```

Open the local URL shown by Streamlit. Add a YouTube link or upload a media file, select the spoken language, and click **Analyze video**. Results include the summary, notes, full transcript, and video chat.

Every analysis transcribes again. Chat and tab changes reuse the current session result; they do not run the pipeline again. Temporary media files are removed after analysis. No transcript folder is created. The download button lets the user explicitly save a transcript.

English and Hindi currently use Whisper's existing translation-to-English behavior. Hinglish uses Sarvam. Chat displays previous messages, but each question is retrieved independently, so ask self-contained questions.

## Configuration

- `GROQ_API_KEY`: required, primary LLM (`openai/gpt-oss-120b`).
- `MISTRAL_API_KEY`: optional fallback after Groq rate-limit retries.
- `COHERE_API_KEY`: required for embeddings.
- `SARVAM_API_KEY`: required for Hinglish.
- `QDRANT_URL`: defaults to `http://localhost:6333`.
- `QDRANT_API_KEY`: set when using an authenticated remote Qdrant instance.

## Hosting

Use `app.py` as the Streamlit entry point. Install `requirements.txt`, FFmpeg, and a current Node.js runtime. YouTube downloads need Node.js; media processing needs FFmpeg. Whisper also downloads its model on first use and needs enough memory to load it.

For Streamlit-hosted deployments, configure the keys above as top-level secrets in the hosting dashboard (TOML syntax). Use a reachable remote Qdrant service and set `QDRANT_URL` and `QDRANT_API_KEY`; localhost on a hosted app is not your laptop. Never commit `.env` or `.streamlit/secrets.toml`.

`packages.txt` lists Linux system packages for Streamlit Community Cloud. Verify the installed Node.js version meets the current yt-dlp runtime requirement before deploying. Hosting platforms may restrict YouTube downloads; file upload is an alternative.

This app has no login or per-user database separation. Keep deployment private for personal use; a public multi-user deployment needs authentication, storage isolation, and usage controls first. The current video filter keeps chat retrieval scoped to its transcript.

Deployment has not been performed or verified against a hosted Qdrant instance.
