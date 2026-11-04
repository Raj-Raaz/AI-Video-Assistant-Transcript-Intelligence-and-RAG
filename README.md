# AI Video Assistant

A Streamlit application that transcribes YouTube videos and uploaded media, generates summaries and structured notes, and answers questions using the video's transcript.

The project uses Whisper and Sarvam for speech processing, Groq with a Mistral fallback for text generation, and Cohere embeddings with Qdrant for retrieval.

## Features

- YouTube links and audio/video uploads.
- Generated titles, summaries, action items, decisions, and unresolved questions.
- Full transcript display and text download.
- Transcript-specific retrieval for video chat.
- Timestamp ranges for retrieved sources, with clickable YouTube links.
- Rate-limit retries and optional Mistral fallback.

## Run locally

Install Python 3.11, FFmpeg, and Node.js. For local Qdrant, install and start Docker Desktop as well.

From the project folder, create an environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Keep your API keys and database configuration in a local `.env` file:

```dotenv
GROQ_API_KEY=your_groq_api_key
COHERE_API_KEY=your_cohere_api_key
MISTRAL_API_KEY=your_mistral_api_key
SARVAM_API_KEY=your_sarvam_api_key
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
```

Mistral is optional for fallback. Sarvam is required only for Hinglish. Keep real keys out of GitHub.

Start Qdrant:

```powershell
docker compose up -d
```

The included Compose file exposes Qdrant on port `6333` and stores its data in a Docker volume.

**Keep the Docker engine running while using local Qdrant.** Stopping Docker Desktop's engine stops the database, and analysis or chat may fail with a connection-refused error. The stored data remains in the Docker volume unless that volume is deleted.

Start the Streamlit app:

```powershell
python -m streamlit run app.py
```

Open the local URL printed in your terminal, usually [http://localhost:8501](http://localhost:8501).

Select **YouTube link** or **Upload a file**, choose the language, and click **Analyze video**. Results appear under **Summary**, **Notes**, and **Full transcript**. The chat section appears below them.

Supported uploads: MP4, MOV, MKV, WebM, MP3, WAV, M4A, and OGG. Temporary media files created by the Streamlit analysis are removed when processing finishes or fails.

The command-line version is also available:

```powershell
python main.py
```

Enter a YouTube video URL or a local media-file path. GitHub repository links are not valid video inputs.

## Timestamp sources and processing

Whisper segment timestamps are adjusted for each audio chunk's offset in the full recording. Retrieval chunks store the start and end times of the speech segments they overlap. Hinglish uses approximate Sarvam processing windows of up to 25 seconds per segment; a retrieved chunk can span several windows.

Chat answers show **Retrieved sources** generated from stored metadata. YouTube links open at the beginning of the indicated range. Uploaded files show times without playback links. These sources are the passages supplied to the model, not independently verified citations for every sentence in the answer.

Each new analysis transcribes the audio and generates outputs and embeddings again. There is no transcript-file cache. Streamlit session state keeps completed results available for chat without repeating analysis. Run a new analysis to obtain timestamps for a video processed before this feature was added.

## Language behavior

| Selection | Processing |
| --- | --- |
| English (`en`) | Whisper with `task="transcribe"` |
| Hindi (`hi`) | Whisper with `task="transcribe"` |
| Hinglish (`hinglish`) | Sarvam's speech-to-text translation endpoint |

Whisper currently detects the spoken language automatically; the English/Hindi selection does not force a language or translate the transcript into English. The current Cohere embedding model is English-specific, so retrieval quality can be weaker for Hindi transcripts. Output language is not explicitly enforced by the LLM prompts.

## Rate limits and fallback

Groq is tried first for each request. On HTTP `429`, the app waits using the provider's `Retry-After` header, or uses increasing delays when the header is missing. It makes up to four attempts. A server-requested wait longer than 60 seconds stops that retry cycle early.

If Groq remains rate limited and `MISTRAL_API_KEY` is configured, the same request is sent to Mistral with its own bounded retries. Other errors, such as invalid credentials, do not trigger fallback. Both providers still have their own usage limits.

`core/llm_utils.py` keeps this shared configuration and retry logic in one place for summarization, extraction, and chat.

## Project files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit inputs, results, session state, and chat UI |
| `main.py` | Processing pipeline and command-line entry point |
| `core/transcriber.py` | Whisper and Sarvam transcription |
| `core/summarize.py` | Title generation and chunk-based summarization |
| `core/extractor.py` | Combined extraction of tasks, decisions, and questions |
| `core/llm_utils.py` | Groq requests, retries, and Mistral fallback |
| `core/vector_store.py` | Cohere embeddings and Qdrant storage/search |
| `core/rag_engine.py` | Transcript retrieval and question answering |
| `utils/audio_processor.py` | Audio download, conversion, and splitting |
| `tests/` | Backend and Streamlit tests |
| `docker-compose.yml` | Local Qdrant service |
| `requirements.txt` | Python dependencies |
| `packages.txt` | Linux system packages for hosting |

## Tests

```powershell
python -m unittest discover -s tests -v
```

Tests cover provider retries and fallback, combined extraction, Qdrant creation and video filtering, and Streamlit interactions. API calls are mocked and Qdrant tests use an in-memory database; passing tests does not verify live API quotas or hosted deployment.

## Deployment

For an online demo that works while your laptop is off, use **Streamlit Community Cloud + Qdrant Cloud**.

| Setup | Laptop and Docker requirement |
| --- | --- |
| Local Streamlit + local Qdrant | Keep the laptop, Streamlit process, and Docker engine running. |
| Local Streamlit + Qdrant Cloud | Keep the laptop and Streamlit process running; local Docker is unnecessary. |
| Streamlit Community Cloud + Qdrant Cloud | No dependency on your laptop or local Docker; both hosted services must be available. |

1. Create a Qdrant Cloud cluster and obtain its cluster endpoint and API key.
2. Push the application code, `requirements.txt`, and `packages.txt` to GitHub.
3. Create a Streamlit Community Cloud app from the repository, using `app.py` as its entry point.
4. Select Python **3.11** in Advanced settings. The current pydub setup requires `audioop`, which is absent from Python 3.13 and newer unless separately supplied.
5. Add the secrets below, save, and deploy.

Provide FFmpeg and a Node.js version supported by yt-dlp. `packages.txt` lists the Linux packages, but verify the available Node.js version on your host.

For Streamlit Community Cloud, set the API keys and Qdrant connection values as top-level secrets in the deployment settings:

```toml
GROQ_API_KEY = "your_groq_api_key"
COHERE_API_KEY = "your_cohere_api_key"
MISTRAL_API_KEY = "your_mistral_api_key"
SARVAM_API_KEY = "your_sarvam_api_key"
QDRANT_URL = "https://your-qdrant-endpoint"
QDRANT_API_KEY = "your_qdrant_api_key"
```

Replace `QDRANT_URL` with the actual Qdrant Cloud cluster endpoint, not a dashboard URL. Omit optional keys if you do not use those features. A hosted app needs a reachable Qdrant service; `localhost:6333` on the hosting server does not connect to Qdrant on your laptop. Allow enough memory and storage for Whisper's model and temporary media.

Streamlit Cloud secrets and your laptop's `.env` are separate configurations. To use Qdrant Cloud locally too, set its endpoint and key in `.env`, then restart your local Streamlit process.

Before sharing the demo, test a short video and a chat question on the deployed URL with local Docker stopped. This checks that the demo uses the hosted database. The latest timestamp changes also need to be pushed to the branch used by Streamlit before they appear online.

## Troubleshooting Qdrant connections

`[WinError 10061]` means the configured endpoint refused the connection. For local Qdrant, open Docker Desktop, wait for the engine to start, then run:

```powershell
docker compose up -d
docker compose ps
```

Check the [local Qdrant dashboard](http://localhost:6333/dashboard), then retry analysis. For Qdrant Cloud, check that the cluster is running and the endpoint and key are configured in the environment where the app runs.

The warning about failing to obtain the Qdrant server version can accompany a connection failure. Disabling the compatibility check does not repair the connection.

## Current limitations

- Transcription and LLM outputs can contain mistakes. Check important names, dates, and figures against the video.
- Chat displays previous messages, but retrieval answers each question independently. Ask self-contained questions.
- Long videos take longer and can hit API rate or token limits.
- YouTube access depends on the video and hosting network; file upload provides another input option.
- Qdrant retains indexed transcripts. **Clear results** clears the browser session, not the database.
- The app has no authentication or per-user database isolation; it is currently intended for personal use.
