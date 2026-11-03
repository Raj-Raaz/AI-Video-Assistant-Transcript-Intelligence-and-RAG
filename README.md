# AI Video Assistant

A Streamlit app that turns YouTube videos and uploaded audio or video files into summaries, structured notes, and a searchable transcript. Ask questions about a video and get answers based on its transcript.

## Features

- Analyze a YouTube video or upload an audio/video file.
- Generate a title and summary.
- Extract action items, key decisions, and open questions in one LLM request.
- Read the full transcript and download it as a text file.
- Chat with the video using transcript retrieval.
- Retry temporary API rate limits and fall back from Groq to Mistral when configured.
- View progress and errors in the Streamlit interface.

## Tech stack

| Component | Technology |
| --- | --- |
| User interface | Streamlit |
| Audio download and processing | yt-dlp, FFmpeg, pydub |
| English / Hindi transcription | Whisper (`base` by default) |
| Hinglish transcription | Sarvam AI |
| Primary LLM | Groq — `openai/gpt-oss-120b` |
| Fallback LLM | Mistral — `mistral-small-latest` |
| Text embeddings | Cohere — `embed-english-v3.0` |
| Vector database | Qdrant |
| Prompt chains and retrieval | LangChain |

## How it works

1. Download the audio from YouTube or read the uploaded file.
2. Convert it to mono, 16 kHz WAV and split it into audio chunks.
3. Transcribe the audio with Whisper or Sarvam.
4. Generate a title, summary, and structured notes from the transcript.
5. Split the transcript into smaller text chunks, embed them with Cohere, and store them in Qdrant.
6. Retrieve relevant chunks for each question and pass them to the LLM to generate an answer.

Audio chunks are used for transcription; text chunks are used for retrieval. The app creates the Qdrant collection when missing and filters retrieval to the current transcript. Repeated indexing of the same transcript uses stable chunk IDs to avoid duplicates.

Every click on **Analyze video** starts a new analysis, including transcription. There is no transcript-file cache. Streamlit keeps the completed result in session state so changing tabs, downloading the transcript, and asking questions do not repeat analysis. Refreshing the browser may reset that session.

## Local setup

### 1. Prerequisites

Install Python, Docker with Docker Compose, FFmpeg, and a current Node.js runtime. Make sure `ffmpeg` and `node` are available on your PATH.

Whisper downloads its model the first time it runs. YouTube downloads require Node.js in this project.

### 2. Install the project

```powershell
git clone https://github.com/Raj-Raaz/AI-Video-Agent.git
cd AI-Video-Agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If you already have the project and an active environment, run only the dependency installation command.

### 3. Add API keys

Create a `.env` file in the project root:

```dotenv
GROQ_API_KEY=your_groq_api_key
COHERE_API_KEY=your_cohere_api_key

# Optional fallback when Groq remains rate limited
MISTRAL_API_KEY=your_mistral_api_key

# Required only for Hinglish
SARVAM_API_KEY=your_sarvam_api_key

# Local Qdrant; change these for a remote instance
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=

# Optional; defaults to base
WHISPER_MODEL=base
```

Groq and Cohere keys are required for analysis. Mistral is optional. Keep real keys out of Git; `.env` and `.streamlit/secrets.toml` are ignored.

### 4. Start Qdrant

```powershell
docker compose up -d
```

The included Compose file exposes Qdrant on port `6333` and stores its data in a Docker volume.

### 5. Start the Streamlit app

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

Use `app.py` as the Streamlit entry point. Install the Python dependencies and provide FFmpeg and a Node.js version supported by yt-dlp. `packages.txt` lists the Linux packages, but verify the available Node.js version on your host.

For Streamlit Community Cloud, set the API keys and Qdrant connection values as top-level secrets in the deployment settings:

```toml
GROQ_API_KEY = "your_groq_api_key"
COHERE_API_KEY = "your_cohere_api_key"
MISTRAL_API_KEY = "your_mistral_api_key"
SARVAM_API_KEY = "your_sarvam_api_key"
QDRANT_URL = "https://your-qdrant-endpoint"
QDRANT_API_KEY = "your_qdrant_api_key"
```

Omit optional keys if you do not use those features. A hosted app needs a reachable Qdrant service; `localhost:6333` on the hosting server does not connect to Qdrant on your laptop. Allow enough memory and storage for Whisper's model and temporary media.

Online deployment has not yet been verified.

## Current limitations

- Transcription and LLM outputs can contain mistakes. Check important names, dates, and figures against the video.
- Chat displays previous messages, but retrieval answers each question independently. Ask self-contained questions.
- Long videos take longer and can hit API rate or token limits.
- YouTube access depends on the video and hosting network; file upload provides another input option.
- Qdrant retains indexed transcripts. **Clear results** clears the browser session, not the database.
- The app has no authentication or per-user database isolation; it is currently intended for personal use.

## Reference

This project follows the video-assistant workflow from [Akarsh Vyas's AI Video Assistant](https://github.com/AkarshVyas/AI-Video-Assistant-), with Qdrant and Cohere used for vector storage and embeddings.
