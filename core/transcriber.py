import whisper
import os
import requests
from pydub import AudioSegment


SARVAM_PIECE_SECONDS = 25


WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")


_model = None


def load_model():
    global _model

    if _model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL}...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded.")

    return _model


def transcribe_audio_whisper(audio_file: str, with_segments=False):
    model = load_model()

    result = model.transcribe(
        audio_file,
        task="transcribe"
    )

    if with_segments:
        return {"text": result["text"], "segments": [
            {"text": s["text"].strip(), "start": float(s["start"]), "end": float(s["end"])}
            for s in result.get("segments", []) if s["text"].strip()
        ]}
    return result["text"]


def send_to_sarvam(audio_file: str) -> str:

    if not SARVAM_API_KEY:
        raise RuntimeError(
            "SARVAM_API_KEY is not set in environment / .env"
        )

    headers = {
        "api-subscription-key": SARVAM_API_KEY
    }

    with open(audio_file, "rb") as f:

        files = {
            "file": (
                os.path.basename(audio_file),
                f,
                "audio/wav"
            )
        }

        data = {
            "model": SARVAM_STT_MODEL,
            "with_diarization": "false"
        }

        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120
        )

    if not response.ok:
        print(f"Sarvam Error: {response.status_code}")
        print(response.text)
        response.raise_for_status()

    return response.json().get("transcript", "")


def transcribe_audio_sarvam(audio_file: str, with_segments=False):

    if not SARVAM_API_KEY:
        raise RuntimeError(
            "SARVAM_API_KEY is not set in environment / .env"
        )

    audio = AudioSegment.from_wav(audio_file)

    piece_ms = SARVAM_PIECE_SECONDS * 1000

    full_transcription = ""
    segments = []

    total_pieces = (
        len(audio) + piece_ms - 1
    ) // piece_ms

    for i, start in enumerate(
        range(0, len(audio), piece_ms)
    ):

        piece = audio[start:start + piece_ms]

        piece_path = f"{audio_file}_sv_{i}.wav"

        piece.export(
            piece_path,
            format="wav"
        )

        try:
            print(
                f"  Sarvam piece "
                f"{i + 1}/{total_pieces}..."
            )

            text = send_to_sarvam(piece_path)

            full_transcription += text + " "
            if text.strip():
                segments.append({"text": text.strip(), "start": start / 1000,
                                 "end": min(start + piece_ms, len(audio)) / 1000})

        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    if with_segments:
        return {"text": full_transcription.strip(), "segments": segments}
    return full_transcription.strip()


def transcribe_audio(
    audio_file: str,
    language: str = "english"
) -> str:

    if language.lower() == "hinglish":
        return transcribe_audio_sarvam(audio_file)

    return transcribe_audio_whisper(audio_file)


def transcribe_audio_chunks(
    audio_chunks: list,
    language: str = "english",
    with_segments=False,
):

    full_transcription = ""
    segments = []
    offset = 0.0

    if language.lower() == "hinglish":
        print("Using Sarvam AI for transcription.")
    else:
        print("Using Whisper for transcription.")

    for i, chunk in enumerate(audio_chunks):

        print(
            f"Transcribing chunk "
            f"{i + 1}/{len(audio_chunks)}..."
        )

        if with_segments:
            transcribe = transcribe_audio_sarvam if language.lower() == "hinglish" else transcribe_audio_whisper
            result = transcribe(chunk, with_segments=True)
            text = " ".join(s["text"] for s in result["segments"]) or result["text"]
            segments.extend({**s, "start": s["start"] + offset, "end": s["end"] + offset}
                            for s in result["segments"])
            offset += len(AudioSegment.from_file(chunk)) / 1000
        else:
            text = transcribe_audio(chunk, language=language)

        full_transcription += text + " "

    print("Transcription complete.")

    if with_segments:
        # Use the same segment text for display and retrieval offset mapping.
        return {"text": full_transcription.strip(),
                "segments": segments}
    return full_transcription.strip()
