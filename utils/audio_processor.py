import os
import shutil

import yt_dlp
from pydub import AudioSegment

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_audio_from_youtube(url:str) -> str:
    node_path = shutil.which("node")
    if node_path is None:
        raise RuntimeError(
            "Downloading from YouTube requires Node.js for yt-dlp's JavaScript "
            "challenge solving. Install Node.js and ensure `node` is available "
            "on PATH."
        )

    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'js_runtimes': {'node': {'path': node_path}},
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info_dict).replace('.webm', '.wav').replace('.m4a', '.wav').replace('.mp4', '.wav')
    return filename


def convert_audio_to_wav(input_file:str) -> str:
    output_file = os.path.splitext(input_file)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_file)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_file, format="wav")
    return output_file

def chunk_audio(input_file:str, chunk_minutes:int=10) -> list:
    chunk_length_ms = chunk_minutes * 60 * 1000
    audio = AudioSegment.from_file(input_file)
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunk_file = os.path.splitext(input_file)[0] + f"_chunk_{i // chunk_length_ms}.wav"
        chunk.export(chunk_file, format="wav")
        chunks.append(chunk_file)
    return chunks

def process_input_audio(input_file:str) -> list:
    if input_file.startswith(("http://", "https://")):
        print(f"Downloading audio from YouTube URL: {input_file}")
        input_file = download_audio_from_youtube(input_file)
    else:
        print(f"Processing local audio file: {input_file}")

    converted_file = convert_audio_to_wav(input_file)
    return chunk_audio(converted_file)
