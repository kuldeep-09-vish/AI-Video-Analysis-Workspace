import os
from pathlib import Path
from urllib.parse import urlparse

import yt_dlp
from pydub import AudioSegment

DOWNLOAD_DIR = Path("downloads")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _is_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except ValueError:
        return False


def download_youtube_audio(url: str) -> str:
    """Download the best available audio and convert it to 16-bit WAV via yt-dlp/FFmpeg."""
    output_template = str(DOWNLOAD_DIR / "%(id)s_%(title).100B.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        original = Path(ydl.prepare_filename(info))
        wav_path = original.with_suffix(".wav")

    if not wav_path.is_file():
        # Some extractors alter the prepared filename. Find the most likely output.
        video_id = str(info.get("id", ""))
        matches = sorted(DOWNLOAD_DIR.glob(f"{video_id}_*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not matches:
            raise FileNotFoundError("yt-dlp finished but the converted WAV file could not be found.")
        wav_path = matches[0]

    return str(wav_path)


def convert_to_wav(input_path: str) -> str:
    """Convert an audio/video file to mono 16 kHz WAV format using pydub/FFmpeg."""
    source = Path(input_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input file not found: {source}")

    output_path = source.with_name(f"{source.stem}_converted.wav")
    audio = AudioSegment.from_file(str(source))
    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    audio.export(str(output_path), format="wav")
    return str(output_path)


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list[str]:
    if chunk_minutes <= 0:
        raise ValueError("chunk_minutes must be greater than zero")

    source = Path(wav_path)
    if not source.is_file():
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    audio = AudioSegment.from_wav(str(source))
    if len(audio) == 0:
        raise ValueError("The input audio file is empty.")

    chunk_ms = chunk_minutes * 60 * 1000
    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = source.with_name(f"{source.stem}_chunk_{i}.wav")
        chunk.export(str(chunk_path), format="wav")
        chunks.append(str(chunk_path))

    return chunks


def process_input(source: str) -> list[str]:
    source = source.strip()
    if not source:
        raise ValueError("A YouTube/media URL or local file path is required.")

    if _is_url(source):
        print("Detected media URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready - {len(chunks)} chunk(s) created.")
    return chunks
