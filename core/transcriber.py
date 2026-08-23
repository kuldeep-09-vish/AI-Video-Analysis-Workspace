import os
from pathlib import Path

import requests
import torch
import whisper
from pydub import AudioSegment

# Sarvam's sync STT-translate API rejects audio longer than 30s.
SARVAM_PIECE_SECONDS = 25
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"

_model = None
_model_name = None


def load_model():
    """Load and cache the configured local Whisper model."""
    global _model, _model_name

    requested_model = os.getenv("WHISPER_MODEL", "small").strip() or "small"
    if _model is None or _model_name != requested_model:
        print(f"Loading Whisper model: {requested_model} ...")
        _model = whisper.load_model(requested_model)
        _model_name = requested_model
        print("Whisper model loaded.")
    return _model


def transcribe_chunk_whisper(chunk_path: str) -> str:
    path = Path(chunk_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio chunk not found: {chunk_path}")

    model = load_model()
    result = model.transcribe(
        str(path),
        task="transcribe",
        fp16=torch.cuda.is_available(),
    )
    return str(result.get("text", "")).strip()


def _send_to_sarvam(piece_path: str) -> str:
    """Send one <=30s WAV file to Sarvam and return the English transcript."""
    api_key = os.getenv("SARVAM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    model = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5").strip() or "saaras:v2.5"
    headers = {"api-subscription-key": api_key}

    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data = {"model": model, "with_diarization": "false"}
        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        detail = response.text[:1000]
        raise RuntimeError(
            f"Sarvam transcription failed ({response.status_code}): {detail}"
        )

    payload = response.json()
    return str(payload.get("transcript", "")).strip()


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """Split audio into <=25s pieces, translate/transcribe each with Sarvam."""
    if not os.getenv("SARVAM_API_KEY", "").strip():
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    path = Path(chunk_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio chunk not found: {chunk_path}")

    audio = AudioSegment.from_file(str(path))
    piece_ms = SARVAM_PIECE_SECONDS * 1000
    transcripts = []
    total_pieces = max(1, (len(audio) + piece_ms - 1) // piece_ms)

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start : start + piece_ms]
        piece_path = path.with_name(f"{path.stem}_sv_{i}.wav")
        piece.export(str(piece_path), format="wav")

        try:
            print(f"  -> Sarvam piece {i + 1}/{total_pieces} ...")
            text = _send_to_sarvam(str(piece_path))
            if text:
                transcripts.append(text)
        finally:
            piece_path.unlink(missing_ok=True)

    return " ".join(transcripts).strip()


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    language = language.strip().lower()
    if language == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)
    if language != "english":
        raise ValueError("language must be either 'english' or 'hinglish'")
    return transcribe_chunk_whisper(chunk_path)


def transcribe_all(chunks: list[str], language: str = "english") -> str:
    if not chunks:
        raise ValueError("No audio chunks were provided for transcription.")

    language = language.strip().lower()
    if language not in {"english", "hinglish"}:
        raise ValueError("language must be either 'english' or 'hinglish'")

    engine = "Sarvam AI" if language == "hinglish" else "Whisper"
    print(f"Using {engine} for transcription.")

    parts = []
    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
        text = transcribe_chunk(chunk, language=language)
        if text:
            parts.append(text)

    transcript = " ".join(parts).strip()
    if not transcript:
        raise RuntimeError("Transcription completed but produced no text.")

    print("Transcription complete.")
    return transcript
