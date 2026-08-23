# AI Video Assistant - Review & Fixes

## Fixed issues

- `.env` is loaded before project modules in both CLI and Streamlit entry points.
- Whisper/Sarvam configuration is read at runtime instead of being frozen during import.
- Whisper now disables FP16 automatically when CUDA is unavailable.
- Sarvam API failures include actionable HTTP error details and temporary pieces are cleaned up.
- Invalid languages, missing files, empty audio, empty transcripts, empty questions, and missing API keys now fail clearly.
- YouTube/media downloads use robust output-path detection instead of replacing only `.webm`/`.m4a` suffixes.
- Local audio/video is normalized to mono 16 kHz 16-bit WAV.
- Fixed the broken `load_rag_chain()` implementation (`retriver` typo and missing vector store argument).
- RAG collections now use unique names, preventing a new meeting from silently mixing with old transcript chunks.
- Switched embeddings import to `langchain_huggingface.HuggingFaceEmbeddings`.
- Added missing `langchain-chroma` and `langchain-text-splitters` dependencies.
- Added conventional lowercase `requirements.txt` for case-sensitive systems.
- Added `.env.example`.
- Streamlit now supports real browser file uploads in addition to URLs/server-local paths.
- Escaped transcript, LLM output, and chat content before placing it in `unsafe_allow_html` blocks.
- Added generated runtime files/directories to `.gitignore`.

## Validation performed

- `python -m compileall` passes for the full project.
- All Python files pass AST parsing and duplicate top-level definition checks.
- Full runtime/API testing was not possible in the review sandbox because project dependencies, model weights, FFmpeg, and private API credentials are not installed there.

## Run locally

1. Install FFmpeg and make sure `ffmpeg` is on PATH.
2. Create a virtual environment.
3. Run `pip install -r requirements.txt`.
4. Copy `.env.example` to `.env` and fill in `MISTRAL_API_KEY` (and `SARVAM_API_KEY` for Hinglish).
5. Start Streamlit with `streamlit run app.py`.

The first Whisper and embedding-model runs can download model weights and therefore require internet access.
