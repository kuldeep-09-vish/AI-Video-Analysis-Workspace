import html
import os
import tempfile
import time
import textwrap

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question


# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NovaMeet AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def render_html(markup: str) -> None:
    """Render custom HTML safely without Markdown indent/code-block issues."""
    st.markdown(textwrap.dedent(markup).strip(), unsafe_allow_html=True)


def safe_html(value) -> str:
    return html.escape(str(value)).replace("\n", "<br>")


for key, default in {
    "result": None,
    "chat_history": [],
    "pipeline_done": False,
    "pipeline_steps": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def step_state(key: str) -> str:
    return st.session_state.pipeline_steps.get(key, "pending")


# -----------------------------------------------------------------------------
# Modern light multi-gradient theme
# -----------------------------------------------------------------------------
render_html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');

:root {
  --bg: #fbfcff;
  --surface: rgba(255,255,255,.88);
  --surface-solid: #ffffff;
  --text: #182033;
  --muted: #667085;
  --border: #e6eaf2;
  --blue: #5b8cff;
  --cyan: #55d8e8;
  --pink: #ff78b7;
  --yellow: #ffd86b;
  --purple: #8b7cff;
  --green: #20b486;
  --shadow: 0 18px 45px rgba(83, 98, 145, .10);
}

html, body, [class*="css"] {
  font-family: 'Inter', sans-serif !important;
  color: var(--text) !important;
}

.stApp {
  background:
    radial-gradient(circle at 8% 5%, rgba(91,140,255,.20), transparent 24%),
    radial-gradient(circle at 92% 8%, rgba(85,216,232,.18), transparent 23%),
    radial-gradient(circle at 88% 86%, rgba(255,120,183,.18), transparent 25%),
    radial-gradient(circle at 22% 92%, rgba(255,216,107,.20), transparent 24%),
    linear-gradient(135deg, #fbfdff 0%, #f8fbff 42%, #fff9fc 72%, #fffdf6 100%) !important;
  min-height: 100vh;
}

.block-container {
  max-width: 1450px;
  padding-top: 1.15rem;
  padding-bottom: 2.5rem;
}

/* Hide Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent !important;}

/* Sidebar */
[data-testid="stSidebar"] {
  background: rgba(255,255,255,.84) !important;
  backdrop-filter: blur(18px);
  border-right: 1px solid rgba(220,226,238,.95) !important;
}
[data-testid="stSidebar"] > div:first-child {
  padding-top: 1rem;
}
[data-testid="stSidebar"] * {color: var(--text) !important;}

.sidebar-brand {
  display:flex;
  align-items:center;
  gap:.8rem;
  padding:.2rem 0 .9rem;
}
.brand-mark {
  width:44px;
  height:44px;
  border-radius:15px;
  display:flex;
  align-items:center;
  justify-content:center;
  background:linear-gradient(135deg,var(--blue),var(--purple),var(--pink));
  box-shadow:0 10px 24px rgba(91,140,255,.24);
  color:#fff !important;
  font-size:1.2rem;
  font-weight:800;
}
.brand-name {
  font-family:'Plus Jakarta Sans',sans-serif;
  font-size:1.02rem;
  font-weight:800;
  letter-spacing:-.02em;
}
.brand-sub {font-size:.74rem;color:var(--muted)!important;margin-top:.1rem;}
.section-label {
  font-size:.69rem;
  font-weight:800;
  letter-spacing:.11em;
  text-transform:uppercase;
  color:#8b94a7 !important;
  margin:1rem 0 .45rem;
}
.side-note {
  margin-top:1rem;
  padding:.9rem;
  border-radius:16px;
  background:linear-gradient(135deg,rgba(91,140,255,.08),rgba(255,120,183,.07),rgba(255,216,107,.10));
  border:1px solid rgba(216,223,237,.9);
  color:#586174 !important;
  font-size:.78rem;
  line-height:1.6;
}

/* Native form controls: high visibility */
.stTextInput input,
.stTextArea textarea,
[data-baseweb="select"] > div {
  background:#fff !important;
  color:#172033 !important;
  border:1px solid #d8deea !important;
  border-radius:13px !important;
  min-height:44px !important;
  box-shadow:0 4px 14px rgba(60,80,125,.04) !important;
}
.stTextInput input:focus,
.stTextArea textarea:focus,
[data-baseweb="select"] > div:focus-within {
  border-color:#8baaff !important;
  box-shadow:0 0 0 3px rgba(91,140,255,.12) !important;
}
.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
  color:#9aa4b6 !important;
  opacity:1 !important;
}
[data-baseweb="select"] * {color:#172033 !important;}
[data-baseweb="popover"], [data-baseweb="menu"] {background:#fff !important;}
[data-baseweb="menu"] * {color:#172033 !important;}
label, [data-testid="stWidgetLabel"] p {
  color:#414b60 !important;
  font-weight:600 !important;
  font-size:.78rem !important;
}

/* File uploader */
[data-testid="stFileUploaderDropzone"] {
  background:linear-gradient(135deg,#ffffff,#fbfcff) !important;
  border:1px dashed #cfd8e8 !important;
  border-radius:16px !important;
  padding:.65rem !important;
}
[data-testid="stFileUploaderDropzone"] button {
  background:linear-gradient(90deg,#eef3ff,#f8f1ff,#fff2f8) !important;
  color:#5d61d6 !important;
  border:1px solid #d9dcff !important;
  border-radius:11px !important;
  font-weight:700 !important;
}
[data-testid="stFileUploaderDropzone"] button * {color:#5d61d6 !important;}
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span {color:#687386 !important;}

/* Buttons */
.stButton > button {
  min-height:44px;
  border:none !important;
  border-radius:13px !important;
  background:linear-gradient(100deg,#5b8cff 0%,#877bff 35%,#ff78b7 68%,#ffc85c 100%) !important;
  background-size:180% 180% !important;
  color:#fff !important;
  font-weight:800 !important;
  box-shadow:0 10px 24px rgba(108,112,220,.22) !important;
  transition:transform .18s ease, box-shadow .18s ease !important;
}
.stButton > button:hover {
  transform:translateY(-1px);
  box-shadow:0 14px 30px rgba(108,112,220,.28) !important;
}
.stButton > button * {color:#fff !important;}
.stButton > button[kind="secondary"] {
  background:#fff !important;
  color:#475167 !important;
  border:1px solid #dfe4ee !important;
  box-shadow:none !important;
}
.stButton > button[kind="secondary"] * {color:#475167 !important;}

/* Main header */
.topbar {
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:1rem;
  padding:1rem 1.2rem;
  border-radius:20px;
  background:rgba(255,255,255,.82);
  backdrop-filter:blur(16px);
  border:1px solid rgba(224,229,240,.95);
  box-shadow:0 10px 30px rgba(80,96,135,.07);
  margin-bottom:1rem;
}
.topbar-title {
  font-family:'Plus Jakarta Sans',sans-serif;
  font-size:1.05rem;
  font-weight:800;
}
.topbar-sub {font-size:.75rem;color:var(--muted);margin-top:.15rem;}
.ready-pill {
  display:inline-flex;
  align-items:center;
  gap:.45rem;
  padding:.48rem .72rem;
  border-radius:999px;
  background:linear-gradient(90deg,rgba(85,216,232,.12),rgba(91,140,255,.10));
  border:1px solid rgba(126,181,221,.25);
  color:#3d7188;
  font-size:.72rem;
  font-weight:700;
}
.ready-dot {width:8px;height:8px;border-radius:50%;background:var(--green);box-shadow:0 0 0 4px rgba(32,180,134,.10);}

/* Hero */
.hero {
  position:relative;
  overflow:hidden;
  border-radius:28px;
  padding:2rem;
  background:rgba(255,255,255,.82);
  border:1px solid rgba(224,229,240,.92);
  box-shadow:var(--shadow);
  margin-bottom:1rem;
}
.hero:before {
  content:'';
  position:absolute;
  width:300px;height:300px;
  left:-90px;top:-140px;
  border-radius:50%;
  background:rgba(91,140,255,.22);
  filter:blur(28px);
}
.hero:after {
  content:'';
  position:absolute;
  width:360px;height:360px;
  right:-120px;bottom:-220px;
  border-radius:50%;
  background:linear-gradient(135deg,rgba(255,120,183,.20),rgba(255,216,107,.22));
  filter:blur(30px);
}
.hero-inner {position:relative;z-index:1;display:grid;grid-template-columns:1.25fr .75fr;gap:1.4rem;align-items:center;}
.hero-badge {
  display:inline-flex;
  padding:.44rem .68rem;
  border-radius:999px;
  background:linear-gradient(90deg,rgba(91,140,255,.10),rgba(255,120,183,.09),rgba(255,216,107,.14));
  border:1px solid rgba(137,150,190,.18);
  color:#5b6590;
  font-size:.72rem;
  font-weight:800;
}
.hero-title {
  font-family:'Plus Jakarta Sans',sans-serif;
  font-size:clamp(2rem,4vw,3.25rem);
  line-height:1.08;
  font-weight:800;
  letter-spacing:-.045em;
  margin:.75rem 0 .6rem;
  background:linear-gradient(92deg,#4c7fff 0%,#7f78ff 34%,#f06eae 66%,#e6a11c 100%);
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
  background-clip:text;
}
.hero-copy {max-width:790px;font-size:.98rem;line-height:1.7;color:#657085;}
.feature-row {display:flex;gap:.5rem;flex-wrap:wrap;margin-top:1rem;}
.feature-chip {
  padding:.45rem .65rem;
  border-radius:10px;
  background:rgba(255,255,255,.76);
  border:1px solid rgba(218,224,237,.95);
  font-size:.72rem;
  font-weight:700;
  color:#596277;
}
.hero-orb {
  min-height:190px;
  border-radius:24px;
  display:flex;
  align-items:center;
  justify-content:center;
  background:
    radial-gradient(circle at 25% 25%,rgba(91,140,255,.85),transparent 32%),
    radial-gradient(circle at 72% 32%,rgba(255,120,183,.78),transparent 32%),
    radial-gradient(circle at 60% 80%,rgba(255,216,107,.82),transparent 34%),
    linear-gradient(135deg,#eaf4ff,#fff2f9,#fff7d9);
  box-shadow:inset 0 0 0 1px rgba(255,255,255,.65),0 16px 38px rgba(99,114,160,.10);
}
.hero-orb-core {
  width:92px;height:92px;border-radius:28px;
  display:flex;align-items:center;justify-content:center;
  background:rgba(255,255,255,.86);
  box-shadow:0 18px 35px rgba(85,95,135,.16);
  font-size:2rem;
}

/* Pipeline */
.pipeline-wrap {
  display:grid;
  grid-template-columns:repeat(6,minmax(0,1fr));
  gap:.7rem;
  margin:1rem 0 1.15rem;
}
.pipeline-step {
  padding:.85rem;
  border-radius:16px;
  background:rgba(255,255,255,.82);
  border:1px solid rgba(221,227,238,.95);
  box-shadow:0 7px 20px rgba(77,91,130,.05);
}
.step-top {display:flex;align-items:center;justify-content:space-between;margin-bottom:.65rem;}
.step-index {font-size:.66rem;font-weight:800;color:#98a2b3;letter-spacing:.08em;}
.step-dot {width:9px;height:9px;border-radius:50%;background:#d7dce7;}
.pipeline-step.active {border-color:#aab8ff;background:linear-gradient(135deg,#fff,#f8f7ff);}
.pipeline-step.active .step-dot {background:linear-gradient(90deg,var(--blue),var(--pink));box-shadow:0 0 0 5px rgba(113,117,255,.10);}
.pipeline-step.done {background:linear-gradient(135deg,#fff,#f5fffb);}
.pipeline-step.done .step-dot {background:var(--green);}
.step-name {font-size:.76rem;font-weight:700;color:#525d72;}

/* Empty state */
.empty-card {
  min-height:320px;
  display:flex;
  align-items:center;
  justify-content:center;
  text-align:center;
  border-radius:24px;
  background:rgba(255,255,255,.72);
  border:1px dashed #d6ddea;
  box-shadow:0 12px 30px rgba(78,92,130,.05);
  padding:2rem;
}
.empty-icon {
  width:64px;height:64px;border-radius:20px;margin:0 auto .9rem;
  display:flex;align-items:center;justify-content:center;font-size:1.7rem;
  background:linear-gradient(135deg,#eaf2ff,#f7efff,#fff0f6,#fff7d9);
  border:1px solid #e5e6f2;
}
.empty-title {font-family:'Plus Jakarta Sans';font-size:1.25rem;font-weight:800;margin-bottom:.45rem;}
.empty-copy {max-width:620px;color:#70798b;font-size:.86rem;line-height:1.7;}
.empty-steps {display:flex;gap:.45rem;flex-wrap:wrap;justify-content:center;margin-top:1rem;}
.empty-step {font-size:.7rem;font-weight:700;padding:.42rem .6rem;border-radius:9px;background:#fff;border:1px solid #e2e7ef;color:#606a7d;}

/* Results */
.session-card {
  padding:1rem 1.15rem;
  border-radius:18px;
  background:linear-gradient(105deg,rgba(91,140,255,.10),rgba(255,255,255,.88),rgba(255,120,183,.08),rgba(255,216,107,.13));
  border:1px solid rgba(219,225,238,.95);
  margin-bottom:.8rem;
}
.session-label {font-size:.66rem;font-weight:800;text-transform:uppercase;letter-spacing:.1em;color:#8791a5;}
.session-title {font-family:'Plus Jakarta Sans';font-size:1.18rem;font-weight:800;margin-top:.25rem;}

.stTabs [data-baseweb="tab-list"] {
  gap:.4rem;
  background:rgba(255,255,255,.72);
  padding:.35rem;
  border-radius:15px;
  border:1px solid #e4e8f0;
}
.stTabs [data-baseweb="tab"] {border-radius:11px;color:#697386;font-weight:700;padding:.55rem .9rem;}
.stTabs [aria-selected="true"] {
  background:linear-gradient(100deg,#edf3ff,#f8f2ff,#fff1f6,#fff8df) !important;
  color:#5660b7 !important;
  box-shadow:0 4px 12px rgba(75,88,125,.06);
}

.content-card {
  height:100%;
  padding:1.1rem;
  border-radius:20px;
  background:rgba(255,255,255,.88);
  border:1px solid #e4e8f0;
  box-shadow:0 10px 28px rgba(74,88,126,.06);
}
.card-head {display:flex;align-items:center;gap:.55rem;margin-bottom:.75rem;}
.card-icon {
  width:32px;height:32px;border-radius:10px;
  display:flex;align-items:center;justify-content:center;
  background:linear-gradient(135deg,#edf3ff,#f7efff,#fff2f7,#fff8de);
  color:#6268c5;font-weight:800;
}
.card-title {font-family:'Plus Jakarta Sans';font-size:.9rem;font-weight:800;}
.card-content {font-size:.88rem;line-height:1.75;color:#4f596d;}

.transcript-box {
  padding:1.15rem;
  border-radius:20px;
  background:#fff;
  border:1px solid #e3e8f0;
  box-shadow:0 9px 24px rgba(78,91,125,.05);
  color:#4d576a;
  font-size:.87rem;
  line-height:1.78;
  max-height:570px;
  overflow-y:auto;
}

/* Chat */
.chat-intro {
  padding:.9rem 1rem;
  border-radius:16px;
  background:linear-gradient(100deg,rgba(91,140,255,.08),rgba(255,120,183,.07),rgba(255,216,107,.12));
  border:1px solid #e3e7f0;
  margin-bottom:.8rem;
}
.chat-title {font-family:'Plus Jakarta Sans';font-weight:800;font-size:.92rem;}
.chat-sub {font-size:.74rem;color:#737d90;margin-top:.15rem;}
.chat-feed {display:flex;flex-direction:column;gap:.75rem;margin-bottom:.9rem;}
.chat-row {display:flex;}
.chat-row.user {justify-content:flex-end;}
.chat-row.assistant {justify-content:flex-start;}
.chat-msg {max-width:78%;}
.chat-meta {font-size:.65rem;font-weight:800;color:#8a94a7;margin-bottom:.25rem;}
.chat-row.user .chat-meta {text-align:right;}
.chat-bubble {padding:.75rem .9rem;border-radius:16px;font-size:.84rem;line-height:1.65;border:1px solid #e2e6ef;}
.chat-row.user .chat-bubble {background:linear-gradient(110deg,#edf3ff,#f7efff,#fff1f6);color:#3b465b;}
.chat-row.assistant .chat-bubble {background:#fff;color:#4d576a;}

[data-testid="stChatInput"] {
  background:#fff !important;
  border:1px solid #dce2ec !important;
  border-radius:17px !important;
  box-shadow:0 10px 28px rgba(72,86,124,.08) !important;
}
[data-testid="stChatInput"] textarea {color:#182033 !important;}
[data-testid="stChatInput"] textarea::placeholder {color:#98a2b3 !important;}

/* Alerts */
[data-testid="stAlert"] {border-radius:14px !important;}

/* Responsive */
@media(max-width:1100px) {
  .pipeline-wrap {grid-template-columns:repeat(3,minmax(0,1fr));}
  .hero-inner {grid-template-columns:1fr;}
  .hero-orb {min-height:150px;}
}
@media(max-width:760px) {
  .block-container {padding-left:.75rem;padding-right:.75rem;}
  .topbar {align-items:flex-start;flex-direction:column;}
  .hero {padding:1.25rem;}
  .pipeline-wrap {grid-template-columns:repeat(2,minmax(0,1fr));}
  .chat-msg {max-width:94%;}
}
</style>
""")


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    render_html("""
    <div class="sidebar-brand">
      <div class="brand-mark">✦</div>
      <div>
        <div class="brand-name">NovaMeet AI</div>
        <div class="brand-sub">Video intelligence workspace</div>
      </div>
    </div>
    <div class="section-label">New analysis</div>
    """)

    source = st.text_input(
        "Media URL or file path",
        placeholder="Paste YouTube URL or local path",
    )

    uploaded_file = st.file_uploader(
        "Upload audio or video",
        type=["wav", "mp3", "m4a", "mp4", "mov", "webm", "mpeg", "mpga"],
        help="Supported: WAV, MP3, M4A, MP4, MOV and WEBM",
    )

    language = st.selectbox("Language", ["english", "hinglish"], index=0)
    run_btn = st.button("✨  Start analysis", use_container_width=True)

    render_html("""
    <div class="side-note">
      <strong>How it works</strong><br>
      Add media, choose the language, and start analysis. NovaMeet creates a transcript, summary, action items, decisions, open questions and searchable AI chat.
    </div>
    """)


# -----------------------------------------------------------------------------
# Header / Hero
# -----------------------------------------------------------------------------
render_html("""
<div class="topbar">
  <div>
    <div class="topbar-title">AI Video Analysis Workspace</div>
    <div class="topbar-sub">Transcribe · Summarise · Extract · Ask AI</div>
  </div>
  <div class="ready-pill"><span class="ready-dot"></span> AI engine ready</div>
</div>
""")

render_html("""
<div class="hero">
  <div class="hero-inner">
    <div>
      <div class="hero-badge">✨ Multi-model meeting intelligence</div>
      <div class="hero-title">Turn every video into clear, useful knowledge.</div>
      <div class="hero-copy">Upload or link a recording, generate a reliable transcript, surface key decisions and action items, then ask questions using the recording as context.</div>
      <div class="feature-row">
        <span class="feature-chip">Fast transcript</span>
        <span class="feature-chip">Smart summary</span>
        <span class="feature-chip">Action extraction</span>
        <span class="feature-chip">Context-aware chat</span>
      </div>
    </div>
    <div class="hero-orb"><div class="hero-orb-core">🎬</div></div>
  </div>
</div>
""")


# -----------------------------------------------------------------------------
# Pipeline status
# -----------------------------------------------------------------------------
steps = [
    ("audio", "01", "Media processing"),
    ("transcript", "02", "Transcription"),
    ("title", "03", "Title generation"),
    ("summary", "04", "Smart summary"),
    ("extract", "05", "Key extraction"),
    ("rag", "06", "Chat indexing"),
]

pipeline_parts = ['<div class="pipeline-wrap">']
for key, number, label in steps:
    state = step_state(key)
    pipeline_parts.append(
        f'<div class="pipeline-step {state}">'
        f'<div class="step-top"><span class="step-index">{number}</span><span class="step-dot"></span></div>'
        f'<div class="step-name">{html.escape(label)}</div>'
        f'</div>'
    )
pipeline_parts.append('</div>')
render_html(''.join(pipeline_parts))


# -----------------------------------------------------------------------------
# Processing pipeline
# -----------------------------------------------------------------------------
if run_btn:
    if not source.strip() and uploaded_file is None:
        st.error("Please add a YouTube URL, local file path, or upload a media file.")
    else:
        st.session_state.pipeline_done = False
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_steps = {}
        progress = st.empty()

        def set_step(name, state):
            st.session_state.pipeline_steps[name] = state

        try:
            progress.info("Preparing your media…")
            set_step("audio", "active")
            effective_source = source.strip()
            uploaded_temp_path = None

            if uploaded_file is not None:
                suffix = os.path.splitext(uploaded_file.name)[1] or ".bin"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    uploaded_temp_path = tmp.name
                effective_source = uploaded_temp_path

            try:
                chunks = process_input(effective_source)
            finally:
                if uploaded_temp_path and os.path.exists(uploaded_temp_path):
                    os.remove(uploaded_temp_path)
            set_step("audio", "done")

            progress.info("Creating transcript…")
            set_step("transcript", "active")
            transcript = transcribe_all(chunks, language)
            set_step("transcript", "done")

            progress.info("Generating title…")
            set_step("title", "active")
            title = generate_title(transcript)
            set_step("title", "done")

            progress.info("Creating smart summary…")
            set_step("summary", "active")
            summary = summarize(transcript)
            set_step("summary", "done")

            progress.info("Extracting key information…")
            set_step("extract", "active")
            action_items = extract_action_items(transcript)
            decisions = extract_key_decisions(transcript)
            questions = extract_questions(transcript)
            set_step("extract", "done")

            progress.info("Building searchable AI context…")
            set_step("rag", "active")
            rag_chain = build_rag_chain(transcript)
            set_step("rag", "done")

            st.session_state.result = {
                "title": title,
                "transcript": transcript,
                "summary": summary,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
                "rag_chain": rag_chain,
            }
            st.session_state.pipeline_done = True
            progress.success("Analysis complete ✨")
            time.sleep(0.35)
            progress.empty()
            st.rerun()

        except Exception as e:
            for key in ["audio", "transcript", "title", "summary", "extract", "rag"]:
                if st.session_state.pipeline_steps.get(key) == "active":
                    st.session_state.pipeline_steps[key] = "pending"
            progress.error(f"Analysis failed: {e}")


# -----------------------------------------------------------------------------
# Results / empty state
# -----------------------------------------------------------------------------
if st.session_state.result:
    r = st.session_state.result

    render_html(f"""
    <div class="session-card">
      <div class="session-label">Current session</div>
      <div class="session-title">{safe_html(r['title'])}</div>
    </div>
    """)

    tab_overview, tab_transcript, tab_chat = st.tabs(["✨ Overview", "📝 Transcript", "💬 Ask AI"])

    with tab_overview:
        left, right = st.columns([1.2, .8], gap="medium")
        with left:
            render_html(f"""
            <div class="content-card">
              <div class="card-head"><div class="card-icon">✦</div><div class="card-title">AI Summary</div></div>
              <div class="card-content">{safe_html(r['summary'])}</div>
            </div>
            """)
        with right:
            render_html(f"""
            <div class="content-card">
              <div class="card-head"><div class="card-icon">✓</div><div class="card-title">Action Items</div></div>
              <div class="card-content">{safe_html(r['action_items'])}</div>
            </div>
            """)

        st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            render_html(f"""
            <div class="content-card">
              <div class="card-head"><div class="card-icon">◆</div><div class="card-title">Key Decisions</div></div>
              <div class="card-content">{safe_html(r['key_decisions'])}</div>
            </div>
            """)
        with c2:
            render_html(f"""
            <div class="content-card">
              <div class="card-head"><div class="card-icon">?</div><div class="card-title">Open Questions</div></div>
              <div class="card-content">{safe_html(r['open_questions'])}</div>
            </div>
            """)

    with tab_transcript:
        render_html(f'<div class="transcript-box">{safe_html(r["transcript"])}</div>')

    with tab_chat:
        render_html("""
        <div class="chat-intro">
          <div class="chat-title">Ask about this recording</div>
          <div class="chat-sub">Answers use your transcript as context.</div>
        </div>
        """)

        if st.session_state.chat_history:
            chat_parts = ['<div class="chat-feed">']
            for msg in st.session_state.chat_history:
                role = "user" if msg["role"] == "user" else "assistant"
                label = "You" if role == "user" else "NovaMeet AI"
                chat_parts.append(
                    f'<div class="chat-row {role}"><div class="chat-msg">'
                    f'<div class="chat-meta">{label}</div>'
                    f'<div class="chat-bubble">{safe_html(msg["content"])}</div>'
                    f'</div></div>'
                )
            chat_parts.append('</div>')
            render_html(''.join(chat_parts))
        else:
            render_html("""
            <div class="empty-card" style="min-height:190px">
              <div>
                <div class="empty-icon">💬</div>
                <div class="empty-title">Start a conversation</div>
                <div class="empty-copy">Ask “What were the main decisions?”, “What tasks were assigned?” or “Summarise the discussion in five points.”</div>
              </div>
            </div>
            """)

        user_input = st.chat_input("Ask anything about this transcript…")
        if user_input:
            with st.spinner("Thinking…"):
                answer = ask_question(r["rag_chain"], user_input.strip())
            st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()

        if st.session_state.chat_history:
            if st.button("Clear conversation", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()

else:
    render_html("""
    <div class="empty-card">
      <div>
        <div class="empty-icon">🎥</div>
        <div class="empty-title">Ready for your first analysis</div>
        <div class="empty-copy">Add a YouTube URL, local file path, or upload media from the sidebar. NovaMeet will create the transcript, summary, actions, decisions, questions and searchable AI chat.</div>
        <div class="empty-steps">
          <span class="empty-step">1 · Add media</span>
          <span class="empty-step">2 · Choose language</span>
          <span class="empty-step">3 · Start analysis</span>
          <span class="empty-step">4 · Ask AI</span>
        </div>
      </div>
    </div>
    """)    