import streamlit as st
import whisper
import tempfile
import os
import pandas as pd
import time

# ------------------------------------------------------------------
# 1. Page Configuration & Custom CSS
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Whisper Tiny Transcriber",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; margin-bottom: 0.25rem; }
    .sub-header { font-size: 1rem; color: #64748b; margin-bottom: 1.5rem; }
    .status-badge { 
        display: inline-block; padding: 0.3rem 0.6rem; border-radius: 20px; 
        font-size: 0.85rem; font-weight: 500; background: #dcfce7; color: #166534; 
    }
    .info-card { 
        padding: 1rem; background: #f8fafc; border-radius: 10px; 
        border: 1px solid #e2e8f0; margin-top: 0.5rem; 
    }
    .stDataFrame th { background-color: #0f172a !important; color: white !important; }
    .stDataFrame tr:hover td { background-color: #f1f5f9 !important; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 2. Header & Model Initialization
# ------------------------------------------------------------------
st.markdown('<div class="main-header">🎙️ Whisper Audio Transcriber</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Fast AI transcription with precise timestamps. Runs locally using OpenAI\'s Tiny model.</div>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return whisper.load_model("tiny")

with st.spinner("📦 Loading model weights..."):
    model = load_model()

st.markdown('<span class="status-badge">✅ Model Ready</span>', unsafe_allow_html=True)

# ------------------------------------------------------------------
# 3. Layout: Upload & Settings
# ------------------------------------------------------------------
col_upload, col_settings = st.columns([2, 1])

with col_upload:
    st.subheader("📁 Upload Audio")
    audio_file = st.file_uploader(
        "Drag & drop or click to browse",
        type=["wav", "mp3", "m4a", "ogg", "flac"],
        label_visibility="collapsed",
        help="Supported: WAV, MP3, M4A, OGG, FLAC"
    )

with col_settings:
    st.subheader("⚙️ Settings")
    language = st.selectbox(
        "Language",
        options=["Auto-detect", "en", "es", "fr", "de", "zh", "ja", "pt", "ru"],
        index=0,
        help="Leave as Auto-detect for best results"
    )
    task = st.radio("Task", ["transcribe", "translate to English"], horizontal=True)
    
    if audio_file:
        audio_bytes = audio_file.getvalue()
        size_mb = len(audio_bytes) / (1024 * 1024)
        st.markdown(f"""
        <div class="info-card">
            📄 <b>Format:</b> {audio_file.type.split('/')[-1].upper()}<br>
            💾 <b>Size:</b> {size_mb:.2f} MB<br>
            🔊 <b>Audio Player:</b>
        </div>
        """, unsafe_allow_html=True)
        st.audio(audio_bytes)
    else:
        st.info("📤 Upload an audio file to begin.")

# ------------------------------------------------------------------
# 4. Transcription & Results
# ------------------------------------------------------------------
if audio_file is not None:
    st.divider()
    
    if st.button("🚀 Generate Transcription", type="primary", use_container_width=True):
        with st.status("Processing audio...", expanded=True) as status:
            try:
                status.write("📂 Preparing audio file...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name

                status.write("🤖 Running Whisper inference...")
                lang_arg = None if language == "Auto-detect" else language
                task_arg = "translate" if "translate" in task else "transcribe"
                
                result = model.transcribe(
                    tmp_path, 
                    language=lang_arg, 
                    task=task_arg, 
                    verbose=False
                )

                status.write("✨ Formatting output...")
                time.sleep(0.3)
                status.update(label="✅ Transcription Complete!", state="complete")

                # Store in session state to persist across reruns
                st.session_state.transcript = result
                st.session_state.transcribed = True

            except Exception as e:
                st.error(f"❌ Transcription failed: {str(e)}")
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

    # Display results if transcription succeeded
    if st.session_state.get("transcribed"):
        result = st.session_state.transcript
        st.subheader("📝 Transcription Results")
        
        tab_table, tab_text, tab_export = st.tabs(["📊 Timestamp Table", "📄 Plain Text", "💾 Export Files"])

        with tab_table:
            segments = []
            for seg in result["segments"]:
                s, e = seg["start"], seg["end"]
                segments.append({
                    "Start": f"{int(s//3600):02d}:{int((s%3600)//60):02d}:{s%60:05.2f}",
                    "End": f"{int(e//3600):02d}:{int((e%3600)//60):02d}:{e%60:05.2f}",
                    "Text": seg["text"].strip()
                })
            st.dataframe(pd.DataFrame(segments), use_container_width=True, hide_index=True, height=350)

        with tab_text:
            st.text_area("Full Transcript", result["text"], height=300, key="full_text_area")

        with tab_export:
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    label="⬇️ Download .TXT",
                    data=result["text"],
                    file_name="transcript.txt",
                    mime="text/plain"
                )
            with col_dl2:
                # Generate proper SRT content
                srt_lines = []
                for i, seg in enumerate(result["segments"], 1):
                    def fmt_srt(sec):
                        ms = int((sec % 1) * 1000)
                        h, m, s = int(sec // 3600), int((sec % 3600) // 60), int(sec % 60)
                        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
                    srt_lines.append(f"{i}\n{fmt_srt(seg['start'])} --> {fmt_srt(seg['end'])}\n{seg['text'].strip()}\n")
                
                st.download_button(
                    label="⬇️ Download .SRT (Subtitles)",
                    data="\n".join(srt_lines),
                    file_name="transcript.srt",
                    mime="text/plain"
                )
