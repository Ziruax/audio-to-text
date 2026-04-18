import streamlit as st
import whisper
import tempfile
import os
import pandas as pd
import time

# ------------------------------------------------------------------
# 1. Page Config & High-Contrast Styling
# ------------------------------------------------------------------
st.set_page_config(page_title="Audio Transcriber", layout="centered", page_icon="🎙️")

st.markdown("""
<style>
    .main-header { font-size: 1.8rem; font-weight: 700; color: #0F172A; margin-bottom: 0.25rem; }
    .sub-text { color: #334155; font-size: 1rem; margin-bottom: 1rem; }
    .stDataFrame th { background-color: #0F172A !important; color: #FFFFFF !important; font-weight: 600; }
    .stDataFrame td { color: #1E293B !important; font-size: 0.95rem; }
    .stSuccess, .stError { font-weight: 500; }
    .export-btn { margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🎙️ Audio to Text</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Upload an audio file to automatically generate a transcript with timestamps.</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------
# 2. Model Loading (Cached)
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    return whisper.load_model("tiny")

with st.spinner("⏳ Loading model (first run only)..."):
    model = load_model()

# ------------------------------------------------------------------
# 3. Helper Functions
# ------------------------------------------------------------------
def format_time(sec: float) -> str:
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:05.2f}"

def generate_srt(segments: list) -> str:
    srt_lines = []
    for i, seg in enumerate(segments, 1):
        def fmt(sec):
            ms = int((sec % 1) * 1000)
            h, m, s = int(sec // 3600), int((sec % 3600) // 60), int(sec % 60)
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
        srt_lines.append(f"{i}\n{fmt(seg['start'])} --> {fmt(seg['end'])}\n{seg['text'].strip()}\n")
    return "\n".join(srt_lines)

# ------------------------------------------------------------------
# 4. Auto-Transcribe Logic
# ------------------------------------------------------------------
uploaded_file = st.file_uploader("📂 Choose an audio file", type=["mp3", "wav", "m4a", "ogg", "flac"])

if uploaded_file is not None:
    # Track current file to prevent duplicate processing
    if st.session_state.get("current_file") != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        st.session_state.transcript = None
        st.session_state.processing = False

    # Automatically run transcription when a new file is uploaded
    if st.session_state.transcript is None and not st.session_state.get("processing"):
        st.session_state.processing = True
        tmp_path = None
        try:
            with st.status("🔊 Transcribing audio...", expanded=True) as status:
                status.write("💾 Preparing file...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name

                status.write("🤖 Running Whisper inference...")
                result = model.transcribe(tmp_path, verbose=False)
                
                status.update(label="✅ Transcription Complete!", state="complete")
                time.sleep(0.4)  # Let UI update smoothly
                st.session_state.transcript = result
        except Exception as e:
            st.error(f"❌ Transcription failed: {str(e)}")
            st.session_state.processing = False
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # ------------------------------------------------------------------
    # 5. Display Results
    # ------------------------------------------------------------------
    if st.session_state.transcript:
        result = st.session_state.transcript
        st.success("✅ Transcription finished")

        st.subheader("🕒 Timestamped Segments")
        segments_df = pd.DataFrame([
            {"Start": format_time(s["start"]), "End": format_time(s["end"]), "Text": s["text"].strip()}
            for s in result["segments"]
        ])
        st.dataframe(segments_df, use_container_width=True, hide_index=True, height=300)

        st.subheader("📄 Full Transcript")
        st.text_area("", result["text"], height=150, label_visibility="collapsed")

        st.subheader("💾 Export")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("⬇️ Download TXT", result["text"], "transcript.txt", "text/plain")
        with col2:
            srt_content = generate_srt(result["segments"])
            st.download_button("⬇️ Download SRT", srt_content, "transcript.srt", "text/plain")
else:
    st.info("👆 Upload an audio file above to begin.")
