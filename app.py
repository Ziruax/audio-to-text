import os
import shutil

# ------------------------------------------------------------------
# 0. Patch FFmpeg path before Whisper loads
# ------------------------------------------------------------------
# Streamlit Cloud's Debian repository mirrors can fail on apt-get.
# This extracts the static binary from imageio-ffmpeg and injects it into PATH.
try:
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_exe)

    if ffmpeg_dir not in os.environ["PATH"]:
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]

    target_ffmpeg = os.path.join(ffmpeg_dir, "ffmpeg")
    if not os.path.exists(target_ffmpeg) and os.path.exists(ffmpeg_exe):
        try:
            os.symlink(ffmpeg_exe, target_ffmpeg)
        except OSError:
            shutil.copyfile(ffmpeg_exe, target_ffmpeg)
except ImportError:
    pass

import streamlit as st
import whisper
import tempfile
import pandas as pd

# ------------------------------------------------------------------
# 1. Page Configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Whisper Audio Transcriber",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ Audio to Text Transcriber")
st.caption("Powered by OpenAI Whisper. Upload an audio file to generate transcripts and subtitles.")

# ------------------------------------------------------------------
# 2. Sidebar Settings
# ------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # 'tiny' and 'base' fit safely within Streamlit Cloud's 1GB RAM quota
    model_size = st.selectbox(
        "Whisper Model Size",
        options=["tiny", "base"],
        index=0,
        help="'tiny' is the fastest. 'base' offers slightly higher accuracy."
    )
    
    task = st.radio(
        "Task",
        options=["Transcribe", "Translate to English"],
        index=0,
        help="'Translate' converts foreign-language audio directly to English text."
    )
    
    st.markdown("---")
    st.caption("💡 Free tier cloud nodes run on shared CPU with 1GB RAM. Larger models risk out-of-memory crashes.")

# ------------------------------------------------------------------
# 3. Model Loader (Cached)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_whisper_model(size: str):
    return whisper.load_model(size)

with st.spinner(f"Loading Whisper '{model_size}' model..."):
    model = get_whisper_model(model_size)

# ------------------------------------------------------------------
# 4. Helpers for Time Formatting
# ------------------------------------------------------------------
def fmt_time(sec: float) -> str:
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:05.2f}"

def fmt_srt(sec: float) -> str:
    ms = int((sec % 1) * 1000)
    h, m, s = int(sec // 3600), int((sec % 3600) // 60), int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

# ------------------------------------------------------------------
# 5. File Upload & Processing
# ------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Choose an audio file", 
    type=["mp3", "wav", "m4a", "ogg", "flac"]
)

if uploaded_file is not None:
    st.audio(uploaded_file, format=uploaded_file.type)

    start_transcription = st.button("🚀 Transcribe Audio", type="primary", use_container_width=True)

    if start_transcription:
        tmp_path = None
        try:
            with st.status("Processing audio...", expanded=True) as status:
                status.write("💾 Writing temporary audio file...")
                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name

                status.write(f"🤖 Running Whisper inference ({task.lower()})...")
                task_action = "translate" if task == "Translate to English" else "transcribe"
                
                result = model.transcribe(
                    tmp_path, 
                    task=task_action, 
                    verbose=False,
                    fp16=False  # Avoids CPU warnings on non-GPU containers
                )
                
                status.update(label="✅ Completed successfully!", state="complete", expanded=False)
                st.session_state["result"] = result
                st.session_state["file_name"] = uploaded_file.name

        except Exception as e:
            st.error(f"❌ Processing failed: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

# ------------------------------------------------------------------
# 6. Display Results
# ------------------------------------------------------------------
if "result" in st.session_state and st.session_state["result"] is not None:
    res = st.session_state["result"]
    base_filename = os.path.splitext(st.session_state.get("file_name", "audio"))[0]

    # Metrics
    col_lang, col_dur = st.columns(2)
    detected_lang = res.get("language", "Unknown").upper()
    col_lang.metric("Detected Language", detected_lang)
    
    segments = res.get("segments", [])
    total_time = segments[-1]["end"] if segments else 0
    col_dur.metric("Audio Duration", fmt_time(total_time))

    # Parse segments
    clean_text = res.get("text", "").strip()
    table_rows = []
    timestamped_lines = []
    srt_blocks = []

    for i, seg in enumerate(segments, 1):
        t_start = fmt_time(seg["start"])
        t_end = fmt_time(seg["end"])
        text = seg["text"].strip()

        table_rows.append({"Start": t_start, "End": t_end, "Segment Text": text})
        timestamped_lines.append(f"[{t_start} -> {t_end}] {text}")
        srt_blocks.append(f"{i}\n{fmt_srt(seg['start'])} --> {fmt_srt(seg['end'])}\n{text}\n")

    timestamped_text = "\n".join(timestamped_lines)
    srt_content = "\n".join(srt_blocks)

    st.markdown("---")

    # Download Bar
    d_col1, d_col2, d_col3 = st.columns(3)
    d_col1.download_button(
        "📄 Plain Text (.txt)",
        data=clean_text,
        file_name=f"{base_filename}_transcript.txt",
        mime="text/plain",
        use_container_width=True
    )
    d_col2.download_button(
        "⏱️ Timestamped (.txt)",
        data=timestamped_text,
        file_name=f"{base_filename}_timestamped.txt",
        mime="text/plain",
        use_container_width=True
    )
    d_col3.download_button(
        "🎬 Subtitles (.srt)",
        data=srt_content,
        file_name=f"{base_filename}.srt",
        mime="text/plain",
        use_container_width=True
    )

    # Output Tabs
    tab1, tab2, tab3 = st.tabs(["📝 Full Transcript", "⏱️ Timestamp Segments", "🎬 SRT Output"])

    with tab1:
        st.text_area("Full Transcript", clean_text, height=350, label_visibility="collapsed")

    with tab2:
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    with tab3:
        st.code(srt_content, language="markdown")
