import streamlit as st
import whisper
import tempfile
import os
import pandas as pd
import time

st.set_page_config(page_title="Audio Transcriber", layout="centered", page_icon="🎙️")

st.title("🎙️ Audio to Text")
st.caption("Upload an audio file to automatically generate a timestamped transcript.")

# ------------------------------------------------------------------
# 1. Initialize Session State (REQUIRED - prevents KeyError)
# ------------------------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = None
if "done" not in st.session_state:
    st.session_state.done = False
if "current_file" not in st.session_state:
    st.session_state.current_file = None

# ------------------------------------------------------------------
# 2. Load Model (Cached)
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    return whisper.load_model("tiny")

with st.spinner("Loading model..."):
    model = load_model()

# ------------------------------------------------------------------
# 3. Helpers
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
# 4. File Upload & Auto-Transcribe
# ------------------------------------------------------------------
uploaded_file = st.file_uploader("Choose audio file", type=["mp3", "wav", "m4a", "ogg", "flac"])

if uploaded_file is not None:
    # Detect new file to trigger transcription automatically
    if st.session_state.current_file != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        st.session_state.result = None
        st.session_state.done = False

    # Run transcription automatically on new upload
    # ✅ FIX: Use .get() to safely check session state keys
    if st.session_state.get("result") is None and not st.session_state.get("done"):
        st.session_state.done = True
        tmp_path = None
        try:
            with st.status("Transcribing audio...", expanded=True) as status:
                status.write("💾 Saving temporary file...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name

                status.write("🤖 Running Whisper inference...")
                res = model.transcribe(tmp_path, verbose=False)
                
                status.update(label="✅ Transcription Complete!", state="complete")
                time.sleep(0.3)
                st.session_state.result = res
        except Exception as e:
            st.error(f"❌ Transcription failed: {str(e)}")
            st.session_state.done = False
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # ------------------------------------------------------------------
    # 5. Display Results
    # ------------------------------------------------------------------
    if st.session_state.get("result"):
        res = st.session_state.result
        st.success("✅ Transcription finished.")

        # Prepare formats
        txt_lines = []
        srt_lines = []
        table_rows = []

        for seg in res["segments"]:
            t_start = fmt_time(seg["start"])
            t_end = fmt_time(seg["end"])
            text = seg["text"].strip()

            table_rows.append({"Start": t_start, "End": t_end, "Text": text})
            txt_lines.append(f"[{t_start} -> {t_end}] {text}")
            srt_lines.append(f"{len(txt_lines)}\n{fmt_srt(seg['start'])} --> {fmt_srt(seg['end'])}\n{text}\n")

        # DOWNLOAD BUTTONS AT TOP
        st.subheader("📥 Download Files")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download Timestamped TXT",
                data="\n".join(txt_lines),
                file_name="transcript.txt",
                mime="text/plain"
            )
        with col2:
            st.download_button(
                label="Download SRT Subtitles",
                data="\n".join(srt_lines),
                file_name="transcript.srt",
                mime="text/plain"
            )

        st.divider()

        # TABLE VIEW
        st.subheader("Timestamped Segments")
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True, height=300)

        # FULL TEXT VIEW
        st.subheader("Full Transcript")
        st.text_area("", "\n".join(txt_lines), height=250, label_visibility="collapsed")
else:
    st.info("👆 Upload an audio file above to begin.")
