import streamlit as st
import whisper
import tempfile
import os
import pandas as pd

st.set_page_config(page_title="Whisper Tiny Transcriber", page_icon="🎙️", layout="wide")
st.title("🎙️ Whisper Tiny Audio Transcriber")
st.markdown("Upload an audio file to generate a transcription with precise timestamps.")

# 1. Load Model (Cached)
@st.cache_resource
def load_model():
    return whisper.load_model("tiny")

with st.spinner("📦 Loading Whisper Tiny model (first run only)..."):
    model = load_model()

# 2. File Uploader
audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a", "ogg", "flac"])

if audio_file is not None:
    st.audio(audio_file)
    
    if st.button("🚀 Generate Transcription", type="primary"):
        with st.spinner("🔊 Transcribing audio..."):
            # Save to temp file safely
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(audio_file.getbuffer())
                tmp_path = tmp.name

            try:
                # verbose=False keeps Streamlit logs clean
                result = model.transcribe(tmp_path, verbose=False)
                st.success("✅ Transcription Complete!")
                
                st.subheader("📝 Timestamped Segments")
                segments = []
                for seg in result["segments"]:
                    start = seg["start"]
                    end = seg["end"]
                    
                    # Format with sub-second precision: HH:MM:SS.mmm
                    start_str = f"{int(start//3600):02d}:{int((start%3600)//60):02d}:{start%60:05.2f}"
                    end_str = f"{int(end//3600):02d}:{int((end%3600)//60):02d}:{end%60:05.2f}"
                    
                    segments.append({
                        "Start": start_str,
                        "End": end_str,
                        "Text": seg["text"].strip()
                    })
                    
                st.dataframe(pd.DataFrame(segments), use_container_width=True, hide_index=True)
                
                # Download full text
                st.download_button(
                    "⬇️ Download Full Transcript (.txt)",
                    result["text"],
                    file_name="transcription.txt",
                    mime="text/plain"
                )
                
            except Exception as e:
                st.error(f"❌ Transcription failed: {e}")
            finally:
                # Always clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
