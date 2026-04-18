import streamlit as st
import torch
import numpy as np
import pandas as pd
import soundfile as sf
import io
from transformers import pipeline

st.set_page_config(page_title="Whisper Tiny Transcriber", layout="wide")
st.title("🎙️ Audio to Transcription with Timestamps")
st.caption("Powered by OpenAI Whisper Tiny via 🤗 Transformers")

# ------------------------------------------------------------------
# 1. Pipeline Loading (Cached)
# ------------------------------------------------------------------
@st.cache_resource
def load_asr_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny",
        processor="openai/whisper-tiny",
        torch_dtype=torch_dtype,
        device=device
    )

with st.spinner("⏳ Loading Whisper Tiny model (first run takes ~15s)..."):
    try:
        asr_pipe = load_asr_pipeline()
    except Exception as e:
        st.error("❌ Model initialization failed")
        st.error(f"Details: {e}")
        st.stop()

# ------------------------------------------------------------------
# 2. Helper Functions
# ------------------------------------------------------------------
def format_seconds(sec: float) -> str:
    if sec is None:
        return "--:--:--"
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:05.2f}"

# ------------------------------------------------------------------
# 3. Streamlit UI
# ------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload an audio file",
    type=["mp3", "wav", "m4a", "ogg", "flac"],
    help="Supported formats: MP3, WAV, M4A, OGG, FLAC"
)

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    if st.button("🚀 Transcribe with Timestamps", type="primary"):
        try:
            with st.spinner("🔊 Processing audio..."):
                # 🔑 BYPASS FFMPEG: Load audio directly into memory as a NumPy array
                audio_bytes = uploaded_file.read()
                audio_data, sampling_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
                
                # Whisper expects 1-channel audio. Convert stereo to mono if needed
                if audio_data.ndim > 1:
                    audio_data = audio_data.mean(axis=1)
                
                # Pass (array, sampling_rate) directly to pipeline
                result = asr_pipe(audio_data, sampling_rate=sampling_rate, return_timestamps=True)

            st.success("✅ Transcription Complete!")

            st.subheader("📝 Full Transcript")
            st.text_area("", result["text"], height=120, label_visibility="collapsed")

            if result.get("chunks"):
                st.subheader("⏱️ Timestamped Segments")
                chunks = []
                for chunk in result["chunks"]:
                    ts = chunk.get("timestamp", (None, None))
                    if ts[0] is not None:
                        chunks.append({
                            "Start": format_seconds(ts[0]),
                            "End": format_seconds(ts[1]),
                            "Text": chunk["text"].strip()
                        })
                st.dataframe(pd.DataFrame(chunks), use_container_width=True, hide_index=True)
            else:
                st.info("⚠️ No timestamp chunks returned. This may happen with very short or silent audio.")

        except Exception as e:
            st.error(f"❌ Transcription failed: {str(e)}")
            st.exception(e)
