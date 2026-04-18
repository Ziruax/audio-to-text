import streamlit as st
import torch
import tempfile
import os
import pandas as pd
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline

# Page config
st.set_page_config(page_title="Whisper Tiny Transcriber", layout="wide")
st.title("🎙️ Audio to Transcription with Timestamps")
st.caption("Powered by OpenAI Whisper Tiny via 🤗 Transformers")

# ------------------------------------------------------------------
# 1. Model Loading & Caching
# ------------------------------------------------------------------
@st.cache_resource
def load_whisper_model():
    """Loads model weights once and caches them in memory."""
    processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
    model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-tiny")
    return processor, model

with st.spinner("⏳ Loading Whisper Tiny model (this runs only once)..."):
    processor, model = load_whisper_model()

# Auto-detect device and optimal dtype
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32
model = model.to(device, dtype=torch_dtype)

# Create ASR pipeline for seamless timestamp handling
asr_pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor,
    torch_dtype=torch_dtype,
    device=device
)

# ------------------------------------------------------------------
# 2. Helper Functions
# ------------------------------------------------------------------
def format_seconds(sec: float) -> str:
    """Convert seconds to HH:MM:SS format."""
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
        # Save uploaded bytes to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        try:
            with st.spinner("🔊 Processing audio... This may take a moment on CPU."):
                # Run inference with timestamps
                result = asr_pipe(tmp_path, return_timestamps=True)

            st.success("✅ Transcription Complete!")

            # Display full transcript
            st.subheader("📝 Full Transcript")
            st.text_area("", result["text"], height=150, label_visibility="collapsed")

            # Display timestamped chunks
            if "chunks" in result and result["chunks"]:
                st.subheader("⏱️ Timestamped Segments")
                
                chunks = []
                for chunk in result["chunks"]:
                    start, end = chunk.get("timestamp", (None, None))
                    chunks.append({
                        "Start": format_seconds(start),
                        "End": format_seconds(end),
                        "Text": chunk["text"].strip()
                    })
                
                df = pd.DataFrame(chunks)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("⚠️ No timestamp chunks returned. This can happen with very short audio clips.")

        except Exception as e:
            st.error(f"❌ Transcription failed: {str(e)}")
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
