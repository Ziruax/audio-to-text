import streamlit as st
import torch
import tempfile
import os
import pandas as pd
import sys
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline

st.set_page_config(page_title="Whisper Tiny Transcriber", layout="wide")
st.title("🎙️ Audio to Transcription with Timestamps")
st.caption("Powered by OpenAI Whisper Tiny via 🤗 Transformers")

# ------------------------------------------------------------------
# 1. Model & Pipeline Loading (Cached & Optimized)
# ------------------------------------------------------------------
@st.cache_resource
def load_asr_pipeline():
    """Loads model, processor, and creates ASR pipeline. Runs only once per container."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    # Load model with memory optimization flags for cloud environments
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "openai/whisper-tiny",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained("openai/whisper-tiny")

    # Official HuggingFace pattern for Whisper ASR pipeline
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor,  # Whisper processor acts as both tokenizer & feature extractor
        torch_dtype=torch_dtype,
        device=device
    )
    return asr_pipe

with st.spinner("⏳ Loading Whisper Tiny model (first run takes ~20s)..."):
    try:
        asr_pipe = load_asr_pipeline()
    except Exception as e:
        st.error("❌ Model initialization failed")
        st.code(f"Python: {sys.version}\nTransformers: {__import__('transformers').__version__}")
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
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        try:
            with st.spinner("🔊 Processing audio..."):
                # run inference with timestamps enabled
                result = asr_pipe(tmp_path, return_timestamps=True)

            st.success("✅ Transcription Complete!")

            st.subheader("📝 Full Transcript")
            st.text_area("", result["text"], height=120, label_visibility="collapsed")

            if result.get("chunks"):
                st.subheader("⏱️ Timestamped Segments")
                chunks = []
                for chunk in result["chunks"]:
                    start, end = chunk.get("timestamp", (None, None))
                    chunks.append({
                        "Start": format_seconds(start),
                        "End": format_seconds(end),
                        "Text": chunk["text"].strip()
                    })
                st.dataframe(pd.DataFrame(chunks), use_container_width=True, hide_index=True)
            else:
                st.info("⚠️ No timestamp chunks returned. This may happen with very short or silent audio.")

        except Exception as e:
            st.error(f"❌ Transcription failed: {str(e)}")
            st.exception(e)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
