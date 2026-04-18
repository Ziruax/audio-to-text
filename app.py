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
# 1. Model & Pipeline Loading (Cached)
# ------------------------------------------------------------------
@st.cache_resource
def load_asr_pipeline():
    """Loads model, processor, and creates ASR pipeline. Runs only once."""
    processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
    model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-tiny")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    # Prepare model first
    model = model.to(device, dtype=torch_dtype)
    
    # Create pipeline (use `processor=`, NOT `tokenizer=`)
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        processor=processor,
        chunk_length_s=30,      # Process in 30s chunks for better memory & timestamps
        stride_length_s=5       # 5s overlap to avoid cutting words
    )
    return asr_pipe

with st.spinner("⏳ Loading Whisper Tiny model (first load takes ~15s)..."):
    asr_pipe = load_asr_pipeline()

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
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        try:
            with st.spinner("🔊 Processing audio..."):
                # Run inference with timestamps
                result = asr_pipe(tmp_path, return_timestamps=True)

            st.success("✅ Transcription Complete!")

            # Full transcript
            st.subheader("📝 Full Transcript")
            st.text_area("", result["text"], height=120, label_visibility="collapsed")

            # Timestamped segments
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
                df = pd.DataFrame(chunks)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("⚠️ No timestamp chunks returned. This may happen with very short or silent audio.")

        except Exception as e:
            st.error(f"❌ Transcription failed: {str(e)}")
            st.exception(e)  # Shows full traceback for debugging
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
