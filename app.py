import streamlit as st
import torch
import numpy as np
from transformers import pipeline
import tempfile
import os
from datetime import timedelta
import librosa

st.set_page_config(page_title="Whisper Tiny Transcriber", layout="wide")
st.title("Audio Transcription with Timestamps (Whisper Tiny)")
st.markdown("Upload an audio file and get a timestamped transcription with word-level precision.")

SAMPLE_RATE = 16000

@st.cache_resource
def load_pipeline():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny",
        device=device,
        chunk_length_s=30,
        stride_length_s=(4, 2),
        return_timestamps="word",
    )

pipe = load_pipeline()

def format_timestamp(seconds):
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    secs = td.seconds % 60
    millis = int(td.microseconds / 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def output_to_segments(result):
    segments = []
    if "chunks" in result:
        for chunk in result["chunks"]:
            start, end = chunk["timestamp"]
            segments.append({
                "start": start,
                "end": end,
                "text": chunk["text"].strip()
            })
    else:
        segments.append({
            "start": 0.0,
            "end": 0.0,
            "text": result.get("text", "").strip()
        })
    return segments

def segments_to_srt(segments):
    srt = []
    for i, seg in enumerate(segments, 1):
        start = format_timestamp(seg["start"])
        end = format_timestamp(seg["end"])
        text = seg["text"]
        srt.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(srt)

def segments_to_text(segments):
    lines = []
    for seg in segments:
        start = format_timestamp(seg["start"])
        end = format_timestamp(seg["end"])
        lines.append(f"[{start} --> {end}] {seg['text']}")
    return "\n".join(lines)

uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a", "flac", "ogg"])

if uploaded_file:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.audio(uploaded_file)

    if st.button("Transcribe with Timestamps"):
        with st.spinner("Transcribing..."):
            try:
                audio, sr = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
                result = pipe(audio)
                segments = output_to_segments(result)

                if segments:
                    st.success("Transcription complete!")

                    with st.expander("Timestamped Transcription", expanded=True):
                        st.text(segments_to_text(segments))

                    col1, col2 = st.columns(2)
                    with col1:
                        srt_data = segments_to_srt(segments)
                        st.download_button(
                            "Download SRT",
                            data=srt_data,
                            file_name="transcription.srt",
                            mime="text/plain"
                        )
                    with col2:
                        txt_data = segments_to_text(segments)
                        st.download_button(
                            "Download TXT (with timestamps)",
                            data=txt_data,
                            file_name="transcription.txt",
                            mime="text/plain"
                        )

                    full_text = " ".join([seg["text"] for seg in segments])
                    with st.expander("Plain Transcription (no timestamps)"):
                        st.write(full_text)
                else:
                    st.error("No speech detected or transcription failed.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
            finally:
                os.unlink(tmp_path)

st.sidebar.markdown(""
### How to use
1. Upload an audio file (MP3, WAV, M4A, etc.)
2. Click **Transcribe with Timestamps**
3. View the timestamped transcription
4. Download as SRT (subtitles) or plain text

### Model Info
- Model: OpenAI Whisper Tiny (39M parameters)
- Library: Hugging Face Transformers
- Timestamps: Word-level, 20ms precision

### Deployment on Streamlit Cloud
Add this `requirements.txt`:
