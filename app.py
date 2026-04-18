import streamlit as st
import torch
import librosa
import numpy as np
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import tempfile
import os
from datetime import timedelta

# Page config
st.set_page_config(page_title="Whisper Tiny Transcriber with Timestamps", layout="wide")
st.title("🎙️ Audio Transcription with Timestamps (Whisper Tiny)")
st.markdown("Upload an audio file (MP3, WAV, M4A) and get a timestamped transcription.")

# Constants
SAMPLE_RATE = 16000  # Whisper expects 16 kHz

# Cache the model and processor
@st.cache_resource
def load_model():
    with st.spinner("Loading Whisper Tiny model... (this may take a minute on first run)"):
        processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
        model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-tiny")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        return processor, model, device

processor, model, device = load_model()

def load_audio(file_path):
    """Load audio file, resample to 16kHz mono, return waveform array."""
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    return audio

def transcribe_with_timestamps(audio_array):
    """Run Whisper model and return segments with timestamps."""
    inputs = processor(audio_array, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_features,
            return_timestamps=True,
            language="en",          # Force English for best results
            task="transcribe"
        )

    # Decode with timestamps - FIX: add return_language=False
    segments = processor.tokenizer._decode_asr(
        generated_ids[0],
        return_timestamps=True,
        return_language=False,      # Required to avoid missing argument error
        time_precision=0.02,
    )
    return segments

def format_timestamp(seconds):
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    secs = td.seconds % 60
    millis = int(td.microseconds / 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def segments_to_srt(segments):
    """Convert segment list to SRT format."""
    srt_lines = []
    for i, seg in enumerate(segments, 1):
        start = format_timestamp(seg["start"])
        end = format_timestamp(seg["end"])
        text = seg["text"].strip()
        srt_lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(srt_lines)

def segments_to_text(segments):
    """Return plain text with timestamps."""
    lines = []
    for seg in segments:
        start = format_timestamp(seg["start"])
        end = format_timestamp(seg["end"])
        text = seg["text"].strip()
        lines.append(f"[{start} --> {end}] {text}")
    return "\n".join(lines)

# Main app
uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a", "flac", "ogg"])

if uploaded_file is not None:
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Audio player preview
    st.audio(uploaded_file, format='audio/wav')

    if st.button("Transcribe with Timestamps"):
        with st.spinner("Processing audio..."):
            try:
                audio_data = load_audio(tmp_path)
                segments = transcribe_with_timestamps(audio_data)

                if segments:
                    st.success("Transcription complete!")

                    with st.expander("📝 View Transcription with Timestamps", expanded=True):
                        st.text(segments_to_text(segments))

                    col1, col2 = st.columns(2)
                    with col1:
                        srt_data = segments_to_srt(segments)
                        st.download_button(
                            label="⬇️ Download as SRT",
                            data=srt_data,
                            file_name="transcription.srt",
                            mime="text/plain"
                        )
                    with col2:
                        plain_text = segments_to_text(segments)
                        st.download_button(
                            label="⬇️ Download as TXT (with timestamps)",
                            data=plain_text,
                            file_name="transcription.txt",
                            mime="text/plain"
                        )

                    # Plain transcription without timestamps
                    full_text = " ".join([seg["text"].strip() for seg in segments])
                    with st.expander("📄 Plain Transcription (no timestamps)"):
                        st.write(full_text)
                else:
                    st.error("No transcription segments returned.")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            finally:
                os.unlink(tmp_path)

# Sidebar info
st.sidebar.markdown("""
### How to use
1. Upload an audio file (MP3, WAV, M4A, etc.)
2. Click **Transcribe with Timestamps**
3. View the timestamped transcription
4. Download as SRT (subtitles) or plain text

### Model Info
- **Model**: OpenAI Whisper Tiny (39M parameters)
- **Library**: Hugging Face Transformers
- **Timestamps**: Word-level, 20ms precision

### Deployment on Streamlit Cloud
- The app will automatically install dependencies from `requirements.txt`
- First run may take ~1 minute to download the model
- Subsequent runs use the cached model
""")

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ using Streamlit and Whisper")
