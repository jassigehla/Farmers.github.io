# Build an AI Punjabi Music App (Gurmukhi + Hindi Lyrics + Beats) on Windows 11 Pro

This guide takes you from zero to a working prototype app.

---

## 1) What you are building

You will build a local web app that can:
1. Generate song lyrics in **Punjabi (Gurmukhi)** and **Hindi (Devanagari)**.
2. Generate a musical track (melody + texture).
3. Generate a separate beat/percussion layer.
4. Create a Hindi/Punjabi vocal line with TTS (prototype singing-like output).
5. Mix all audio into one exportable `.wav` file.

Tech stack used:
- **Python 3.10/3.11**
- **Streamlit** for UI
- **OpenAI API** for lyrics generation
- **Audiocraft MusicGen** for music generation
- **Coqui TTS** for Hindi/Punjabi-friendly voice synthesis
- **Pydub + ffmpeg** for audio mixing

> Note: True high-quality AI singing in Punjabi is still advanced; this tutorial gives a practical production-style prototype.

---

## 2) Prerequisites (Windows 11 Pro)

### 2.1 Install Python
1. Download Python from: https://www.python.org/downloads/
2. Install Python 3.10 or 3.11.
3. During install, check **Add Python to PATH**.

Verify in PowerShell:

```powershell
python --version
pip --version
```

### 2.2 Install Git
Download: https://git-scm.com/download/win

Verify:

```powershell
git --version
```

### 2.3 Install ffmpeg
1. Download ffmpeg build (e.g., from gyan.dev builds).
2. Extract to `C:\ffmpeg`.
3. Add `C:\ffmpeg\bin` to Environment Variables `Path`.

Verify:

```powershell
ffmpeg -version
```

### 2.4 (Recommended) Install Visual C++ Build Tools
Some Python packages need C++ tools.
Install from Visual Studio Build Tools installer.

---

## 3) Create project folder

Open PowerShell and run:

```powershell
mkdir C:\ai-punjabi-music-app
cd C:\ai-punjabi-music-app
```

Create and activate virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If script execution is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

---

## 4) Create project files

Run:

```powershell
ni requirements.txt
ni .env
ni app.py
mkdir outputs
```

Now paste the following content into each file.

---

## 5) `requirements.txt`

```txt
streamlit==1.37.1
openai==1.40.6
python-dotenv==1.0.1
torch==2.3.1
torchaudio==2.3.1
audiocraft==1.3.0
TTS==0.22.0
pydub==0.25.1
soundfile==0.12.1
numpy==1.26.4
```

Install dependencies:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 6) `.env`

```env
OPENAI_API_KEY=your_openai_api_key_here
```

How to get key:
1. Go to OpenAI platform.
2. Create API key.
3. Paste into `.env`.

---

## 7) `app.py` (full app code)

```python
import os
import uuid
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torchaudio
from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment
from TTS.api import TTS

# MusicGen imports
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write


# -------------------------
# Setup
# -------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY missing in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------
# AI helper functions
# -------------------------
def generate_lyrics(theme: str, mood: str, style: str, length_lines: int):
    """
    Generates bilingual lyrics:
    - Punjabi in Gurmukhi
    - Hindi in Devanagari
    - Roman transliteration (optional singing reference)
    """
    system_prompt = (
        "You are an expert Punjabi + Hindi lyricist. "
        "Write poetic, musical lyrics suitable for songs. "
        "Return exact sections with headings: "
        "[PUNJABI_GURMUKHI], [HINDI_DEVANAGARI], [ROMAN]."
    )

    user_prompt = f"""
Theme: {theme}
Mood: {mood}
Style/Genre: {style}
Length: around {length_lines} lines.

Rules:
- Punjabi section must be in Gurmukhi script only.
- Hindi section must be in Devanagari script only.
- Roman section should be singable transliteration.
- Keep hooks/chorus catchy.
- Avoid explicit content.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.9,
    )

    return response.choices[0].message.content


def generate_music(prompt: str, duration_sec: int, out_stem: str):
    """
    Generate musical background with MusicGen.
    """
    model = MusicGen.get_pretrained("facebook/musicgen-small")
    model.set_generation_params(duration=duration_sec)

    wav = model.generate([prompt])  # shape: [B, C, T]
    out_path = OUTPUT_DIR / f"{out_stem}_music"
    audio_write(str(out_path), wav[0].cpu(), model.sample_rate, strategy="loudness")

    return str(out_path) + ".wav"


def generate_beat(prompt: str, duration_sec: int, out_stem: str):
    """
    Generate beat/percussion loop track.
    """
    beat_prompt = f"{prompt}, strong drums, clean percussion loop, club-ready beat"

    model = MusicGen.get_pretrained("facebook/musicgen-small")
    model.set_generation_params(duration=duration_sec)

    wav = model.generate([beat_prompt])
    out_path = OUTPUT_DIR / f"{out_stem}_beat"
    audio_write(str(out_path), wav[0].cpu(), model.sample_rate, strategy="loudness")

    return str(out_path) + ".wav"


def synthesize_vocal(text: str, out_stem: str):
    """
    Synthesize vocal line (spoken/sing-like prototype) with Coqui TTS.
    For better Hindi/Punjabi support, you can switch to multilingual models.
    """
    # Example multilingual model (downloads first time)
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(DEVICE)

    # This model supports speaker cloning if speaker_wav is provided.
    # For quick prototype, default voice is used.
    out_path = OUTPUT_DIR / f"{out_stem}_vocal.wav"

    tts.tts_to_file(
        text=text,
        language="hi",  # Hindi; Punjabi can be approximated, or switch model as needed
        file_path=str(out_path),
    )

    return str(out_path)


def normalize(seg: AudioSegment, target_dbfs=-16.0):
    change = target_dbfs - seg.dBFS
    return seg.apply_gain(change)


def mix_tracks(music_path: str, beat_path: str, vocal_path: str, out_stem: str):
    """
    Mix 3 tracks with level balancing.
    """
    music = AudioSegment.from_file(music_path)
    beat = AudioSegment.from_file(beat_path)
    vocal = AudioSegment.from_file(vocal_path)

    # Match lengths (trim or pad)
    max_len = max(len(music), len(beat), len(vocal))

    def fit(seg):
        if len(seg) < max_len:
            return seg + AudioSegment.silent(duration=max_len - len(seg))
        return seg[:max_len]

    music = fit(music)
    beat = fit(beat)
    vocal = fit(vocal)

    # Balance levels
    music = normalize(music - 5)   # slightly lower
    beat = normalize(beat - 3)
    vocal = normalize(vocal + 2)   # vocal slightly louder

    mixed = music.overlay(beat).overlay(vocal)

    out_path = OUTPUT_DIR / f"{out_stem}_final_mix.wav"
    mixed.export(out_path, format="wav")
    return str(out_path)


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Punjabi AI Music Studio", layout="wide")
st.title("🎵 Punjabi AI Music Studio")
st.caption("Generate Punjabi + Hindi lyrics, AI music, beats, and final mix.")

with st.sidebar:
    st.header("Song Controls")
    theme = st.text_input("Theme", "Ishq in rainy season")
    mood = st.selectbox("Mood", ["Romantic", "Sad", "Party", "Motivational", "Sufi"])
    style = st.selectbox("Style", ["Punjabi Pop", "Folk", "Bhangra", "Lo-fi", "Bollywood"])
    duration = st.slider("Music duration (sec)", 10, 60, 20, 5)
    lines = st.slider("Lyrics length (lines)", 6, 24, 12, 2)

col1, col2 = st.columns(2)

with col1:
    if st.button("1) Generate Lyrics"):
        with st.spinner("Writing lyrics..."):
            lyrics = generate_lyrics(theme, mood, style, lines)
            st.session_state["lyrics"] = lyrics

    if "lyrics" in st.session_state:
        st.subheader("Generated Lyrics")
        st.text_area("Lyrics Output", st.session_state["lyrics"], height=350)

with col2:
    if st.button("2) Generate Full Song"):
        if "lyrics" not in st.session_state:
            st.warning("Please generate lyrics first.")
        else:
            with st.spinner("Generating music, beat, vocals, and mix..."):
                run_id = str(uuid.uuid4())[:8]

                music_prompt = f"{style}, {mood}, Punjabi commercial track, rich melody, studio quality"
                beat_prompt = f"{style}, {mood}, Punjabi dhol + modern drums"

                music_path = generate_music(music_prompt, duration, run_id)
                beat_path = generate_beat(beat_prompt, duration, run_id)

                # Use short lyrical text for TTS vocal (first lines)
                vocal_text = st.session_state["lyrics"][:400]
                vocal_path = synthesize_vocal(vocal_text, run_id)

                final_mix_path = mix_tracks(music_path, beat_path, vocal_path, run_id)

                st.success("Song generated!")
                st.audio(music_path)
                st.audio(beat_path)
                st.audio(vocal_path)
                st.audio(final_mix_path)

                with open(final_mix_path, "rb") as f:
                    st.download_button(
                        "Download Final Mix",
                        data=f,
                        file_name=Path(final_mix_path).name,
                        mime="audio/wav",
                    )

st.markdown("---")
st.markdown("### Notes")
st.markdown(
    "- First run may be slow because AI models are downloaded.\n"
    "- For better quality, use GPU (NVIDIA + latest CUDA-compatible PyTorch).\n"
    "- You can replace TTS with a dedicated singing model for pro output."
)
```

---

## 8) Run the app

In PowerShell (inside project folder with venv activated):

```powershell
streamlit run app.py
```

Then open browser URL shown by Streamlit (usually `http://localhost:8501`).

---

## 9) How to use the app

1. Enter **Theme**, select **Mood** and **Style**.
2. Click **Generate Lyrics**.
3. Verify Gurmukhi + Hindi text.
4. Click **Generate Full Song**.
5. Listen to individual tracks and final mix.
6. Click **Download Final Mix**.

---

## 10) Recommended improvements (next level)

1. **Real singing model** integration (instead of standard TTS).
2. Add **chord progression control** and BPM selector.
3. Add **hook + verse + bridge structure** prompts.
4. Add **karaoke subtitle export** (`.srt`).
5. Add **database** to save projects (SQLite/PostgreSQL).
6. Build API backend (FastAPI) + React frontend for team collaboration.

---

## 11) Common issues and fixes

### Issue: `ffmpeg not found`
Fix Path variable and restart terminal, then run:

```powershell
ffmpeg -version
```

### Issue: `torch` installation failure
Use Python 3.10/3.11 and upgrade pip:

```powershell
python -m pip install --upgrade pip
```

### Issue: GPU not detected
Check CUDA-compatible PyTorch install from official PyTorch site and NVIDIA driver update.

### Issue: Slow generation
- First run downloads models.
- Use shorter duration (10–15 sec) while testing.
- Use GPU if available.

---

## 12) Productionization checklist

1. Add user authentication.
2. Add rate limits and API quotas.
3. Store generated assets in cloud storage.
4. Add moderation/safety filters for lyrics.
5. Add legal policy for copyrighted style prompts.
6. Add queue workers for long-running generation.

---

## 13) Quick command recap (copy-paste)

```powershell
mkdir C:\ai-punjabi-music-app
cd C:\ai-punjabi-music-app
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
ni requirements.txt
ni .env
ni app.py
mkdir outputs
pip install -r requirements.txt
streamlit run app.py
```

You now have an end-to-end Windows 11 AI Punjabi music generator prototype with bilingual lyrics and beats.
