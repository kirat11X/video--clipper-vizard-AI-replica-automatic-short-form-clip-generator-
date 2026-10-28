
## 🎬 video-clipper (Vizard AI Replica **Automatic Short-form Clip Generator**)
> **video-clipper** is an end-to-end, fully local system that automatically converts long-form videos (YouTube videos, podcasts, stand-up comedy, streams) into short-form vertical clips optimized for TikTok, Instagram Reels, and YouTube Shorts — without relying on any external SaaS tools.
It replicates the **core functionality of Vizard AI**, but is:
* **fully open-source**
* **modular**
* **transparent**
* **tunable by design**
**video-clipper** is a multimodal video understanding pipeline that automatically:

* analyzes long-form videos,
* detects high-engagement moments,
* selects non-overlapping clips,
* and renders vertical short-form videos with captions.

Unlike commercial tools, this system is **built from scratch** using open-source components and exposes **every scoring and decision step** for full control and experimentation.

---

## 🚀 Key Features

### 🔹 End-to-End Automation

* YouTube video ingestion
* Audio, text, and visual analysis
* Intelligent clip selection
* Vertical video rendering with captions

### 🔹 Multimodal Intelligence

* **Audio signals**: volume, pitch, silence, excitement
* **Semantic signals**: Hinglish-aware NLP, novelty detection, questions, punchlines
* **Visual signals**: face presence, motion, scene dynamics

### 🔹 Clip Selection Engine

* Recall-first candidate detection
* Structural guards (duration caps, merge limits)
* Non-Maximum Suppression (NMS) for overlap removal
* Tunable thresholds for quality vs quantity

### 🔹 Shorts-Ready Rendering

* 9:16 vertical format
* Background blur + foreground crop
* Loudness normalization
* Burned-in captions from Whisper timestamps
* GPU-accelerated encoding (NVENC)

---

## 🏗️ Architecture Overview

```text
YouTube URL
   ↓
Video Download (yt-dlp)
   ↓
Audio Extraction (ffmpeg)
   ↓
Whisper Transcription (word-level)
   ↓
Signal Extraction
   ├── Audio Analysis
   ├── Semantic Analysis
   └── Visual Analysis
   ↓
Clip Selector
   ├── Scoring
   ├── Structural Guards
   └── Temporal NMS
   ↓
Video Renderer
   ↓
Upload-Ready Shorts
```

---

## 🧪 Supported Content Types

* 🎙️ Podcasts & interviews
* 🎤 Stand-up comedy (English / Hindi / Hinglish)
* 🎮 Gaming streams
* 📚 Educational content
* 📰 Commentary & opinion videos

---

## 🛠️ Tech Stack

* **Python 3.10+**
* **yt-dlp** – video ingestion
* **ffmpeg** – audio/video processing
* **faster-whisper** – transcription
* **librosa / numpy** – audio analysis
* **OpenCV / MediaPipe** – visual analysis
* **FFmpeg NVENC** – GPU rendering

---

## 🖥️ Clipper Studio (web UI)

```bash
streamlit run app.py        # http://localhost:8501
```

A local Streamlit app over the whole pipeline - upload a file or paste a URL, then:

1. **Source & Analyze** - pick the Whisper model / language, toggle the hook LLM, cap the download quality, and run the analysis as a background job with live progress and a cancel button. Finished steps are skipped, so tweaking and re-running is cheap.
2. **Signals & Transcript** - interactive timeline of audio/face/motion signals with your picks overlaid, plus a searchable transcript.
3. **Score & Clips** - choose a genre preset, edit every signal weight (and save your own presets), pick a platform, set minimum score / clip length / spacing / overlap / max clips, and see *why* each moment scored the way it did. Trim, rename, add or drop clips in the final list.
4. **Render & Export** - output size (9:16, 4:5, 1:1), crop-to-fill or blurred-background framing with a crop-focus slider, caption font / size / colour / position, padding, loudness, encoder; quick preview of any clip, then render and download (single files or a zip).

Each project lives in its own folder under `projects/`; the original `pipeline_data/` run shows up as an existing project. Docker: `docker compose up --build ui` (add `-f docker-compose.gpu.yml` for an NVIDIA GPU).

## ⚙️ Project Structure

```text
video-clipper/
├── pipeline_data/
│   ├── video/
│   ├── audio/
│   ├── frames/
│   └── outputs/
├── audio_analysis.py
├── text_analysis.py
├── visual_analysis.py
├── clip_selector.py
├── video_renderer.py
├── audio_validator.py
├── requirements.txt
└── README.md
```

---

## 🧩 Design Philosophy

* **Signals over black-box models**
* **Structure before thresholds**
* **Recall first, precision later**
* **Explainable decisions at every step**

This makes the system ideal for:

* research
* experimentation
* learning multimodal AI
* building creator tools

---

## 📊 Why This Is Different from SaaS Tools

| Feature          | video-clipper | Vizard AI |
| ---------------- | ------------- | --------- |
| Fully local      | ✅             | ❌         |
| Open-source      | ✅             | ❌         |
| Custom scoring   | ✅             | ❌         |
| Hinglish(Hindi+
  English) support | ✅             | ❌         |
| Debuggable       | ✅             | ❌         |

---

## 🚧 Current Status

* ✅ Core pipeline complete
* ✅ Clip selection stable
* ✅ Rendering production-ready
* ⏳ Platform presets (TikTok / Reels / Shorts)
* ⏳ Optional hook-LLM integration

---

## 🧠 Future Improvements

* Platform-specific scoring profiles
* Face-tracking smart crops
* Feedback-driven weight learning
* Auto-upload integrations
* Web UI / API layer

---
⚠️ IMPORTANT INSTALL NOTES
FFmpeg (required, system-level)
sudo apt install ffmpeg


or on Fedora:

sudo dnf install ffmpeg

GPU Support (Optional, Recommended)

For NVIDIA GPUs:

pip install torch --index-url https://download.pytorch.org/whl/cu121


If not using GPU, CPU torch works fine.

## ⚠️ Disclaimer

This project is an **educational and experimental reimplementation** inspired by tools like Vizard AI.
It is **not affiliated** with or endorsed by Vizard AI.

---

## ⭐ Why You Might Star This Repo

* You want to understand **how viral clipping actually works**
* You’re building creator tools
* You care about explainable AI systems
* You want a serious open-source alternative

---
