##video-clipper
(subtitle: Vizard AI–style automatic short-form clip generator)
> **video-clipper** is an end-to-end, fully local system that automatically converts long-form videos (YouTube videos, podcasts, stand-up comedy, streams) into short-form vertical clips optimized for TikTok, Instagram Reels, and YouTube Shorts — without relying on any external SaaS tools.
It replicates the **core functionality of Vizard AI**, but is:
* **fully open-source**
* **modular**
* **transparent**
* **tunable by design**


## 🎬 video-clipper (Vizard AI Replica)

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
| Hinglish support | ✅             | ❌         |
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
