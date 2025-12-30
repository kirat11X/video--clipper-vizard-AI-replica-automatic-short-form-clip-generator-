import json
from pathlib import Path
import os
import sys
from typing import Iterable
import ctypes
import ctypes.util


def _cudnn_available() -> bool:
    """Best-effort check that cuDNN shared libs are available for CUDA inference.

    Some GPU backends (e.g., CTranslate2) may abort the process if cuDNN cannot be
    loaded at runtime. We probe for the presence of cuDNN libraries up-front so we
    can fall back to CPU instead of crashing.
    """
    candidates = []

    # The crash you hit is specifically about libcudnn_ops.* not being loadable.
    # Require that library (not just libcudnn.so) to be present.
    for name in ("cudnn_ops",):
        lib = ctypes.util.find_library(name)
        if lib:
            candidates.append(lib)

    # Add common SONAMEs as fallbacks.
    candidates.extend([
        "libcudnn_ops.so.9.1.0",
        "libcudnn_ops.so.9.1",
        "libcudnn_ops.so.9",
        "libcudnn_ops.so",
    ])

    seen = set()
    for lib in candidates:
        if lib in seen:
            continue
        seen.add(lib)
        try:
            ctypes.CDLL(lib)
            return True
        except Exception:
            continue

    return False


def _iter_nvidia_package_lib_dirs() -> Iterable[str]:
    """Yield candidate directories that contain NVIDIA-provided shared libraries.

    When installed via pip, NVIDIA CUDA/cuDNN wheels place .so files under
    site-packages/nvidia/*/lib. Those paths are not always on the dynamic loader
    search path, which can cause hard aborts when GPU backends dlopen cuDNN.
    """
    try:
        import nvidia  # type: ignore

        nvidia_root = os.path.dirname(nvidia.__file__)
        # Typical layout: site-packages/nvidia/<component>/lib
        for entry in os.listdir(nvidia_root):
            lib_dir = os.path.join(nvidia_root, entry, "lib")
            if os.path.isdir(lib_dir):
                yield lib_dir
    except Exception:
        return


def _prepend_to_ld_library_path(paths: Iterable[str]) -> None:
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    existing_parts = [p for p in existing.split(":") if p]

    new_parts = []
    seen = set(existing_parts)
    for p in paths:
        if p and p not in seen:
            new_parts.append(p)
            seen.add(p)

    if new_parts:
        os.environ["LD_LIBRARY_PATH"] = ":".join(new_parts + existing_parts)


def _configure_cuda_runtime() -> None:
    """Best-effort CUDA runtime configuration for faster-whisper GPU mode."""
    _prepend_to_ld_library_path(_iter_nvidia_package_lib_dirs())

    # Preload cuDNN libraries (absolute paths) to avoid dlopen search path issues.
    try:
        import nvidia.cudnn  # type: ignore

        cudnn_root = os.path.dirname(nvidia.cudnn.__file__)
        cudnn_lib_dir = os.path.join(cudnn_root, "lib")
        if os.path.isdir(cudnn_lib_dir):
            # Load the major-versioned shared libs first for best compatibility.
            names = sorted(
                (f for f in os.listdir(cudnn_lib_dir) if f.startswith("libcudnn") and ".so" in f),
                key=lambda s: (0 if s.endswith(".so.9") else 1, s),
            )
            for fname in names:
                candidate = os.path.join(cudnn_lib_dir, fname)
                try:
                    ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
                except Exception:
                    continue
    except Exception:
        pass

class AudioTranscriber:
    def __init__(self, model_size="large-v3", device="cpu", compute_type="int8"):
        """
        Initializes the Whisper model.
        
        Args:
            model_size (str): 'tiny', 'base', 'small', 'medium', 'large-v2'.
                              'small' is a good balance for testing.
                              'medium' or ';' recommended for final production.
            device (str): 'cuda' if you have an NVIDIA GPU, else 'cpu'.
            compute_type (str): 'float16' for GPU, 'int8' for CPU.
        """
        if device == "cuda":
            _configure_cuda_runtime()

        print(f"🧠 Loading Whisper model: {model_size} on {device}...")
        try:
            # Import here so LD_LIBRARY_PATH changes (above) can take effect before
            # any GPU backend attempts to dlopen CUDA/cuDNN libraries.
            from faster_whisper import WhisperModel
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            print("✅ Model loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def transcribe(self, audio_path):
        """
        Transcribes audio and extracts word-level timestamps.
        
        Args:
            audio_path (str): Path to the .wav file.
            
        Returns:
            dict: Structured data containing full text and word segments.
        """
        print(f"🎙️  Transcribing audio: {audio_path} (this may take a moment)...")
        
        # Run transcription with word_timestamps=True (CRITICAL for clipping)
        segments, info = self.model.transcribe(
            str(audio_path), 
            word_timestamps=True,
            beam_size=5
        )

        # Process generator into a list
        # We need to iterate through segments to get the data
        full_transcript = []
        words_data = []

        print(f"   Detected language: {info.language} (Probability: {info.language_probability:.2f})")

        for segment in segments:
            # Store full sentence segments
            full_transcript.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })
            
            # Store individual words (Critical for 'kinetic typography' and precise cuts)
            if segment.words:
                for word in segment.words:
                    words_data.append({
                        "word": word.word.strip(),
                        "start": word.start,
                        "end": word.end,
                        "score": word.probability
                    })

        result = {
            "language": info.language,
            "language_probability": info.language_probability, # Added metadata
            "duration": info.duration,
            "segments": full_transcript,
            "words": words_data
        }
        
        print(f"✅ Transcription complete! Processed {len(words_data)} words.")
        return result

    def save_to_json(self, data, output_path):
        """Helper to save results to JSON for the next step in the pipeline."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"💾 Transcript saved to: {output_path}")

if __name__ == "__main__":
    # 1. Setup paths (Matching the folder structure from Step 1)
    # Assuming you ran video_ingestion.py and have data in 'pipeline_data'
    WORKSPACE_DIR = Path("pipeline_data")
    AUDIO_FILE = WORKSPACE_DIR / "audio" / "audio.wav"
    OUTPUT_FILE = WORKSPACE_DIR / "transcription.json"
    
    if not AUDIO_FILE.exists():
        print(f"❌ Audio file not found at {AUDIO_FILE}")
        print("   Did you run video_ingestion.py first?")
    else:
        # 2. Initialize Transcriber
        # NOTE: CUDA transcription requires a working CUDA + cuDNN runtime.
        # If cuDNN is missing/misconfigured, the backend can abort the process.
        _configure_cuda_runtime()
        use_cuda = _cudnn_available()
        device = "cuda" if use_cuda else "cpu"
        compute_type = "float16" if use_cuda else "int8"

        if use_cuda:
            print("⚡ Using CUDA (cuDNN detected).")
        else:
            print("⚠️  cuDNN not detected; falling back to CPU to avoid a crash.")
            print("   To enable GPU, install a compatible cuDNN (e.g., libcudnn_ops.so.9*) and ensure it's on LD_LIBRARY_PATH.")

        transcriber = AudioTranscriber(model_size="large-v3", device=device, compute_type=compute_type)

        # 3. Run Transcription
        transcript_data = transcriber.transcribe(AUDIO_FILE)
        
        # 4. Save
        transcriber.save_to_json(transcript_data, OUTPUT_FILE)