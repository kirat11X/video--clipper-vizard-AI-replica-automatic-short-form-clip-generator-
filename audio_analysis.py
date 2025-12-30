import librosa
import numpy as np
import json
from pathlib import Path

class AudioAnalyzer:
    def __init__(self):
        print("🎛️  Initializing Audio Analyzer (Librosa)...")

    def analyze_excitement(self, audio_path, window_size=0.5, vol_weight=0.6, pitch_weight=0.4):
        """
        Analyzes audio for excitement levels using volume and pitch.
        
        Args:
            audio_path (Path): Path to the audio file.
            window_size (float): Analysis window size in seconds.
            vol_weight (float): Weight for volume in excitement calculation (default 0.6).
            pitch_weight (float): Weight for pitch in excitement calculation (default 0.4).
        """
        print(f"🌊 Loading audio: {audio_path}")
        y, sr = librosa.load(audio_path, sr=None)

        # ---- RMS (coarse, clip-level) ----
        rms_hop = int(sr * window_size)
        rms = librosa.feature.rms(
            y=y,
            frame_length=2048,
            hop_length=rms_hop
        )[0]

        # ---- Pitch (fine-grained, stable) ----
        pitch_hop = int(sr * 0.01)  # 10ms
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr,
            frame_length=2048,
            hop_length=pitch_hop
        )

        f0 = np.nan_to_num(f0)

        # ---- Aggregate pitch into RMS windows ----
        pitch_per_window = []
        samples_per_window = rms_hop

        for i in range(len(rms)):
            start = int(i * samples_per_window / pitch_hop)
            end = int((i + 1) * samples_per_window / pitch_hop)
            # Handle potential index out of bounds for the last frame
            if start >= len(f0):
                pitch_slice = []
            else:
                pitch_slice = f0[start:min(end, len(f0))]

            pitch_per_window.append(
                np.mean(pitch_slice) if len(pitch_slice) > 0 else 0
            )

        pitch_per_window = np.array(pitch_per_window)

        # ---- Normalize ----
        # RMS Normalization (0-1)
        rms_norm = (rms - rms.min()) / (rms.max() - rms.min() + 1e-6)
        
        # Pitch Normalization (Percentile Clipped)
        # Fix 1: Clip pitch outliers (5th to 95th percentile) before normalizing
        # This prevents one scream from compressing the rest of the data.
        p_low, p_high = np.percentile(pitch_per_window, [5, 95])
        pitch_clipped = np.clip(pitch_per_window, p_low, p_high)
        
        pitch_norm = (
            (pitch_clipped - pitch_clipped.min()) /
            (pitch_clipped.max() - pitch_clipped.min() + 1e-6)
        )

        # ---- Final Scores ----
        results = []
        times = np.arange(len(rms_norm)) * window_size

        for i, t in enumerate(times):
            # Fix 3: Use configurable weights instead of hardcoded values
            excitement = vol_weight * rms_norm[i] + pitch_weight * pitch_norm[i]
            
            # Fix 2: Explicitly model silence (inverse of volume)
            # Crucial for detecting pauses before punchlines
            silence_score = 1.0 - rms_norm[i]
            
            results.append({
                "time": round(float(t), 2),
                "volume_score": round(float(rms_norm[i]), 4),
                "pitch_score": round(float(pitch_norm[i]), 4),
                "silence_score": round(float(silence_score), 4), # Added silence metric
                "excitement_score": round(float(excitement), 4)
            })

        print(f"✅ Analysis complete. Generated {len(results)} points.")
        return results

    def detect_spikes(self, data, spike_threshold=0.25):
        """ Legacy method, keeping for compatibility if needed. """
        spikes = []
        for i in range(1, len(data)):
            delta = data[i]["excitement_score"] - data[i-1]["excitement_score"]
            if delta > spike_threshold:
                spikes.append({
                    "time": data[i]["time"],
                    "delta": round(delta, 3)
                })
        return spikes

    def detect_audio_spikes(self, data, min_delta=0.15):
        """
        Optimized detection of sudden emotional spikes.
        Uses list comprehension for cleaner execution.
        """
        return [
            {
                "time": data[i]["time"],
                "delta": round(
                    data[i]["excitement_score"] - data[i-1]["excitement_score"], 3
                ),
                "excitement": data[i]["excitement_score"]
            }
            for i in range(1, len(data))
            if data[i]["excitement_score"] - data[i-1]["excitement_score"] >= min_delta
        ]

    def save_to_json(self, data, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"💾 Audio analysis saved to: {output_path}")

if __name__ == "__main__":
    # Setup paths
    WORKSPACE_DIR = Path("pipeline_data")
    AUDIO_FILE = WORKSPACE_DIR / "audio" / "audio.wav"
    OUTPUT_FILE = WORKSPACE_DIR / "audio_analysis.json"
    
    if not AUDIO_FILE.exists():
        print("❌ Audio file not found. Run previous steps first.")
    else:
        analyzer = AudioAnalyzer()
        # Analyze in 0.5 second chunks for granularity
        # Weights can now be tuned here if needed (e.g., vol_weight=0.7 for podcasts)
        analysis_data = analyzer.analyze_excitement(
            AUDIO_FILE, 
            window_size=0.5,
            vol_weight=0.6,
            pitch_weight=0.4
        )
        
        # Optional: Run the new spike detection to test it
        spikes = analyzer.detect_audio_spikes(analysis_data)
        print(f"⚡ Detected {len(spikes)} emotional spikes.")
        
        analyzer.save_to_json(analysis_data, OUTPUT_FILE)