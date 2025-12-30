import json
import numpy as np
from pathlib import Path
from pyAudioAnalysis import ShortTermFeatures
from pyAudioAnalysis import audioBasicIO

class AudioValidator:
    def __init__(self, workspace_dir="pipeline_data"):
        self.workspace_dir = Path(workspace_dir)
        self.audio_path = self.workspace_dir / "audio" / "audio.wav"
        
        # We no longer use binary thresholds for rejection inside the validator.
        # We use scaling factors to return a confidence score.
        
    def extract_features(self, start_time, end_time):
        """
        Extracts mid-level features for a specific time window using pyAudioAnalysis.
        """
        if not self.audio_path.exists():
            return None

        # Load audio (cached reading would be better for prod, but this is fine for now)
        try:
            fs, s = audioBasicIO.read_audio_file(str(self.audio_path))
        except Exception:
            return None
        
        # Convert time to samples
        start_sample = int(start_time * fs)
        end_sample = int(end_time * fs)
        
        # Safety check for duration
        if end_sample <= start_sample:
            return None
            
        # Slice audio for the clip
        s_clip = s[start_sample:end_sample]
        
        # Check if clip is stereo, convert to mono if necessary
        if len(s_clip.shape) > 1:
            s_clip = np.mean(s_clip, axis=1)

        # Extract features
        # Window: 50ms, Step: 25ms (standard for speech analysis)
        win_size = 0.050 * fs
        step_size = 0.025 * fs
        
        try:
            # F: Feature Matrix, f_names: Feature Names
            F, f_names = ShortTermFeatures.feature_extraction(
                s_clip, 
                fs, 
                win_size, 
                step_size
            )
            return F, f_names
        except Exception:
            return None

    def validate_clip(self, clip):
        """
        Validates a single clip candidate.
        Returns a granular confidence score (0.0 - 1.0) and a reason.
        """
        start = clip['start']
        end = clip['end']
        duration = end - start
        
        if duration < 3.0:
            return 0.0, "Too short (<3s)"
            
        result = self.extract_features(start, end)
        if result is None:
            return 0.0, "Feature extraction failed"
            
        F, f_names = result

        # --- Feature Indices in pyAudioAnalysis ---
        # 0: Zero Crossing Rate
        # 2: Entropy of Energy
        
        # 1. Analyze Energy Entropy (Measure of sudden changes / excitement)
        entropy_seq = F[2, :]
        avg_entropy = np.mean(entropy_seq)
        
        # 2. Analyze Zero Crossing Rate (Voice activity proxy)
        zcr_seq = F[0, :]
        avg_zcr = np.mean(zcr_seq)
        
        # --- FIX 4: Scaled Scoring Logic ---
        # Instead of binary thresholds, we scale the values to a 0-1 range.
        # Normal speech entropy is ~2.5-3.5. We saturate at 4.0.
        entropy_score = min(avg_entropy / 4.0, 1.0)
        
        # Normal speech ZCR is 0.05-0.15. We saturate at 0.15.
        zcr_score = min(avg_zcr / 0.15, 1.0)
        
        # Weighted Confidence
        # Entropy (Dynamics) is 60%, ZCR (Content) is 40%
        confidence = (0.6 * entropy_score) + (0.4 * zcr_score)
        
        reason = f"Ent:{avg_entropy:.2f}->{entropy_score:.2f}, ZCR:{avg_zcr:.2f}->{zcr_score:.2f}"
        return round(confidence, 3), reason

    def run_validation(self, clips_file="suggested_clips.json"):
        """
        Standalone runner (legacy mode), mostly for testing.
        Real usage is now inside clip_selector.py.
        """
        input_path = self.workspace_dir / clips_file
        output_path = self.workspace_dir / "validated_clips.json"
        
        if not input_path.exists():
            print(f"❌ Clips file not found: {input_path}")
            return

        with open(input_path, 'r', encoding='utf-8') as f:
            clips = json.load(f)
            
        print(f"🕵️  Validating {len(clips)} clips...")
        
        validated_clips = []
        
        for clip in clips:
            score, reason = self.validate_clip(clip)
            clip['validation_score'] = score
            clip['validation_reason'] = reason
            
            if score >= 0.5:
                validated_clips.append(clip)
                print(f"   ✅ {score}: {clip.get('trigger_word', '?')} ({reason})")
            else:
                print(f"   ❌ {score}: {clip.get('trigger_word', '?')} ({reason})")
                
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(validated_clips, f, indent=2)

if __name__ == "__main__":
    validator = AudioValidator("pipeline_data")
    validator.run_validation()