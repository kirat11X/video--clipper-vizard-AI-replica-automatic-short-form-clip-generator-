import json
import numpy as np
from pathlib import Path
# IMPORT VALIDATOR TO FUSE PIPELINE
from audio_validator import AudioValidator

class ClipSelector:
    # 2. Minimum clip length (FIX: GENRE-AWARE)
    MIN_CLIP_LEN = 10.0 
    MAX_CLIP_LEN = 45.0
    
    # 3. Thresholds (FIX: QUALITY ↔ QUANTITY TRADE)
    HARD_REJECT = 2.3   # (UPDATED) Lowered to 2.3 to increase recall
    SOFT_ACCEPT = 3.0   # (UPDATED) Lowered to 3.0 for preferred quality

    # Structural Guards (FIX: PREVENT GIANT CLIPS)
    ABS_MAX_CLIP_LEN = 60.0  # Hard cap for Shorts
    MAX_MERGE_GAP = 6.0      # Max silence/gap allowed between merged clips

    def __init__(self, workspace_dir="pipeline_data"):
        self.workspace_dir = Path(workspace_dir)
        self.transcription_path = self.workspace_dir / "transcription.json"
        self.analysis_path = self.workspace_dir / "audio_analysis.json"
        self.semantic_path = self.workspace_dir / "semantic_analysis.json"
        self.visual_path = self.workspace_dir / "visual_analysis.json"
        
        # Initialize Validator
        self.validator = AudioValidator(workspace_dir)
        
    def load_data(self):
        """Loads the computed data from previous steps."""
        if not self.transcription_path.exists() or not self.analysis_path.exists():
            raise FileNotFoundError("❌ Missing data. Run transcription and audio analysis first.")
            
        with open(self.transcription_path, 'r', encoding='utf-8') as f:
            self.transcript = json.load(f)
            
        with open(self.analysis_path, 'r', encoding='utf-8') as f:
            self.analysis = json.load(f)
            
        # Load Semantic Data (Optional but recommended)
        if self.semantic_path.exists():
            with open(self.semantic_path, 'r', encoding='utf-8') as f:
                self.semantics = json.load(f)
        else:
            print("⚠️ Semantic analysis not found. Using default neutral score.")
            self.semantics = []

        # Load Visual Data (Updated for Event Architecture)
        if self.visual_path.exists():
             with open(self.visual_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Support new Event format and legacy List format
                if isinstance(data, list):
                    self.visuals = data # Legacy
                    self.visual_events = []
                else:
                    self.visuals = data.get("time_series", [])
                    self.visual_events = data.get("events", [])
        else:
            print("⚠️ Visual analysis not found. Using default neutral score.")
            self.visuals = []
            self.visual_events = []
            
        print("✅ Data loaded: Transcript, Audio, Semantic, and Visual Analysis.")

    def find_word_at_time(self, time_in_seconds):
        words = self.transcript['words']
        for i, word in enumerate(words):
            if word['start'] <= time_in_seconds <= word['end']:
                return i, word
        
        closest_idx = -1
        min_diff = float('inf')
        for i, word in enumerate(words):
            if word['end'] <= time_in_seconds:
                diff = time_in_seconds - word['end']
                if diff < min_diff:
                    min_diff = diff
                    closest_idx = i
        
        if closest_idx != -1:
            return closest_idx, words[closest_idx]
        return 0, words[0] if words else None

    def find_sentence_bounds(self, time_in_seconds):
        for segment in self.transcript['segments']:
            if segment['start'] <= time_in_seconds <= segment['end']:
                return segment['start'], segment['end']
        return time_in_seconds, time_in_seconds + 5

    def _safe_score(self, item, *keys, default=0.5):
        """Return the first present numeric score from item[key] among keys."""
        for k in keys:
            if isinstance(item, dict) and k in item:
                try:
                    return float(item[k])
                except (TypeError, ValueError):
                    continue
        return float(default)

    def temporal_nms(self, clips, iou_threshold=0.35):
        """
        Applies Non-Maximum Suppression to remove overlapping clips.
        Keeps highest scoring clips.
        """
        if not clips:
            return []

        # Sort clips by score descending (highest score first)
        clips = sorted(clips, key=lambda x: x["score"], reverse=True)
        selected = []

        def iou(a, b):
            # Calculate Intersection
            start_max = max(a["start"], b["start"])
            end_min = min(a["end"], b["end"])
            inter = max(0, end_min - start_max)
            
            # Calculate Union
            duration_a = a["end"] - a["start"]
            duration_b = b["end"] - b["start"]
            union = duration_a + duration_b - inter
            
            return inter / union if union > 0 else 0

        for c in clips:
            # Select clip if it doesn't overlap significantly with already selected clips
            if all(iou(c, s) < iou_threshold for s in selected):
                selected.append(c)

        # Return sorted by time for chronological order
        return sorted(selected, key=lambda x: x["start"])

    def detect_smart_cuts(self):
        """
        Identifies potential viral clips based on excitement spikes and semantic coherence.
        """
        print("✂️  Detecting smart cuts...")
        candidates = []
        
        # 1. Identify Spikes (Audio)
        spikes = []
        for i in range(1, len(self.analysis)):
            curr = self.analysis[i]
            prev = self.analysis[i-1]
            delta = curr['excitement_score'] - prev['excitement_score']
            
            # 1. Spike detection (FIX: Widen net slightly)
            # Old: if delta >= 0.10 or curr['excitement_score'] >= 0.65:
            if delta >= 0.08 or curr['excitement_score'] >= 0.62:
                spikes.append(curr)

        print(f"   Found {len(spikes)} initial excitement spikes.")

        # 2. Expand Spikes to Clips
        for spike in spikes:
            center_time = spike['time']
            
            # Start: 10s before spike (buildup)
            # End: 5s after spike (reaction)
            raw_start = max(0, center_time - 10)
            raw_end = min(self.transcript['duration'], center_time + 5)
            
            # Snap to sentence boundaries
            start_sentence_start, _ = self.find_sentence_bounds(raw_start)
            _, end_sentence_end = self.find_sentence_bounds(raw_end)
            
            clip_start = start_sentence_start
            clip_end = end_sentence_end
            
            duration = clip_end - clip_start
            
            # 2. Minimum clip length check (FIX: GENRE-AWARE)
            if duration < self.MIN_CLIP_LEN:
                 # Try to extend backwards to capture setup
                 clip_start = max(0, clip_end - self.MIN_CLIP_LEN)
                 duration = clip_end - clip_start

            if duration > self.MAX_CLIP_LEN:
                # Truncate if too long, keeping the punchline (end)
                clip_start = clip_end - self.MAX_CLIP_LEN
                duration = self.MAX_CLIP_LEN

            # Validate Candidate
            candidate = {
                "start": round(clip_start, 2),
                "end": round(clip_end, 2),
                "duration": round(duration, 2),
                "trigger_time": center_time,
                "spike_score": spike['excitement_score']
            }
            
            # Step 2: Absolute safety guard (New)
            # Prevents malformed giant clips from entering scoring
            # FIX: Use ABS_MAX_CLIP_LEN instead of hardcoded 90.0
            if candidate["duration"] > self.ABS_MAX_CLIP_LEN:
                continue

            # Score and Filter
            scored_candidate = self.score_candidate(candidate)
            if scored_candidate:
                candidates.append(scored_candidate)

        # Merge overlapping clips
        merged_clips = self.merge_overlapping_clips(candidates)
        print(f"   Merged into {len(merged_clips)} clips before NMS.")
        
        # 3. Apply Temporal NMS (Selection Phase)
        final_clips = self.temporal_nms(merged_clips)
        
        # Note: No MAX_CLIPS cap applied per instruction (unlimited)
        
        print(f"✅ Found {len(final_clips)} valid clips after NMS and filtering.")
        return final_clips

    def score_candidate(self, clip):
        """
        Aggregates Audio, Semantic, and Visual scores.
        Applies Fix 3 (Hard/Soft thresholds) and Fix 4 (Entropy penalty).
        """
        start = clip['start']
        end = clip['end']
        
        # 1. Audio Score (Avg Excitement)
        audio_scores = [
            p['excitement_score'] for p in self.analysis 
            if start <= p['time'] <= end
        ]
        avg_audio = np.mean(audio_scores) if audio_scores else 0
        
        # 2. Semantic Score (Updated Aggregation)
        # FIX: Use weighted max to prevent dilution of punchlines
        semantic_scores = [
            self._safe_score(s, 'score', 'semantic_score', default=0.3) for s in self.semantics
            if (s['start'] >= start and s['end'] <= end) or (start <= s['start'] <= end)
        ]
        
        if semantic_scores:
            max_sem = np.max(semantic_scores)
            mean_sem = np.mean(semantic_scores)
            avg_semantic = (max_sem * 0.6) + (mean_sem * 0.4)
        else:
            # FIX: Lower neutral default (was 0.5) to avoid boosting weak clips
            avg_semantic = 0.3 
        
        # 3. Visual Score
        visual_scores = [
            self._safe_score(v, 'score', 'visual_score', default=0.5) for v in self.visuals
            if start <= v['time'] <= end
        ]
        avg_visual = np.mean(visual_scores) if visual_scores else 0.5
        
        # 4. Run Validator (Structure, Content, Entropy)
        val_result = self.validator.validate_clip({"start": start, "end": end})
        
        # Fix: Parse tuple return from validator (score, reason)
        entropy = 5.0 # Default healthy entropy
        val_reason = ""
        
        if isinstance(val_result, tuple):
            _, val_reason = val_result
            # Parse entropy from reason string "Ent:X.XX->..."
            if "Ent:" in val_reason:
                try:
                    # Extract raw entropy from "Ent:X.XX->..."
                    entropy_part = val_reason.split("Ent:")[1].split("->")[0]
                    entropy = float(entropy_part)
                except (IndexError, ValueError):
                    pass
        elif isinstance(val_result, dict):
             # Fallback if validator returns dict
            entropy = val_result.get('entropy', 5.0)
            val_reason = val_result.get('reason', "")
        
        # Base Score Calculation
        # Weights: Audio (40%), Semantic (30%), Visual (30%)
        base_score = (0.4 * avg_audio) + (0.3 * avg_semantic) + (0.3 * avg_visual)
        
        # Apply Multipliers
        final_score = base_score * 10 # Scale 0-10
        
        # 4. Validator entropy penalty (FIX: Less aggressive for comedy)
        penalty_mult = 1.0
        if entropy < 2.8:
            penalty_mult = 0.75 # Preserves storytelling/monologues
            
        final_score *= penalty_mult
        
        # Add visual density bonus (if events exist)
        event_count = len([e for e in self.visual_events if start <= e['time'] <= end])
        if event_count > 2:
            final_score += 0.5

        # 3. Final score threshold (FIX: Two-tier threshold)
        if final_score < self.HARD_REJECT:
            return None
            
        return {
            "start": start,
            "end": end,
            "duration": clip['duration'],
            "score": round(final_score, 2),
            "components": {
                "audio": round(avg_audio, 2),
                "semantic": round(avg_semantic, 2),
                "visual": round(avg_visual, 2),
                "entropy": round(entropy, 2)
            },
            "validation_reason": val_reason
        }

    def merge_overlapping_clips(self, clips):
        """Merges clips that overlap or are close, up to a structural limit."""
        if not clips:
            return []
            
        # Sort by start time
        clips.sort(key=lambda x: x['start'])
        
        merged = []
        current = clips[0]
        
        for next_clip in clips[1:]:
            # Calculate gap: next_start - current_end
            # Negative gap means overlap
            gap = next_clip['start'] - current['end']
            
            # FIX: Structural Guard Logic
            # 1. Allow merge if overlap OR small gap (MAX_MERGE_GAP)
            # 2. BUT only if resulting clip is within ABS_MAX_CLIP_LEN
            
            can_merge = False
            if gap < self.MAX_MERGE_GAP:
                potential_end = max(current['end'], next_clip['end'])
                potential_duration = potential_end - current['start']
                if potential_duration <= self.ABS_MAX_CLIP_LEN:
                    can_merge = True
                    
            if can_merge:
                current['end'] = max(current['end'], next_clip['end'])
                # Keep max score
                current['score'] = max(current['score'], next_clip['score'])
                
                current['duration'] = current['end'] - current['start']
                if "Merged" not in current.get('validation_reason', ""):
                    current['validation_reason'] = current.get('validation_reason', "") + " [Merged]"
                
                # Merge component scores
                for key in ['audio', 'semantic', 'visual', 'entropy']:
                    current['components'][key] = max(current['components'].get(key, 0), next_clip['components'].get(key, 0))
            else:
                merged.append(current)
                current = next_clip
        
        merged.append(current)
        return merged

    def save_clips(self, clips, output_filename="suggested_clips.json"):
        path = self.workspace_dir / output_filename
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(clips, f, indent=2)
        print(f"💾 Clip candidates saved to: {path}")

if __name__ == "__main__":
    selector = ClipSelector("pipeline_data")
    try:
        selector.load_data()
        clips = selector.detect_smart_cuts()
        selector.save_clips(clips)
    except Exception as e:
        print(f"⚠️ Error in selector: {e}")