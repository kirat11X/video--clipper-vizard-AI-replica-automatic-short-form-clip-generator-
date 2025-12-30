import json
import csv
import argparse
import numpy as np
from pathlib import Path
from copy import deepcopy

# Import ranking logic directly to simulate scoring without re-running full pipeline
from rank_clips import load_weights, read_windows, score_windows, apply_cutoff, enforce_spacing

class GenreDetector:
    """
    Lightweight rule-based genre inference engine.
    Uses early-stage signals (first 2-4 mins) to classify video content.
    """
    def __init__(self, workspace_dir):
        self.workspace_dir = Path(workspace_dir)
        
    def detect(self):
        print("🕵️  Detecting genre from signals...")
        
        # Load raw data
        try:
            with open(self.workspace_dir / "audio_analysis.json") as f:
                audio = json.load(f)
            with open(self.workspace_dir / "semantic_analysis.json") as f:
                semantics = json.load(f)
            with open(self.workspace_dir / "visual_analysis.json") as f:
                visuals_raw = json.load(f)
                if isinstance(visuals_raw, list): visuals = visuals_raw
                else: visuals = visuals_raw.get("time_series", [])
        except Exception as e:
            print(f"⚠️  Could not load all signals for detection: {e}")
            return "podcast" # Default fallback

        # --- EXTRACT METRICS (First 180s / 3 mins) ---
        limit_time = 180.0
        
        # Audio Metrics
        audio_slice = [x for x in audio if x['time'] < limit_time]
        if not audio_slice: return "podcast"
        
        avg_vol = sum(x['volume_score'] for x in audio_slice) / len(audio_slice)
        excitement_spikes = sum(1 for x in audio_slice if x['excitement_score'] > 0.7)
        
        # Semantic Metrics
        sem_slice = [x for x in semantics if x['start'] < limit_time]
        question_count = sum(1 for x in sem_slice if "Question" in x.get('signals', []))
        emotion_count = sum(1 for x in sem_slice if "Emotional" in str(x.get('signals', [])))
        story_count = sum(1 for x in sem_slice if "Story" in str(x.get('signals', [])))
        
        # Visual Metrics
        vis_slice = [x for x in visuals if x['time'] < limit_time]
        if vis_slice:
            avg_motion = sum(x['motion_score'] for x in vis_slice) / len(vis_slice)
            avg_face = sum(x['face_score'] for x in vis_slice) / len(vis_slice)
        else:
            avg_motion, avg_face = 0.0, 0.0

        print(f"   [Metrics] Motion: {avg_motion:.2f}, Face: {avg_face:.2f}, Qs: {question_count}, Emotion: {emotion_count}")

        # --- SCORING LOGIC ---
        scores = {
            "podcast": 0,
            "gaming": 0,
            "educational": 0,
            "interview": 0,
            "comedy": 0,
            "vlog": 0
        }

        # 1. Visual Rules
        if avg_motion > 0.4:
            scores["gaming"] += 3
            scores["vlog"] += 2
        elif avg_motion < 0.1:
            scores["podcast"] += 2
            scores["interview"] += 2
            
        if avg_face > 0.6:
            scores["interview"] += 2
            scores["vlog"] += 1
            scores["educational"] += 1
        elif avg_face < 0.1:
            scores["gaming"] += 1 # Screen recording often has no face or small face cam

        # 2. Semantic Rules
        if question_count > 3:
            scores["interview"] += 3
            scores["educational"] += 1
            scores["podcast"] += 1
            
        if story_count > 2:
            scores["podcast"] += 2
            scores["vlog"] += 2
            
        if emotion_count > 4:
            scores["comedy"] += 3
            scores["gaming"] += 2

        # 3. Audio Rules
        if excitement_spikes > 10:
            scores["gaming"] += 3
            scores["comedy"] += 2
        elif excitement_spikes < 2:
            scores["educational"] += 2
            scores["podcast"] += 1

        # Select Winner
        best_genre = max(scores, key=scores.get)
        print(f"✅ Auto-Detected Genre: {best_genre.upper()} (Scores: {scores})")
        
        # Map simple categories to the detailed ones in weights.json if needed
        # For now, we assume 1:1 mapping or simple fallback
        return best_genre

class PipelineEvaluator:
    def __init__(self, workspace_dir="pipeline_data", ground_truth_file="ground_truth.csv"):
        self.workspace_dir = Path(workspace_dir)
        self.ground_truth_path = self.workspace_dir / ground_truth_file
        self.signals_path = self.workspace_dir / "signals.csv"
        
    def load_ground_truth(self):
        """
        Loads manual labels.
        Expected CSV format: start, end, label (1=Good, 0=Bad)
        """
        if not self.ground_truth_path.exists():
            print(f"❌ Ground truth file not found: {self.ground_truth_path}")
            print("   Create a CSV with columns: start, end, label")
            return []
            
        labels = []
        with open(self.ground_truth_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels.append({
                    "start": float(row["start"]),
                    "end": float(row["end"]),
                    "label": int(row["label"])
                })
        return labels

    def get_iou(self, clip_a, clip_b):
        """Calculates Intersection over Union (IoU) for two time windows."""
        start_a, end_a = clip_a["start"], clip_a["end"]
        start_b, end_b = clip_b["start"], clip_b["end"]
        
        inter_start = max(start_a, start_b)
        inter_end = min(end_a, end_b)
        
        if inter_end <= inter_start: return 0.0
        
        intersection = inter_end - inter_start
        union = (end_a - start_a) + (end_b - start_b) - intersection
        
        return intersection / union if union > 0 else 0.0

    def match_predictions(self, predictions, ground_truth, iou_threshold=0.3):
        """
        Matches model predictions to ground truth labels.
        """
        tp = 0 # True Positive (Model picked GOOD clip)
        fp = 0 # False Positive (Model picked BAD/Unlabeled clip)
        fn = 0 # False Negative (Model missed GOOD clip)
        
        # Track which GT items were matched
        matched_gt = set()
        
        for pred in predictions:
            match_found = False
            for i, gt in enumerate(ground_truth):
                if self.get_iou(pred, gt) >= iou_threshold:
                    if gt["label"] == 1:
                        tp += 1
                        matched_gt.add(i)
                        match_found = True
                        break # Count as one match
                    elif gt["label"] == 0:
                        # Explicit negative match (Model picked what human rejected)
                        fp += 1
                        matched_gt.add(i)
                        match_found = True
                        break
            
            if not match_found:
                # Prediction didn't match any label -> Treat as FP (strict) or ignore?
                # Usually treat as FP implies "noise" unless dataset is sparse.
                fp += 1

        # Count False Negatives (Good clips model missed)
        for i, gt in enumerate(ground_truth):
            if gt["label"] == 1 and i not in matched_gt:
                fn += 1
                
        return tp, fp, fn

    def calculate_metrics(self, tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

    def run_evaluation(self, genre="podcast", platform="shorts", ablate=None):
        """
        Runs the ranking logic and compares against ground truth.
        Optional: 'ablate' takes a list of signal names to zero out.
        """
        # 1. Load Data & Config
        if not self.signals_path.exists():
            print("❌ Signals CSV missing. Run run_pipeline.py first.")
            return None

        # --- AUTO DETECT GENRE ---
        target_genre = genre
        if genre == "auto":
            detector = GenreDetector(self.workspace_dir)
            target_genre = detector.detect()
            # Fallback check against known genres in weights.json (conceptually)
            # For now, just a basic safety list
            if target_genre not in ["podcast", "gaming", "educational", "interview", "sports", "news", "comedy", "music", "tech_reviews", "lifestyle_vlog", "beauty_fashion", "asmr_relax", "film_reviews", "diy_howto", "cooking", "politics"]:
                 target_genre = "podcast" # Safety net

        try:
            weights, cutoff, min_spacing, signal_order = load_weights("weights.json", target_genre)
        except Exception as e:
            print(f"❌ Failed to load weights for genre '{target_genre}': {e}")
            return None
        
        # ABLATION LOGIC: Zero out specific weights
        if ablate:
            indices_to_zero = [i for i, s in enumerate(signal_order) if s in ablate]
            for idx in indices_to_zero:
                weights[idx] = 0.0
            # Renormalize? Usually for ablation we just drop the signal to see impact.
            # If we renormalize, we test "substitution". If not, we test "loss".
            # Let's simple-drop (score will decrease, might hit cutoff less).
        
        # 2. Simulate Pipeline
        try:
            windows = read_windows(self.signals_path, signal_order)
        except Exception as e:
            print(f"❌ Failed to read signals: {e}")
            return None

        # Apply Platform Constraints (from rank_clips.py logic - simplified here or imported)
        # For true fidelity, we should import filter_by_constraints from rank_clips
        # But score_windows and apply_cutoff are enough for scoring logic check
        
        # Note: score_windows in rank_clips now accepts platform for bias application
        # If your rank_clips.py version doesn't support platform arg in score_windows yet, 
        # this line assumes it does based on your "Platform-Specific Profiles" request.
        # If not, use: windows = score_windows(windows, weights)
        try:
            windows = score_windows(windows, weights, platform)
        except TypeError:
             # Fallback if score_windows hasn't been updated to take platform
             windows = score_windows(windows, weights)

        predictions = apply_cutoff(windows, cutoff)
        predictions = enforce_spacing(predictions, min_spacing)
        
        # 3. Match & Score
        ground_truth = self.load_ground_truth()
        if not ground_truth: return None
        
        tp, fp, fn = self.match_predictions(predictions, ground_truth)
        p, r, f1 = self.calculate_metrics(tp, fp, fn)
        
        return {
            "precision": p,
            "recall": r,
            "f1": f1,
            "tp": tp, "fp": fp, "fn": fn,
            "clip_count": len(predictions)
        }

    def run_ablation_study(self, genre="podcast"):
        print(f"\n🔬 STARTING ABLATION STUDY (Genre: {genre})")
        print("-" * 60)
        print(f"{'Experiment':<25} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'ΔF1':<6}")
        print("-" * 60)
        
        # Baseline
        base = self.run_evaluation(genre=genre)
        if not base: return
        
        print(f"{'Baseline (All Signals)':<25} | {base['precision']:.2f}   | {base['recall']:.2f}   | {base['f1']:.2f}   | -")
        
        # Ablations
        signals_to_test = [
            "audio_excitation", 
            "semantic_novelty", 
            "hook_phrase_score", 
            "face_motion",
            "laughter_or_reaction"
        ]
        
        for sig in signals_to_test:
            res = self.run_evaluation(genre=genre, ablate=[sig])
            if res:
                delta = res['f1'] - base['f1']
                print(f"{'- ' + sig:<25} | {res['precision']:.2f}   | {res['recall']:.2f}   | {res['f1']:.2f}   | {delta:+.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--genre", default="podcast") # Can accept 'auto'
    args = parser.parse_args()
    
    evaluator = PipelineEvaluator()
    evaluator.run_ablation_study(genre=args.genre)