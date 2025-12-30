import json
import csv
import argparse
import numpy as np

# --- 2️⃣ PLATFORM-SPECIFIC PROFILES ---
PLATFORM_PROFILES = {
    "tiktok": {
        "boost": {
            "audio_excitation": 1.25,
            "hook_phrase_score": 1.30,
            "face_motion": 1.20
        },
        "penalty": {
            "silence_breaks": 0.70 # Penalty for long silences
        },
        "constraints": {
            "min_length": 7.0,
            "max_length": 45.0, # Reject If > 45s
            "optimal_min": 7.0,
            "optimal_max": 30.0
        }
    },
    "reels": {
        "boost": {
            "sentiment_intensity": 1.20,
            "face_presence": 1.15
        },
        "penalty": {
            "scene_change_rate": 0.85 # Penalty for abrupt cuts if high
        },
        "constraints": {
            "min_length": 6.0, # Reject If < 6s
            "max_length": 60.0, 
            "optimal_min": 10.0,
            "optimal_max": 45.0
        }
    },
    "shorts": {
        "boost": {
            "semantic_novelty": 1.25,
            # "speech_clarity": 1.15 # Fallback or skip if missing
        },
        "penalty": {
            "scene_change_rate": 0.80 # Penalty for chaotic pacing
        },
        "constraints": {
            "min_length": 10.0, # Reject If < 10s
            "max_length": 60.0,
            "optimal_min": 15.0,
            "optimal_max": 60.0
        }
    }
}

def load_weights(path, genre):
    """Loads weights and config for a specific genre from JSON."""
    with open(path, "r") as f:
        data = json.load(f)
    
    if genre not in data["genres"]:
        raise ValueError(f"Genre '{genre}' not found in config. Available: {list(data['genres'].keys())}")
        
    weights = np.array(data["genres"][genre])
    cutoff = data["default_cutoffs"].get(genre, 0.75)
    min_spacing = data.get("min_spacing_seconds", 20)
    signal_order = data["signal_order"]
    
    return weights, cutoff, min_spacing, signal_order

def read_windows(csv_path, signal_order):
    """Reads the CSV containing normalized signal windows."""
    windows = []
    try:
        with open(csv_path, newline="", encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    s = np.array([float(row[k]) for k in signal_order])
                    windows.append({
                        "start": float(row["window_start"]),
                        "end": float(row["window_end"]),
                        "signals": s,
                        "signal_dict": {k: float(row[k]) for k in signal_order} # Keep dict for name-based lookup
                    })
                except KeyError as e:
                    print(f"⚠️ Warning: Missing signal column {e} in CSV. Skipping row.")
                except ValueError:
                    print(f"⚠️ Warning: Invalid numeric data in row. Skipping.")
    except FileNotFoundError:
        print(f"❌ Signals CSV not found: {csv_path}")
        return []
    return windows

def filter_by_constraints(windows, platform):
    """Filters windows based on Platform Clip Length Constraints."""
    profile = PLATFORM_PROFILES.get(platform, PLATFORM_PROFILES["shorts"])
    constraints = profile["constraints"]
    
    valid_windows = []
    for w in windows:
        duration = w["end"] - w["start"]
        
        # Hard Rules
        if duration < constraints["min_length"]:
            continue
        if duration > constraints["max_length"]:
            continue
            
        valid_windows.append(w)
        
    return valid_windows

def score_windows(windows, weights, platform):
    """Calculates the score with Platform Bias."""
    profile = PLATFORM_PROFILES.get(platform, PLATFORM_PROFILES["shorts"])
    
    for w in windows:
        # 1. Base Genre Score
        base_score = float(np.dot(w["signals"], weights))
        
        # 2. Platform Bias Multipliers
        bias_multiplier = 1.0
        
        # Apply Boosts
        for signal_name, boost_factor in profile["boost"].items():
            val = w["signal_dict"].get(signal_name, 0.0)
            if val > 0.6: # Only boost if signal is strong
                # Apply boost proportional to signal strength
                # boost_factor 1.25 means 25% boost max
                bias_multiplier *= (1.0 + (boost_factor - 1.0) * val)
        
        # Apply Penalties
        for signal_name, penalty_factor in profile["penalty"].items():
            val = w["signal_dict"].get(signal_name, 0.0)
            if val > 0.6: # Only penalize if negative signal is strong
                bias_multiplier *= penalty_factor
        
        # 3. Final Score Calculation (Normalized 1-10)
        # Note: Length bonus and hook bonus are implicitly handled by signal weights 
        # and boost multipliers in this simplified architecture.
        # We clamp the result to be safe.
        final_score = base_score * bias_multiplier * 10.0
        w["score"] = min(final_score, 10.0)
        
    return windows

def apply_cutoff(windows, cutoff):
    # Cutoff is originally 0-1, so we scale it to 1-10 for comparison
    scaled_cutoff = cutoff * 10.0
    # Slightly relax cutoff since we are using a different scoring scale now
    return [w for w in windows if w["score"] >= (scaled_cutoff * 0.8)]

def enforce_spacing(windows, min_spacing):
    windows = sorted(windows, key=lambda x: x["score"], reverse=True)
    selected = []
    for w in windows:
        is_far_enough = True
        for s in selected:
            if abs(w["start"] - s["start"]) < min_spacing:
                is_far_enough = False
                break
        if is_far_enough:
            selected.append(w)
    return sorted(selected, key=lambda x: x["start"])

def main():
    parser = argparse.ArgumentParser(description="Rank and select viral clips based on genre weights and platform profiles.")
    parser.add_argument("--signals", required=True, help="Path to input CSV containing signal windows")
    parser.add_argument("--weights", default="weights.json", help="Path to weights configuration JSON")
    parser.add_argument("--genre", required=True, help="Target genre")
    parser.add_argument("--platform", default="shorts", choices=["tiktok", "reels", "shorts"], help="Target platform")
    parser.add_argument("--out", default="ranked_clips.csv", help="Output path")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Ranker | Genre: {args.genre} | Platform: {args.platform}")
    
    try:
        weights, cutoff, min_spacing, signal_order = load_weights(args.weights, args.genre)
        
        windows = read_windows(args.signals, signal_order)
        if not windows:
            print("❌ No valid windows found.")
            return
            
        # 1. Apply Constraints FIRST (Efficiency)
        windows = filter_by_constraints(windows, args.platform)
        print(f"   {len(windows)} windows valid for platform constraints.")

        # 2. Score with Bias
        windows = score_windows(windows, weights, args.platform)
        
        # 3. Filter by Score
        high_score_windows = apply_cutoff(windows, cutoff)
        print(f"   {len(high_score_windows)} windows passed score threshold")
        
        # 4. Spacing
        final_clips = enforce_spacing(high_score_windows, min_spacing)
        
        with open(args.out, "w", newline="", encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["start", "end", "score"])
            for w in final_clips:
                writer.writerow([w["start"], w["end"], round(w["score"], 4)])
                
        print(f"✅ Saved {len(final_clips)} ranked clips to {args.out}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()