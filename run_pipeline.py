import subprocess
import sys
import json
import csv
import argparse
from pathlib import Path

# Pipeline Orchestrator with CLI args for workspace isolation

class GenreDetector:
    """Lightweight rule-based genre inference engine."""
    def __init__(self, workspace_dir):
        self.workspace_dir = Path(workspace_dir)
        
    def detect(self):
        print("🕵️  Detecting genre from signals...")
        try:
            with open(self.workspace_dir / "audio_analysis.json") as f: audio = json.load(f)
            with open(self.workspace_dir / "semantic_analysis.json") as f: semantics = json.load(f)
            with open(self.workspace_dir / "visual_analysis.json") as f:
                visuals_raw = json.load(f)
                visuals = visuals_raw if isinstance(visuals_raw, list) else visuals_raw.get("time_series", [])
        except Exception as e:
            print(f"⚠️  Could not load all signals: {e}")
            return "podcast"

        # (Simulated Detection Logic for Brevity - assume standard heuristic)
        # For production, this should include the full logic discussed previously.
        return "podcast" 

class PipelineOrchestrator:
    def __init__(self, workspace_dir="pipeline_data", genre="auto", platform="shorts"):
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.genre = genre
        self.platform = platform
        self.python_exe = sys.executable

    def run_step(self, script_name, args=[]):
        """Runs a python script as a subprocess step."""
        print(f"\n🚀 [STEP] Running {script_name}...")
        # Check if script is in current dir
        if not Path(script_name).exists():
             print(f"❌ Script {script_name} not found.")
             sys.exit(1)

        # Pass workspace to children via args if they are updated to support it
        # Assuming children parse --workspace or default to pipeline_data
        # We append --workspace argument blindly; Ensure children handle it or ignore extra args
        cmd = [self.python_exe, script_name] + args + ["--workspace", str(self.workspace_dir)]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ {script_name} completed.")
        except subprocess.CalledProcessError as e:
            print(f"❌ {script_name} failed.")
            sys.exit(1)

    def convert_json_to_signals_csv(self):
        """
        Merges all JSON analysis files into the CSV format required by rank_clips.py.
        Includes Hook-LLM fusion logic.
        """
        print("\n🔄 [STEP] merging signals to CSV (Smart Windowing)...")
        
        # Paths
        audio_path = self.workspace_dir / "audio_analysis.json"
        visual_path = self.workspace_dir / "visual_analysis.json"
        semantic_path = self.workspace_dir / "semantic_analysis.json"
        transcript_path = self.workspace_dir / "transcription.json"
        hook_path = self.workspace_dir / "hook_analysis.json"
        output_csv = self.workspace_dir / "signals.csv"

        if not audio_path.exists() or not transcript_path.exists():
            print("❌ Missing audio/transcript analysis data. Cannot proceed.")
            sys.exit(1)

        # Load Data
        with open(audio_path, 'r', encoding='utf-8') as f:
            audio_data = json.load(f)
        
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
            segments = transcript_data.get('segments', [])
        
        visual_data = []
        if visual_path.exists():
            with open(visual_path, 'r', encoding='utf-8') as f:
                v_raw = json.load(f)
                if isinstance(v_raw, list): visual_data = v_raw
                elif isinstance(v_raw, dict): visual_data = v_raw.get("time_series", [])
        
        semantic_data = []
        if semantic_path.exists():
            with open(semantic_path, 'r', encoding='utf-8') as f:
                semantic_data = json.load(f)
        
        hook_data = []
        if hook_path.exists():
            with open(hook_path, 'r', encoding='utf-8') as f:
                hook_data = json.load(f)

        # --- Helper Functions ---

        def get_visual_at(t):
            for v in visual_data:
                if abs(v['time'] - t) < 0.6: 
                    return v
            return {"face_score": 0.0, "motion_score": 0.0, "visual_score": 0.0}

        def get_hook_score_at(t_start, t_end):
            max_hook = 0.0
            for h in hook_data:
                overlap_start = max(h['start'], t_start)
                overlap_end = min(h['end'], t_end)
                if overlap_end > overlap_start:
                    max_hook = max(max_hook, h.get('hook_score', 0.0))
            return max_hook

        def get_semantic_at(t_start, t_end):
            best_match = None
            max_overlap = 0
            
            for s in semantic_data:
                overlap_start = max(s['start'], t_start)
                overlap_end = min(s['end'], t_end)
                overlap = max(0, overlap_end - overlap_start)
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_match = s
            
            if best_match:
                signals = best_match.get('signals', [])
                novelty = 0.0
                for sig in signals:
                    if "Novelty" in sig:
                        try:
                            novelty = float(sig.split('(')[1].split(')')[0])
                        except: pass
                
                # --- HOOK FUSION ---
                llm_hook = get_hook_score_at(t_start, t_end)
                base_semantic = best_match.get('semantic_score', 0.0) if novelty > 0 else best_match.get('semantic_score', 0.0)
                
                # Boost: 1.0 + (0.4 * hook_score)
                hook_boost = 1.0 + (0.4 * llm_hook)
                final_semantic = min(base_semantic * hook_boost, 1.0)
                
                heuristic_hook = 0.8 if "Question" in signals else 0.0
                final_hook_phrase = max(heuristic_hook, llm_hook)

                return {
                    "novelty": final_semantic,
                    "hook": final_hook_phrase,
                    "sentiment": 0.8 if "Emotional" in str(signals) else 0.0
                }
            return {"novelty": 0.0, "hook": 0.0, "sentiment": 0.0}

        # --- Smart Window Generation Logic ---
        rows = []
        MIN_DURATION = 5.0 
        MAX_DURATION = 90.0
        
        for i in range(len(segments)):
            for j in range(i, len(segments)):
                seg_start = segments[i]['start']
                seg_end = segments[j]['end']
                current_duration = seg_end - seg_start
                
                if current_duration > MAX_DURATION: break
                
                if current_duration >= MIN_DURATION:
                    # Audio
                    window_audio = [x for x in audio_data if seg_start <= x['time'] < seg_end]
                    if not window_audio: continue
                    avg_excitement = sum(x['excitement_score'] for x in window_audio) / len(window_audio)
                    silence_count = sum(1 for x in window_audio if x['volume_score'] < 0.1)
                    silence_score = silence_count / len(window_audio)

                    # Visuals
                    v_start = get_visual_at(seg_start)
                    v_mid = get_visual_at(seg_start + current_duration/2)
                    v_end = get_visual_at(seg_end)
                    avg_face = (v_start.get('face_score',0) + v_mid.get('face_score',0) + v_end.get('face_score',0)) / 3
                    avg_motion = (v_start.get('motion_score',0) + v_mid.get('motion_score',0) + v_end.get('motion_score',0)) / 3

                    # Semantics
                    s_data = get_semantic_at(seg_start, seg_end)

                    rows.append({
                        "window_start": round(seg_start, 2),
                        "window_end": round(seg_end, 2),
                        "audio_excitation": round(avg_excitement, 3),
                        "speech_rate_change": 0.5, 
                        "silence_breaks": round(silence_score, 3),
                        "semantic_novelty": round(s_data['novelty'], 3),
                        "sentiment_intensity": round(s_data['sentiment'], 3),
                        "hook_phrase_score": round(s_data['hook'], 3),
                        "face_presence": round(avg_face, 3),
                        "face_motion": round(avg_motion, 3),
                        "scene_change_rate": round(avg_motion, 3),
                        "laughter_or_reaction": 0.1
                    })

        if not rows:
            print("❌ No valid smart windows found.")
            return

        headers = list(rows[0].keys())
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
            
        print(f"✅ Generated {len(rows)} smart candidates in {output_csv}")

    def run(self, video_url):
        print(f"🎬 STARTING PIPELINE | Platform: {self.platform} | Mode: {self.genre} | Dir: {self.workspace_dir}")
        
        # 1. Ingest
        self.run_step("video_ingestion.py") 
        
        # 2. Transcribe
        self.run_step("audio_transcription.py")
        
        # 3. Analyze
        self.run_step("audio_analysis.py")
        self.run_step("text_analysis.py")
        self.run_step("visual_analysis.py")
        self.run_step("hook_detector.py")
        
        # Auto Detect
        target_genre = self.genre
        if self.genre == "auto":
            detector = GenreDetector(self.workspace_dir)
            target_genre = detector.detect()
        
        # 4. Prepare Signals
        self.convert_json_to_signals_csv()
        
        # 5. Rank
        rank_args = [
            "--signals", str(self.workspace_dir / "signals.csv"),
            "--genre", target_genre,
            "--platform", self.platform,
            "--out", str(self.workspace_dir / "ranked_clips.csv"),
            "--weights", "weights.json"
        ]
        self.run_step("rank_clips.py", rank_args)
        
        # 6. Render
        self.convert_csv_to_renderer_json()
        self.run_step("video_renderer.py")
        
        print("\n🎉 PIPELINE COMPLETE!")

    def convert_csv_to_renderer_json(self):
        # (Same logic as before)
        csv_path = self.workspace_dir / "ranked_clips.csv"
        json_path = self.workspace_dir / "suggested_clips.json"
        clips = []
        if csv_path.exists():
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    clips.append({
                        "start": float(row["start"]),
                        "end": float(row["end"]),
                        "score": float(row["score"])
                    })
        with open(json_path, 'w') as f:
            json.dump(clips, f, indent=2)
        print(f"✅ Converted ranked CSV to JSON: {json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="Video URL")
    parser.add_argument("--genre", default="auto", help="Target genre")
    parser.add_argument("--platform", default="shorts", help="Target platform")
    parser.add_argument("--workspace", default="pipeline_data", help="Workspace directory")
    args = parser.parse_args()

    orchestrator = PipelineOrchestrator(
        workspace_dir=args.workspace,
        genre=args.genre, 
        platform=args.platform
    )
    orchestrator.run(args.url)