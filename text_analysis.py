import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import re
import torch

class TextAnalyzer:
    def __init__(self, workspace_dir="pipeline_data"):
        self.workspace_dir = Path(workspace_dir)
        self.transcription_path = self.workspace_dir / "transcription.json"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load lightweight but powerful embedding model
        print("🧠 Loading Semantic Model (all-MiniLM-L6-v2)...")
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        except Exception as e:
            print(f"⚠️  Model load failed: {e}")
            self.model = None

    def load_transcript(self):
        if not self.transcription_path.exists():
            raise FileNotFoundError("❌ Transcript not found.")
        
        with open(self.transcription_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data  # Returning full data object to access 'language'

    def merge_segments(self, segments, target_duration=8.0):
        """
        Merges small Whisper segments into larger semantic windows (6-10s).
        This fixes the issue where split sentences get 0 semantic score.
        """
        if not segments:
            return []
            
        merged = []
        current_block = {
            "start": segments[0]['start'],
            "end": segments[0]['end'],
            "text": segments[0]['text'].strip(),
        }
        
        for i in range(1, len(segments)):
            seg = segments[i]
            
            # Calculate current duration if we were to add this segment
            curr_duration = seg['end'] - current_block['start']
            
            # Merge if under target duration (default 8s is sweet spot for standup setups)
            if curr_duration <= target_duration:
                current_block['end'] = seg['end']
                current_block['text'] += " " + seg['text'].strip()
            else:
                # Finalize current block
                merged.append(current_block)
                # Start new block
                current_block = {
                    "start": seg['start'],
                    "end": seg['end'],
                    "text": seg['text'].strip(),
                }
        
        # Append the last block
        merged.append(current_block)
        return merged

    def analyze_heuristics(self, text, language="en"):
        """
        Scores text based on viral linguistic patterns, with expanded Hinglish support.
        """
        score = 0.0
        signals = []
        text_lower = text.lower()
        
        # --- UNIVERSAL SIGNALS ---
        
        # 1. Questions (Hooks)
        if "?" in text:
            score += 0.3
            signals.append("Question")
            
        # 2. Emphasis via repetition
        if re.search(r'\b(\w+)\s+\1\b', text_lower):
            score += 0.2
            signals.append("Repetition")
            
        # 3. Laughter markers (in transcript)
        if any(x in text_lower for x in ["(laughter)", "haha", "audience"]):
            score += 0.5
            signals.append("Laughter")

        # --- HINGLISH / HINDI SUPPORT (FIX: EXPANDED LEXICON) ---
        # Checks Roman Hinglish + Devanagari regardless of detected language flag
        
        HINGLISH_QUESTIONS = [
            "kya", "kyu", "kaise", "kaunsa", "kaunsi", 
            "sach mein", "matlab", "samajh", "kyon", 
            "kab", "kahan", "kisko", "kiska"
        ]
        
        HINGLISH_STORY = [
            "ek baar", "pata hai", "samjhe", "sun", "dekho", 
            "hua kya", "bola", "kehta", "lagta hai"
        ]

        # Check Hinglish Questions
        if any(w in text_lower for w in HINGLISH_QUESTIONS):
            score += 0.25
            if "Question" not in signals:
                signals.append("Question (HI)")
                
        # Check Hinglish Storytelling
        if any(w in text_lower for w in HINGLISH_STORY):
            score += 0.2
            signals.append("Story (HI)")

        # Native Hindi script check
        if any(w in text_lower for w in ["क्या", "क्यों", "कैसे", "मतलब"]):
            score += 0.25
            signals.append("Question (Devanagari)")

        return min(score, 1.0), signals

    def analyze_semantics(self):
        """
        Main pipeline for semantic scoring.
        """
        print("🧠 Running Semantic Analysis...")
        data = self.load_transcript()
        raw_segments = data['segments']
        language = data.get('language', 'en')
        
        # FIX A: Merge segments before analysis
        # Rolling window of ~8 seconds helps capture full jokes/setups
        segments = self.merge_segments(raw_segments, target_duration=8.0)
        print(f"   Merged {len(raw_segments)} raw segments into {len(segments)} semantic windows.")
        
        results = []
        
        # Compute embeddings if model loaded
        embeddings = None
        novelty_scores = []
        if self.model:
            texts = [s['text'] for s in segments]
            embeddings = self.model.encode(texts, convert_to_tensor=True)
            if torch.is_tensor(embeddings):
                embeddings = embeddings.to(self.device)
            
            # Compute novelty (distance from average)
            # 1. Calculate centroid (keep on same device as embeddings)
            centroid = embeddings.mean(dim=0)
            
            # 2. Calculate cosine similarity to centroid for each
            # Lower similarity = Higher Novelty
            novelty_scores = []
            for emb in embeddings:
                sim = util.cos_sim(emb.unsqueeze(0), centroid.unsqueeze(0)).item()
                novelty_scores.append(1.0 - sim) # 0 to 1 (1 = very unique)
        
        for i, seg in enumerate(segments):
            text = seg['text']
            
            # 1. Heuristic Score
            h_score, h_signals = self.analyze_heuristics(text, language)
            
            # FIX: Down-weight "Question-only" hits to avoid false positive rhetorical questions
            if h_signals == ["Question (Devanagari)"]:
                h_score *= 0.85
            
            # 2. Novelty Score
            n_score = novelty_scores[i] if embeddings is not None else 0.0
            
            # Combined Score Logic
            final_score = h_score
            
            # FIX C: Boost novelty fallback
            # If heuristic missed it (0.0), check if it's semantically unique
            # Raised threshold to 0.25 and multiplier to 0.7 for standup
            if h_score < 0.25:
                final_score = max(h_score, n_score * 0.7)
                if final_score > h_score and final_score > 0.3:
                    h_signals.append(f"Novelty ({n_score:.2f})")

            # Boost storytelling automatically
            if "Story (HI)" in h_signals or "Repetition" in h_signals:
                final_score += 0.15

            # FIX D: Relax short-text penalty
            word_count = len(text.split())
            if word_count < 3: 
                final_score *= 0.7 # Less brutal than 0.5 (was < 4)
            
            # Cap at 1.0
            final_score = min(final_score, 1.0)
            
            results.append({
                "start": seg['start'],
                "end": seg['end'],
                "text": text,
                "score": round(final_score, 3), # Matched key to clip_selector
                "signals": h_signals
            })
            
        print(f"✅ Analyzed {len(results)} merged text windows.")
        return results

    def save_results(self, results):
        output_path = self.workspace_dir / "semantic_analysis.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 Semantic analysis saved to: {output_path}")

if __name__ == "__main__":
    analyzer = TextAnalyzer("pipeline_data")
    try:
        results = analyzer.analyze_semantics()
        analyzer.save_results(results)
    except Exception as e:
        print(f"❌ Error: {e}")