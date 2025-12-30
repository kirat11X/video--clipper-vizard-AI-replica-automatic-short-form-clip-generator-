import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

class HookDetector:
    def __init__(self, workspace_dir="pipeline_data", model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        """
        Initializes the Tiny Hook-LLM.
        """
        self.workspace_dir = Path(workspace_dir)
        self.transcription_path = self.workspace_dir / "transcription.json"
        self.output_path = self.workspace_dir / "hook_analysis.json"
        
        print(f"🧠 Loading Hook-LLM: {model_name}...")
        try:
            # Check for GPU
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"   Device: {self.device}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                device_map=self.device, 
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).eval()
            print("✅ Model loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None

    def _get_prompt(self, text_chunk):
        """Constructs the strict classifier prompt."""
        system_prompt = (
            "You are a viral content classifier. "
            "Task: Determine whether the following text would stop a scrolling user within the first 3 seconds. "
            "Rules:\n"
            "- Score high (0.8-1.0) ONLY if curiosity, tension, controversy, or surprise exists.\n"
            "- Score low (0.0-0.4) for greetings, context buildup, filler, or boring facts.\n"
            "- hook_type must be one of: ['question', 'contrarian', 'reveal', 'emotional', 'none']\n"
            "Output valid JSON only. No explanation."
        )
        
        user_prompt = f"Text: \"{text_chunk}\"\nJSON:"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def analyze_hooks(self):
        if not self.model:
            print("⚠️ Skipping hook detection (Model not loaded).")
            return []

        if not self.transcription_path.exists():
            print("❌ Transcript not found.")
            return []

        with open(self.transcription_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            segments = data.get('segments', [])

        print("🎣 Scanning transcript for viral hooks...")
        results = []
        
        # Analyze sliding windows (3-sentence chunks)
        stride = 2
        window_size = 3
        
        for i in range(0, len(segments), stride):
            window = segments[i : i+window_size]
            if not window: break
            
            text_chunk = " ".join([s['text'].strip() for s in window])
            start_time = window[0]['start']
            end_time = window[-1]['end']
            
            if len(text_chunk.split()) < 5: continue

            prompt = self._get_prompt(text_chunk)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=64, 
                    temperature=0.1, 
                    do_sample=False
                )
            
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Robust JSON Parsing
            try:
                json_str = response.strip()
                # Handle markdown fences
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0].strip()
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].strip()
                
                analysis = json.loads(json_str)
                
                hook_score = float(analysis.get("hook_score", 0.0))
                hook_type = analysis.get("hook_type", "none")
                
                if hook_score > 0.4:
                    print(f"   found hook ({hook_score:.2f} - {hook_type}): \"{text_chunk[:40]}...\"")
                    results.append({
                        "start": start_time,
                        "end": end_time,
                        "text_snippet": text_chunk,
                        "hook_score": hook_score,
                        "hook_type": hook_type
                    })
                    
            except Exception:
                continue

        print(f"✅ Hook detection complete. Found {len(results)} potential hooks.")
        return results

    def save_results(self, results):
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Hook analysis saved to: {self.output_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Support running with --workspace arg if provided by orchestrator
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--workspace", default="pipeline_data")
        args, _ = parser.parse_known_args()
        workspace = args.workspace
    else:
        workspace = "pipeline_data"

    detector = HookDetector(workspace)
    hooks = detector.analyze_hooks()
    detector.save_results(hooks)