import cv2
import mediapipe as mp
import numpy as np
import json
import argparse
from pathlib import Path
import os

class VisualAnalyzer:
    def __init__(self, workspace_dir="pipeline_data"):
        self.workspace_dir = Path(workspace_dir)
        self.frames_dir = self.workspace_dir / "frames"
        self.output_file = self.workspace_dir / "visual_analysis.json"
        
        # Face Detection
        self.face_backend = "opencv"
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
            print("👀 MediaPipe Face Detection enabled.")
            self.face_backend = "mediapipe"
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        else:
            cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def analyze(self):
        print(f"🎞️  Analyzing frames...")
        frames = sorted(list(self.frames_dir.glob("*.jpg")))
        if not frames:
            print("❌ No frames found.")
            return

        time_series = []
        events = []
        prev_gray = None
        prev_face = 0.0
        prev_motion = 0.0
        
        for i, frame_path in enumerate(frames):
            img = cv2.imread(str(frame_path))
            if img is None: continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 1. Face Score
            face_score = 0.0
            if self.face_backend == "mediapipe":
                results = self.face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if results.detections:
                    # Score based on confidence + size of largest face
                    best_detection = max(results.detections, key=lambda d: d.location_data.relative_bounding_box.width)
                    face_score = 0.5 + (best_detection.score[0] * 0.5) # range 0.5 - 1.0
            else:
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                if len(faces) > 0:
                    face_score = 0.6 # Basic presence score for Haar
            
            # 2. Motion Score
            motion_score = 0.0
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                # Fix B: Reduce motion saturation (denominator 0.25 -> 0.4)
                motion_score = min(np.count_nonzero(thresh) / (gray.size * 0.4), 1.0)

            # 3. Events & Visual Score
            timestamp = i + 1.0
            
            # Fix A: Blend instead of max for smoother signal
            visual_score = (0.6 * face_score) + (0.4 * motion_score)
            
            # Fix C: Penalize no-face clips
            if face_score == 0.0:
                visual_score *= 0.6
            
            if abs(face_score - prev_face) > 0.15:
                events.append({"time": timestamp, "event": "FACE_CHANGE", "strength": abs(face_score - prev_face)})
            if motion_score - prev_motion > 0.15:
                events.append({"time": timestamp, "event": "MOTION_SPIKE", "strength": motion_score - prev_motion})

            time_series.append({
                "time": timestamp,
                "face_score": round(face_score, 3),
                "motion_score": round(motion_score, 3),
                "visual_score": round(visual_score, 3)
            })
            
            prev_gray = gray
            prev_face = face_score
            prev_motion = motion_score
            
            if i % 50 == 0: print(f"   Frame {i}/{len(frames)}...", end="\r")

        output = {"time_series": time_series, "events": events}
        with open(self.output_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n✅ Visual analysis saved.")

if __name__ == "__main__":
    analyzer = VisualAnalyzer()
    analyzer.analyze()