import json
import subprocess
import argparse
from pathlib import Path
import os

class VideoRenderer:
    def __init__(self, workspace_dir="pipeline_data"):
        self.workspace_dir = Path(workspace_dir)
        # Find video
        self.video_path = self.workspace_dir / "video" / "input_video.mp4"
        if not self.video_path.exists():
            videos = list((self.workspace_dir / "video").glob("*.*"))
            if videos: self.video_path = videos[0]
            
        self.clips_path = self.workspace_dir / "suggested_clips.json"
        self.output_dir = self.workspace_dir / "shorts"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.transcription_path = self.workspace_dir / "transcription.json"
        
        with open(self.transcription_path, 'r', encoding='utf-8') as f:
            self.transcript = json.load(f)

    def _time_to_srt(self, seconds):
        ms = int((seconds - int(seconds)) * 1000)
        s = int(seconds)
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def _escape_path(self, path):
        """
        Escapes paths for FFmpeg filters (subtitles=...).
        Handles Windows backslashes, colons, and special chars.
        """
        return (
            str(path)
            .replace("\\", "/")
            .replace(":", "\\:")
            .replace("'", "\\'")
            .replace(",", "\\,")
            .replace("[", "\\[")
            .replace("]", "\\]")
        )

    def render(self):
        if not self.clips_path.exists(): return

        with open(self.clips_path, 'r') as f:
            clips = json.load(f)

        print(f"🎬 Rendering {len(clips)} clips...")

        for i, clip in enumerate(clips):
            start = clip['start']
            end = clip['end']
            duration = end - start
            out_name = self.output_dir / f"clip_{i+1}_{clip['score']}.mp4"
            
            # 1. Generate temp SRT for this clip
            srt_path = self.output_dir / "temp.srt"
            
            # Find relevant words
            words_in_clip = [
                w for w in self.transcript['words'] 
                if w['start'] >= start and w['end'] <= end
            ]
            
            # Chunk words for readable captions (3 words per line)
            with open(srt_path, 'w', encoding='utf-8') as srt:
                idx = 1
                for j in range(0, len(words_in_clip), 3):
                    chunk = words_in_clip[j:j+3]
                    if not chunk: continue
                    
                    # Timing padding for better readability
                    t0 = max(0, chunk[0]['start'] - start - 0.1)
                    t1 = max(0, chunk[-1]['end'] - start + 0.1)
                    text = " ".join([w['word'] for w in chunk])
                    
                    srt.write(f"{idx}\n{self._time_to_srt(t0)} --> {self._time_to_srt(t1)}\n{text}\n\n")
                    idx += 1

            # Escape path for FFmpeg
            srt_esc = self._escape_path(srt_path)
            
            # Complex Filter: Blur BG + Center Crop + Subs
            # Fix 1: Correct crop logic to prevent crashes on different aspect ratios
            vf = (
                f"[0:v]scale=-2:ih,boxblur=20:1[bg];"
                f"[0:v]scale=-2:ih,crop=ih*9/16:ih:(iw-ih*9/16)/2:0[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2,"
                f"subtitles='{srt_esc}':force_style='Alignment=10,Fontsize=24,Fontname=Noto Sans Devanagari,PrimaryColour=&H00FFFFFF,Outline=2,Shadow=1,MarginV=50'"
            )

            # Fix 2: Move -ss after -i for frame-accurate seeking (slower but precise)
            cmd = [
                "ffmpeg", "-y",
                "-i", str(self.video_path),
                "-ss", str(start), 
                "-t", str(duration),
                "-filter_complex", vf,
                "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
                "-c:v", "h264_nvenc", # RTX Optimized
                "-preset", "p4",      # Performance/Quality balance
                "-b:v", "5M",
                "-c:a", "aac", "-b:a", "192k",
                str(out_name)
            ]
            
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                print(f"   ✅ Saved: {out_name.name}")
            except Exception as e:
                print(f"   ❌ Failed to render clip {i}: {e}")
            finally:
                if srt_path.exists(): srt_path.unlink()

if __name__ == "__main__":
    renderer = VideoRenderer()
    renderer.render()