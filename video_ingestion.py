import os
import shutil
import subprocess
import sys
from pathlib import Path
import yt_dlp

class VideoIngestor:
    def __init__(self, output_dir="workspace"):
        """
        Initializes the ingestion pipeline.
        
        Args:
            output_dir (str): Base directory for all processing artifacts.
        """
        self.base_dir = Path(output_dir)
        self.video_dir = self.base_dir / "video"
        self.audio_dir = self.base_dir / "audio"
        self.frames_dir = self.base_dir / "frames"
        
        # Create directories
        for path in [self.video_dir, self.audio_dir, self.frames_dir]:
            path.mkdir(parents=True, exist_ok=True)
            
        print(f"📁 Workspace initialized at: {self.base_dir.absolute()}")

    def import_local(self, file_path):
        """Uses a video file already on disk instead of downloading one."""
        src = Path(file_path).expanduser().resolve()
        if not src.exists():
            raise RuntimeError(f"Video file not found: {src}")

        dest = self.video_dir / f"input_video{src.suffix.lower()}"
        # Drop any previous input so the renderer never picks up a stale video.
        for old in self.video_dir.glob("input_video.*"):
            if old.resolve() != src:
                old.unlink()
        if src != dest.resolve():
            shutil.copy2(src, dest)

        info = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(dest)],
            capture_output=True, text=True,
        )
        try:
            duration = float(info.stdout.strip())
        except ValueError:
            duration = None
        print(f"✅ Using local video: {dest}")
        return {"video_path": dest, "duration": duration, "title": src.stem}

    def download_video(self, url, max_height=None):
        """
        Downloads video from YouTube/Twitch using yt-dlp.
        Returns a dictionary containing path, duration, and title.
        """
        print(f"⬇️  Downloading video from: {url}")
        
        # Output template to force specific filename structure
        output_template = str(self.video_dir / "input_video.%(ext)s")
        
        cap = f"[height<={int(max_height)}]" if max_height else ""
        # Prefer H.264 (cheap to decode) when a height cap is set, then any mp4.
        video_fmt = (
            f"bestvideo{cap}[vcodec^=avc1][ext=mp4]+bestaudio[ext=m4a]/"
            f"bestvideo{cap}[ext=mp4]+bestaudio[ext=m4a]/best{cap}[ext=mp4]/best{cap}/best"
            if max_height else
            'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'
        )
        ydl_opts = {
            'format': video_fmt,
            'outtmpl': output_template,
            'quiet': True,
            'no_warnings': True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                
                # Extract metadata
                duration = info.get("duration")
                title = info.get("title")
                
                print(f"✅ Download complete: {filename}")
                
                return {
                    "video_path": Path(filename),
                    "duration": duration,
                    "title": title
                }
        except Exception as e:
            # CRITICAL FIX: Raise exception instead of sys.exit(1)
            # This allows the caller (worker/job queue) to handle the failure
            raise RuntimeError(f"Download failed: {e}")

    def extract_audio(self, video_path):
        """
        Extracts audio from video to WAV format (16kHz mono).
        Optimized for Whisper Speech-to-Text.
        """
        output_path = self.audio_dir / "audio.wav"
        print("🔊 Extracting audio...")

        # FFmpeg command: 
        # -vn (no video) 
        # -acodec pcm_s16le (wav codec) 
        # -ar 16000 (16kHz sample rate - best for Whisper) 
        # -ac 1 (mono channel)
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-y", # Overwrite output
            str(output_path)
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✅ Audio extracted: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            # CRITICAL FIX: Raise exception instead of sys.exit(1)
            raise RuntimeError(f"Audio extraction failed: {e}")

    def extract_frames(self, video_path, fps=1):
        """
        Extracts frames from video as JPG images.
        
        Args:
            fps (int): Frames per second to extract. 
                       1 fps is standard for general visual analysis.
        """
        print(f"🎞️  Extracting frames at {fps} FPS...")
        
        # Pattern for output filenames (frame_0001.jpg, frame_0002.jpg, etc.)
        output_pattern = self.frames_dir / "frame_%04d.jpg"
        
        # FFmpeg command:
        # -vf fps={fps} (filter video to specific frames per second)
        # -q:v 2 (quality setting for jpg, 2 is very high quality)
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vf", f"fps={fps}",
            "-q:v", "2",
            "-y",
            str(output_pattern)
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Count generated frames
            frame_count = len(list(self.frames_dir.glob("*.jpg")))
            print(f"✅ Frames extracted: {frame_count} images saved to {self.frames_dir}")
            return self.frames_dir
        except subprocess.CalledProcessError as e:
            # CRITICAL FIX: Raise exception instead of sys.exit(1)
            raise RuntimeError(f"Frame extraction failed: {e}")

    def run(self, url=None, file=None, max_height=None):
        """Orchestrates the full ingestion process (URL download or local file)."""
        print("🚀 Starting Video Ingestion Pipeline...")
        
        # 1. Download (or import a local file)
        # Returns a dict with path and metadata
        if file:
            download_result = self.import_local(file)
        else:
            download_result = self.download_video(url, max_height=max_height)
        video_path = download_result["video_path"]
        duration = download_result["duration"]
        title = download_result["title"]
        
        # 2. Extract Audio (Critical for Whisper)
        audio_path = self.extract_audio(video_path)
        
        # 3. Extract Frames (Critical for Visual Analysis)
        frames_dir = self.extract_frames(video_path, fps=1)
        
        print("\n✨ Ingestion Complete!")
        print(f"   Video: {video_path}")
        print(f"   Audio: {audio_path}")
        print(f"   Frames: {frames_dir}")
        print(f"   Duration: {duration}s")
        
        return {
            "video_path": str(video_path),
            "audio_path": str(audio_path),
            "frames_dir": str(frames_dir),
            "duration": duration,
            "title": title
        }

if __name__ == "__main__":
    import argparse

    # Default URL is a sample video for testing; pass --url to use your own.
    TEST_URL = "https://www.youtube.com/watch?v=UZzQpP5JHuc&pp=ygUHc3RhbmR1cA%3D%3D"

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=TEST_URL, help="Video URL")
    parser.add_argument("--file", default=None, help="Local video file (skips the download)")
    parser.add_argument("--max-height", type=int, default=None, help="Cap download resolution, e.g. 1080")
    parser.add_argument("--workspace", default="pipeline_data")
    args = parser.parse_args()

    try:
        ingestor = VideoIngestor(output_dir=args.workspace)
        result = ingestor.run(args.url, file=args.file, max_height=args.max_height)
        print("\nPipeline Result:", result)
    except Exception as e:
        print(f"\n💥 Pipeline failed: {e}")
        sys.exit(1)
