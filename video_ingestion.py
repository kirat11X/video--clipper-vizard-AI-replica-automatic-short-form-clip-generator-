import os
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

    def download_video(self, url):
        """
        Downloads video from YouTube/Twitch using yt-dlp.
        Returns a dictionary containing path, duration, and title.
        """
        print(f"⬇️  Downloading video from: {url}")
        
        # Output template to force specific filename structure
        output_template = str(self.video_dir / "input_video.%(ext)s")
        
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
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

    def run(self, url):
        """Orchestrates the full ingestion process."""
        print("🚀 Starting Video Ingestion Pipeline...")
        
        # 1. Download
        # Returns a dict with path and metadata
        download_result = self.download_video(url)
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
    # Example Usage
    # You can change this URL to any YouTube video for testing
    TEST_URL = "https://www.youtube.com/watch?v=UZzQpP5JHuc&pp=ygUHc3RhbmR1cA%3D%3D" 
    
    try:
        ingestor = VideoIngestor(output_dir="pipeline_data")
        result = ingestor.run(TEST_URL)
        print("\nPipeline Result:", result)
    except Exception as e:
        print(f"\n💥 Pipeline failed: {e}")
        sys.exit(1)