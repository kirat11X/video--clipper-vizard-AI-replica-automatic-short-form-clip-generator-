import json
import re
import subprocess
import argparse
import sys
from pathlib import Path

# Everything the UI can tweak. Anything missing from --options falls back to these.
DEFAULT_OPTIONS = {
    # Canvas
    "width": 1080,
    "height": 1920,
    "layout": "fill_crop",       # fill_crop = crop the sides | fit_blur = whole frame over a blurred copy
    "crop_focus_x": 0.5,         # 0 = left edge, 0.5 = centre, 1 = right edge (fill_crop only)
    "blur_strength": 30,         # fit_blur background blur
    # Clip timing
    "pad_before": 0.0,           # seconds added before each clip
    "pad_after": 0.0,            # seconds added after each clip
    # Captions
    "captions": True,
    "words_per_line": 3,
    "font": "Noto Sans Devanagari",
    "font_size": 80,             # pixels on the output canvas
    "bold": True,
    "font_color": "#FFFFFF",
    "outline_color": "#000000",
    "outline": 3,
    "shadow": 1,
    "position": "bottom",        # bottom | middle | top
    "margin_v": 300,             # distance from the top/bottom edge in pixels
    # Audio
    "normalize_audio": True,
    "loudness": -16,             # LUFS target
    # Video encoding
    "encoder": "auto",           # auto | h264_nvenc | libx264 | libopenh264
    "bitrate": "8M",
}

# Preference order when encoder == "auto".
ENCODER_PREFERENCE = ["h264_nvenc", "libx264", "libopenh264"]

# ASS numpad alignment: bottom-centre, middle-centre, top-centre
_ALIGNMENT = {"bottom": 2, "middle": 5, "top": 8}


def available_encoders():
    """H.264 encoders this machine's ffmpeg was built with, in preference order."""
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, check=True
        ).stdout
    except Exception:
        return []
    return [e for e in ENCODER_PREFERENCE if re.search(rf"\b{e}\b", out)]


def _ass_color(hex_color):
    """#RRGGBB -> ASS &H00BBGGRR&"""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        h = "FFFFFF"
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H00{b}{g}{r}&".upper()


class VideoRenderer:
    def __init__(self, workspace_dir="pipeline_data", options=None, out_dir=None, clips_path=None):
        self.workspace_dir = Path(workspace_dir)
        self.options = {**DEFAULT_OPTIONS, **(options or {})}

        # Find video
        self.video_path = self.workspace_dir / "video" / "input_video.mp4"
        if not self.video_path.exists():
            videos = sorted((self.workspace_dir / "video").glob("input_video.*"))
            if not videos:
                videos = sorted((self.workspace_dir / "video").glob("*.*"))
            if videos:
                self.video_path = videos[0]

        self.clips_path = Path(clips_path) if clips_path else self.workspace_dir / "suggested_clips.json"
        self.output_dir = Path(out_dir) if out_dir else self.workspace_dir / "shorts"
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

    def _write_srt(self, srt_path, start, end):
        """Writes captions for [start, end] with times relative to the clip start."""
        n = max(1, int(self.options["words_per_line"]))
        words = [
            w for w in self.transcript.get('words', [])
            if w['end'] > start and w['start'] < end
        ]
        count = 0
        with open(srt_path, 'w', encoding='utf-8') as srt:
            for j in range(0, len(words), n):
                chunk = words[j:j + n]
                if not chunk:
                    continue
                # Small padding for readability, clamped to the clip.
                t0 = max(0, chunk[0]['start'] - start - 0.1)
                t1 = min(end - start, max(0, chunk[-1]['end'] - start + 0.1))
                text = " ".join(w['word'] for w in chunk)
                count += 1
                srt.write(f"{count}\n{self._time_to_srt(t0)} --> {self._time_to_srt(t1)}\n{text}\n\n")
        return count

    def _filter_graph(self, srt_path):
        o = self.options
        W, H = int(o["width"]), int(o["height"])

        if o["layout"] == "fit_blur":
            graph = (
                f"[0:v]split=2[a][b];"
                f"[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
                f"boxblur={int(o['blur_strength'])}:3[bg];"
                f"[b]scale={W}:{H}:force_original_aspect_ratio=decrease[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2"
            )
        else:
            fx = min(1.0, max(0.0, float(o["crop_focus_x"])))
            graph = (
                f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,"
                f"crop={W}:{H}:(in_w-out_w)*{fx:.3f}:(in_h-out_h)/2"
            )

        if o["captions"] and srt_path is not None:
            # libass sizes SRT captions in a 288px-tall script space, not output pixels,
            # so convert the pixel values exposed to the user.
            k = 288.0 / H
            style = (
                f"Fontname={o['font']},Fontsize={float(o['font_size']) * k:.2f},Bold={1 if o['bold'] else 0},"
                f"PrimaryColour={_ass_color(o['font_color'])},OutlineColour={_ass_color(o['outline_color'])},"
                f"BorderStyle=1,Outline={float(o['outline']) * k:.2f},Shadow={float(o['shadow']) * k:.2f},"
                f"Alignment={_ALIGNMENT.get(o['position'], 2)},MarginV={round(float(o['margin_v']) * k)}"
            )
            graph += (
                f",subtitles='{self._escape_path(srt_path)}':original_size={W}x{H}"
                f":force_style='{style}'"
            )
        return graph + "[v]"

    def _encoder_args(self, encoder):
        br = str(self.options["bitrate"])
        if encoder == "h264_nvenc":
            return ["-c:v", "h264_nvenc", "-preset", "p4", "-b:v", br]
        if encoder == "libx264":
            return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21"]
        return ["-c:v", encoder, "-b:v", br]

    def _candidates(self):
        choice = self.options["encoder"]
        if choice and choice != "auto":
            return [choice]
        found = available_encoders()
        return found or ["libx264"]

    def render(self):
        if not self.clips_path.exists():
            print(f"❌ No clips file at {self.clips_path}")
            return 0

        with open(self.clips_path, 'r', encoding='utf-8') as f:
            clips = json.load(f)

        o = self.options
        total = len(clips)
        print(f"🎬 Rendering {total} clips -> {self.output_dir} ({o['width']}x{o['height']}, {o['layout']})")
        print(f"PROGRESS 0/{total}", flush=True)

        done = 0
        for i, clip in enumerate(clips):
            start = max(0.0, float(clip['start']) - float(o["pad_before"]))
            end = float(clip['end']) + float(o["pad_after"])
            duration = end - start
            label = re.sub(r"[^A-Za-z0-9._-]+", "_", str(clip.get("name") or clip.get("score", "")))
            out_name = self.output_dir / f"clip_{i+1}_{label}.mp4"

            srt_path = self.output_dir / f".captions_{i+1}.srt"
            has_caps = o["captions"] and self._write_srt(srt_path, start, end) > 0

            vf = self._filter_graph(srt_path if has_caps else None)
            audio = ["-af", f"loudnorm=I={o['loudness']}:TP=-1.5:LRA=11"] if o["normalize_audio"] else []

            ok = False
            last_err = ""
            for encoder in self._candidates():
                # -ss before -i seeks without decoding everything up to the clip start.
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
                    "-i", str(self.video_path),
                    "-filter_complex", vf,
                    "-map", "[v]", "-map", "0:a:0?",
                    *self._encoder_args(encoder),
                    "-pix_fmt", "yuv420p",
                    *audio,
                    "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart",
                    str(out_name),
                ]
                proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
                if proc.returncode == 0:
                    ok = True
                    print(f"   ✅ Saved: {out_name.name} ({encoder})")
                    break
                last_err = (proc.stderr or "").strip().splitlines()[-3:]
                print(f"   ⚠️  {encoder} failed on clip {i+1}: {' | '.join(last_err)}")

            if ok:
                done += 1
            else:
                print(f"   ❌ Failed to render clip {i+1}")
                if out_name.exists():
                    out_name.unlink()
            if srt_path.exists():
                srt_path.unlink()
            print(f"PROGRESS {i+1}/{total}", flush=True)

        print(f"🎉 Rendered {done}/{total} clips")
        return done


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="pipeline_data")
    parser.add_argument("--options", default=None, help="JSON file with render options (see DEFAULT_OPTIONS)")
    parser.add_argument("--clips", default=None, help="Clips JSON (default: <workspace>/suggested_clips.json)")
    parser.add_argument("--out-dir", default=None, help="Output folder (default: <workspace>/shorts)")
    args = parser.parse_args()

    opts = {}
    if args.options:
        with open(args.options, 'r', encoding='utf-8') as f:
            opts = json.load(f)

    renderer = VideoRenderer(args.workspace, options=opts, out_dir=args.out_dir, clips_path=args.clips)
    rendered = renderer.render()
    sys.exit(0 if rendered else 1)
