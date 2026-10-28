"""Logic behind the Streamlit UI (app.py): workspaces, scoring, clip selection, background jobs.

Kept free of Streamlit imports so it can be tested and reused from scripts.
"""
import copy
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import rank_clips

ROOT = Path(__file__).resolve().parent
PROJECTS_DIR = ROOT / "projects"
LEGACY_WORKSPACE = ROOT / "pipeline_data"
USER_PRESETS_FILE = ROOT / "user_presets.json"
WEIGHTS_FILE = ROOT / "weights.json"

# Pipeline steps in execution order: id -> (label, files that prove the step is done)
ANALYZE_STEPS = {
    "ingest": ("Download / import video", ["video", "audio/audio.wav", "frames/frame_0001.jpg"]),
    "transcribe": ("Transcribe speech (Whisper)", ["transcription.json"]),
    "audio": ("Audio excitement analysis", ["audio_analysis.json"]),
    "text": ("Semantic analysis", ["semantic_analysis.json"]),
    "visual": ("Visual analysis (faces, motion)", ["visual_analysis.json"]),
    "hooks": ("Hook detection (small LLM)", ["hook_analysis.json"]),
    "signals": ("Merge signals into candidate windows", ["signals.csv"]),
}
# A step is re-run automatically when something it depends on was re-run.
STEP_DEPS = {
    "ingest": [],
    "transcribe": ["ingest"],
    "audio": ["ingest"],
    "visual": ["ingest"],
    "text": ["transcribe"],
    "hooks": ["transcribe"],
    "signals": ["transcribe", "audio", "visual", "text", "hooks"],
}


# --------------------------------------------------------------------------- workspaces

def slugify(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("-.") or "project"


def list_projects():
    """[(label, path)] - the original pipeline_data run first, then projects/<name>."""
    out = []
    if LEGACY_WORKSPACE.exists():
        out.append(("pipeline_data (existing run)", LEGACY_WORKSPACE))
    if PROJECTS_DIR.exists():
        out += [(p.name, p) for p in sorted(PROJECTS_DIR.iterdir()) if p.is_dir()]
    return out


def create_project(name):
    path = PROJECTS_DIR / slugify(name)
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_video(ws):
    vids = sorted((Path(ws) / "video").glob("input_video.*")) or sorted((Path(ws) / "video").glob("*.*"))
    return vids[0] if vids else None


def _step_done(ws, step_id):
    ws = Path(ws)
    for rel in ANALYZE_STEPS[step_id][1]:
        if rel == "video":
            if find_video(ws) is None:
                return False
        elif not (ws / rel).exists():
            return False
    return True


def workspace_status(ws):
    """{step_id: bool} for every analysis step."""
    return {sid: _step_done(ws, sid) for sid in ANALYZE_STEPS}


def probe_video(path):
    """Duration / resolution / codec of a video, or {} if ffprobe fails."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name,width,height:format=duration,size",
             "-of", "json", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout
        d = json.loads(out)
        s = d["streams"][0]
        return {
            "codec": s.get("codec_name"), "width": s.get("width"), "height": s.get("height"),
            "duration": float(d["format"]["duration"]), "size_mb": int(d["format"]["size"]) / 1e6,
        }
    except Exception:
        return {}


def load_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def fmt_time(seconds):
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# --------------------------------------------------------------------------- scoring

def weights_config():
    return load_json(WEIGHTS_FILE, {})


def signal_order():
    return weights_config()["signal_order"]


def user_presets():
    return load_json(USER_PRESETS_FILE, {})


def save_user_preset(name, weights):
    presets = user_presets()
    presets[name] = {s: float(w) for s, w in weights.items()}
    save_json(USER_PRESETS_FILE, presets)


def delete_user_preset(name):
    presets = user_presets()
    presets.pop(name, None)
    save_json(USER_PRESETS_FILE, presets)


def genre_names():
    return list(weights_config()["genres"].keys())


def preset_weights(name):
    """Weights (signal -> float) for a built-in genre or a saved user preset."""
    order = signal_order()
    custom = user_presets()
    if name in custom:
        return {s: float(custom[name].get(s, 0.0)) for s in order}
    return dict(zip(order, [float(x) for x in weights_config()["genres"][name]]))


def default_min_score(genre):
    """The score the CLI ranker effectively applies: cutoff x 10 x 0.8."""
    cfg = weights_config()
    return round(cfg["default_cutoffs"].get(genre, 0.75) * 10 * 0.8, 1)


def default_scoring(genre="podcast", platform="shorts"):
    cons = rank_clips.PLATFORM_PROFILES[platform]["constraints"]
    cfg = weights_config()
    return {
        "preset": genre,
        "platform": platform,
        "weights": preset_weights(genre),
        "min_score": default_min_score(genre),
        "min_len": float(cons["min_length"]),
        "max_len": float(cons["max_length"]),
        "min_spacing": float(cfg.get("min_spacing_seconds", 20)),
        "max_clips": 8,
        "allow_overlap": False,
        "platform_bias": True,
    }


def platform_defaults(platform):
    cons = rank_clips.PLATFORM_PROFILES[platform]["constraints"]
    return float(cons["min_length"]), float(cons["max_length"])


def build_profile(settings):
    profile = copy.deepcopy(rank_clips.PLATFORM_PROFILES[settings["platform"]])
    profile["constraints"]["min_length"] = float(settings["min_len"])
    profile["constraints"]["max_length"] = float(settings["max_len"])
    if not settings.get("platform_bias", True):
        profile["boost"], profile["penalty"] = {}, {}
    return profile


def read_windows(signals_csv):
    return rank_clips.read_windows(str(signals_csv), signal_order())


def _transcript_text(segments, start, end, limit=220):
    text = " ".join(s["text"] for s in segments if s["end"] > start and s["start"] < end).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def score_candidates(windows, settings):
    """Every window that passes the length filter, scored with the user's weights."""
    order = signal_order()
    profile = build_profile(settings)
    weights = np.array([float(settings["weights"].get(s, 0.0)) for s in order])
    pool = [dict(w) for w in windows]
    pool = rank_clips.filter_by_constraints(pool, profile)
    pool = rank_clips.score_windows(pool, weights, profile)
    for w in pool:
        base = float(np.dot(w["signals"], weights)) * 10.0
        w["base"] = base
        w["bias"] = (w["score"] / base) if base > 1e-9 else 1.0
    return pool


def select_clips(pool, settings):
    """Pick the best windows above the score floor, honouring spacing / overlap / count."""
    passed = [w for w in pool if w["score"] >= float(settings["min_score"])]
    spacing = float(settings["min_spacing"])
    limit = int(settings.get("max_clips") or 0)
    chosen = []
    for w in sorted(passed, key=lambda x: x["score"], reverse=True):
        if limit and len(chosen) >= limit:
            break
        ok = True
        for c in chosen:
            if abs(w["start"] - c["start"]) < spacing:
                ok = False
                break
            if not settings.get("allow_overlap") and w["start"] < c["end"] and c["start"] < w["end"]:
                ok = False
                break
        if ok:
            chosen.append(w)
    return sorted(chosen, key=lambda x: x["start"])


def candidates_frame(clips, settings, segments):
    order = signal_order()
    weights = settings["weights"]
    rows = []
    for w in clips:
        contrib = {s: float(w["signal_dict"][s]) * float(weights.get(s, 0.0)) * 10.0 for s in order}
        top = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        rows.append({
            "start": round(w["start"], 2),
            "end": round(w["end"], 2),
            "duration": round(w["end"] - w["start"], 1),
            "score": round(w["score"], 2),
            "platform_x": round(w["bias"], 2),
            "driven_by": ", ".join(f"{s.replace('_', ' ')} {v:+.1f}" for s, v in top if abs(v) >= 0.05),
            "text": _transcript_text(segments, w["start"], w["end"]),
        })
    return pd.DataFrame(rows, columns=["start", "end", "duration", "score", "platform_x", "driven_by", "text"])


def contributions(window_signals, weights):
    """Per-signal contribution to the base score (signal value x weight x 10)."""
    order = signal_order()
    return pd.DataFrame({
        "signal": order,
        "contribution": [float(window_signals[s]) * float(weights.get(s, 0.0)) * 10.0 for s in order],
        "value": [float(window_signals[s]) for s in order],
    })


def find_window(windows, start, end):
    for w in windows:
        if abs(w["start"] - start) < 0.011 and abs(w["end"] - end) < 0.011:
            return w
    return None


# --------------------------------------------------------------------------- explore data

def timeline_frame(ws, smooth=5):
    """One row per second: audio excitement/volume plus face and motion scores."""
    ws = Path(ws)
    audio = load_json(ws / "audio_analysis.json", [])
    visual = load_json(ws / "visual_analysis.json", [])
    if isinstance(visual, dict):
        visual = visual.get("time_series", [])
    if not audio:
        return pd.DataFrame()
    a = pd.DataFrame(audio)
    a["t"] = np.floor(a["time"]).astype(int)
    df = a.groupby("t")[["excitement_score", "volume_score"]].mean()
    if visual:
        v = pd.DataFrame(visual)
        v["t"] = np.floor(v["time"]).astype(int)
        df = df.join(v.groupby("t")[["face_score", "motion_score"]].mean(), how="left")
    df = df.rename(columns={
        "excitement_score": "audio excitement", "volume_score": "volume",
        "face_score": "face presence", "motion_score": "motion",
    }).fillna(0.0)
    if smooth and smooth > 1:
        df = df.rolling(int(smooth), min_periods=1, center=True).mean()
    df.index.name = "second"
    return df.reset_index()


def transcript_segments(ws):
    return load_json(Path(ws) / "transcription.json", {}).get("segments", [])


# --------------------------------------------------------------------------- clip lists

FINAL_COLUMNS = ["keep", "start", "end", "name", "score"]


def load_final_clips(ws):
    data = load_json(Path(ws) / "final_clips.json", [])
    df = pd.DataFrame(data, columns=FINAL_COLUMNS)
    df["keep"] = df["keep"].fillna(True).astype(bool)
    return df


def save_final_clips(ws, df):
    """Persist the shortlist and mirror the kept clips to suggested_clips.json for the renderer."""
    ws = Path(ws)
    rows = []
    for r in df.to_dict("records"):
        if pd.isna(r.get("start")) or pd.isna(r.get("end")):
            continue
        rows.append({
            "keep": bool(r.get("keep", True)),
            "start": float(r["start"]), "end": float(r["end"]),
            "name": "" if pd.isna(r.get("name")) else str(r.get("name") or ""),
            "score": None if pd.isna(r.get("score")) else float(r["score"]),
        })
    save_json(ws / "final_clips.json", rows)
    save_json(ws / "suggested_clips.json", [
        {"start": r["start"], "end": r["end"], "score": r["score"] if r["score"] is not None else 0.0,
         **({"name": r["name"]} if r["name"] else {})}
        for r in rows if r["keep"] and r["end"] > r["start"]
    ])


# --------------------------------------------------------------------------- background jobs

def job_paths(ws):
    ws = Path(ws)
    return ws / "job_status.json", ws / "job.log"


def job_status(ws):
    status_file, _ = job_paths(ws)
    st = load_json(status_file)
    if st and st.get("state") == "running":
        # A crashed/killed runner leaves "running" behind; detect it by pid.
        try:
            os.kill(int(st["pid"]), 0)
        except (OSError, ValueError, TypeError):
            st["state"] = "failed"
            st["message"] = "Job process disappeared (killed or crashed). See the log."
    return st


def read_log(ws, lines=60):
    _, log_file = job_paths(ws)
    try:
        with open(log_file, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 60_000))
            text = f.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(text.splitlines()[-lines:])


def start_job(ws, kind, config):
    """Launch job_runner.py detached so it survives page reloads. Returns the pid."""
    ws = Path(ws)
    st = job_status(ws)
    if st and st.get("state") == "running":
        raise RuntimeError("A job is already running in this project.")
    cfg_path = ws / f"job_{kind}_config.json"
    save_json(cfg_path, config)
    _, log_file = job_paths(ws)
    log = open(log_file, "wb")
    proc = subprocess.Popen(
        [sys.executable, "-u", str(ROOT / "job_runner.py"), kind, "--workspace", str(ws), "--config", str(cfg_path)],
        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    log.close()
    # Give the runner a moment to write its first status so the UI never sees a stale one.
    for _ in range(40):
        st = load_json(job_paths(ws)[0])
        if st and st.get("pid") == proc.pid:
            break
        time.sleep(0.05)
    return proc.pid


def cancel_job(ws):
    st = load_json(job_paths(ws)[0])
    if not st or st.get("state") != "running":
        return False
    try:
        os.killpg(os.getpgid(int(st["pid"])), signal.SIGTERM)
    except (OSError, ValueError):
        return False
    return True


def render_runs(ws):
    """Rendered clips grouped by folder, newest first: [(folder_label, [mp4 paths])]."""
    shorts = Path(ws) / "shorts"
    if not shorts.exists():
        return []
    groups = {}
    for p in shorts.rglob("*.mp4"):
        groups.setdefault(p.parent, []).append(p)
    runs = []
    for folder, files in groups.items():
        label = "earlier renders" if folder == shorts else folder.name
        files.sort(key=lambda p: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", p.name)])
        runs.append((label, folder, files, max(f.stat().st_mtime for f in files)))
    runs.sort(key=lambda r: r[3], reverse=True)
    return [(label, files) for label, _, files, _ in runs]


# --------------------------------------------------------------------------- misc helpers

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3"]


def whisper_cached(name):
    """True if the faster-whisper model is already on disk (so no download is needed)."""
    hub = Path(os.environ.get("HF_HUB_CACHE") or (Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"))
    return (hub / f"models--Systran--faster-whisper-{name}").exists()


def cut_preview(video, start, end, out_path, width=640):
    """Fast low-res cut of the source so the browser never has to load the full video."""
    from video_renderer import available_encoders
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    encs = available_encoders() or ["libx264"]
    for enc in encs:
        args = ["-c:v", enc] + (["-crf", "28", "-preset", "veryfast"] if enc == "libx264" else ["-b:v", "1500k"])
        cmd = ["ffmpeg", "-y", "-ss", f"{max(0.0, start):.3f}", "-t", f"{end - start:.3f}", "-i", str(video),
               "-vf", f"scale={width}:-2", *args, "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k",
               "-movflags", "+faststart", str(out_path)]
        if subprocess.run(cmd, capture_output=True).returncode == 0:
            return out_path
    return None


def system_fonts():
    """Font families known to fontconfig (what the caption renderer can use)."""
    try:
        out = subprocess.run(["fc-list", ":", "family"], capture_output=True, text=True, check=True).stdout
    except Exception:
        return []
    return sorted({line.split(",")[0].strip() for line in out.splitlines() if line.strip()})
