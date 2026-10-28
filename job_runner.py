"""Runs an analysis or render job in the background for the UI.

    python job_runner.py analyze --workspace W --config cfg.json
    python job_runner.py render  --workspace W --config cfg.json

Progress is written to <workspace>/job_status.json (polled by the UI) and all output goes to
stdout, which the UI redirects to <workspace>/job.log.
"""
import argparse
import datetime
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import clip_engine as ce

PY = sys.executable
ROOT = ce.ROOT


class Job:
    def __init__(self, ws, kind, steps):
        self.ws = Path(ws)
        self.status_file, _ = ce.job_paths(self.ws)
        self.state = {
            "kind": kind, "state": "running", "pid": os.getpid(),
            "started": time.time(), "finished": None, "message": "",
            "progress": 0.0,
            "steps": [{"id": sid, "label": label, "state": "pending", "note": ""} for sid, label in steps],
        }
        self.child = None
        self.write()

    def write(self):
        ce.save_json(self.status_file, self.state)

    def step(self, sid):
        return next(s for s in self.state["steps"] if s["id"] == sid)

    def set_step(self, sid, state, note=""):
        s = self.step(sid)
        s["state"], s["note"] = state, note
        done = sum(1 for x in self.state["steps"] if x["state"] in ("done", "skipped"))
        self.state["progress"] = done / max(1, len(self.state["steps"]))
        self.write()

    def finish(self, state, message=""):
        self.state.update(state=state, message=message, finished=time.time())
        if state == "done":
            self.state["progress"] = 1.0
        self.write()

    def run(self, cmd, sid=None):
        """Run a child process, tee its output to the log, and parse 'PROGRESS i/n' lines."""
        print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
        self.child = subprocess.Popen(
            [str(c) for c in cmd], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        pat = re.compile(rb"PROGRESS (\d+)/(\d+)")
        for raw in iter(self.child.stdout.readline, b""):
            sys.stdout.buffer.write(raw)
            sys.stdout.buffer.flush()
            m = pat.search(raw)
            if m and sid:
                i, n = int(m.group(1)), max(1, int(m.group(2)))
                self.state["progress"] = i / n
                self.step(sid)["note"] = f"{i}/{n} clips"
                self.write()
        code = self.child.wait()
        self.child = None
        return code


def analyze(job, ws, cfg):
    src = cfg.get("source", {"type": "existing"})
    force = set(cfg.get("force", []))
    run_hooks = cfg.get("hooks", False)
    ran = set()

    def should_run(sid):
        if sid in force or not ce._step_done(ws, sid):
            return True
        return any(dep in ran for dep in ce.STEP_DEPS[sid])

    commands = {
        "ingest": lambda: [PY, "video_ingestion.py", "--workspace", ws,
                           *(["--file", src["path"]] if src["type"] == "file" else ["--url", src["url"]]),
                           *(["--max-height", src["max_height"]] if src.get("max_height") else [])],
        "transcribe": lambda: [PY, "audio_transcription.py", "--workspace", ws,
                               "--model", cfg.get("whisper_model", "large-v3"),
                               "--device", cfg.get("device", "auto"),
                               *(["--language", cfg["language"]] if cfg.get("language") else [])],
        "audio": lambda: [PY, "audio_analysis.py", "--workspace", ws,
                          "--vol-weight", cfg.get("vol_weight", 0.6), "--pitch-weight", 1 - cfg.get("vol_weight", 0.6)],
        "text": lambda: [PY, "text_analysis.py", "--workspace", ws],
        "visual": lambda: [PY, "visual_analysis.py", "--workspace", ws],
        "hooks": lambda: [PY, "hook_detector.py", "--workspace", ws],
    }

    for sid, _ in [(s["id"], s["label"]) for s in job.state["steps"]]:
        if sid == "ingest" and src["type"] == "existing":
            if ce.find_video(ws) is None:
                raise RuntimeError("This project has no video yet. Upload a file or paste a URL.")
            job.set_step(sid, "skipped", "using the video already in the project")
            continue
        if sid == "hooks" and not run_hooks:
            job.set_step(sid, "skipped", "disabled")
            continue
        if not should_run(sid):
            job.set_step(sid, "skipped", "already done")
            continue

        job.set_step(sid, "running")
        if sid == "signals":
            # Same merge the CLI orchestrator does, run in-process.
            import run_pipeline
            try:
                run_pipeline.PipelineOrchestrator(str(ws)).convert_json_to_signals_csv()
            except SystemExit:
                job.set_step(sid, "failed", "missing audio/transcript data")
                raise RuntimeError("Could not build candidate windows (missing audio or transcript data).")
        else:
            code = job.run(commands[sid](), sid)
            if code != 0:
                job.set_step(sid, "failed", f"exit code {code}")
                raise RuntimeError(f"Step '{sid}' failed (exit code {code}). See the log.")
        if not ce._step_done(ws, sid) and sid != "hooks":
            job.set_step(sid, "failed", "no output produced")
            raise RuntimeError(f"Step '{sid}' finished without producing its output. See the log.")
        ran.add(sid)
        job.set_step(sid, "done")


def render(job, ws, cfg):
    job.set_step("render", "running")
    out_dir = Path(cfg["out_dir"])
    code = job.run([PY, "video_renderer.py", "--workspace", ws, "--clips", cfg["clips"],
                    "--options", cfg["options"], "--out-dir", out_dir], "render")
    if code != 0:
        job.set_step("render", "failed", "no clips were rendered")
        raise RuntimeError("Rendering failed. See the log.")
    job.set_step("render", "done", f"saved to {out_dir.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["analyze", "render"])
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    ws = Path(args.workspace).resolve()
    cfg = ce.load_json(args.config, {})
    steps = ([(sid, label) for sid, (label, _) in ce.ANALYZE_STEPS.items()]
             if args.kind == "analyze" else [("render", "Render clips")])
    job = Job(ws, args.kind, steps)

    def on_term(signum, frame):
        if job.child is not None:
            job.child.terminate()
        job.finish("cancelled", "Cancelled by user.")
        os._exit(143)

    signal.signal(signal.SIGTERM, on_term)

    print(f"[{datetime.datetime.now():%H:%M:%S}] {args.kind} job started in {ws}", flush=True)
    try:
        (analyze if args.kind == "analyze" else render)(job, ws, cfg)
    except Exception as e:  # noqa: BLE001 - surfaced to the UI
        print(f"\n❌ {e}", flush=True)
        job.finish("failed", str(e))
        sys.exit(1)
    print(f"\n✅ {args.kind} job finished", flush=True)
    job.finish("done", "Finished.")


if __name__ == "__main__":
    main()
