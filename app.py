"""Clipper Studio - Streamlit UI for the video-clipper pipeline.

    streamlit run app.py
"""
import hashlib
import io
import json
import shutil
import time
import zipfile
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import clip_engine as ce
from video_renderer import DEFAULT_OPTIONS, VideoRenderer, available_encoders

alt.data_transformers.disable_max_rows()

st.set_page_config(page_title="Clipper Studio", page_icon="🎬", layout="wide")
ss = st.session_state

SIGNAL_HELP = {
    "audio_excitation": "Loudness and pitch energy of the speech in the window.",
    "speech_rate_change": "Placeholder - the pipeline feeds a constant, so it has no effect yet.",
    "silence_breaks": "Share of near-silent time. Usually weighted negative.",
    "semantic_novelty": "How different the passage is from the rest of the video (boosted by detected hooks).",
    "sentiment_intensity": "Emotional language detected in the passage.",
    "hook_phrase_score": "Questions / scroll-stopping lines (uses the hook LLM if you enabled it).",
    "face_presence": "How much of the window has a face on screen.",
    "face_motion": "Visual movement while a face is present.",
    "scene_change_rate": "Visual change over the window (currently the same measure as face motion).",
    "laughter_or_reaction": "Placeholder - the pipeline feeds a constant, so it has no effect yet.",
}
LANGUAGES = ["", "hi", "en", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "ur",
             "es", "fr", "de", "pt", "it", "ja", "ko", "zh", "ar", "ru"]
OUTPUT_SIZES = {
    "1080 × 1920 · 9:16 (Shorts / Reels / TikTok)": (1080, 1920),
    "720 × 1280 · 9:16 (smaller file)": (720, 1280),
    "1080 × 1350 · 4:5 (Instagram feed)": (1080, 1350),
    "1080 × 1080 · 1:1 (square)": (1080, 1080),
}
HEIGHT_CAPS = {"Best available": 0, "2160p": 2160, "1440p": 1440, "1080p (recommended)": 1080, "720p": 720, "480p": 480}
STEP_ICON = {"pending": "⬜", "running": "⏳", "done": "✅", "skipped": "⏭️", "failed": "❌"}
SCORING_KEYS = ["preset", "platform", "min_score", "len", "min_spacing", "max_clips", "allow_overlap", "platform_bias"]


# =========================================================================== state

def load_project_state(ws):
    """Fill session_state from the project's saved settings (or defaults)."""
    saved = ce.load_json(ws / "ui_state.json", {})
    order = ce.signal_order()

    sc = {**ce.default_scoring(), **saved.get("scoring", {})}
    sc["weights"] = {**ce.default_scoring()["weights"], **saved.get("scoring", {}).get("weights", {})}
    for s in order:
        ss[f"w_{s}"] = float(sc["weights"][s])
    ss["sc_preset"] = sc["preset"] if sc["preset"] in ce.genre_names() + list(ce.user_presets()) else "podcast"
    ss["sc_platform"] = sc["platform"]
    ss["sc_min_score"] = float(sc["min_score"])
    ss["sc_len"] = (float(sc["min_len"]), float(sc["max_len"]))
    ss["sc_min_spacing"] = float(sc["min_spacing"])
    ss["sc_max_clips"] = int(sc["max_clips"])
    ss["sc_allow_overlap"] = bool(sc["allow_overlap"])
    ss["sc_platform_bias"] = bool(sc["platform_bias"])

    rd = {**DEFAULT_OPTIONS, **saved.get("render", {})}
    for k, v in rd.items():
        ss[f"rd_{k}"] = v
    size = next((label for label, wh in OUTPUT_SIZES.items() if wh == (rd["width"], rd["height"])), list(OUTPUT_SIZES)[0])
    ss["rd_size"] = size

    an = {**AN_DEFAULTS, **saved.get("analysis", {})}
    for k, v in an.items():
        ss[f"an_{k}"] = v

    ss["final_base"] = ce.load_final_clips(ws)
    ss["final_current"] = ss["final_base"].copy()
    ss["final_ver"] = 0
    ss["_loaded_ws"] = str(ws)
    ss["_saved_state"] = json.dumps(saved, sort_keys=True)


def scoring_from_state():
    d = ce.default_scoring()
    lo, hi = ss.get("sc_len", (d["min_len"], d["max_len"]))
    return {
        "preset": ss.get("sc_preset", d["preset"]), "platform": ss.get("sc_platform", d["platform"]),
        "weights": {s: float(ss.get(f"w_{s}", d["weights"][s])) for s in ce.signal_order()},
        "min_score": float(ss.get("sc_min_score", d["min_score"])), "min_len": float(lo), "max_len": float(hi),
        "min_spacing": float(ss.get("sc_min_spacing", d["min_spacing"])),
        "max_clips": int(ss.get("sc_max_clips", d["max_clips"])),
        "allow_overlap": bool(ss.get("sc_allow_overlap", d["allow_overlap"])),
        "platform_bias": bool(ss.get("sc_platform_bias", d["platform_bias"])),
    }


def render_options_from_state():
    opts = {k: ss.get(f"rd_{k}", v) for k, v in DEFAULT_OPTIONS.items()}
    opts["width"], opts["height"] = OUTPUT_SIZES.get(ss.get("rd_size"), (DEFAULT_OPTIONS["width"], DEFAULT_OPTIONS["height"]))
    return opts


AN_DEFAULTS = {"whisper_model": "large-v3", "language": "", "device": "auto", "vol_weight": 0.6,
               "hooks": False, "height_cap": "1080p (recommended)"}


def analysis_from_state():
    return {k: ss.get(f"an_{k}", v) for k, v in AN_DEFAULTS.items()}


def persist(ws):
    state = json.loads(ss.get("_saved_state") or "{}")
    state["render"] = render_options_from_state()
    state["analysis"] = analysis_from_state()
    if ss.get("_scoring_rendered"):  # the scoring widgets only exist once signals exist
        state["scoring"] = scoring_from_state()
    dumped = json.dumps(state, sort_keys=True)
    if dumped != ss.get("_saved_state"):
        ce.save_json(ws / "ui_state.json", state)
        ss["_saved_state"] = dumped


# =========================================================================== callbacks

def cb_apply_preset():
    name = ss["sc_preset"]
    for s, v in ce.preset_weights(name).items():
        ss[f"w_{s}"] = v
    if name in ce.genre_names():
        ss["sc_min_score"] = ce.default_min_score(name)


def cb_platform():
    ss["sc_len"] = ce.platform_defaults(ss["sc_platform"])


def cb_save_preset():
    name = ss["new_preset_name"].strip()
    if name:
        ce.save_user_preset(name, {s: ss[f"w_{s}"] for s in ce.signal_order()})
        ss["sc_preset"] = name
        ss["new_preset_name"] = ""


def cb_delete_preset():
    ce.delete_user_preset(ss["sc_preset"])
    ss["sc_preset"] = "podcast"
    cb_apply_preset()


def _final_set(df):
    ss["final_base"] = df.reset_index(drop=True)
    ss["final_current"] = ss["final_base"].copy()
    ss["final_ver"] += 1
    ce.save_final_clips(Path(ss["_loaded_ws"]), ss["final_base"])


def cb_add_to_final(rows):
    """rows: DataFrame of candidates to append (start, end, score)."""
    cur = ss["final_current"].copy()
    have = {(round(r.start, 1), round(r.end, 1)) for r in cur.itertuples() if pd.notna(r.start)}
    add = [{"keep": True, "start": r.start, "end": r.end, "name": "", "score": r.score}
           for r in rows.itertuples() if (round(r.start, 1), round(r.end, 1)) not in have]
    if add:
        cur = pd.concat([cur, pd.DataFrame(add)], ignore_index=True)
    _final_set(cur.sort_values("start"))


def cb_add_selected():
    picked = ss.get("_cands_df")
    rows = ss.get("cand_table", {}).get("selection", {}).get("rows", [])
    if picked is not None and rows:
        cb_add_to_final(picked.iloc[rows])


def cb_add_all():
    if ss.get("_cands_df") is not None:
        cb_add_to_final(ss["_cands_df"])


def cb_clear_final():
    _final_set(pd.DataFrame(columns=ce.FINAL_COLUMNS))


def cb_sort_final():
    _final_set(ss["final_current"].sort_values("start"))


# =========================================================================== cached data

@st.cache_data(show_spinner=False)
def cached_windows(csv_path, mtime):
    return ce.read_windows(csv_path)


@st.cache_data(show_spinner=False)
def cached_rank(csv_path, mtime, settings_json):
    settings = json.loads(settings_json)
    pool = ce.score_candidates(cached_windows(csv_path, mtime), settings)
    picks = ce.select_clips(pool, settings)
    return {
        "scores": [w["score"] for w in pool], "total": len(pool),
        "passed": sum(1 for w in pool if w["score"] >= settings["min_score"]), "picks": picks,
    }


@st.cache_data(show_spinner=False)
def cached_timeline(ws, mtime, smooth):
    return ce.timeline_frame(ws, smooth)


@st.cache_data(show_spinner=False)
def cached_segments(ws, mtime):
    return ce.transcript_segments(ws)


@st.cache_data(show_spinner=False)
def cached_fonts():
    return ce.system_fonts()


@st.cache_data(show_spinner=False)
def cached_encoders():
    return available_encoders()


def mtime(path):
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return 0.0


# =========================================================================== jobs

def job_body(ws_str, kind):
    ws = Path(ws_str)
    st_ = ce.job_status(ws)
    flag = f"_job_running_{kind}"
    if not st_ or st_.get("kind") != kind:
        return
    running = st_["state"] == "running"
    if running:
        ss[flag] = True
    elif ss.pop(flag, False):
        st.rerun()  # job just finished: refresh the rest of the app

    label = {"running": "⏳ Running", "done": "✅ Finished", "failed": "❌ Failed", "cancelled": "⏹ Cancelled"}[st_["state"]]
    elapsed = (st_["finished"] or time.time()) - st_["started"]
    st.markdown(f"**{kind.capitalize()} job · {label}** · {int(elapsed // 60)}m {int(elapsed % 60)}s")
    st.progress(min(1.0, st_.get("progress", 0.0)))
    for s in st_["steps"]:
        note = f" — {s['note']}" if s["note"] else ""
        st.write(f"{STEP_ICON.get(s['state'], '⬜')} {s['label']}{note}")
    if st_.get("message") and st_["state"] in ("failed", "cancelled"):
        st.error(st_["message"])
    if running and st.button("⏹ Cancel job", key=f"cancel_{kind}"):
        ce.cancel_job(ws)
        st.rerun()
    with st.expander("Log", expanded=(st_["state"] == "failed")):
        st.code(ce.read_log(ws) or "(empty)", language=None, height=260)


def job_panel(ws, kind):
    st_ = ce.job_status(ws)
    running = bool(st_ and st_.get("kind") == kind and st_["state"] == "running")
    st.fragment(job_body, run_every=2 if running else None)(str(ws), kind)


def any_job_running(ws):
    st_ = ce.job_status(ws)
    return bool(st_ and st_["state"] == "running")


# =========================================================================== sidebar / project

def sidebar():
    st.sidebar.title("🎬 Clipper Studio")
    projects = ce.list_projects()

    with st.sidebar.expander("➕ New project", expanded=not projects):
        name = st.text_input("Project name", key="new_project_name", placeholder="my-podcast-ep12")
        if st.button("Create project", disabled=not name.strip()):
            path = ce.create_project(name)
            ss["project"] = path.name
            ss["new_project_name"] = ""
            st.rerun()
        projects = ce.list_projects()

    if not projects:
        st.info("Create a project in the sidebar to get started.")
        st.stop()

    labels = [p[0] for p in projects]
    paths = dict(projects)
    if ss.get("project") not in labels:
        ss["project"] = labels[0]
    label = st.sidebar.selectbox("Project", labels, key="project")
    ws = paths[label]

    st.sidebar.caption(f"📁 `{ws}`")
    encs = cached_encoders()
    st.sidebar.caption("🎞️ Video encoders: " + (", ".join(encs) if encs else "none found - install ffmpeg"))
    return ws


# =========================================================================== tabs

def tab_source(ws):
    status = ce.workspace_status(ws)
    busy = any_job_running(ws)
    video = ce.find_video(ws)

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("1 · Video")
        if video:
            info = ce.probe_video(video)
            if info:
                a, b, c, d = st.columns(4)
                a.metric("Length", ce.fmt_time(info["duration"]))
                b.metric("Resolution", f"{info['width']}×{info['height']}")
                c.metric("Codec", info["codec"])
                d.metric("Size", f"{info['size_mb']:.0f} MB")
            st.caption(f"Current video: `{video.name}`")

        modes = (["Keep current video"] if video else []) + ["Upload a file", "YouTube / URL", "File on this computer"]
        mode = st.radio("Source", modes, horizontal=True, key="src_mode")
        source = {"type": "existing"}
        ready = bool(video) if mode == "Keep current video" else False
        if mode == "Upload a file":
            up = st.file_uploader("Video file", type=["mp4", "mov", "mkv", "webm", "avi", "m4v"], key="src_upload")
            ready = up is not None
            source = {"type": "upload", "file": up}
        elif mode == "YouTube / URL":
            url = st.text_input("Video URL", key="src_url", placeholder="https://www.youtube.com/watch?v=…")
            cap = st.selectbox("Max download quality", list(HEIGHT_CAPS), key="an_height_cap",
                               help="1080p keeps downloads small and makes decoding and rendering much faster than 4K.")
            ready = bool(url.strip())
            source = {"type": "url", "url": url.strip(), "max_height": HEIGHT_CAPS[cap]}
        elif mode == "File on this computer":
            path = st.text_input("Path to a video file", key="src_path", placeholder="/home/you/videos/talk.mp4")
            ready = bool(path.strip()) and Path(path.strip()).expanduser().is_file()
            if path.strip() and not ready:
                st.warning("File not found.")
            source = {"type": "file", "path": str(Path(path.strip()).expanduser())}

        st.subheader("2 · Analysis settings")
        c1, c2 = st.columns(2)
        c1.selectbox(
            "Whisper model", ce.WHISPER_MODELS, key="an_whisper_model",
            format_func=lambda m: f"{m}  ({'on disk' if ce.whisper_cached(m) else 'downloads on first use'})",
            help="Bigger = more accurate, especially for Hindi / Hinglish, but slower and needs more VRAM.")
        c2.selectbox("Spoken language", LANGUAGES, key="an_language", accept_new_options=True,
                     format_func=lambda x: x or "Auto-detect")
        c1.radio("Device", ["auto", "cuda", "cpu"], horizontal=True, key="an_device")
        c2.slider("Audio excitement: loudness ↔ pitch", 0.0, 1.0, key="an_vol_weight", step=0.05,
                  help="1.0 = only loudness counts, 0.0 = only pitch variation counts.")
        st.toggle("Detect hooks with a small LLM (Qwen2.5-1.5B) - slower, boosts hook scores", key="an_hooks")

        steps = list(ce.ANALYZE_STEPS)
        st.multiselect(
            "Re-run these steps even if already done", steps, key="an_force",
            format_func=lambda s: f"{STEP_ICON['done' if status[s] else 'pending']} {ce.ANALYZE_STEPS[s][0]}",
            help="Steps that depend on a re-run step are redone automatically.")

        label = "▶ Run analysis" if source["type"] != "existing" or not all(status.values()) else "▶ Re-run with these settings"
        if st.button(label, type="primary", disabled=busy or not ready, width="stretch"):
            cfg = {
                "whisper_model": ss["an_whisper_model"], "language": ss["an_language"] or None,
                "device": ss["an_device"], "vol_weight": ss["an_vol_weight"], "hooks": ss["an_hooks"],
                "force": list(ss.get("an_force", [])),
            }
            if source["type"] == "upload":
                (ws / "video").mkdir(parents=True, exist_ok=True)
                for old in (ws / "video").glob("input_video.*"):
                    old.unlink()
                dest = ws / "video" / f"input_video{Path(source['file'].name).suffix.lower()}"
                with open(dest, "wb") as f:
                    shutil.copyfileobj(source["file"], f)
                cfg["source"] = {"type": "file", "path": str(dest)}
            else:
                cfg["source"] = source
            if cfg["source"]["type"] != "existing":
                cfg["force"] = sorted(set(cfg["force"]) | {"ingest"})
                for stale in ("final_clips.json", "suggested_clips.json"):
                    (ws / stale).unlink(missing_ok=True)  # they point into the previous video
                ss["final_base"] = pd.DataFrame(columns=ce.FINAL_COLUMNS)
                ss["final_current"] = ss["final_base"].copy()
                ss["final_ver"] += 1
            persist(ws)
            ce.start_job(ws, "analyze", cfg)
            st.rerun()
        if source["type"] != "existing":
            st.caption("A new video replaces the project's current one and clears its clip list.")

    with right:
        st.subheader("Pipeline status")
        for sid, (label, _) in ce.ANALYZE_STEPS.items():
            st.write(f"{STEP_ICON['done' if status[sid] else 'pending']} {label}")
        if all(status.values()):
            st.success("Analysis complete - head to **Score & Clips**.")
        st.divider()
        job_panel(ws, "analyze")


def tab_explore(ws, final_df, picks):
    sig = ws / "signals.csv"
    tl_file = ws / "audio_analysis.json"
    if not tl_file.exists():
        st.info("Run the analysis first to see signals and the transcript.")
        return

    st.subheader("Signal timeline")
    c1, c2 = st.columns([3, 1])
    options = ["audio excitement", "volume", "face presence", "motion"]
    shown = c1.multiselect("Signals", options, default=["audio excitement", "motion"], key="tl_signals")
    smooth = c2.slider("Smoothing (s)", 1, 30, 8, key="tl_smooth")
    tl = cached_timeline(str(ws), mtime(tl_file) + mtime(ws / "visual_analysis.json"), smooth)
    shown = [s for s in shown if s in tl.columns]
    if tl.empty or not shown:
        st.caption("Nothing to plot yet.")
    else:
        step = max(1, len(tl) // 2500)
        long = tl.iloc[::step].melt("second", value_vars=shown, var_name="signal", value_name="value")
        lines = alt.Chart(long).mark_line().encode(
            x=alt.X("second:Q", title="time (s)"), y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1]), title=None),
            color=alt.Color("signal:N", legend=alt.Legend(orient="bottom", title=None)))
        bands = [{"start": p["start"], "end": p["end"], "kind": "auto pick"} for p in picks]
        bands += [{"start": r.start, "end": r.end, "kind": "final list"}
                  for r in final_df.itertuples() if r.keep and pd.notna(r.start) and pd.notna(r.end)]
        chart = lines
        if bands:
            rects = alt.Chart(pd.DataFrame(bands)).mark_rect(opacity=0.28).encode(
                x="start:Q", x2="end:Q",
                color=alt.Color("kind:N", scale=alt.Scale(domain=["auto pick", "final list"], range=["#8a8f98", "#e4572e"]),
                                legend=alt.Legend(orient="bottom", title=None)))
            chart = rects + lines
        st.altair_chart(chart.properties(height=260).interactive(bind_y=False), width="stretch")
        st.caption("Grey = clips the current scoring settings would pick · orange = your final list. Drag to pan, scroll to zoom.")

    st.subheader("Transcript")
    segs = cached_segments(str(ws), mtime(ws / "transcription.json"))
    q = st.text_input("Search", key="tr_search", placeholder="Find a word or phrase…")
    kept = [(r.start, r.end) for r in final_df.itertuples() if r.keep and pd.notna(r.start)]
    rows = []
    for s in segs:
        if q and q.lower() not in s["text"].lower():
            continue
        in_clip = next((i + 1 for i, (a, b) in enumerate(kept) if s["end"] > a and s["start"] < b), None)
        rows.append({"time": ce.fmt_time(s["start"]), "text": s["text"], "in final clip": f"#{in_clip}" if in_clip else ""})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=380)


def tab_clips(ws, windows, rank, segments):
    left, right = st.columns([2, 3], gap="large")

    with left:
        st.subheader("Scoring")
        options = ce.genre_names() + list(ce.user_presets())
        st.selectbox("Preset (signal weights)", options, key="sc_preset", on_change=cb_apply_preset,
                     format_func=lambda n: f"★ {n}" if n in ce.user_presets() else n,
                     help="Starting weights tuned per content type. Edit any weight below, then save your own preset.")
        st.selectbox("Platform", list(ce.rank_clips.PLATFORM_PROFILES), key="sc_platform", on_change=cb_platform,
                     help="Each platform boosts different signals and has its own clip-length limits.")
        st.toggle("Apply platform boosts / penalties", key="sc_platform_bias")

        with st.expander("Signal weights", expanded=False):
            for s in ce.signal_order():
                st.slider(s.replace("_", " "), -0.5, 1.0, key=f"w_{s}", step=0.01, help=SIGNAL_HELP.get(s))
            total = sum(ss[f"w_{s}"] for s in ce.signal_order())
            st.caption(f"Weights sum to **{total:.2f}** (the built-in presets sum to ≈0.93 → max score ≈10).")
            st.button("↺ Reset to preset", on_click=cb_apply_preset)
            c1, c2 = st.columns([3, 2])
            c1.text_input("Save as preset", key="new_preset_name", placeholder="my-comedy-mix", label_visibility="collapsed")
            c2.button("💾 Save preset", on_click=cb_save_preset, width="stretch")
            if ss["sc_preset"] in ce.user_presets():
                st.button("🗑 Delete this preset", on_click=cb_delete_preset)

        st.subheader("Selection")
        st.slider("Minimum score", 0.0, 10.0, key="sc_min_score", step=0.1,
                  help="Only moments scoring at least this are considered. The histogram on the right shows where to set it.")
        st.slider("Clip length (seconds)", 3.0, 90.0, key="sc_len", step=1.0)
        st.slider("Minimum gap between clip starts (s)", 0.0, 180.0, key="sc_min_spacing", step=5.0)
        c1, c2 = st.columns(2)
        c1.number_input("Max clips (0 = no limit)", 0, 100, key="sc_max_clips")
        c2.checkbox("Allow overlapping clips", key="sc_allow_overlap")

    with right:
        c1, c2, c3 = st.columns(3)
        c1.metric("Candidate moments", f"{rank['total']:,}", help="Windows that fit the clip-length range")
        c2.metric("Above minimum score", f"{rank['passed']:,}")
        c3.metric("Picked", len(rank["picks"]))

        if rank["scores"]:
            hist = alt.Chart(pd.DataFrame({"score": rank["scores"]})).mark_bar(opacity=0.8).encode(
                x=alt.X("score:Q", bin=alt.Bin(maxbins=40), title="moment score"), y=alt.Y("count()", title=None))
            rule = alt.Chart(pd.DataFrame({"x": [ss["sc_min_score"]]})).mark_rule(color="#e4572e", strokeWidth=2).encode(x="x:Q")
            st.altair_chart((hist + rule).properties(height=110), width="stretch")

        st.subheader("Suggested clips")
        cands = ce.candidates_frame(rank["picks"], scoring_from_state(), segments).sort_values("score", ascending=False)
        cands = cands.reset_index(drop=True)
        ss["_cands_df"] = cands
        if cands.empty:
            st.warning("No moments pass the current settings - lower the minimum score or widen the clip length.")
        else:
            st.dataframe(
                cands, hide_index=True, width="stretch", key="cand_table", on_select="rerun", selection_mode="multi-row",
                column_config={
                    "start": st.column_config.NumberColumn("start (s)", format="%.1f"),
                    "end": st.column_config.NumberColumn("end (s)", format="%.1f"),
                    "duration": st.column_config.NumberColumn("len (s)", format="%.1f"),
                    "score": st.column_config.ProgressColumn("score", min_value=0, max_value=10, format="%.2f"),
                    "platform_x": st.column_config.NumberColumn("platform ×", format="%.2f"),
                    "driven_by": "driven by",
                    "text": st.column_config.TextColumn("transcript", width="large"),
                })
            b1, b2 = st.columns(2)
            b1.button("➕ Add selected rows to final list", on_click=cb_add_selected, width="stretch")
            b2.button("➕ Add all suggestions", on_click=cb_add_all, width="stretch")

            sel = ss.get("cand_table", {}).get("selection", {}).get("rows", [])
            if sel:
                row = cands.iloc[sel[0]]
                win = ce.find_window(rank["picks"], row.start, row.end)
                with st.container(border=True):
                    st.markdown(f"**Why this moment?** {ce.fmt_time(row.start)}–{ce.fmt_time(row.end)} · score **{row.score:.2f}**")
                    if win is not None:
                        contrib = ce.contributions(win["signal_dict"], ss_weights())
                        st.bar_chart(contrib.set_index("signal")["contribution"], horizontal=True, height=260)
                        st.caption(f"Base {win['base']:.2f} × platform {win['bias']:.2f} = {row.score:.2f} (capped at 10)")
                    st.write(row.text)
                    video = ce.find_video(ws)
                    if video and st.button("▶ Preview this moment", key=f"pv_{row.start}_{row.end}"):
                        out = ce.cut_preview(video, row.start, row.end, ws / "_preview" / "moment.mp4")
                        if out:
                            st.video(out.read_bytes(), format="video/mp4")
                        else:
                            st.error("Could not cut a preview (is ffmpeg installed?).")

        st.subheader("Final clips")
        st.caption("Edit start/end to trim, rename, untick to skip, or add rows manually. Changes save automatically.")
        edited = st.data_editor(
            ss["final_base"], num_rows="dynamic", width="stretch", hide_index=True, key=f"final_editor_{ss['final_ver']}",
            column_config={
                "keep": st.column_config.CheckboxColumn("render", default=True),
                "start": st.column_config.NumberColumn("start (s)", format="%.1f", step=0.1, min_value=0.0, required=True),
                "end": st.column_config.NumberColumn("end (s)", format="%.1f", step=0.1, min_value=0.0, required=True),
                "name": st.column_config.TextColumn("name (optional)"),
                "score": st.column_config.NumberColumn("score", format="%.2f", disabled=True),
            })
        edited["keep"] = edited["keep"].fillna(True).astype(bool)
        if not edited.reset_index(drop=True).equals(ss["final_current"].reset_index(drop=True)):
            ss["final_current"] = edited.reset_index(drop=True)
            ce.save_final_clips(ws, edited)
        kept = edited[edited["keep"] & edited["start"].notna() & edited["end"].notna()]
        bad = kept[kept["end"] <= kept["start"]]
        if len(bad):
            st.error(f"{len(bad)} clip(s) end before they start - fix them before rendering.")
        a, b, c = st.columns(3)
        a.metric("Clips to render", len(kept))
        b.metric("Total length", ce.fmt_time((kept["end"] - kept["start"]).clip(lower=0).sum()))
        c.button("↕ Sort by start", on_click=cb_sort_final, width="stretch")
        st.button("🗑 Clear list", on_click=cb_clear_final)


def ss_weights():
    return {s: float(ss[f"w_{s}"]) for s in ce.signal_order()}


def tab_render(ws, final_df):
    kept = final_df[final_df["keep"] & final_df["start"].notna() & final_df["end"].notna() & (final_df["end"] > final_df["start"])]
    busy = any_job_running(ws)

    left, right = st.columns([2, 3], gap="large")
    with left:
        st.subheader("Look")
        st.selectbox("Output size", list(OUTPUT_SIZES), key="rd_size")
        st.radio("Framing", ["fill_crop", "fit_blur"], key="rd_layout", horizontal=True,
                 format_func=lambda x: {"fill_crop": "Fill (crop the sides)", "fit_blur": "Fit + blurred background"}[x],
                 help="Fill crops the 16:9 picture to vertical. Fit keeps the whole frame over a blurred copy of itself.")
        c1, c2 = st.columns(2)
        c1.slider("Crop focus (left ↔ right)", 0.0, 1.0, key="rd_crop_focus_x", step=0.05,
                  disabled=ss["rd_layout"] != "fill_crop", help="Where the vertical crop sits horizontally. Move it to follow the speaker.")
        c2.slider("Background blur", 5, 60, key="rd_blur_strength", disabled=ss["rd_layout"] != "fit_blur")
        c1, c2 = st.columns(2)
        c1.slider("Pad before (s)", 0.0, 5.0, key="rd_pad_before", step=0.1)
        c2.slider("Pad after (s)", 0.0, 5.0, key="rd_pad_after", step=0.1)

        st.subheader("Captions")
        st.toggle("Burn in captions", key="rd_captions")
        off = not ss["rd_captions"]
        fonts = cached_fonts()
        if ss["rd_font"] not in fonts:
            fonts = [ss["rd_font"]] + fonts
        c1, c2 = st.columns(2)
        c1.selectbox("Font", fonts, key="rd_font", disabled=off, accept_new_options=True)
        c2.slider("Words per line", 1, 8, key="rd_words_per_line", disabled=off)
        c1.slider("Font size (px)", 30, 160, key="rd_font_size", disabled=off)
        c2.radio("Position", ["bottom", "middle", "top"], key="rd_position", horizontal=True, disabled=off)
        c1.slider("Distance from edge (px)", 0, 700, key="rd_margin_v", step=10, disabled=off,
                  help="Keep captions clear of the platform's own buttons - about 300 px from the bottom works well.")
        c2.checkbox("Bold", key="rd_bold", disabled=off)
        c1, c2 = st.columns(2)
        c1.color_picker("Text colour", key="rd_font_color", disabled=off)
        c2.color_picker("Outline colour", key="rd_outline_color", disabled=off)
        c1.slider("Outline", 0, 10, key="rd_outline", disabled=off)
        c2.slider("Shadow", 0, 10, key="rd_shadow", disabled=off)

        st.subheader("Audio & encoding")
        c1, c2 = st.columns(2)
        c1.toggle("Normalize loudness", key="rd_normalize_audio")
        c2.slider("Target (LUFS)", -24, -9, key="rd_loudness", disabled=not ss["rd_normalize_audio"])
        encs = ["auto"] + cached_encoders()
        c1.selectbox("Encoder", encs, key="rd_encoder", help="Auto prefers NVIDIA NVENC, then libx264, then OpenH264.")
        c2.text_input("Bitrate", key="rd_bitrate", help="e.g. 8M. Ignored by libx264, which uses constant quality.")

    with right:
        st.subheader("Preview this look")
        if kept.empty:
            st.info("Add clips to your final list on the **Score & Clips** tab first.")
        else:
            names = [f"#{i + 1} · {ce.fmt_time(r.start)}–{ce.fmt_time(r.end)}" + (f" · {r.name}" if r.name else "")
                     for i, r in enumerate(kept.itertuples())]
            pick = st.selectbox("Clip", range(len(kept)), format_func=lambda i: names[i], key="rd_preview_clip")
            if st.button("🔍 Render a quick preview"):
                row = kept.iloc[pick]
                opts = render_options_from_state()
                f = 540 / opts["width"]
                opts.update(width=540, height=round(opts["height"] * f / 2) * 2, bitrate="3M",
                            font_size=opts["font_size"] * f, outline=opts["outline"] * f,
                            shadow=opts["shadow"] * f, margin_v=opts["margin_v"] * f)
                prev = ws / "_preview"
                shutil.rmtree(prev, ignore_errors=True)
                prev.mkdir(parents=True)
                ce.save_json(prev / "clip.json", [{"start": float(row.start), "end": float(row.end), "score": 0}])
                with st.spinner("Rendering preview…"):
                    VideoRenderer(ws, options=opts, out_dir=prev, clips_path=prev / "clip.json").render()
                out = next(prev.glob("*.mp4"), None)
                if out:
                    ss["_preview_bytes"] = out.read_bytes()
                else:
                    st.error("Preview failed - see the terminal for details.")
            if ss.get("_preview_bytes"):
                col, _ = st.columns([1, 1])
                col.video(ss["_preview_bytes"], format="video/mp4")

        st.subheader("Render")
        if kept.empty:
            st.caption("Nothing to render yet.")
        else:
            st.write(f"**{len(kept)}** clip(s) · {ce.fmt_time((kept['end'] - kept['start']).sum())} total · "
                     f"{OUTPUT_SIZES[ss['rd_size']][0]}×{OUTPUT_SIZES[ss['rd_size']][1]}")
        if st.button("🎬 Render clips", type="primary", disabled=busy or kept.empty, width="stretch"):
            run_dir = ws / "shorts" / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
            opts = render_options_from_state()
            ce.save_json(ws / "render_options.json", opts)
            clips = [{"start": float(r.start), "end": float(r.end), "score": 0.0 if pd.isna(r.score) else round(float(r.score), 4),
                      **({"name": r.name} if r.name else {})} for r in kept.itertuples()]
            ce.save_json(ws / "render_clips.json", clips)
            persist(ws)
            ce.start_job(ws, "render", {"clips": str(ws / "render_clips.json"),
                                        "options": str(ws / "render_options.json"), "out_dir": str(run_dir)})
            st.rerun()
        job_panel(ws, "render")

    st.divider()
    st.subheader("Rendered clips")
    runs = ce.render_runs(ws)
    if not runs:
        st.caption("Nothing rendered yet.")
    for n, (label, files) in enumerate(runs):
        with st.expander(f"{label} · {len(files)} clip(s)", expanded=(n == 0)):
            def make_zip(files=files):
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
                    for f in files:
                        z.write(f, f.name)
                return buf.getvalue()
            st.download_button("⬇ Download all (.zip)", make_zip, file_name=f"{ws.name}_{label}.zip",
                               mime="application/zip", key=f"zip_{label}")
            cols = st.columns(3)
            for i, f in enumerate(files):
                with cols[i % 3]:
                    st.video(f.read_bytes() if f.stat().st_size < 60e6 else str(f), format="video/mp4")
                    st.caption(f"{f.name} · {f.stat().st_size / 1e6:.1f} MB")
                    st.download_button("⬇ Download", f.read_bytes() if f.stat().st_size < 60e6 else f.open("rb"),
                                       file_name=f.name, mime="video/mp4", key=f"dl_{label}_{f.name}")


# =========================================================================== main

def keep_widget_state():
    """Streamlit drops a widget's value on any run where it isn't drawn; re-assigning keeps it."""
    for k in list(ss.keys()):
        if k.startswith(("w_", "sc_", "rd_", "an_")):
            ss[k] = ss[k]


def main():
    ws = sidebar()
    if ss.get("_loaded_ws") != str(ws):
        load_project_state(ws)
    keep_widget_state()

    st.title("Clipper Studio")
    st.caption("Turn long videos into ranked, captioned, vertical shorts - fully local.")

    sig = ws / "signals.csv"
    have_signals = sig.exists()
    windows, rank, segments, picks = [], {"scores": [], "total": 0, "passed": 0, "picks": []}, [], []
    if have_signals:
        windows = cached_windows(str(sig), mtime(sig))
        rank = cached_rank(str(sig), mtime(sig), json.dumps(scoring_from_state(), sort_keys=True))
        segments = cached_segments(str(ws), mtime(ws / "transcription.json"))
        picks = rank["picks"]

    t1, t2, t3, t4 = st.tabs(["① Source & Analyze", "② Signals & Transcript", "③ Score & Clips", "④ Render & Export"])
    with t1:
        tab_source(ws)
    with t2:
        tab_explore(ws, ss["final_current"], picks)
    with t3:
        if have_signals:
            ss["_scoring_rendered"] = True
            tab_clips(ws, windows, rank, segments)
        else:
            st.info("Run the analysis in step ① first - it produces the moments this tab scores.")
    with t4:
        tab_render(ws, ss["final_current"])

    persist(ws)


main()
