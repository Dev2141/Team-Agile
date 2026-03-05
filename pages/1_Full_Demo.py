"""
pages/1_Full_Demo.py — ThreatSense AI-DVR Full Pipeline Demo
New dedicated page: upload your own video → YOLO scan → Ollama VLM → rich results.
"""

import asyncio
import base64
import os
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# ── Page config ──
st.set_page_config(
    page_title="ThreatSense · Full Pipeline Demo",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Premium CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

html, body, [data-testid="stAppViewContainer"] {
    background-color: #080b12 !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] { display: none; }

.demo-hero {
    text-align: center;
    padding: 40px 20px 30px;
}
.demo-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00f2ff 0%, #6366f1 50%, #f43f5e 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 12px;
}
.demo-subtitle {
    color: #64748b;
    font-size: 1rem;
    max-width: 600px;
    margin: 0 auto;
    line-height: 1.6;
}
.pipeline-stage {
    background: rgba(17, 24, 39, 0.7);
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 16px;
    backdrop-filter: blur(10px);
}
.pipeline-stage.active {
    border-color: #6366f1;
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.15);
}
.pipeline-stage.done {
    border-color: #10b981;
    box-shadow: 0 0 12px rgba(16, 185, 129, 0.1);
}
.pipeline-stage.threat-done {
    border-color: #f43f5e;
    box-shadow: 0 0 20px rgba(244, 63, 94, 0.15);
}
.stage-header {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.8rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #6366f1;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.stage-header.done { color: #10b981; }
.stage-header.threat { color: #f43f5e; }
.keyframe-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 10px;
    margin-top: 12px;
}
.keyframe-tile {
    border-radius: 6px;
    overflow: hidden;
    border: 1px solid #1e293b;
    background: #0f172a;
    font-size: 0.7rem;
    color: #475569;
}
.keyframe-tile img { width: 100%; display: block; }
.keyframe-label { padding: 4px 8px; text-align: center; }
.verdict-card {
    border-radius: 12px;
    padding: 28px;
    margin-top: 20px;
    background: rgba(17, 24, 39, 0.9);
    backdrop-filter: blur(20px);
}
.verdict-threat {
    border: 2px solid #f43f5e;
    box-shadow: 0 0 40px rgba(244, 63, 94, 0.2);
}
.verdict-safe {
    border: 2px solid #10b981;
    box-shadow: 0 0 40px rgba(16, 185, 129, 0.15);
}
.verdict-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    margin-bottom: 16px;
}
.verdict-threat .verdict-title { color: #f43f5e; }
.verdict-safe .verdict-title { color: #10b981; }
.detail-section {
    background: rgba(0,0,0,0.3);
    border-radius: 8px;
    padding: 16px;
    margin-top: 16px;
    border-left: 3px solid #6366f1;
}
.detail-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #6366f1;
    font-weight: 700;
    margin-bottom: 8px;
}
.detail-text {
    font-size: 0.95rem;
    color: #cbd5e1;
    line-height: 1.7;
}
.risk-tag {
    display: inline-block;
    background: rgba(244, 63, 94, 0.15);
    border: 1px solid rgba(244, 63, 94, 0.4);
    color: #f87171;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.75rem;
    margin: 3px;
    font-weight: 600;
}
.action-box {
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 8px;
    padding: 14px;
    margin-top: 12px;
    font-size: 0.9rem;
    color: #a5b4fc;
}
.conf-meter {
    height: 8px;
    border-radius: 4px;
    background: #1e293b;
    margin-top: 8px;
    overflow: hidden;
}
.conf-fill-threat { background: linear-gradient(90deg, #f43f5e, #ff7c8e); height: 100%; border-radius: 4px; transition: width 0.8s ease; }
.conf-fill-safe   { background: linear-gradient(90deg, #10b981, #34d399); height: 100%; border-radius: 4px; transition: width 0.8s ease; }
.back-btn {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 20px;
    background: transparent;
    border: 1px solid #1e293b;
    border-radius: 8px;
    color: #64748b;
    font-size: 0.85rem;
    cursor: pointer;
    text-decoration: none;
    margin-bottom: 24px;
    transition: all 0.2s;
}
.back-btn:hover {
    border-color: #6366f1;
    color: #a5b4fc;
}
.yolo-stat {
    display: inline-flex;
    background: rgba(0, 242, 255, 0.08);
    border: 1px solid rgba(0, 242, 255, 0.2);
    border-radius: 6px;
    padding: 6px 14px;
    font-size: 0.8rem;
    color: #00f2ff;
    margin-right: 8px;
    margin-top: 8px;
    font-family: 'Orbitron', sans-serif;
}
</style>
""", unsafe_allow_html=True)


# ── Cache YOLO ──
@st.cache_resource
def load_yolo():
    model_path = Path(__file__).parent.parent / "yolov8n.pt"
    return YOLO(str(model_path) if model_path.exists() else "yolov8n.pt")


# ══════════════════════════════════════════════════════════════════════════════
# HERO SECTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="demo-hero">
    <div class="demo-title">🎬 Full Pipeline Demo</div>
    <div class="demo-subtitle">
        Upload any video → YOLOv8 person detection → keyframe extraction →
        Ollama VLM deep intent analysis → detailed threat report
    </div>
</div>
""", unsafe_allow_html=True)

# Back button
st.markdown('<a href="/" target="_self" class="back-btn">← Back to Live Dashboard</a>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# VIDEO UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="pipeline-stage">
    <div class="stage-header">📂 Stage 0 — Upload Your Video</div>
</div>
""", unsafe_allow_html=True)

col_up, col_cfg = st.columns([3, 1])
with col_up:
    uploaded_video = st.file_uploader(
        "Upload a security camera video",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        help="Upload any camera footage. Supports MP4, AVI, MOV, MKV, WebM.",
        label_visibility="collapsed",
    )

with col_cfg:
    sample_fps    = st.slider("Sample rate (FPS)", 1, 5, 2, help="Frames extracted per second for YOLO scanning")
    max_keyframes = st.slider("Max keyframes → VLM", 2, 6, 5, help="Maximum keyframes sent to the Ollama VLM")
    context_hint  = st.text_input("Context hint (optional)", placeholder="e.g. night-time outdoor parking lot")

if uploaded_video is None:
    st.info("👆 Upload a video file above to begin the full pipeline demo.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# RUN PIPELINE BUTTON
# ══════════════════════════════════════════════════════════════════════════════
run = st.button("▶️ Run Full Analysis Pipeline", type="primary", use_container_width=True)
if not run and "demo_result" not in st.session_state:
    st.stop()

if run:
    # Clear any previous result
    if "demo_result" in st.session_state:
        del st.session_state["demo_result"]
    if "demo_keyframes" in st.session_state:
        del st.session_state["demo_keyframes"]
    if "demo_yolo_stats" in st.session_state:
        del st.session_state["demo_yolo_stats"]

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE EXECUTION
# ══════════════════════════════════════════════════════════════════════════════
if run:
    import tempfile, sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from analyser import OllamaAnalyser

    yolo = load_yolo()
    analyser = OllamaAnalyser(
        ollama_url=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"),
        model=os.getenv("OLLAMA_MODEL", "qwen3-vl:235b-cloud"),
    )

    # ── Save uploaded video to temp file ──
    suffix = Path(uploaded_video.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_video.read())
        tmp_path = tmp.name

    # ── STAGE 1: YOLO Scan ──
    stage1_ph = st.empty()
    stage1_ph.markdown("""
    <div class="pipeline-stage active">
        <div class="stage-header">⚡ Stage 1 — YOLOv8 Person Detection</div>
        <div style="color:#94a3b8;font-size:0.85rem;">Scanning all frames for persons…</div>
    </div>
    """, unsafe_allow_html=True)

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        st.error("❌ Could not open the video file. Please try another format.")
        st.stop()

    video_fps       = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_interval = max(1, int(video_fps / sample_fps))

    frames_with_person: list[tuple[int, bytes]] = []
    all_sampled_frames:  list[tuple[int, bytes]] = []
    person_frame_ids: set[int] = set()
    frame_idx = 0
    last_pct  = -1

    yolo_prog = st.progress(0.0, text="Stage 1: YOLOv8 scanning…")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_interval == 0:
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            jpeg_bytes = jpeg.tobytes()
            all_sampled_frames.append((frame_idx, jpeg_bytes))

            results = yolo(frame, verbose=False)
            person_found = False
            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.35:
                        # Draw bounding box on frame
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 255), 2)
                        label = f"Person {conf:.0%}"
                        cv2.putText(frame, label, (x1, max(y1 - 8, 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 255), 1, cv2.LINE_AA)
                        person_found = True

            if person_found:
                _, annotated_jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frames_with_person.append((frame_idx, annotated_jpeg.tobytes()))
                person_frame_ids.add(frame_idx)

            pct = min(0.99, frame_idx / max(total_frames, 1))
            if int(pct * 20) > int(last_pct * 20):
                yolo_prog.progress(pct, text=f"Stage 1: Scanning frame {frame_idx}/{total_frames}…")
                last_pct = pct

        frame_idx += 1

    cap.release()
    yolo_prog.progress(1.0, text="Stage 1 complete ✓")
    time.sleep(0.3)

    # Store YOLO stats
    st.session_state["demo_yolo_stats"] = {
        "total_scanned": len(all_sampled_frames),
        "person_frames": len(frames_with_person),
        "total_video_frames": total_frames,
    }

    stage1_ph.markdown(
        f'<div class="pipeline-stage done"><div class="stage-header done">✓ Stage 1 — YOLOv8 Complete</div>'
        f'<div style="margin-top:8px;">'
        f'<span class="yolo-stat">📹 {total_frames} Total Frames</span>'
        f'<span class="yolo-stat">🔍 {len(all_sampled_frames)} Sampled</span>'
        f'<span class="yolo-stat">🧍 {len(frames_with_person)} Person Detections</span>'
        f'</div></div>',
        unsafe_allow_html=True)

    # ── STAGE 2: Build Video Clip ──
    stage2_ph = st.empty()
    stage2_ph.markdown(
        '<div class="pipeline-stage active"><div class="stage-header">🎬 Stage 2 — Building Event Clip</div>'
        '<div style="color:#94a3b8;font-size:0.85rem;">Assembling annotated video clip from detected frames…</div></div>',
        unsafe_allow_html=True)

    clip_pool = frames_with_person if frames_with_person else all_sampled_frames
    clip_bytes = None
    try:
        import tempfile as _tf, shutil as _sh, subprocess as _sp

        # Decode first frame to get W×H
        first_arr = cv2.imdecode(np.frombuffer(clip_pool[0][1], np.uint8), cv2.IMREAD_COLOR)
        h, w = first_arr.shape[:2]

        ffmpeg_bin = _sh.which("ffmpeg") or "ffmpeg"
        out_tmp = _tf.NamedTemporaryFile(suffix=".mp4", delete=False)
        out_tmp.close()

        # ffmpeg reads raw JPEG frames from stdin, outputs H.264 mp4 (browser-compatible)
        cmd = [
            ffmpeg_bin, "-y",
            "-f", "image2pipe",
            "-framerate", str(sample_fps),
            "-i", "pipe:0",
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            "-crf", "23",
            out_tmp.name,
        ]
        proc = _sp.Popen(cmd, stdin=_sp.PIPE, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
        all_jpeg_data = b"".join(jpeg_bytes for _, jpeg_bytes in clip_pool)
        proc.communicate(input=all_jpeg_data)

        with open(out_tmp.name, "rb") as f:
            clip_bytes = f.read()
        os.unlink(out_tmp.name)
        st.session_state["demo_clip"] = clip_bytes if clip_bytes else None
    except Exception:
        st.session_state["demo_clip"] = None


    clip_count = len(clip_pool)
    stage2_ph.markdown(
        f'<div class="pipeline-stage done"><div class="stage-header done">✓ Stage 2 — Event Clip Built</div>'
        f'<div style="color:#94a3b8;font-size:0.8rem;margin-top:6px;">'
        f'{clip_count} annotated frames assembled into MP4 clip</div></div>',
        unsafe_allow_html=True)

    # ── STAGE 3: Keyframe Selection ──
    stage3_ph = st.empty()
    stage3_ph.markdown(
        '<div class="pipeline-stage active"><div class="stage-header">🎯 Stage 3 — Keyframe Selection</div>'
        '<div style="color:#94a3b8;font-size:0.85rem;">Selecting most informative frames for VLM…</div></div>',
        unsafe_allow_html=True)
    time.sleep(0.3)

    if frames_with_person:
        pool = frames_with_person
        label = "person-detected"
    else:
        # No persons: sample evenly from all frames
        pool = all_sampled_frames
        label = "uniform (no persons found)"

    # Evenly spaced selection
    n = min(max_keyframes, len(pool))
    step = max(1, len(pool) // n)
    selected = pool[::step][:n]
    keyframe_jpegs = [f[1] for f in selected]
    keyframe_ids   = [f[0] for f in selected]

    stage3_ph.markdown(
        f'<div class="pipeline-stage done"><div class="stage-header done">✓ Stage 3 — {len(keyframe_jpegs)} Keyframes Selected ({label})</div></div>',
        unsafe_allow_html=True)

    st.session_state["demo_keyframes"] = list(zip(keyframe_ids, keyframe_jpegs))

    # ── STAGE 4: Ollama VLM ──
    stage4_ph = st.empty()
    stage4_ph.markdown(
        f'<div class="pipeline-stage active"><div class="stage-header">🧠 Stage 4 — Ollama VLM Analysis</div>'
        f'<div style="color:#94a3b8;font-size:0.85rem;">Sending {len(keyframe_jpegs)} keyframe(s) to model… (may take 30–120s)</div></div>',
        unsafe_allow_html=True)

    frame_info = (
        f"sampled from the video at {sample_fps} FPS, "
        f"frames #{', '.join(str(i) for i in keyframe_ids)}"
    )
    ctx = context_hint.strip() if context_hint else "Security camera footage uploaded by user"
    if frames_with_person:
        ctx += f". YOLOv8 detected {len(frames_with_person)} person-containing frames."

    vlm_prog = st.progress(0.0, text="Stage 4: Waiting for Ollama response…")

    async def run_analysis():
        return await analyser.analyse_images(keyframe_jpegs, context=ctx, frame_info=frame_info)

    result = asyncio.run(run_analysis())
    vlm_prog.progress(1.0, text="Stage 4: Analysis complete ✓")
    time.sleep(0.2)

    if result is None:
        stage4_ph.markdown(
            '<div class="pipeline-stage" style="border-color:#f59e0b;">'
            '<div class="stage-header" style="color:#f59e0b;">⚠ Stage 4 — Ollama Unavailable</div>'
            '<div style="color:#94a3b8;font-size:0.85rem;">Could not reach Ollama at 127.0.0.1:11434.</div></div>',
            unsafe_allow_html=True)
        st.error("Ollama VLM did not respond. Ensure `ollama serve` is running with `qwen3-vl:235b-cloud`.")
        st.stop()

    verdict_label = '⚠ Stage 4 — THREAT DETECTED' if result.is_threat else '✓ Stage 4 — Analysis Complete'
    stage4_ph.markdown(
        f'<div class="pipeline-stage {"threat-done" if result.is_threat else "done"}">'  
        f'<div class="stage-header {"threat" if result.is_threat else "done"}">{verdict_label}</div></div>',
        unsafe_allow_html=True)

    st.session_state["demo_result"] = result

    # Cleanup temp file
    try:
        os.unlink(tmp_path)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# SHOW RESULTS (persistent after run)
# ══════════════════════════════════════════════════════════════════════════════

# ── Event Video Clip ──
if st.session_state.get("demo_clip"):
    st.markdown("---")
    st.markdown(
        '<div class="pipeline-stage done" style="padding-bottom:6px;">'  
        '<div class="stage-header done">🎬 Annotated Event Clip (YOLO Detections)</div></div>',
        unsafe_allow_html=True)
    st.video(st.session_state["demo_clip"])

# ── Keyframes sent to VLM ──
if "demo_keyframes" in st.session_state and st.session_state["demo_keyframes"]:
    st.markdown(
        '<div class="pipeline-stage done" style="padding-bottom:6px;margin-top:12px;">'  
        '<div class="stage-header done">🖼️ Keyframes Sent to VLM</div></div>',
        unsafe_allow_html=True)

    kf_cols = st.columns(min(len(st.session_state["demo_keyframes"]), 6))
    for idx, (fid, jpeg_bytes) in enumerate(st.session_state["demo_keyframes"]):
        with kf_cols[idx % len(kf_cols)]:
            img_b64 = base64.b64encode(jpeg_bytes).decode()
            st.markdown(
                f'<div class="keyframe-tile"><img src="data:image/jpeg;base64,{img_b64}" />'
                f'<div class="keyframe-label">Frame #{fid}</div></div>',
                unsafe_allow_html=True)

if "demo_result" in st.session_state and st.session_state["demo_result"]:
    result = st.session_state["demo_result"]
    conf_pct = int(result.confidence * 100)
    is_threat = result.is_threat
    conf_cls  = "conf-fill-threat" if is_threat else "conf-fill-safe"
    border    = "#f43f5e" if is_threat else "#10b981"
    icon      = "⚠️" if is_threat else "✅"
    headline  = "THREAT DETECTED" if is_threat else "NO THREAT — ALL CLEAR"
    title_col = "#f43f5e" if is_threat else "#10b981"

    st.markdown("---")

    # ── Verdict headline ──
    st.markdown(
        f'<div style="font-family:Orbitron,sans-serif;font-size:1.8rem;font-weight:700;'
        f'color:{title_col};margin-bottom:16px;padding:20px 24px 0;">{icon} {headline}</div>',
        unsafe_allow_html=True,
    )

    # ── Type + Confidence badges ──
    type_label = result.type.upper().replace("_", " ")
    col_t, col_c, col_sp = st.columns([1, 1, 3])
    with col_t:
        st.markdown(
            '<div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:1.5px;color:#64748b;">Threat Type</div>',
            unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:1.2rem;font-weight:700;color:#e2e8f0;margin-top:4px;">{type_label}</div>',
            unsafe_allow_html=True)
    with col_c:
        st.markdown(
            '<div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:1.5px;color:#64748b;">Confidence</div>',
            unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:1.2rem;font-weight:700;color:#e2e8f0;margin-top:4px;">{conf_pct}%</div>'
            f'<div class="conf-meter"><div class="{conf_cls}" style="width:{conf_pct}%"></div></div>',
            unsafe_allow_html=True)

    # ── Headline finding ──
    st.markdown(
        f'<div class="detail-section" style="border-left-color:{border};margin-top:12px;">'
        f'<div class="detail-label" style="color:{border};">🔍 Headline Finding</div>'
        f'<div class="detail-text" style="font-size:1.05rem;font-weight:600;color:#f1f5f9;">'
        f'{result.reason}</div></div>',
        unsafe_allow_html=True,
    )

    # ── Detailed columns ──
    col_a, col_b = st.columns(2)

    with col_a:
        if result.scene_description:
            st.markdown(
                '<div class="detail-section">'
                '<div class="detail-label">🏙️ Scene Description</div>'
                f'<div class="detail-text">{result.scene_description}</div></div>',
                unsafe_allow_html=True)

        if result.risk_factors:
            tags = "".join(f'<span class="risk-tag">⚡ {rf}</span>' for rf in result.risk_factors)
            st.markdown(
                '<div class="detail-section" style="border-left-color:#f59e0b;">'
                '<div class="detail-label" style="color:#f59e0b;">⚠️ Risk Factors</div>'
                f'<div style="margin-top:6px;">{tags}</div></div>',
                unsafe_allow_html=True)

    with col_b:
        if result.behaviour_analysis:
            st.markdown(
                '<div class="detail-section">'
                '<div class="detail-label">🧍 Behaviour Analysis</div>'
                f'<div class="detail-text">{result.behaviour_analysis}</div></div>',
                unsafe_allow_html=True)

        if result.recommended_action:
            st.markdown(
                '<div class="action-box">'
                '<div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:1.5px;'
                'color:#6366f1;margin-bottom:8px;font-weight:700;">🛡️ Recommended Action</div>'
                f'<div style="font-size:0.9rem;color:#c7d2fe;">{result.recommended_action}</div></div>',
                unsafe_allow_html=True)

    # ── YOLO stats ──
    if "demo_yolo_stats" in st.session_state:
        stats = st.session_state["demo_yolo_stats"]
        st.markdown(
            f'<div style="margin-top:20px;padding:12px 18px;background:rgba(0,242,255,0.04);'
            f'border:1px solid rgba(0,242,255,0.1);border-radius:8px;font-size:0.8rem;color:#475569;">'
            f'<strong style="color:#00f2ff;">YOLOv8 Stats:</strong> '
            f'{stats["total_video_frames"]} total frames · '
            f'{stats["total_scanned"]} sampled · '
            f'{stats["person_frames"]} person detections</div>',
            unsafe_allow_html=True)

    if not is_threat:
        st.balloons()

    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
    if st.button("🔄 Analyse Another Video", use_container_width=True):
        for k in ["demo_result", "demo_keyframes", "demo_yolo_stats"]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()



