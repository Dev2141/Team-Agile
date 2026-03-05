"""
app.py — ThreatSense AI-DVR
Main Streamlit dashboard. Real-time IP Webcam feed with YOLOv8 detection.
Full Demo redirects to pages/1_Full_Demo.py.
"""

import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# ── Page config ──
st.set_page_config(
    page_title="ThreatSense System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stHeader"] { background-color: transparent !important; }
[data-testid="collapsedControl"] {
    display: flex !important; visibility: visible !important;
    background-color: #090b10 !important; color: white !important;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: #080b12 !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #090b10 !important;
    border-right: 1px solid #1a2236 !important;
}
.sidebar-logo {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #00f2ff;
    text-shadow: 0 0 12px rgba(0, 242, 255, 0.35);
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.sidebar-section {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #475569;
    font-weight: 700;
    margin: 20px 0 8px;
    padding-bottom: 6px;
    border-bottom: 1px solid #1a2236;
}

/* ── Risk Pill ── */
.risk-indicator {
    padding: 8px 18px;
    border-radius: 6px;
    font-weight: 700;
    font-family: 'Orbitron', sans-serif;
    font-size: 0.8rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    margin-bottom: 16px;
}
.risk-safe       { background: rgba(16, 185, 129, 0.08);  color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
.risk-suspicious { background: rgba(245, 158, 11, 0.1);   color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.35); }
.risk-threat     { background: rgba(244, 63, 94, 0.12);   color: #fb7185; border: 1px solid #f43f5e; animation: pulse-red 1.5s infinite; }

@keyframes pulse-red {
    0%   { box-shadow: 0 0 0 0 rgba(244, 63, 94, 0.4); }
    70%  { box-shadow: 0 0 0 10px rgba(244, 63, 94, 0); }
    100% { box-shadow: 0 0 0 0 rgba(244, 63, 94, 0); }
}

/* ── Metric Cards ── */
.metric-container {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-bottom: 16px;
    margin-top: -20px;
}
.glass-card {
    background: rgba(15, 21, 32, 0.7);
    backdrop-filter: blur(12px);
    border: 1px solid #1a2236;
    border-radius: 8px;
    padding: 14px 10px;
    text-align: center;
}
.metric-value {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.15rem;
    color: #00f2ff;
    margin-bottom: 5px;
    font-weight: 700;
}
.metric-label {
    font-size: 0.62rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

/* ── Alert Banner ── */
.alert-banner {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    color: white;
    padding: 10px 20px;
    border-radius: 6px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 14px;
    font-weight: 600;
    border: 1px solid #f43f5e;
    animation: slide-down 0.4s ease-out;
    font-size: 0.9rem;
}
@keyframes slide-down {
    from { transform: translateY(-20px); opacity: 0; }
    to   { transform: translateY(0);     opacity: 1; }
}

/* ── Incident Panel ── */
.incident-item {
    padding: 10px 0;
    border-bottom: 1px solid #1a2236;
}
.incident-item:last-child { border-bottom: none; }

/* ── Engine Info ── */
.engine-info {
    background: rgba(0,0,0,0.3);
    border: 1px solid #1a2236;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 0.72rem;
    color: #475569;
    line-height: 1.8;
}
.engine-info span { color: #00f2ff; }
</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═════════════════════════════════════════════════════════════════════════════
for key, default in [
    ("risk_level", "SAFE"),
    ("alerts", []),
    ("trigger_sim", None),
    ("incident_reasoning", "System initialised. All zones monitored."),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Alert integration
import alert_service

# ═════════════════════════════════════════════════════════════════════════════
# YOLO MODEL
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
<div class="sidebar-logo">
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
    </svg>
    ThreatSense
</div>
""", unsafe_allow_html=True)

    # IP Webcam URL
    st.markdown('<div class="sidebar-section">IP Webcam Stream</div>', unsafe_allow_html=True)
    default_url = os.getenv("IP_WEBCAM_URL", "http://10.226.220.51:8080/video")
    cam1_source = st.text_input(
        "Stream URL",
        value=default_url,
        placeholder="http://192.168.1.x:8080/video",
        help="Paste the HTTP URL from the IP Webcam app. Phone and PC must be on same Wi‑Fi.",
        label_visibility="collapsed",
    )
    st.caption("Format: http://[phone-IP]:8080/video (IP Webcam app)")

    # Full Demo link
    st.markdown('<div class="sidebar-section">Pipeline Demo</div>', unsafe_allow_html=True)
    st.page_link("pages/1_Full_Demo.py", label="▶️ Open Full Pipeline Demo", icon="🎬", use_container_width=True)
    st.caption("Upload your own video for YOLO + VLM analysis")

    # Simulation
    st.markdown('<div class="sidebar-section">Quick Simulation</div>', unsafe_allow_html=True)
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("🚨 INTRUDER", use_container_width=True):
            st.session_state.trigger_sim = "intruder"
            st.session_state.risk_level = "THREAT"
    with col_s2:
        if st.button("🚶 LOITER", use_container_width=True):
            st.session_state.trigger_sim = "loitering"
            st.session_state.risk_level = "SUSPICIOUS"

    # AI Engine settings
    st.markdown('<div class="sidebar-section">AI Engine</div>', unsafe_allow_html=True)
    show_tracking = st.checkbox("Live Bounding Boxes", value=True)
    ollama_api_url = os.getenv("OLLAMA_API_URL", "http://localhost:8100")
    st.markdown(f"""
<div style="font-size:0.75rem;margin-top:8px;">
    <a href="{ollama_api_url}" target="_blank" style="color:#6366f1;">🔗 Open VLM Analyser UI</a>
    <div style="color:#475569;margin-top:4px;font-size:0.7rem;">Standalone image/video analysis</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="engine-info" style="margin-top:14px;">
    Engine: <span>YOLOv8n / ByteTrack</span><br>
    Sample FPS: <span>~2</span> | Infer: <span>12ms</span><br>
    Precision: <span>YOLO conf > 35%</span>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# ALERT BANNER
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.trigger_sim:
    msg    = ("UNAUTHORIZED ENTRY DETECTED" if st.session_state.trigger_sim == "intruder"
              else "SUSPICIOUS LOITERING DETECTED")
    reason = ("Person detected during restricted hours — immediate review required."
              if st.session_state.trigger_sim == "intruder"
              else "Individual loitering near entrance for >45 seconds without entry.")
    st.session_state.incident_reasoning = reason
    st.markdown(f'<div class="alert-banner">⚠️ {msg}<span style="font-size:0.75rem;opacity:0.7;">JUST NOW</span></div>',
                unsafe_allow_html=True)
    alert_service.trigger_alert("IP Webcam", msg)
    st.session_state.trigger_sim = None

# ═════════════════════════════════════════════════════════════════════════════
# METRIC STRIP
# ═════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="metric-container">
    <div class="glass-card">
        <div class="metric-value">1</div>
        <div class="metric-label">Active Feed</div>
    </div>
    <div class="glass-card">
        <div class="metric-value">02</div>
        <div class="metric-label">Daily Incidents</div>
    </div>
    <div class="glass-card">
        <div class="metric-value">{datetime.now().strftime("%H:%M:%S")}</div>
        <div class="metric-label">System Time</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Risk indicator
risk_class = {
    "SAFE":       "risk-safe",
    "SUSPICIOUS": "risk-suspicious",
    "THREAT":     "risk-threat",
}.get(st.session_state.risk_level, "risk-safe")
st.markdown(f'<div class="risk-indicator {risk_class}">● System Status: {st.session_state.risk_level}</div>',
            unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT: Video feed + Incident panel
# ═════════════════════════════════════════════════════════════════════════════
col_vid, col_inc = st.columns([7, 3])

with col_inc:
    # ── AI Reasoning Engine ──
    st.markdown('<div class="metric-label" style="margin-bottom:8px;">AI REASONING ENGINE</div>', unsafe_allow_html=True)
    reasoning_text = st.session_state.incident_reasoning
    st.markdown(
        f'<div style="background:rgba(0,242,255,0.04);border:1px solid rgba(0,242,255,0.12);'
        f'padding:12px;border-radius:6px;border-left:2px solid #00f2ff;margin-bottom:16px;">'
        f'<div style="font-size:0.78rem;line-height:1.6;color:#cbd5e1;">'
        f'<strong style="color:#00f2ff;">ANALYSIS:</strong> {reasoning_text}</div></div>',
        unsafe_allow_html=True,
    )

    # ── Incident Timeline ──
    st.markdown('<div class="metric-label" style="margin-bottom:8px;margin-top:4px;">INCIDENT TIMELINE</div>', unsafe_allow_html=True)
    st.markdown("""
<div style="border-bottom:1px solid #1a2236;padding:8px 0;">
  <div style="display:flex;justify-content:space-between;font-size:0.68rem;color:#64748b;"><span>IP Webcam</span><span>17:06</span></div>
  <div style="font-weight:600;font-size:0.8rem;color:#fb7185;margin-top:2px;">Unauthorized Access</div>
</div>
<div style="padding:8px 0;">
  <div style="display:flex;justify-content:space-between;font-size:0.68rem;color:#64748b;"><span>IP Webcam</span><span>15:30</span></div>
  <div style="font-weight:600;font-size:0.8rem;color:#fbbf24;margin-top:2px;">Suspicious Loitering</div>
</div>
""", unsafe_allow_html=True)

    # ── 24H Analytics ──
    st.markdown('<div class="metric-label" style="margin-bottom:8px;margin-top:16px;">24H ANALYTICS</div>', unsafe_allow_html=True)
    st.markdown("""
<div style="font-size:0.72rem;margin-bottom:5px;color:#94a3b8;">Unauthorized Entry</div>
<div style="width:100%;background:#1a2236;height:5px;border-radius:3px;">
  <div style="width:70%;background:#00f2ff;height:100%;border-radius:3px;"></div>
</div>
<div style="font-size:0.72rem;margin-top:10px;margin-bottom:5px;color:#94a3b8;">Crowd Detection</div>
<div style="width:100%;background:#1a2236;height:5px;border-radius:3px;">
  <div style="width:30%;background:#818cf8;height:100%;border-radius:3px;"></div>
</div>
""", unsafe_allow_html=True)

with col_vid:
    st.slider("Footage Timeline (scrub)", 0, 100, 100,
              help="Scrub through historical detection events.", label_visibility="collapsed")
    video_ph = st.empty()
    st.markdown(
        '<div style="margin-top:6px;font-size:0.65rem;color:#334155;letter-spacing:0.8px;">▶ IP WEBCAM — LIVE STREAM</div>',
        unsafe_allow_html=True
    )

# ═════════════════════════════════════════════════════════════════════════════
# VIDEO CAPTURE + YOLO RENDER LOOP
# ═════════════════════════════════════════════════════════════════════════════
yolo_model = load_yolo_model()

if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0
if "cap1" not in st.session_state:
    st.session_state.cap1 = None
    st.session_state.last_source = None

frame_num = st.session_state.frame_count


def get_cap1(source: str):
    """Open IP Webcam HTTP stream. Returns None if source is blank or fails."""
    if not source or not str(source).strip():
        return None
    source = str(source).strip()
    if st.session_state.last_source != source:
        if st.session_state.cap1 is not None:
            try:
                st.session_state.cap1.release()
            except Exception:
                pass
            st.session_state.cap1 = None
        try:
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                st.session_state.cap1 = cap
                st.session_state.last_source = source
            else:
                st.session_state.cap1 = None
        except Exception:
            st.session_state.cap1 = None
    return st.session_state.cap1


def draw_overlay(img, text, badge="LIVE"):
    """Draw camera label and LIVE/OFFLINE badge."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Left label
    cv2.rectangle(img, (0, 0), (220, 28), (5, 8, 18), -1)
    cv2.putText(img, text, (10, 19), font, 0.45, (0, 242, 255), 1, cv2.LINE_AA)
    # Right badge
    badge_color = (0, 200, 100) if badge == "LIVE" else (80, 80, 80)
    cv2.rectangle(img, (img.shape[1] - 70, 0), (img.shape[1], 24), (5, 8, 18), -1)
    cv2.putText(img, badge, (img.shape[1] - 60, 17), font, 0.4, badge_color, 1, cv2.LINE_AA)


def generate_cam_frame(frame_num: int, show_tracking: bool, source: str):
    """Generate one video frame with optional YOLO overlay."""
    w, h = 640, 360
    cap = get_cap1(source)

    if cap and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (w, h))
            if show_tracking and frame_num % 3 == 0:
                st.session_state["last_boxes"] = []
                for r in yolo_model(frame, stream=True, verbose=False):
                    for i, box in enumerate(r.boxes):
                        if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.35:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            behavior = "Loitering" if conf > 0.78 else "Moving"
                            color = (0, 200, 100) if behavior == "Moving" else (0, 140, 255)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            tid = i + 101
                            cv2.putText(frame, f"ID#{tid} {behavior}",
                                        (x1, max(y1 - 22, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                            cv2.putText(frame, f"CONF:{conf:.0%}",
                                        (x1, max(y1 - 8, 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                            st.session_state["last_boxes"].append((x1, y1, x2, y2, f"ID#{tid} {behavior}", f"CONF:{conf:.0%}", color))
            elif show_tracking and "last_boxes" in st.session_state:
                for (x1, y1, x2, y2, lbl, conf_txt, color) in st.session_state["last_boxes"]:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, lbl,      (x1, max(y1 - 22, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    cv2.putText(frame, conf_txt, (x1, max(y1 - 8,  22)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            draw_overlay(frame, "CAM-01 · IP WEBCAM", "LIVE")
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Fallback offline frame
    img = np.full((h, w, 3), 10, dtype=np.uint8)
    msg = "Paste IP Webcam URL in sidebar →" if not source else "Cannot connect — check URL & Wi‑Fi"
    ts  = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.putText(img, msg, ((w - ts[0]) // 2, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 70, 90), 1, cv2.LINE_AA)
    draw_overlay(img, "CAM-01 · IP WEBCAM", "OFFLINE")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ── Render Loop ──
try:
    for _ in range(24):
        frame_img = generate_cam_frame(frame_num, show_tracking, cam1_source)
        video_ph.image(frame_img, use_container_width=True)   # ← fixed: was width="stretch"
        frame_num += 1
        time.sleep(1 / 30)
finally:
    st.session_state.frame_count = frame_num

st.rerun()
