"""
app.py — ThreatSense AI-DVR
Streamlit live-dashboard with real-time RTSP/Webcam integration for CAM-01.
"""

import os
import time
from datetime import datetime

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

# ── Clean Enterprise CSS (Antigravity) ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');

/* Global Overrides */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stHeader"] { background-color: transparent !important; }
[data-testid="collapsedControl"] { display: flex !important; visibility: visible !important; background-color: #090b10 !important; color: white !important; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0c0e14 !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif;
}

/* Sidebar - Cyberpunk Control */
[data-testid="stSidebar"] {
    background-color: #090b10 !important;
    border-right: 1px solid #1e293b !important;
    width: 300px !important;
}
.sidebar-logo {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #00f2ff;
    text-shadow: 0 0 10px rgba(0, 242, 255, 0.4);
    margin-bottom: 30px;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Risk Indicator Pill */
.risk-indicator {
    padding: 6px 16px;
    border-radius: 4px;
    font-weight: 700;
    font-family: 'Orbitron', sans-serif;
    font-size: 0.85rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    margin-bottom: 15px;
}
.risk-safe { background: rgba(34, 197, 94, 0.1); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3); }
.risk-suspicious { background: rgba(234, 179, 8, 0.1); color: #fde047; border: 1px solid rgba(234, 179, 8, 0.3); }
.risk-threat { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid #ef4444; animation: pulse-red 1.5s infinite; }

@keyframes pulse-red {
    0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
    70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
    100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
}

/* Professional Analytics Cards */
.metric-container {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 20px;
    margin-top: -30px;
}
.glass-card {
    background: rgba(17, 24, 39, 0.6);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 8px;
    padding: 12px;
    text-align: center;
}
.metric-value {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.2rem;
    color: #00f2ff;
    margin-bottom: 4px;
}
.metric-label {
    font-size: 0.65rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Floating Alert Overlay */
.alert-banner {
    background: #ef4444;
    color: white;
    padding: 10px 20px;
    border-radius: 4px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 15px;
    font-weight: 600;
    animation: slide-down 0.4s ease-out;
}
@keyframes slide-down {
    from { transform: translateY(-50px); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
}

/* Feed Container Design */
.feed-box {
    border: 1px solid #1e293b;
    background: black;
    border-radius: 4px;
    overflow: hidden;
}
.view-header {
    background: #0f172a;
    padding: 6px 12px;
    border-bottom: 1px solid #1e293b;
    font-size: 0.75rem;
    display: flex;
    justify-content: space-between;
    color: #94a3b8;
}
</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
# ── Session State Logic ──
if "risk_level" not in st.session_state:
    st.session_state.risk_level = "SAFE"
if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "trigger_sim" not in st.session_state:
    st.session_state.trigger_sim = None
if "incident_reasoning" not in st.session_state:
    st.session_state.incident_reasoning = "System initialised. All zones monitored."

# ── Alert Integration (Audio) ──
import alert_service

with st.sidebar:
    st.markdown("""
<div class="sidebar-logo">
    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
    </svg>
    ThreatSense
</div>
""", unsafe_allow_html=True)
    
    st.markdown('<p class="metric-label">System Operations</p>', unsafe_allow_html=True)
    view_mode = st.radio("Display Mode", 
        ["Grid View", "CAM-01 (Front)", "CAM-02 (Lobby)", "CAM-03 (Parking)", "CAM-04 (Alley)"],
        label_visibility="collapsed")
    
    st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
    st.markdown('<p class="metric-label">Camera Configuration</p>', unsafe_allow_html=True)
    cam1_source = st.text_input("Source ID / Stream URL", value="0")

    st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
    st.markdown('<p class="metric-label">Health Monitor</p>', unsafe_allow_html=True)
    st.markdown("""
<div style="font-size: 0.8rem; background: rgba(30,41,59,0.3); padding: 10px; border-radius: 4px;">
    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;"><span>CAM-01</span><span style="color:#4ade80">ONLINE</span></div>
    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;"><span>CAM-02</span><span style="color:#4ade80">ONLINE</span></div>
    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;"><span>CAM-03</span><span style="color:#ef4444">OFFLINE</span></div>
    <div style="display: flex; justify-content: space-between;"><span>CAM-04</span><span style="color:#4ade80">ONLINE</span></div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
    st.markdown('<p class="metric-label">Demo Simulation</p>', unsafe_allow_html=True)
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("🚨 INTRUDER", use_container_width=True):
            st.session_state.trigger_sim = "intruder"
            st.session_state.risk_level = "THREAT"
    with col_s2:
        if st.button("🚶 LOITER", use_container_width=True):
            st.session_state.trigger_sim = "loitering"
            st.session_state.risk_level = "SUSPICIOUS"

    st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
    st.markdown('<p class="metric-label">AI Engine</p>', unsafe_allow_html=True)
    show_tracking = st.checkbox("Live Overlay", value=True)
    
    st.markdown("""
<div style="font-size: 0.7rem; color: #64748b; margin-top: 10px;">
    Engine: YOLOv8 / ByteTrack<br>
    FPS: ~24.5 | Latency: 12ms<br>
    GPU: Active [TENSORRT]
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

# Top Alert Runner
if st.session_state.trigger_sim:
    msg = "UNAUTHORIZED ENTRY DETECTED - CAM-04" if st.session_state.trigger_sim == "intruder" else "SUSPICIOUS LOITERING - CAM-01"
    reason = "Person detected in back alley (CAM-04) during restricted hours." if st.session_state.trigger_sim == "intruder" else "Individual hovering near front entrance for >45s without entry."
    st.session_state.incident_reasoning = reason
    
    st.markdown(f'<div class="alert-banner">⚠️ {msg} <span>JUST NOW</span></div>', unsafe_allow_html=True)
    # Trigger audio beep via alert_service (non-blocking)
    alert_service.trigger_alert("SYSTEM", msg)
    st.session_state.trigger_sim = None

# Stats Grid
st.markdown(f"""
<div class="metric-container">
    <div class="glass-card">
        <div class="metric-value">4</div>
        <div class="metric-label">Active Feeds</div>
    </div>
    <div class="glass-card">
        <div class="metric-value">02</div>
        <div class="metric-label">Daily Incidents</div>
    </div>
    <div class="glass-card">
        <div class="metric-value">{datetime.now().strftime("%H:%M:%S")}</div>
        <div class="metric-label">Internal Time</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Risk Level Indicator
risk_class = "risk-safe" if st.session_state.risk_level == "SAFE" else "risk-suspicious" if st.session_state.risk_level == "SUSPICIOUS" else "risk-threat"
st.markdown(f'<div class="risk-indicator {risk_class}">System Status: {st.session_state.risk_level}</div>', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT GRID
# ═════════════════════════════════════════════════════════════════════════════
col_vid, col_inc = st.columns([7.8, 2.2])

with col_inc:
    panel_html = f"""<div class="glass-card" style="height: 100%;">
<p class="metric-label" style="margin-bottom: 12px;">AI Reasoning Engine</p>
<div style="background: rgba(0, 242, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); padding: 12px; border-radius: 4px; border-left: 2px solid #00f2ff; margin-bottom: 20px;">
<p style="font-size: 0.8rem; line-height: 1.4; color: #e2e8f0; margin: 0;">
<b>ANALYSIS:</b> {st.session_state.incident_reasoning}
</p>
</div>
<p class="metric-label" style="margin-bottom: 12px;">Incident Timeline</p>
<div class="incident-item-pro">
<div style="display: flex; justify-content: space-between; font-size: 0.7rem; color: #94a3b8;"><span>CAM-04</span><span>17:06</span></div>
<div style="font-weight: 600; font-size: 0.8rem; color: #f87171;">Unauthorized Access</div>
</div>
<div class="incident-item-pro">
<div style="display: flex; justify-content: space-between; font-size: 0.7rem; color: #94a3b8;"><span>CAM-01</span><span>15:30</span></div>
<div style="font-weight: 600; font-size: 0.8rem; color: #fde047;">Suspicious Loitering</div>
</div>
<div style="margin-top: 30px;"></div>
<p class="metric-label" style="margin-bottom: 12px;">Incident Snapshots</p>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
<div style="aspect-ratio: 16/9; background: #1a1e2e; border: 1px solid #334155; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-size: 0.6rem; color: #475569;">EVT-882</div>
<div style="aspect-ratio: 16/9; background: #1a1e2e; border: 1px solid #334155; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-size: 0.6rem; color: #475569;">EVT-881</div>
</div>
<div style="margin-top: 30px;"></div>
<p class="metric-label" style="margin-bottom: 12px;">24h Analytics</p>
<div style="display: flex; flex-direction: column; gap: 8px;">
<div style="font-size: 0.7rem; margin-bottom: 4px;">Unauthorized Entry</div>
<div style="width: 100%; background: #1e293b; height: 6px; border-radius: 3px;"><div style="width: 70%; background: #00f2ff; height: 100%; border-radius: 3px;"></div></div>
<div style="font-size: 0.7rem; margin-top: 8px; margin-bottom: 4px;">Crowd Detection</div>
<div style="width: 100%; background: #1e293b; height: 6px; border-radius: 3px;"><div style="width: 30%; background: #c084fc; height: 100%; border-radius: 3px;"></div></div>
</div>
</div>"""
    st.markdown(panel_html, unsafe_allow_html=True)

with col_vid:
    # Timeline Player
    st.markdown('<div style="margin-top: -10px; margin-bottom: 10px;">', unsafe_allow_html=True)
    st.slider("Incident Timeline (Footage Review)", 0, 100, 100, help="Scrub through historical detection events.")
    st.markdown('</div>', unsafe_allow_html=True)
    if view_mode == "Grid View":
        vr1c1, vr1c2 = st.columns(2)
        vr2c1, vr2c2 = st.columns(2)
        ph1 = vr1c1.empty()
        ph2 = vr1c2.empty()
        ph3 = vr2c1.empty()
        ph4 = vr2c2.empty()
        video_phs = [ph1, ph2, ph3, ph4]
    else:
        ph_single = st.empty()
        video_phs = [ph_single]
    
    st.markdown(
        '<div style="margin-top: 10px; font-size: 0.65rem; color: #475569; letter-spacing: 0.5px;">SECURE TRANSMISSION | STREAM_ID: TS-X82-01 | ENCRYPTION: AES-256-GCM</div>',
        unsafe_allow_html=True
    )

# ── Load YOLOv8 Model ──
@st.cache_resource
def load_yolo_model():
    # 'yolov8n.pt' is the smallest, fastest model. It will auto-download on first run.
    return YOLO("yolov8n.pt")

yolo_model = load_yolo_model()

# ═════════════════════════════════════════════════════════════════════════════
# THE VIDEO GENERATOR
# ═════════════════════════════════════════════════════════════════════════════

if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0

frame_num = st.session_state.frame_count

# Persistence for VideoCapture
if "cap1" not in st.session_state:
    st.session_state.cap1 = None
    st.session_state.last_source = None

def get_cap1(source):
    if st.session_state.last_source != source:
        if st.session_state.cap1 is not None:
            st.session_state.cap1.release()
        
        try:
            # Handle webcam ID vs RTSP string
            src = int(source) if source.isdigit() else source
            cap = cv2.VideoCapture(src)
            if cap.isOpened():
                st.session_state.cap1 = cap
                st.session_state.last_source = source
            else:
                st.session_state.cap1 = None
        except:
            st.session_state.cap1 = None
    return st.session_state.cap1

def draw_cam_pill(img, text, source_type="LIVE", bg_color=(80, 70, 70)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 0.5, 1)[0]
    cv2.rectangle(img, (0, 0), (text_size[0] + 30, 32), bg_color, -1)
    cv2.circle(img, (text_size[0] + 30, 16), 16, bg_color, -1)
    cv2.putText(img, text, (15, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Source Type indicator (Top Right)
    w = img.shape[1]
    s_size = cv2.getTextSize(source_type, font, 0.4, 1)[0]
    cv2.rectangle(img, (w - s_size[0] - 20, 0), (w, 24), (0, 0, 0), -1)
    color = (0, 255, 0) if source_type == "LIVE" else (0, 165, 255)
    cv2.putText(img, source_type, (w - s_size[0] - 10, 16), font, 0.4, color, 1, cv2.LINE_AA)

def generate_cam1(frame_num: int, show_tracking: bool, source):
    """CAM-01: Real-time Video Stream (Webcam/RTSP)."""
    w, h = 640, 360
    cap = get_cap1(source)
    
    if cap and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Resize to dashboard standards
            frame = cv2.resize(frame, (w, h))
            if show_tracking:
                # YOLOv8 Inference
                results = yolo_model(frame, stream=True, verbose=False)
                for r in results:
                    boxes = r.boxes
                    for i, box in enumerate(boxes):
                        if int(box.cls[0]) == 0:  # person
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            
                            # Mock Tracking ID and Behavior
                            track_id = i + 101
                            behavior = "Loitering" if conf > 0.8 else "Moving"
                            color = (0, 255, 0) if behavior == "Moving" else (0, 165, 255)
                            
                            # Draw Bounding Box & Label
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            
                            # Label Panel
                            label_txt = f"ID #{track_id} | {behavior}"
                            cv2.putText(frame, label_txt, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                            cv2.putText(frame, f"CONF: {conf:.0%}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            draw_cam_pill(frame, "CAM-01 (Front Gate)", "LIVE")
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Fallback if stream fails
    img = np.ones((h, w, 3), dtype=np.uint8) * 40
    text = "NO VIDEO SOURCE - Check Sidebar Config"
    tsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.putText(img, text, ((w - tsize[0]) // 2, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)
    draw_cam_pill(img, "CAM-01 (Front Gate)", "ERROR", (20, 20, 40))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def generate_aux_stream(label: str, frame_num: int):
    """Auxiliary online CCTV footages simulation."""
    w, h = 640, 360
    img = np.ones((h, w, 3), dtype=np.uint8) * 60
    
    if "Lobby" in label:
        cv2.line(img, (0, h), (w//2, h//2), (100, 100, 100), 2)
        cv2.line(img, (w, h), (w//2, h//2), (100, 100, 100), 2)
        cv2.rectangle(img, (w//2-40, h//2), (w//2+40, h-40), (80, 80, 80), -1) 
    elif "Parking" in label:
        for i in range(5):
            y = int(h*0.4) + i*40
            cv2.line(img, (0, y), (w, y), (80, 80, 80), 1)
            cv2.putText(img, f"P-{i+1}", (20, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
            
    np.random.seed((frame_num // 3) % 50)
    noise = np.random.randint(0, 15, (h, w, 3), dtype=np.uint8)
    img = cv2.add(img, noise)
    
    draw_cam_pill(img, label, "ONLINE", (60, 60, 70))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def generate_cam4(frame_num: int, show_tracking: bool):
    """CAM-04: Real-time Back Alley (Mock)."""
    w, h = 640, 360
    f4 = np.ones((h, w, 3), dtype=np.uint8) * 20
    np.random.seed((frame_num // 2) % 100)
    noise = np.random.randint(0, 20, (h, w, 3), dtype=np.uint8)
    f4 = cv2.add(f4, noise)
    cv2.fillPoly(f4, [np.array([[0,0], [180,0], [120,h], [0,h]])], (10,10,10))
    cv2.fillPoly(f4, [np.array([[w,0], [w-180,0], [w-120,h], [w,h]])], (10,10,10))
    
    if show_tracking:
        # YOLOv8 Inference on mock alley
        results = yolo_model(f4, stream=True, verbose=False)
        for r in results:
            boxes = r.boxes
            for i, box in enumerate(boxes):
                if int(box.cls[0]) == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    # Enhanced Display
                    track_id = i + 505
                    risk_score = int(conf * 100)
                    
                    cv2.rectangle(f4, (x1, y1), (x2, y2), (200, 200, 255), 2)
                    cv2.putText(f4, f"PERSON #{track_id}", (x1, y1-22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 255), 1)
                    cv2.putText(f4, f"RISK: {risk_score}%", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 255), 1)
    
    draw_cam_pill(f4, "CAM-04 (Back Alley)", "LIVE", (20, 20, 30))
    return cv2.cvtColor(f4, cv2.COLOR_BGR2RGB)

# ── Render Loop ──
try:
    for _ in range(12):
        if view_mode == "Grid View":
            video_phs[0].image(generate_cam4(frame_num, show_tracking), width="stretch")
            video_phs[1].image(generate_cam1(frame_num, show_tracking, cam1_source), width="stretch")
            video_phs[2].image(generate_aux_stream("CAM-02 (Lobby)", frame_num), width="stretch")
            video_phs[3].image(generate_aux_stream("CAM-03 (Parking)", frame_num), width="stretch")
        else:
            if "Alley" in view_mode:
                video_phs[0].image(generate_cam4(frame_num, show_tracking), width="stretch")
            elif "Front" in view_mode:
                video_phs[0].image(generate_cam1(frame_num, show_tracking, cam1_source), width="stretch")
            else:
                video_phs[0].image(generate_aux_stream(view_mode, frame_num), width="stretch")
        
        frame_num += 1
        time.sleep(1 / 20)
finally:
    st.session_state.frame_count = frame_num

st.rerun()
