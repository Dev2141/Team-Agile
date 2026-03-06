# VICSTA Hackathon – Grand Finale
**VIT College, Kondhwa Campus | 5th – 6th March**

---

## Team Details

- **Team Name:** Team Agile  
- **Members:**  
[Dev Padhariya](https://www.github.com/Dev2141)  
[Harshit Jain](https://www.github.com/HarshitJain26-2)  
[Harsh Chaudhari](https://www.github.com/harsch00)  
[Kartik Jadao](https://www.github.com/Mr-kartik20)  
- **Domain:** Productivity & Security  

---

## Project

**Problem:** ThreatSense AI-DVR  

**Solution:**  

Traditional CCTV systems are reactive — they record footage but require a human operator to watch screens all day to spot threats. This creates a massive blind spot: most incidents go unnoticed until after the fact. **ThreatSense AI-DVR** solves this by building a fully autonomous, AI-powered security surveillance pipeline that watches your cameras in real time, detects humans the moment they appear, records a clip of the event, and then uses a large Vision-Language Model (VLM) to *understand* what is happening — distinguishing genuine threats (intrusion, loitering, assault, theft) from harmless activity (delivery workers, residents, pets) with natural-language reasoning.

---

## About ThreatSense AI-DVR

### 🎯 What It Does

ThreatSense AI-DVR turns any Android phone (via the *IP Webcam* app) into an intelligent, always-on security camera with three layers of intelligence:

| Layer | Technology | Role |
|---|---|---|
| **Live Feed** | IP Webcam (HTTP MJPEG stream) | Streams video over Wi-Fi |
| **Object Detection** | YOLOv8n | Detects humans in every frame |
| **Threat Analysis** | Ollama + Qwen3-VL 235B | Understands *what* the person is doing |

---

### 🏗️ Architecture Overview

```
Android Phone (IP Webcam App)
        │  HTTP MJPEG stream
        ▼
┌─────────────────────────────────────────────────┐
│          app.py  (Streamlit Dashboard)          │
│                                                 │
│  1. Stream ingestion via OpenCV / FFMPEG        │
│  2. YOLOv8n person detection (every 3rd frame)  │
│  3. Auto-record clip (8 s) on N consecutive     │
│     person detections                           │
│  4. Extract 5 keyframes from the clip           │
│  5. Send keyframes → OllamaAnalyser             │
│  6. Display verdict + clip + reasoning in UI    │
└─────────────────────────────────────────────────┘
        │  keyframes (JPEG bytes, base64)
        ▼
┌─────────────────────────────────────────────────┐
│       analyser.py  (OllamaAnalyser)             │
│                                                 │
│  POST /api/generate → Ollama (Qwen3-VL 235B)   │
│  Returns structured JSON:                       │
│    verdict · type · confidence · reason         │
│    scene_description · behaviour_analysis       │
│    risk_factors · recommended_action            │
└─────────────────────────────────────────────────┘
        │  alert (is_threat = true)
        ▼
┌─────────────────────────────────────────────────┐
│       alert_service.py                          │
│  Desktop notification + audio beep (non-block)  │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│       api.py  (FastAPI – standalone VLM API)    │
│                                                 │
│  GET  /          – Web UI for manual upload     │
│  GET  /health    – Ollama health check          │
│  POST /analyse/images  – Analyse 1–6 images     │
│  POST /analyse/video   – Analyse a video file   │
└─────────────────────────────────────────────────┘
```

---

### 📂 File Structure

```
ThreatSense/
├── app.py              # Main Streamlit dashboard (live camera + detection)
├── analyser.py         # OllamaAnalyser — sends keyframes to VLM, parses JSON
├── api.py              # FastAPI REST API — standalone image/video analysis UI
├── alert_service.py    # Desktop notifications + audio alert (non-blocking)
├── init_db.py          # Database initialisation (incident logging)
├── requirements.txt    # All Python dependencies
├── start.ps1           # PowerShell launcher (starts all services)
├── yolov8n.pt          # YOLOv8 nano model weights
├── pages/
│   └── 1_Full_Demo.py  # Streamlit multi-page: upload & analyse any video
└── templates/
    └── index.html      # Jinja2 template for FastAPI web UI
```

---

### 🔬 How the Detection Pipeline Works

#### Step 1 — Live Stream Ingestion
The Streamlit dashboard connects to an Android phone running the **IP Webcam** app over the local Wi-Fi network. OpenCV reads the MJPEG stream frame-by-frame using the FFMPEG backend for low-latency decoding.

#### Step 2 — YOLOv8n Person Detection
Every 3rd frame is passed through **YOLOv8 nano** (`yolov8n.pt`). If a bounding box for class `person` is detected above the confidence threshold (default: 35%), the frame is marked as containing a person. Bounding boxes, track IDs, behaviour labels (*Moving* / *Loitering*) and confidence scores are drawn onto the frame overlay.

#### Step 3 — Automatic Clip Recording
A state machine manages clip recording:
- **Trigger:** 3 consecutive frames with a detected person start a recording.
- **Buffer:** All frames are accumulated in a rolling in-memory buffer.
- **Duration:** Recording captures for the configured duration (default 8 s).
- **Cooldown:** A 90-frame cooldown prevents duplicate clips for the same event.

#### Step 4 — Keyframe Extraction & FFmpeg Encoding
When the clip completes:
1. **5 keyframes** are evenly sampled from the clip buffer.
2. The full clip is encoded to browser-playable **H.264 MP4** via `ffmpeg` (libx264, ultrafast preset) and displayed in the dashboard for operator review.

#### Step 5 — VLM Threat Analysis (Ollama + Qwen3-VL 235B)
The 5 keyframes are sent as base64 JPEG images to **Ollama** running `qwen3-vl:235b-cloud` via `/api/generate`. The model is given a detailed security-analyst system prompt and asked to return a structured JSON verdict:

```json
{
  "verdict": "THREAT",
  "type": "intrusion",
  "confidence": 0.87,
  "reason": "Person scaled the perimeter fence and entered restricted area.",
  "scene_description": "...",
  "behaviour_analysis": "...",
  "risk_factors": ["Forced entry", "After hours"],
  "recommended_action": "Dispatch security to north gate immediately."
}
```

The model distinguishes 8 threat categories: `theft`, `assault`, `vandalism`, `trespass`, `intrusion`, `loitering`, `suspicious_package`, `other`. A built-in retry mechanism re-prompts the model once if the JSON is malformed, ensuring reliability.

#### Step 6 — Real-Time Alert
If `is_threat = true`:
- The **risk indicator** in the UI switches to 🔴 **THREAT** with a pulsing red animation.
- A **desktop push notification** is fired via `plyer` (Windows/macOS/Linux).
- A **1000 Hz audio beep** (`winsound.Beep`) alerts operators who may not be watching the screen.
- All operations run on daemon threads so the detection pipeline is never blocked.

---

### 🖥️ Dashboard Features

The Streamlit dashboard (`app.py`) is built with a premium dark UI (Orbitron + Inter fonts, glassmorphism cards, animated alerts):

| Feature | Description |
|---|---|
| **Live Video Feed** | Real-time IP webcam stream with YOLO bounding boxes |
| **Risk Status Pill** | SAFE / SUSPICIOUS / THREAT with pulsing animation |
| **AI Reasoning Engine** | Ollama's natural-language finding displayed instantly |
| **Latest Clip Playback** | H.264 MP4 video of the detected event, embedded in-browser |
| **Incident Timeline** | Running count of auto-detected events |
| **24H Analytics** | Visual bars for Unauthorized Entry & Crowd Detection |
| **Quick Simulation** | 🚨 INTRUDER and 🚶 LOITER buttons for live demo testing |
| **Adjustable Settings** | Clip duration, YOLO confidence, bounding box toggle |

---

### 🌐 Standalone VLM API (FastAPI)

`api.py` exposes a standalone REST API and web UI for manual threat analysis without the live camera:

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web UI — drag-and-drop image/video upload |
| `/health` | GET | Checks Ollama connectivity and model availability |
| `/analyse/images` | POST | Upload 1–6 JPEG/PNG images, returns threat verdict |
| `/analyse/video` | POST | Upload any video; ffmpeg extracts frames → VLM analysis |

---

### 🚀 How to Run

**Prerequisites:** Python 3.10+, FFmpeg in PATH, Ollama running locally, Android phone with *IP Webcam* app on the same Wi-Fi

#### 1. Install dependencies
```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

#### 2. Start all services (Windows — one click)
```powershell
.\start.ps1
```
This launches:
- **Ollama** (VLM backend)
- **Uvicorn** → FastAPI analyser API on `http://localhost:8100`
- **Streamlit** dashboard on `http://localhost:8501`

#### 3. Connect your phone camera
- Install *IP Webcam* from the Play Store on your Android phone
- Start the server in the app
- Paste the URL shown (e.g. `http://192.168.1.x:8080/video`) into the sidebar

---

### ⚙️ Configuration

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `qwen3-vl:235b-cloud` | VLM model to use |
| `IP_WEBCAM_URL` | `http://192.0.0.4:8080/video` | Default camera stream URL |
| `PERSON_TRIGGER_FRAMES` | `3` | Consecutive person frames to trigger recording |
| `CLIP_DURATION_SEC` | `8` | Clip length in seconds |
| `MAX_KEYFRAMES_VLM` | `5` | Keyframes sent to Ollama |
| `CLIP_COOLDOWN_FRAMES` | `90` | Frames between clips (~3 s at 30 fps) |

---

### 🧠 AI Models Used

| Model | Usage |
|---|---|
| **YOLOv8n** (Ultralytics) | Real-time person detection in video frames |
| **Qwen3-VL 235B** (via Ollama) | Vision-language threat classification and reasoning |

---

## Attribution

| Library / Tool | Purpose |
|---|---|
| [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) | Real-time object detection (`yolov8n.pt`) |
| [Ollama](https://ollama.com/) | Local LLM/VLM inference runtime |
| [Qwen3-VL 235B](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | Vision-Language Model for threat reasoning |
| [Streamlit](https://streamlit.io/) | Interactive Python web dashboard |
| [FastAPI](https://fastapi.tiangolo.com/) | REST API framework for standalone VLM endpoint |
| [OpenCV](https://opencv.org/) | Video stream capture and frame processing |
| [FFmpeg](https://ffmpeg.org/) | Video encoding (H.264 MP4) and frame extraction |
| [httpx](https://www.python-httpx.org/) | Async HTTP client for Ollama API calls |
| [Plyer](https://github.com/kivy/plyer) | Cross-platform desktop notifications |
| [IP Webcam (Android)](https://play.google.com/store/apps/details?id=com.pas.webcam) | Turns an Android phone into an IP camera |
| [Google Fonts — Orbitron & Inter](https://fonts.google.com/) | Dashboard typography |
