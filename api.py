"""
Ollama VLM threat-detection API (backend pipeline from ollama_pipeline).

Endpoints
---------
GET  /                     — browser UI (image/video upload)
GET  /health               — check Ollama reachability + model
POST /analyse/images       — send 1-6 JPEG images as form files, get JSON verdict
POST /analyse/video        — upload a video file, extract frames, get JSON verdict

Run
---
    uvicorn api:app --host 0.0.0.0 --port 8100 --reload
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request

from analyser import OllamaAnalyser, AnalysisResult

# --------------------------------------------------------------------------- #
# Logging                                                                      #
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# App                                                                          #
# --------------------------------------------------------------------------- #

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3-vl:235b-cloud")

app = FastAPI(
    title="Ollama VLM Threat Analyser",
    description="Standalone intent-based threat detection using Ollama Qwen-VL",
    version="1.0.0",
)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
analyser = OllamaAnalyser(ollama_url=OLLAMA_URL, model=OLLAMA_MODEL)


# --------------------------------------------------------------------------- #
# Routes                                                                       #
# --------------------------------------------------------------------------- #

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "model": OLLAMA_MODEL})


@app.get("/health")
async def health():
    ok, message = await analyser.health_check()
    return JSONResponse({
        "ok": ok,
        "message": message,
        "ollama_url": OLLAMA_URL,
        "model": OLLAMA_MODEL,
    }, status_code=200 if ok else 503)


@app.post("/analyse/images")
async def analyse_images(
    images: list[UploadFile] = File(..., description="1–6 JPEG/PNG image files"),
    context: str = Form("", description="Optional context hint, e.g. 'person near bicycle'"),
):
    """
    Upload 1 to 6 image files and get a threat verdict.

    Returns JSON:
        {
          "verdict":    "THREAT" | "BENIGN",
          "type":       "theft|assault|vandalism|trespass|intrusion|none",
          "confidence": 0.0 – 1.0,
          "reason":     "one sentence",
          "is_threat":  true | false
        }
    """
    if not images:
        raise HTTPException(400, "At least one image required")
    if len(images) > 6:
        raise HTTPException(400, "Maximum 6 images per request")

    jpegs: list[bytes] = []
    for img in images:
        data = await img.read()
        if not data:
            raise HTTPException(400, f"Empty file: {img.filename}")
        jpegs.append(data)

    logger.info("Analysing %d image(s) with context=%r", len(jpegs), context)
    result = await analyser.analyse_images(jpegs, context=context)

    if result is None:
        raise HTTPException(503, "Ollama returned no result — check that it is running and the model is loaded")

    logger.info("Result: %s / %s (conf=%.2f)", result.verdict, result.type, result.confidence)
    return JSONResponse(result.to_dict())


@app.post("/analyse/video")
async def analyse_video(
    file: UploadFile = File(..., description="Video file (MP4, MOV, AVI, MKV, etc.)"),
    fps: int = Form(2, description="Frames per second to sample (1–10)"),
    max_keyframes: int = Form(6, description="Max keyframes sent to VLM (1–6)"),
    context: str = Form("", description="Optional context hint"),
):
    """
    Upload a video file. Frames are extracted, suspicious ones are sent to Qwen-VL.

    Returns JSON:
        {
          "verdict":         "THREAT" | "BENIGN",
          "type":            "...",
          "confidence":      0.0–1.0,
          "reason":          "...",
          "is_threat":       true | false,
          "total_frames":    120,
          "suspicious_frames": 18,
          "keyframes_to_vlm":  6
        }
    """
    fps = max(1, min(fps, 10))
    max_keyframes = max(1, min(max_keyframes, 6))
    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"

    tmp_dir = Path(tempfile.mkdtemp(prefix="ollama_upload_"))
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    tmp_video = tmp_dir / f"input{suffix}"

    try:
        content = await file.read()
        if not content:
            raise HTTPException(400, "Empty file")
        tmp_video.write_bytes(content)
        logger.info("Video saved: %s (%d bytes), fps=%d", tmp_video.name, len(content), fps)

        # ---- Extract frames ----
        cmd = [
            ffmpeg_bin, "-loglevel", "error",
            "-i", str(tmp_video),
            "-vf", f"fps={fps}",
            "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "5",
            "pipe:1",
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
        except asyncio.TimeoutError:
            raise HTTPException(504, "Frame extraction timed out — try a shorter video")
        except FileNotFoundError:
            raise HTTPException(503, "ffmpeg not found in PATH — please install ffmpeg")

        if proc.returncode != 0:
            err = stderr.decode(errors="replace")[:500]
            raise HTTPException(422, f"ffmpeg failed: {err}")

        # ---- Split MJPEG stream into frames ----
        frame_jpegs: list[bytes] = []
        raw = stdout
        while True:
            s = raw.find(b"\xff\xd8")
            if s == -1:
                break
            e = raw.find(b"\xff\xd9", s + 2)
            if e == -1:
                break
            frame_jpegs.append(raw[s: e + 2])
            raw = raw[e + 2:]

        if not frame_jpegs:
            raise HTTPException(422, "No frames extracted — is the file a valid video?")

        total_frames = len(frame_jpegs)
        logger.info("Extracted %d frames", total_frames)

        # ---- Simple motion/object-presence pre-filter ----
        # Select frames evenly spread from the video to send to VLM
        # (no YOLO here — this module is Ollama-only)
        n_kf = min(max_keyframes, total_frames)
        step = max(1, total_frames // n_kf)
        keyframes = frame_jpegs[::step][:n_kf]

        logger.info("Sending %d keyframes to Ollama", len(keyframes))
        result = await analyser.analyse_images(keyframes, context=context)

        if result is None:
            raise HTTPException(503, "Ollama returned no result — check that it is running")

        logger.info("Result: %s / %s (conf=%.2f)", result.verdict, result.type, result.confidence)

        return JSONResponse({
            **result.to_dict(),
            "total_frames":     total_frames,
            "keyframes_to_vlm": len(keyframes),
        })

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
