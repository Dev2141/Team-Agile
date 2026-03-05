AI-POWERED DVR - MODULE 1: EDGE DETECTION
==========================================

SETUP:
------
1. Install dependencies:
   pip install -r requirements.txt

2. YOLO models will auto-download on first run
   - yolov8m.pt is used by default (52MB, balanced speed/accuracy)
   - Models are downloaded from Ultralytics automatically

FILES:
------
1. stage1_detection_video.py  - Main script (optimized, auto-detects videos)
2. stage1_detection.py         - RTSP stream version (for phone cameras)
3. requirements.txt            - Python dependencies

USAGE:
------
Local:
  python stage1_detection_video.py

Google Colab:
  1. Upload videos to Colab
  2. Upload stage1_detection_video.py
  3. Run: !python stage1_detection_video.py

OUTPUT:
-------
Clips saved to: ../incident_clips/
Videos read from: ../videos/

PERFORMANCE:
------------
- CPU: 8-15 FPS
- GPU (local): 20-30 FPS
- GPU (Colab T4): 30-50 FPS

GITHUB:
-------
Repository: https://github.com/Dev2141/Team-Agile
Branch: YOLO-Processing
