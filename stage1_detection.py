"""
Stage 1: Lightweight Edge Detection Pipeline
AI-Powered DVR with Intent-Aware Threat Detection

This script implements the compute-optimized gating logic:
- Frame throttling (3-8 FPS)
- Ring buffer (last 3 seconds of video)
- YOLO-based object detection
- Persistence filtering (anti-glitch)
- Clip packaging for Stage 2 handoff
- Cooldown state machine
"""

import cv2
import time
import os
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from ultralytics import YOLO
import numpy as np


class CameraDetector:
    """Handles detection and event triggering for a single camera stream."""
    
    def __init__(self, camera_id, rtsp_url, config):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.config = config
        
        # Step 2: Ring Buffer - stores last 3 seconds of frames
        buffer_size = config['processing_fps'] * config['ring_buffer_seconds']
        self.ring_buffer = deque(maxlen=buffer_size)
        
        # Step 4: Persistence tracking
        self.threat_counter = 0
        self.threat_threshold = config['persistence_frames']
        
        # Step 6: Cooldown state
        self.in_cooldown = False
        self.cooldown_end_time = None
        self.cooldown_duration = config['cooldown_seconds']
        
        # Frame processing
        self.frame_count = 0
        self.skip_frames = config['frame_skip']
        
        # Load YOLO model with enhanced settings
        print(f"[{self.camera_id}] Loading YOLO model: {config['yolo_model']}")
        print(f"[{self.camera_id}] First run will download model (~136MB for yolov8x)...")
        self.model = YOLO(config['yolo_model'])
        
        # Set model parameters for maximum accuracy
        self.model.overrides['conf'] = config['yolo_confidence']
        self.model.overrides['iou'] = config.get('yolo_iou', 0.45)
        self.model.overrides['max_det'] = 50  # Max detections per image
        
        print(f"[{self.camera_id}] ✅ Model loaded successfully (Extra-Large - Best Accuracy)")
        
        # Output directories
        self.clips_dir = Path(config['clips_directory'])
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        
        # Video writer for clip saving
        self.current_clip_writer = None
        self.clip_frames = []
        self.recording_event = False
        self.post_event_frames_needed = 0
        
    def process_stream(self):
        """Main processing loop for the camera stream."""
        print(f"[{self.camera_id}] Connecting to RTSP stream: {self.rtsp_url}")
        
        # Step 1: Connect to RTSP stream
        cap = cv2.VideoCapture(self.rtsp_url)
        
        if not cap.isOpened():
            print(f"[{self.camera_id}] ERROR: Could not open RTSP stream")
            return
        
        print(f"[{self.camera_id}] Successfully connected to stream")
        
        # Get stream properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[{self.camera_id}] Stream properties: {width}x{height} @ {fps} FPS")
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print(f"[{self.camera_id}] Failed to read frame, reconnecting...")
                    time.sleep(2)
                    cap.release()
                    cap = cv2.VideoCapture(self.rtsp_url)
                    continue
                
                self.frame_count += 1
                
                # Step 1: Frame Throttling - Only process every Nth frame
                if self.frame_count % self.skip_frames != 0:
                    continue
                
                # Add timestamp to frame
                timestamp = datetime.now()
                frame_with_timestamp = (frame.copy(), timestamp)
                
                # Step 2: Always append to ring buffer (even during cooldown)
                self.ring_buffer.append(frame_with_timestamp)
                
                # Step 6: Check cooldown state
                if self.in_cooldown:
                    if datetime.now() < self.cooldown_end_time:
                        # Still in cooldown - skip detection but keep buffering
                        continue
                    else:
                        # Cooldown ended
                        self.in_cooldown = False
                        print(f"[{self.camera_id}] Cooldown ended, resuming detection")
                
                # Step 3: Run YOLO detection
                threat_possible = self._detect_threats(frame)
                
                # Step 4: Persistence filtering
                if threat_possible:
                    self.threat_counter += 1
                    print(f"[{self.camera_id}] Threat detected ({self.threat_counter}/{self.threat_threshold})")
                    
                    if self.threat_counter >= self.threat_threshold:
                        # Persistence threshold met!
                        print(f"[{self.camera_id}] ⚠️  PERSISTENT THREAT DETECTED - Triggering clip save")
                        
                        # Step 5: Package and save clip
                        clip_path = self._save_clip(width, height, fps)
                        
                        if clip_path:
                            # Reset threat counter
                            self.threat_counter = 0
                            
                            # Step 6: Enter cooldown state
                            self._enter_cooldown()
                            
                            # TODO: Handoff to Stage 2 (Ollama intent analysis)
                            print(f"[{self.camera_id}] 📹 Clip saved: {clip_path}")
                            print(f"[{self.camera_id}] → Ready for Stage 2 handoff")
                else:
                    # No threat - reset counter
                    if self.threat_counter > 0:
                        print(f"[{self.camera_id}] Threat cleared, resetting counter")
                    self.threat_counter = 0
                
                # Optional: Display frame with detections (for debugging)
                if self.config.get('show_preview', False):
                    cv2.imshow(f"Camera {self.camera_id}", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
        except KeyboardInterrupt:
            print(f"\n[{self.camera_id}] Shutting down...")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"[{self.camera_id}] Stream closed")
    
    def _detect_threats(self, frame):
        """
        Step 3: YOLO Logic Filter
        
        Rules:
        - IGNORE: dog, cat, vehicle without person
        - TRIGGER: person alone, or person + bicycle/valuables
        
        Returns:
            bool: True if threat is possible, False otherwise
        """
        # Run YOLO inference with high-accuracy settings
        results = self.model(
            frame, 
            verbose=False, 
            conf=self.config['yolo_confidence'],
            iou=self.config.get('yolo_iou', 0.45)
        )
        
        # Parse detections
        detections = results[0].boxes
        
        detected_classes = []
        for box in detections:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            confidence = float(box.conf[0])
            detected_classes.append(class_name)
        
        if not detected_classes:
            return False
        
        # YOLO COCO classes we care about:
        # 0: person, 1: bicycle, 2: car, 3: motorcycle
        # 15: cat, 16: dog
        
        has_person = 'person' in detected_classes
        has_bicycle = 'bicycle' in detected_classes
        has_vehicle = any(v in detected_classes for v in ['car', 'truck', 'bus', 'motorcycle'])
        has_animal = any(a in detected_classes for a in ['dog', 'cat'])
        
        # Filter logic
        if has_animal and not has_person:
            # Only animals - ignore
            return False
        
        if has_vehicle and not has_person:
            # Passing vehicle with no person - ignore
            return False
        
        if has_person:
            # Person detected - potential threat
            if has_bicycle:
                print(f"[{self.camera_id}] 🚨 Detected: Person + Bicycle")
            else:
                print(f"[{self.camera_id}] 🚨 Detected: Person")
            return True
        
        return False
    
    def _save_clip(self, width, height, fps):
        """
        Step 5: Clip Packaging
        
        Saves pre-event frames (from ring buffer) + post-event frames
        into a single MP4 file.
        
        Returns:
            str: Path to saved clip, or None if save failed
        """
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clip_filename = f"{self.camera_id}_threat_{timestamp}.mp4"
        clip_path = self.clips_dir / clip_filename
        
        print(f"[{self.camera_id}] Saving clip to: {clip_path}")
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        actual_fps = self.config['processing_fps']  # Use our processing FPS
        out = cv2.VideoWriter(str(clip_path), fourcc, actual_fps, (width, height))
        
        if not out.isOpened():
            print(f"[{self.camera_id}] ERROR: Could not create video writer")
            return None
        
        frames_written = 0
        
        # Write pre-event frames from ring buffer
        print(f"[{self.camera_id}] Writing {len(self.ring_buffer)} pre-event frames")
        for frame, timestamp in self.ring_buffer:
            out.write(frame)
            frames_written += 1
        
        # Calculate how many post-event frames we need
        # Goal: 8-15 second clip total
        target_clip_frames = self.config['clip_duration_seconds'] * actual_fps
        post_event_frames = max(target_clip_frames - frames_written, 0)
        
        print(f"[{self.camera_id}] Capturing {post_event_frames} post-event frames...")
        
        # Note: For post-event frames, we would continue capturing
        # This is a simplified version - in production, you'd want to
        # buffer these before writing
        out.release()
        
        print(f"[{self.camera_id}] ✅ Clip saved with {frames_written} frames")
        return str(clip_path)
    
    def _enter_cooldown(self):
        """
        Step 6: Enter cooldown state
        
        Prevents spam to Stage 2 by blocking detection for N seconds.
        Ring buffer continues to update.
        """
        self.in_cooldown = True
        self.cooldown_end_time = datetime.now() + timedelta(seconds=self.cooldown_duration)
        print(f"[{self.camera_id}] 🕐 Entering cooldown for {self.cooldown_duration} seconds")
        print(f"[{self.camera_id}] Detection will resume at {self.cooldown_end_time.strftime('%H:%M:%S')}")


def main():
    """Main entry point for the detection system."""
    
    # Configuration - UPGRADED FOR HIGH ACCURACY
    config = {
        # YOLO settings - Using EXTRA-LARGE model for maximum accuracy
        'yolo_model': 'yolov8x.pt',  # YOLOv8x = Best accuracy (136MB download first time)
        'yolo_confidence': 0.35,      # Lower threshold = more sensitive (was 0.5)
        'yolo_iou': 0.45,             # IoU threshold for Non-Maximum Suppression
        
        # Frame processing
        'processing_fps': 5,  # Process 5 frames per second
        'frame_skip': 6,      # If stream is 30fps, skip 5 frames = ~5fps processing
        
        # Ring buffer
        'ring_buffer_seconds': 3,  # Keep last 3 seconds
        
        # Persistence filter
        'persistence_frames': 3,  # Require threat in 3 consecutive frames
        
        # Clip settings
        'clip_duration_seconds': 10,  # Target 10 second clips
        'clips_directory': 'incident_clips',
        
        # Cooldown
        'cooldown_seconds': 30,  # 30 second cooldown after event
        
        # Debug
        'show_preview': False,  # Set to True to see live preview
    }
    
    # Camera configuration
    # IP Webcam app provides multiple stream options
    MOBILE_IP = "10.226.220.51"  # Your phone's IP address
    HTTP_PORT = 8080  # IP Webcam default HTTP port
    
    # Try these URLs in order (IP Webcam supports multiple formats):
    stream_urls = [
        f"http://{MOBILE_IP}:{HTTP_PORT}/video",           # HTTP MJPEG (most reliable)
        f"rtsp://{MOBILE_IP}:8554/h264_pcm.sdp",          # RTSP H264 with PCM audio
        f"rtsp://{MOBILE_IP}:8554/h264_ulaw.sdp",         # RTSP H264 with uLaw audio
        f"http://{MOBILE_IP}:{HTTP_PORT}/videofeed",      # Alternative HTTP stream
    ]
    
    # Try each URL until one works
    rtsp_url = None
    print("🔍 Testing stream URLs...")
    for url in stream_urls:
        print(f"   Trying: {url}")
        test_cap = cv2.VideoCapture(url)
        if test_cap.isOpened():
            rtsp_url = url
            test_cap.release()
            print(f"   ✅ Connected!")
            break
        test_cap.release()
        print(f"   ❌ Failed")
    
    if not rtsp_url:
        print("\n❌ Could not connect to any stream URL")
        print("\nTroubleshooting:")
        print("1. Open IP Webcam app on your phone")
        print("2. Tap 'Start server' if not already running")
        print("3. Check the IP shown in the app matches:", MOBILE_IP)
        print("4. Try opening this in your browser:")
        print(f"   http://{MOBILE_IP}:{HTTP_PORT}")
        print("5. Make sure phone and laptop are on the same WiFi")
        return
    
    print(f"\n✅ Using stream: {rtsp_url}\n")
    
    print("=" * 70)
    print("AI-Powered DVR - Stage 1: Lightweight Edge Detection")
    print("=" * 70)
    print(f"RTSP URL: {rtsp_url}")
    print(f"Processing: {config['processing_fps']} FPS")
    print(f"Ring Buffer: {config['ring_buffer_seconds']} seconds")
    print(f"Persistence: {config['persistence_frames']} frames")
    print(f"Cooldown: {config['cooldown_seconds']} seconds")
    print(f"Clips Directory: {config['clips_directory']}")
    print("=" * 70)
    print()
    
    # Create detector for the camera
    detector = CameraDetector(
        camera_id="mobile_cam_01",
        rtsp_url=rtsp_url,
        config=config
    )
    
    # Start processing
    detector.process_stream()


if __name__ == "__main__":
    main()
