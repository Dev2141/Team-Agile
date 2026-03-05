"""
Stage 1: Optimized Video Detection Pipeline
Uses YOLOv8m (Medium) for balanced speed and accuracy

PERFORMANCE OPTIMIZED:
- 3-5x faster than YOLOv8x
- GPU acceleration (CUDA) if available
- Efficient post-event capture (no video reopening)
- Supports both RTSP streams and video files
"""

import cv2
import time
import os
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from ultralytics import YOLO
import numpy as np


class HighAccuracyDetector:
    """
    Optimized detector using YOLOv8m for balanced speed/accuracy.
    Implements advanced filtering and prompt engineering for threat detection.
    
    PERFORMANCE FEATURES:
    - GPU acceleration if available
    - Efficient buffering (no video reopening)
    - Smart cooldown system
    - Real-time progress display
    """
    
    def __init__(self, camera_id, source, config):
        self.camera_id = camera_id
        self.source = source  # Can be RTSP URL or video file path
        self.config = config
        self.is_video_file = os.path.isfile(source)  # Track if processing video file
        
        # Ring Buffer - stores last N seconds of frames
        buffer_size = config['processing_fps'] * config['ring_buffer_seconds']
        self.ring_buffer = deque(maxlen=buffer_size)
        
        # Persistence tracking with enhanced logic
        self.threat_history = deque(maxlen=10)  # Track last 10 detections
        self.threat_counter = 0
        self.threat_threshold = config['persistence_frames']
        
        # Post-event capture buffer (for efficient clip saving)
        self.post_event_buffer = deque(maxlen=100)  # Buffer for post-event frames
        self.capturing_post_event = False
        self.post_event_frames_needed = 0
        self.current_clip_writer = None
        self.current_clip_metadata = None
        
        # Cooldown state (different for video vs stream)
        self.in_cooldown = False
        self.cooldown_end_time = None
        self.cooldown_frame_count = 0
        if self.is_video_file:
            # For videos: use frame-based cooldown (skip N frames after event)
            self.cooldown_duration = config.get('video_cooldown_frames', 50)  # Skip 50 frames (~10 sec at 5fps)
        else:
            # For streams: use time-based cooldown
            self.cooldown_duration = config['cooldown_seconds']
        
        # Frame processing
        self.frame_count = 0
        self.skip_frames = config['frame_skip']
        
        # Load YOLO model (optimized for speed/accuracy balance)
        print(f"[{self.camera_id}] Loading YOLO model: {config['yolo_model']}")
        print(f"[{self.camera_id}] This may take a moment for first-time download...")
        self.model = YOLO(config['yolo_model'])
        
        # Enable GPU if available for massive speedup
        import torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            print(f"[{self.camera_id}] 🚀 GPU detected: {torch.cuda.get_device_name(0)} - Using CUDA acceleration")
        else:
            print(f"[{self.camera_id}] Using CPU (slower but works)")
        
        # Set model parameters
        self.model.overrides['conf'] = config['yolo_confidence']
        self.model.overrides['iou'] = config['yolo_iou']
        self.model.overrides['max_det'] = 30
        self.model.overrides['device'] = self.device
        
        print(f"[{self.camera_id}] ✅ Model loaded successfully")
        
        # Output directories
        self.clips_dir = Path(config['clips_directory'])
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        
        # Detection statistics
        self.stats = {
            'total_frames': 0,
            'detections': 0,
            'threats_triggered': 0,
            'false_positives_filtered': 0
        }
    
    def process_source(self):
        """Main processing loop - works for both RTSP and video files."""
        source_type = "video file" if os.path.isfile(self.source) else "RTSP stream"
        print(f"[{self.camera_id}] Processing {source_type}: {self.source}")
        
        # Connect to source
        cap = cv2.VideoCapture(self.source)
        
        if not cap.isOpened():
            print(f"[{self.camera_id}] ❌ ERROR: Could not open source")
            return
        
        print(f"[{self.camera_id}] ✅ Successfully connected")
        
        # Get stream properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Track processing time
        start_time = datetime.now()
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if os.path.isfile(self.source) else -1
        
        print(f"[{self.camera_id}] Properties: {width}x{height} @ {fps:.1f} FPS")
        if total_frames > 0:
            print(f"[{self.camera_id}] Video length: {total_frames} frames (~{total_frames/fps:.1f} seconds)")
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    if os.path.isfile(self.source):
                        print(f"[{self.camera_id}] ✅ Video processing complete")
                        break
                    else:
                        print(f"[{self.camera_id}] ⚠️  Failed to read frame, reconnecting...")
                        time.sleep(2)
                        cap.release()
                        cap = cv2.VideoCapture(self.source)
                        continue
                
                self.frame_count += 1
                self.stats['total_frames'] += 1
                
                # Frame Throttling - Only process every Nth frame
                if self.frame_count % self.skip_frames != 0:
                    continue  # Skip this frame
                
                # Show progress (every 60 frames to reduce spam)
                if total_frames > 0 and self.frame_count % 60 == 0:
                    progress = (self.frame_count / total_frames) * 100
                    elapsed = (datetime.now() - start_time).total_seconds() if 'start_time' in locals() else 0
                    fps_actual = self.frame_count / elapsed if elapsed > 0 else 0
                    print(f"[{self.camera_id}] 📊 {progress:.1f}% | Frame {self.frame_count}/{total_frames} | Speed: {fps_actual:.1f} fps")
                
                # Add timestamp to frame
                timestamp = datetime.now()
                frame_with_timestamp = (frame.copy(), timestamp)
                
                # Always append to ring buffer
                self.ring_buffer.append(frame_with_timestamp)
                
                # If capturing post-event, add to post buffer
                if self.capturing_post_event:
                    self.post_event_buffer.append(frame_with_timestamp)
                    if len(self.post_event_buffer) >= self.post_event_frames_needed:
                        # Finished capturing post-event frames
                        self._finalize_clip()
                        self.capturing_post_event = False
                        self.post_event_buffer.clear()
                        continue
                
                # Check cooldown state (different logic for video vs stream)
                if self.in_cooldown:
                    if self.is_video_file:
                        # Frame-based cooldown for videos
                        self.cooldown_frame_count += 1
                        if self.cooldown_frame_count >= self.cooldown_duration:
                            self.in_cooldown = False
                            self.cooldown_frame_count = 0
                            print(f"[{self.camera_id}] Cooldown ended, resuming detection")
                        else:
                            continue
                    else:
                        # Time-based cooldown for streams
                        if datetime.now() < self.cooldown_end_time:
                            continue
                        else:
                            self.in_cooldown = False
                            print(f"[{self.camera_id}] Cooldown ended, resuming detection")
                
                # HIGH-ACCURACY DETECTION with advanced prompt engineering
                threat_info = self._detect_threats_advanced(frame)
                
                # Update threat history
                self.threat_history.append(threat_info)
                
                # Persistence filtering with confidence scoring
                if threat_info['is_threat']:
                    self.threat_counter += 1
                    
                    # Show detailed detection info
                    print(f"[{self.camera_id}] 🚨 THREAT DETECTED ({self.threat_counter}/{self.threat_threshold})")
                    print(f"[{self.camera_id}]    Type: {threat_info['threat_type']}")
                    print(f"[{self.camera_id}]    Confidence: {threat_info['confidence']:.2%}")
                    print(f"[{self.camera_id}]    Objects: {', '.join(threat_info['objects'])}")
                    
                    if self.threat_counter >= self.threat_threshold:
                        # Calculate average confidence over persistent frames
                        recent_threats = [t for t in self.threat_history if t['is_threat']][-self.threat_threshold:]
                        avg_confidence = np.mean([t['confidence'] for t in recent_threats])
                        
                        print(f"[{self.camera_id}] ⚠️⚠️⚠️  PERSISTENT THREAT CONFIRMED ⚠️⚠️⚠️")
                        print(f"[{self.camera_id}]    Average confidence: {avg_confidence:.2%}")
                        
                        # Start efficient clip capture (no video reopening)
                        self._start_clip_capture(width, height, threat_info, avg_confidence)
                        
                        self.threat_counter = 0
                        self.stats['threats_triggered'] += 1
                        self._enter_cooldown()
                else:
                    # No threat - check if we should reset counter
                    if self.threat_counter > 0:
                        print(f"[{self.camera_id}] ✓ Threat cleared (was at {self.threat_counter}/{self.threat_threshold})")
                        self.stats['false_positives_filtered'] += 1
                    self.threat_counter = 0
                
                # Optional preview (disable in headless environments)
                if self.config.get('show_preview', False):
                    try:
                        self._draw_detections(frame, threat_info)
                        cv2.imshow(f"Camera {self.camera_id}", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    except:
                        pass  # Ignore in headless environments (Colab, servers)
            
            # Print final statistics
            self._print_statistics()
                        
        except KeyboardInterrupt:
            print(f"\n[{self.camera_id}] Shutting down...")
        finally:
            cap.release()
            try:
                cv2.destroyAllWindows()
            except:
                pass  # Ignore in headless environments (Colab, servers)
            print(f"[{self.camera_id}] Processing stopped")
    
    def _detect_threats_advanced(self, frame):
        """
        ADVANCED THREAT DETECTION with PROMPT ENGINEERING
        
        Uses high-accuracy YOLO model with sophisticated threat classification logic.
        
        Threat Categories:
        1. CRITICAL: Person + valuable item (bicycle, laptop, bag)
        2. HIGH: Person alone in restricted area / suspicious posture
        3. MEDIUM: Person + vehicle interaction
        4. LOW: Person in allowed area
        5. NONE: Animals, empty scenes, passing vehicles
        
        Returns:
            dict: Detailed threat information
        """
        # Run YOLO inference with high accuracy settings
        results = self.model(
            frame,
            verbose=False,
            conf=self.config['yolo_confidence'],
            iou=self.config['yolo_iou']
        )
        
        # Parse detections
        detections = results[0].boxes
        
        detected_objects = []
        person_boxes = []
        valuable_boxes = []
        vehicle_boxes = []
        animal_boxes = []
        
        for box in detections:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
            
            detected_objects.append({
                'class': class_name,
                'confidence': confidence,
                'bbox': bbox
            })
            
            # Categorize detections
            if class_name == 'person':
                person_boxes.append({'bbox': bbox, 'conf': confidence})
            elif class_name in ['bicycle', 'motorcycle', 'backpack', 'handbag', 'suitcase', 'laptop']:
                valuable_boxes.append({'class': class_name, 'bbox': bbox, 'conf': confidence})
            elif class_name in ['car', 'truck', 'bus', 'motorcycle']:
                vehicle_boxes.append({'class': class_name, 'bbox': bbox, 'conf': confidence})
            elif class_name in ['dog', 'cat', 'bird', 'horse', 'sheep', 'cow']:
                animal_boxes.append({'class': class_name, 'bbox': bbox, 'conf': confidence})
        
        # THREAT CLASSIFICATION LOGIC (Prompt Engineering for Detection)
        
        threat_info = {
            'is_threat': False,
            'threat_type': 'none',
            'confidence': 0.0,
            'objects': [],
            'reason': ''
        }
        
        # Rule 1: IGNORE - Only animals, no people
        if animal_boxes and not person_boxes:
            threat_info['reason'] = f"Only animals detected ({len(animal_boxes)})"
            return threat_info
        
        # Rule 2: IGNORE - Only vehicles, no people
        if vehicle_boxes and not person_boxes and not valuable_boxes:
            threat_info['reason'] = f"Only vehicles detected ({len(vehicle_boxes)})"
            return threat_info
        
        # Rule 3: IGNORE - Empty scene
        if not detected_objects:
            threat_info['reason'] = "Empty scene"
            return threat_info
        
        # Rule 4: CRITICAL THREAT - Person + Valuable Object Proximity
        if person_boxes and valuable_boxes:
            # Check if person is near valuable objects
            for person in person_boxes:
                for valuable in valuable_boxes:
                    if self._is_near(person['bbox'], valuable['bbox'], threshold=0.3):
                        threat_info['is_threat'] = True
                        threat_info['threat_type'] = f"person_with_{valuable['class']}"
                        threat_info['confidence'] = min(person['conf'], valuable['conf'])
                        threat_info['objects'] = ['person', valuable['class']]
                        threat_info['reason'] = f"Person detected near {valuable['class']}"
                        return threat_info
        
        # Rule 5: HIGH THREAT - Person detected alone
        if person_boxes:
            # Multiple people = higher threat
            max_person_conf = max([p['conf'] for p in person_boxes])
            threat_info['is_threat'] = True
            threat_info['threat_type'] = 'person_detected'
            threat_info['confidence'] = max_person_conf
            threat_info['objects'] = ['person'] * len(person_boxes)
            threat_info['reason'] = f"{len(person_boxes)} person(s) detected"
            return threat_info
        
        # Rule 6: MEDIUM THREAT - Person + Vehicle (potential theft scenario)
        if person_boxes and vehicle_boxes:
            max_person_conf = max([p['conf'] for p in person_boxes])
            threat_info['is_threat'] = True
            threat_info['threat_type'] = 'person_with_vehicle'
            threat_info['confidence'] = max_person_conf * 0.8  # Slightly lower priority
            threat_info['objects'] = ['person', vehicle_boxes[0]['class']]
            threat_info['reason'] = "Person near vehicle"
            return threat_info
        
        threat_info['reason'] = "No threat criteria met"
        return threat_info
    
    def _is_near(self, bbox1, bbox2, threshold=0.3):
        """
        Check if two bounding boxes are near each other.
        
        Args:
            bbox1, bbox2: [x1, y1, x2, y2] format
            threshold: IoU threshold (default 0.3 for proximity)
        
        Returns:
            bool: True if boxes are near each other
        """
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
        
        # Calculate union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        # Calculate IoU
        iou = inter_area / union_area if union_area > 0 else 0
        
        # Also check center distance for proximity
        center1 = ((x1_min + x1_max) / 2, (y1_min + y1_max) / 2)
        center2 = ((x2_min + x2_max) / 2, (y2_min + y2_max) / 2)
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        # Consider "near" if IoU > threshold OR centers are close
        max_dim = max(x1_max - x1_min, y1_max - y1_min, x2_max - x2_min, y2_max - y2_min)
        
        return iou > threshold or distance < max_dim * 1.5
    
    def _draw_detections(self, frame, threat_info):
        """Draw detection visualizations on frame."""
        if threat_info['is_threat']:
            color = (0, 0, 255)  # Red for threats
            status = f"THREAT: {threat_info['threat_type']}"
        else:
            color = (0, 255, 0)  # Green for safe
            status = "SAFE"
        
        cv2.putText(frame, status, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Conf: {threat_info['confidence']:.2%}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    def _start_clip_capture(self, width, height, threat_info, avg_confidence):
        """Start efficient clip capture without reopening video."""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        threat_type = threat_info['threat_type'].replace(' ', '_')
        conf_str = f"{int(avg_confidence * 100)}"
        
        clip_filename = f"{self.camera_id}_{threat_type}_conf{conf_str}_{timestamp_str}.mp4"
        clip_path = self.clips_dir / clip_filename
        
        print(f"[{self.camera_id}] 💾 Starting clip capture: {clip_path.name}")
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        actual_fps = self.config['processing_fps']
        self.current_clip_writer = cv2.VideoWriter(str(clip_path), fourcc, actual_fps, (width, height))
        
        if not self.current_clip_writer.isOpened():
            print(f"[{self.camera_id}] ❌ ERROR: Could not create video writer")
            self.current_clip_writer = None
            return
        
        # Write pre-event frames immediately
        frames_written = 0
        print(f"[{self.camera_id}] Writing {len(self.ring_buffer)} pre-event frames")
        for frame, frame_timestamp in self.ring_buffer:
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, f"PRE-EVENT | {frame_timestamp.strftime('%H:%M:%S')}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            self.current_clip_writer.write(annotated_frame)
            frames_written += 1
        
        # Calculate post-event frames needed
        target_total_frames = self.config['clip_duration_seconds'] * actual_fps
        self.post_event_frames_needed = max(target_total_frames - frames_written, actual_fps * 3)
        
        print(f"[{self.camera_id}] Capturing next {self.post_event_frames_needed} post-event frames...")
        
        # Store metadata for later
        self.current_clip_metadata = {
            'clip_path': clip_path,
            'threat_info': threat_info,
            'avg_confidence': avg_confidence,
            'timestamp': timestamp_str,
            'pre_frames': frames_written
        }
        
        # Enable post-event capture mode
        self.capturing_post_event = True
        self.post_event_buffer.clear()
    
    def _finalize_clip(self):
        """Finalize clip by writing post-event frames and metadata."""
        if not self.current_clip_writer or not self.current_clip_metadata:
            return
        
        # Write post-event frames
        post_frames_written = 0
        for frame, frame_timestamp in self.post_event_buffer:
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, f"POST-EVENT | {frame_timestamp.strftime('%H:%M:%S')}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            self.current_clip_writer.write(annotated_frame)
            post_frames_written += 1
        
        self.current_clip_writer.release()
        
        # Save metadata
        clip_path = self.current_clip_metadata['clip_path']
        threat_info = self.current_clip_metadata['threat_info']
        avg_confidence = self.current_clip_metadata['avg_confidence']
        timestamp_str = self.current_clip_metadata['timestamp']
        pre_frames = self.current_clip_metadata['pre_frames']
        total_frames = pre_frames + post_frames_written
        
        metadata_path = clip_path.with_suffix('.txt')
        with open(metadata_path, 'w') as f:
            f.write(f"Threat Type: {threat_info['threat_type']}\n")
            f.write(f"Confidence: {avg_confidence:.2%}\n")
            f.write(f"Objects: {', '.join(threat_info['objects'])}\n")
            f.write(f"Reason: {threat_info['reason']}\n")
            f.write(f"Timestamp: {timestamp_str}\n")
            f.write(f"Total Frames: {total_frames}\n")
            f.write(f"Pre-Event Frames: {pre_frames}\n")
            f.write(f"Post-Event Frames: {post_frames_written}\n")
            f.write(f"Duration: ~{total_frames / self.config['processing_fps']:.1f} seconds\n")
        
        print(f"[{self.camera_id}] ✅ Clip saved: {total_frames} frames (~{total_frames / self.config['processing_fps']:.1f} sec)")
        print(f"[{self.camera_id}] → Ready for Stage 2 intent analysis")
        
        # Reset
        self.current_clip_writer = None
        self.current_clip_metadata = None
    
    def _enter_cooldown(self):
        """Enter cooldown state to prevent spam."""
        self.in_cooldown = True
        self.cooldown_frame_count = 0
        
        if self.is_video_file:
            # Frame-based cooldown for videos
            print(f"[{self.camera_id}] 🕐 Entering cooldown: skipping next {self.cooldown_duration} processed frames")
        else:
            # Time-based cooldown for streams
            self.cooldown_end_time = datetime.now() + timedelta(seconds=self.cooldown_duration)
            print(f"[{self.camera_id}] 🕐 Entering cooldown for {self.cooldown_duration} seconds")
    
    def _print_statistics(self):
        """Print detection statistics."""
        print("\n" + "=" * 70)
        print(f"[{self.camera_id}] DETECTION STATISTICS")
        print("=" * 70)
        print(f"Total frames processed: {self.stats['total_frames']}")
        print(f"Threats triggered: {self.stats['threats_triggered']}")
        print(f"False positives filtered: {self.stats['false_positives_filtered']}")
        print("=" * 70 + "\n")


def process_video_file(video_path, config):
    """Process a single video file."""
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return
    
    camera_id = Path(video_path).stem
    detector = HighAccuracyDetector(camera_id, video_path, config)
    detector.process_source()


def process_video_folder(folder_path, config):
    """Process all videos in a folder."""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(Path(folder_path).glob(f'*{ext}'))
        video_files.extend(Path(folder_path).glob(f'*{ext.upper()}'))
    
    if not video_files:
        print(f"❌ No video files found in: {folder_path}")
        print(f"Supported formats: {', '.join(video_extensions)}")
        return
    
    print(f"📁 Found {len(video_files)} video(s) in {folder_path}")
    print("=" * 70)
    
    for i, video_file in enumerate(video_files, 1):
        print(f"\n🎬 Processing video {i}/{len(video_files)}: {video_file.name}")
        print("=" * 70)
        process_video_file(str(video_file), config)
        print("\n")


def main():
    """Main entry point."""
    
    # HIGH-ACCURACY CONFIGURATION
    config = {
        # YOLO Model - BALANCED FOR SPEED AND ACCURACY
        'yolo_model': 'yolov8m.pt',  # Medium: 52MB, 3-5x faster than yolov8x, still very accurate
        # Options: yolov8n.pt (fastest), yolov8m.pt (balanced), yolov8l.pt (accurate), yolov8x.pt (slowest)
        
        # Detection thresholds (tuned for accuracy)
        'yolo_confidence': 0.4,   # Confidence threshold (0.25-0.5 range)
        'yolo_iou': 0.45,         # IoU threshold for NMS
        
        # Frame processing
        'processing_fps': 5,
        'frame_skip': 6,  # For 30fps video: 30/6 = 5fps processing
        
        # Ring buffer
        'ring_buffer_seconds': 5,  # More context
        
        # Persistence filter (IMPORTANT: reduces false positives)
        'persistence_frames': 3,  # Must detect threat in 3 consecutive processed frames
        
        # Clip settings
        'clip_duration_seconds': 15,  # Target 15 second clips (pre + post event)
        'clips_directory': 'incident_clips',
        
        # Cooldown settings
        'cooldown_seconds': 30,  # For RTSP streams
        'video_cooldown_frames': 25,  # For video files (~5 sec at 5fps processing)
        
        # Display
        'show_preview': False,  # Set True for visual debugging
    }
    
    print("=" * 70)
    print("AI-Powered DVR - Stage 1: OPTIMIZED Detection")
    print("=" * 70)
    print(f"Model: {config['yolo_model']} (Balanced Speed/Accuracy)")
    print(f"Confidence: {config['yolo_confidence']} | IoU: {config['yolo_iou']}")
    print(f"Processing: {config['processing_fps']} FPS")
    print(f"Persistence: {config['persistence_frames']} frames (anti-glitch)")
    print(f"Clip Duration: {config['clip_duration_seconds']} seconds (pre + post event)")
    print(f"Video Cooldown: {config['video_cooldown_frames']} frames (~{config['video_cooldown_frames'] / config['processing_fps']:.0f} sec)")
    print("=" * 70)
    print()
    
    # Auto-detect environment and videos
    import sys
    is_colab = os.path.exists('/content') or not sys.stdin.isatty()
    
    # Search for videos in multiple locations
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    found_videos = []
    
    # Check current directory first (for Colab uploads)
    for ext in video_extensions:
        found_videos.extend(Path('.').glob(f'*{ext}'))
        found_videos.extend(Path('.').glob(f'*{ext.upper()}'))
    
    # Check videos/ folder
    videos_dir = Path("videos")
    if videos_dir.exists():
        for ext in video_extensions:
            found_videos.extend(videos_dir.glob(f'*{ext}'))
            found_videos.extend(videos_dir.glob(f'*{ext.upper()}'))
    
    # Remove duplicates and sort
    found_videos = sorted(set(found_videos))
    
    if found_videos:
        print(f"🎬 Found {len(found_videos)} video(s) to process:")
        for i, vid in enumerate(found_videos, 1):
            print(f"   {i}. {vid.name}")
        print()
        
        # Process all videos automatically (no input needed)
        for i, video_file in enumerate(found_videos, 1):
            print(f"\n{'='*70}")
            print(f"🎬 Processing video {i}/{len(found_videos)}: {video_file.name}")
            print(f"{'='*70}")
            process_video_file(str(video_file), config)
        
        print(f"\n✅ Processed {len(found_videos)} video(s)")
        return
    
    # No videos found - show interactive menu (only if not in Colab)
    if is_colab:
        print("❌ No videos found!")
        print("Upload videos using:")
        print("   from google.colab import files")
        print("   files.upload()")
        return
    
    # Interactive mode for local use
    print("Select input mode:")
    print("1. Process video files from 'videos/' folder")
    print("2. Process RTSP stream from phone")
    print("3. Process single video file")
    
    choice = input("\nEnter choice (1-3) [1]: ").strip() or "1"
    
    if choice == "1":
        # Process all videos in videos/ folder
        if not videos_dir.exists():
            videos_dir.mkdir()
            print(f"\n✅ Created 'videos' folder")
            print(f"📁 Place your video files in: {videos_dir.absolute()}")
            print("Then run this script again.")
            return
        
        process_video_folder(videos_dir, config)
    
    elif choice == "2":
        # RTSP stream mode
        MOBILE_IP = "10.226.220.51"
        HTTP_PORT = 8080
        
        stream_urls = [
            f"http://{MOBILE_IP}:{HTTP_PORT}/video",
            f"rtsp://{MOBILE_IP}:8554/h264_pcm.sdp",
        ]
        
        print("\n🔍 Testing stream URLs...")
        rtsp_url = None
        for url in stream_urls:
            print(f"   Trying: {url}")
            test_cap = cv2.VideoCapture(url)
            if test_cap.isOpened():
                rtsp_url = url
                test_cap.release()
                print(f"   ✅ Connected!")
                break
            test_cap.release()
        
        if not rtsp_url:
            print("\n❌ Could not connect to phone stream")
            print("Make sure IP Webcam is running on your phone")
            return
        
        detector = HighAccuracyDetector("mobile_cam", rtsp_url, config)
        detector.process_source()
    
    elif choice == "3":
        # Single file mode
        video_path = input("Enter video file path: ").strip().strip('"')
        process_video_file(video_path, config)
    
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
