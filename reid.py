import cv2
import torch
import numpy as np
import os
import time
import json
from ultralytics import YOLO
import supervision as sv
from torchreid.utils import FeatureExtractor
from collections import defaultdict

# GPU acceleration support - try multiple backends
try:
    import torch_directml
    DIRECTML_AVAILABLE = True
except ImportError:
    DIRECTML_AVAILABLE = False

try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except ImportError:
    IPEX_AVAILABLE = False

def get_device():
    """Prioritize GPU backends: IPEX (XPU) > CUDA > CPU.
    
    Note: DirectML is skipped due to PyTorch inference mode incompatibilities.
    DirectML support in YOLOv8 is still experimental.
    """
    # Try Intel IPEX XPU first
    if IPEX_AVAILABLE and torch.xpu.is_available():
        print("[Device] Using Intel Arc GPU (XPU)")
        return 'xpu'
    
    # Try NVIDIA CUDA
    if torch.cuda.is_available():
        print("[Device] Using NVIDIA GPU (CUDA)")
        return 'cuda'
    
    # Fallback to CPU
    print("[Device] Using CPU (DirectML skipped - experimental support)")
    return 'cpu'

# ------------------------------------------------------------------ #
#  RE-ID ENGINE                                                      #
# ------------------------------------------------------------------ #
class ReIDEngine:
    def __init__(self, model_weights):
        self.device = get_device()
        print(f"[ReID] Initializing OSNet on {self.device}...")

        model_basename = os.path.basename(model_weights).lower() if model_weights else ''
        if 'x0_75' in model_basename:
            model_name = 'osnet_x0_75'
        elif 'ain' in model_basename:
            model_name = 'osnet_ain_x1_0'
        elif 'x1' in model_basename or 'x1_0' in model_basename:
            model_name = 'osnet_x1_0'
        else:
            model_name = 'osnet_ain_x1_0'

        self.extractor = FeatureExtractor(
            model_name=model_name,
            model_path=model_weights,
            device=self.device
        )
        self.detector = YOLO('yolov8n.pt')
        if self.device in ['cuda', 'xpu']:
            self.detector.to(self.device)
        
        self.p1_tracker = sv.ByteTrack()
        self.target_gallery = []
        self.max_gallery_size = 15
        self.stop_search = False  # Flag to stop search_video immediately

    def full_reset(self):
        """Clears all internal states for a fresh start."""
        self.target_gallery = []
        self.p1_tracker = sv.ByteTrack()  # Reset tracker state
        self.stop_search = False
        print("[ReID] Engine fully reset.")

    def get_features(self, crop):
        if crop is None or crop.size == 0:
            return None
        try:
            feat = self.extractor([crop])[0].cpu().numpy()
            norm = np.linalg.norm(feat)
            return feat / norm if norm != 0 else None
        except Exception:
            return None

    def process_frame(self, frame, selected_tid=None):
        results = self.detector(frame, classes=[0], verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = self.p1_tracker.update_with_detections(detections)
        annotated = frame.copy()
        current_data = []
        for xyxy, tid in zip(detections.xyxy, detections.tracker_id):
            x1, y1, x2, y2 = map(int, xyxy)
            current_data.append((xyxy, tid))
            color = (0, 255, 255) if tid == selected_tid else (0, 0, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        return annotated, current_data

    def search_video(self, video_path, gallery, progress_callback, match_callback, stop_check, save_dir="matches", frame_skip=5):
        """
        Search for matches in video.
        stop_check:  A callable that returns True if search should stop.
        frame_skip:  Process every Nth frame only (default=5). Progress tracking
                     still counts all frames so the progress bar stays accurate.
        """
        os.makedirs(save_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            progress_callback(100)
            return

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        raw_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = raw_total if raw_total > 50 else 0
        if total_frames == 0:
            probe = cv2.VideoCapture(video_path)
            while probe.grab(): total_frames += 1
            probe.release()
        
        p2_tracker = sv.ByteTrack()
        count, match_count = 0, 0
        matched_tids = set()

        while cap.isOpened():
            # CRITICAL: Check if main app wants to stop (called every frame)
            if stop_check():
                print("[ReID] Background search interrupted by user.")
                break

            ret, frame = cap.read()
            if not ret: break
            count += 1

            # Progress update (based on all frames, not just processed ones)
            if count % 10 == 0:
                progress_callback(min(int((count / total_frames) * 99), 99))

            # Skip non-sampled frames — saves ~80% of YOLO + ReID compute
            if count % frame_skip != 0:
                continue

            results = self.detector(frame, classes=[0], verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = p2_tracker.update_with_detections(detections)

            if detections.tracker_id is not None:
                h, w = frame.shape[:2]
                for xyxy, tid in zip(detections.xyxy, detections.tracker_id):
                    x1, y1, x2, y2 = max(0, int(xyxy[0])), max(0, int(xyxy[1])), min(w, int(xyxy[2])), min(h, int(xyxy[3]))
                    if x2 <= x1 or y2 <= y1: continue
                    crop = frame[y1:y2, x1:x2]
                    feat = self.get_features(crop)
                    if feat is None: continue
                    score = float(np.max(np.dot(gallery, feat)))
                    if score > 0.75 and tid not in matched_tids:
                        matched_tids.add(tid)
                        msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                        mins, secs = int(msec // 60000), int((msec // 1000) % 60)
                        match_count += 1
                        filepath = os.path.join(save_dir, f"{video_name}__ts_{mins:02d}m{secs:02d}s__match{match_count:03d}.jpg")
                        cv2.imwrite(filepath, crop)
                        match_callback(filepath, video_name, f"{mins}:{secs:02d}")
        cap.release()
        time.sleep(0.4)
        # Only call progress callback if not interrupted
        if not stop_check():
            progress_callback(100)
    
    def reset_search(self):
        """Reset the search flag for a fresh start."""
        self.stop_search = False


# ------------------------------------------------------------------ #
#  ANALYTICS ENGINE                                                  #
# ------------------------------------------------------------------ #
class AnalyticsEngine:
    def __init__(self, model_name='yolov8n.pt'):
        self.device = get_device()
        print(f"[Analytics] Initializing YOLOv8 on {self.device}...")
        self.model = YOLO(model_name)
        if self.device in ['cuda', 'xpu']:
            self.model.to(self.device)
            
        # Settings
        self.heatmap = None
        self.heatmap_alpha = 0.45  # Transparency of the overlay
        self.max_count = 0
        self.scale_factor = 0.5    # Resolution of math (0.5 = 1/4 the area for speed)
        
        # Trajectory state
        self.track_history = defaultdict(lambda: [])

    def reset(self, frame_shape):
        """Clears all data for a fresh recording session."""
        h, w = frame_shape[:2]
        self.heatmap = np.zeros((int(h * self.scale_factor), 
                                 int(w * self.scale_factor)), dtype=np.float32)
        self.max_count = 0
        self.track_history = defaultdict(lambda: [])

    def process_analytics_frame(self, frame, show_heatmap=True):
        """Main processing loop for counting, trails, and density."""
        results = self.model.track(frame, persist=True, classes=[0], conf=0.3, verbose=False)[0]
        annotated_frame = frame.copy()
        current_count = 0

        # Draw a subtle "Cyber Grid" for professional look
        self._draw_grid(annotated_frame)

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            current_count = len(track_ids)
            self.max_count = max(self.max_count, current_count)

            for box, tid in zip(boxes, track_ids):
                x1, y1, x2, y2 = box.astype(int)
                # Detection point: Feet (bottom center)
                cx, cy = (x1 + x2) // 2, y2
                
                # 1. Update Heatmap with High Intensity
                self._update_heatmap_dense(cx, cy)
                
                # 2. Update and Draw Trajectory Trails
                self.track_history[tid].append((cx, cy))
                if len(self.track_history[tid]) > 40: # Buffer length
                    self.track_history[tid].pop(0)
                
                if len(self.track_history[tid]) > 1:
                    points = np.array(self.track_history[tid], dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 255, 255), thickness=2)

                # 3. Draw Bounding Box and ID
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 229, 255), 1)
                cv2.putText(annotated_frame, f"TARGET {tid}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 229, 255), 2)

        # 4. Apply the Dense Heatmap Overlay
        if show_heatmap and self.heatmap is not None:
            annotated_frame = self._apply_heatmap_overlay(annotated_frame)
            
        return annotated_frame, current_count

    def _update_heatmap_dense(self, x, y):
        """Adds aggressive heat values to a local area."""
        sx, sy = int(x * self.scale_factor), int(y * self.scale_factor)
        h, w = self.heatmap.shape
        
        # heat_intensity: Increase this (e.g., to 20.0) for even thicker red areas
        heat_intensity = 12.0 
        
        # Apply a 5x5 brush to ensure trails are thick
        for i in range(-2, 3):
            for j in range(-2, 3):
                if 0 <= sx+i < w and 0 <= sy+j < h:
                    dist = np.sqrt(i**2 + j**2)
                    self.heatmap[sy+j, sx+i] += heat_intensity / (dist + 1)

    def _apply_heatmap_overlay(self, frame):
        """Processes raw values into a smooth, vibrant heatmap overlay."""
        # A. Blur the raw accumulation
        blurred = cv2.GaussianBlur(self.heatmap, (35, 35), 0)
        
        # B. Logarithmic Scaling: Makes walking trails almost as bright as loitering spots
        log_heatmap = np.log1p(blurred) 
        
        # C. Normalize and threshold to remove background "blue" noise
        norm = cv2.normalize(log_heatmap, None, 0, 255, cv2.NORM_MINMAX)
        norm = np.uint8(norm)
        _, thresh = cv2.threshold(norm, 30, 255, cv2.THRESH_BINARY)
        
        # D. Smooth the thresholded mask
        final_mask = cv2.GaussianBlur(thresh, (25, 25), 0)
        
        # E. Upscale and Colorize
        h, w = frame.shape[:2]
        heatmap_resized = cv2.resize(final_mask, (w, h))
        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        
        # Only overlay where there is actual 'heat' (prevents blue tint on everything)
        # We use a weighted blend only on the heat areas
        return cv2.addWeighted(frame, 1.0, heatmap_color, self.heatmap_alpha, 0)

    def _draw_grid(self, frame):
        """Tech-style grid overlay."""
        h, w = frame.shape[:2]
        grid_size = 80
        for i in range(0, w, grid_size):
            cv2.line(frame, (i, 0), (i, h), (50, 50, 50), 1)
        for j in range(0, h, grid_size):
            cv2.line(frame, (0, j), (w, j), (50, 50, 50), 1)

    def get_final_heatmap(self, background_frame):
        """Generates the desaturated high-contrast final report."""
        if self.heatmap is None: return background_frame
        
        blurred = cv2.GaussianBlur(self.heatmap, (51, 51), 0)
        log_hm = np.log1p(blurred)
        norm = cv2.normalize(log_hm, None, 0, 255, cv2.NORM_MINMAX)
        
        h, w = background_frame.shape[:2]
        heatmap_resized = cv2.resize(np.uint8(norm), (w, h))
        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_HOT)
        
        # Desaturate background
        gray_bg = cv2.cvtColor(cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        report = cv2.addWeighted(gray_bg, 0.4, heatmap_color, 0.6, 0)
        
        # Branding/Metadata on report
        cv2.putText(report, "SSF VISION: CROWD DENSITY REPORT", (30, 50), 
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        cv2.putText(report, f"PEAK OCCUPANCY: {self.max_count} PERSONS", (30, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
        
        return report
        

# ------------------------------------------------------------------ #
#  ZONE ENGINE (New)                                                 #
# ------------------------------------------------------------------ #
class ZoneEngine:
    def __init__(self, model_name='yolov8n.pt'):
        self.device = get_device()
        print(f"[Zone] Initializing YOLOv8 on {self.device}...")
        self.model = YOLO(model_name)
        # Move model to GPU device if available
        if self.device in ['cuda', 'xpu']:
            self.model.to(self.device)
        self.zones = []  # List of polygons (np.array)
        self.alert_cooldown = {} # track_id: last_alert_time
        self.COOLDOWN_TIME = 5.0 # seconds between alerts for same ID

    def add_zone(self, points_list):
        """Add a new polygon zone from a list of (x,y) tuples."""
        if len(points_list) >= 3:
            self.zones.append(np.array(points_list, dtype=np.int32))

    def clear_zones(self):
        self.zones = []
        self.alert_cooldown = {}

        # Add this method to your ZoneEngine class in reid.py
    def draw_preview(self, frame, current_points):
        """Draws the current zones and the points being clicked in real-time."""
        annotated = frame.copy()
        # Draw existing saved zones
        for poly in self.zones:
            cv2.polylines(annotated, [poly], True, (0, 255, 0), 2)
        
        # Draw the points currently being added
        for p in current_points:
            cv2.circle(annotated, p, 5, (0, 255, 255), -1)
        if len(current_points) > 1:
            pts = np.array(current_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], False, (0, 255, 255), 2)
        return annotated

    def process_frame(self, frame, timestamp_str, save_dir="zone_alerts"):
        """Process frame for intrusion detection with metadata logging."""
        results = self.model.track(frame, persist=True, classes=[0], conf=0.3, verbose=False)[0]
        annotated = frame.copy()
        alerts = []
        h, w = frame.shape[:2]

        # 1. Draw Zones (Greens for normal, will use Red text for alerts)
        for i, poly in enumerate(self.zones):
            cv2.polylines(annotated, [poly], True, (0, 255, 0), 2)
            cv2.putText(annotated, f"RESTRICTED ZONE {i}", (poly[0][0], poly[0][1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            ids = results.boxes.id.cpu().numpy().astype(int)
            confidences = results.boxes.conf.cpu().numpy() if hasattr(results.boxes, 'conf') else [0.0] * len(ids)

            for box, tid, conf in zip(boxes, ids, confidences):
                x1, y1, x2, y2 = box.astype(int)
                # Detection point: Feet (bottom center)
                feet_x, feet_y = (x1 + x2) // 2, y2
                
                # Check point against all defined zones
                intruded_zone = -1
                for i, poly in enumerate(self.zones):
                    if cv2.pointPolygonTest(poly, (float(feet_x), float(feet_y)), False) >= 0:
                        intruded_zone = i
                        break
                
                if intruded_zone != -1:
                    color = (0, 0, 255) # Red for intrusion
                    now = time.time()
                    # Trigger alert if not in cooldown
                    if tid not in self.alert_cooldown or (now - self.alert_cooldown[tid]) > self.COOLDOWN_TIME:
                        self.alert_cooldown[tid] = now
                        
                        # Create organized folder structure: save_dir/zone_{zone_id}/
                        zone_folder = os.path.join(save_dir, f"zone_{intruded_zone}")
                        os.makedirs(zone_folder, exist_ok=True)
                        
                        # Save Evidence Crop
                        crop = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
                        image_filename = f"person_{tid}_{int(now*1000)}.jpg"
                        image_filepath = os.path.join(zone_folder, image_filename)
                        cv2.imwrite(image_filepath, crop)
                        
                        # Save Metadata JSON
                        metadata = {
                            "zone_id": intruded_zone,
                            "person_id": int(tid),
                            "timestamp": timestamp_str,
                            "unix_timestamp": now,
                            "detection_confidence": float(conf),
                            "bounding_box": {
                                "x1": int(x1),
                                "y1": int(y1),
                                "x2": int(x2),
                                "y2": int(y2),
                                "width": int(x2 - x1),
                                "height": int(y2 - y1)
                            },
                            "frame_dimensions": {
                                "width": w,
                                "height": h
                            },
                            "image_file": image_filename,
                            "alert_message": f"ZONE {intruded_zone} INTRUSION | ID:{tid} | {timestamp_str}"
                        }
                        
                        metadata_filename = f"person_{tid}_{int(now*1000)}.json"
                        metadata_filepath = os.path.join(zone_folder, metadata_filename)
                        with open(metadata_filepath, 'w') as f:
                            json.dump(metadata, f, indent=2)
                        
                        # Record alert
                        alerts.append(metadata["alert_message"])
                else:
                    color = (255, 0, 0) # Blue for normal tracked person

                # Draw Visuals
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.circle(annotated, (feet_x, feet_y), 5, color, -1)

        return annotated, alerts