import cv2
from ultralytics import YOLO
import numpy as np
from collections import Counter, defaultdict, deque
import time
import json
import os
from pathlib import Path
import threading
import queue
import datetime
import math
import pygame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

class EnhancedObjectDetector:
    def __init__(self):
        """Initialize enhanced detector with advanced features"""
        print("🚀 Loading Enhanced Detection System...")
        
        # Load AI model
        self.model = YOLO('yolov8n.pt')
        
        # Enhanced tracking system
        self.object_tracks = {}
        self.next_track_id = 1
        self.track_history = defaultdict(lambda: deque(maxlen=30))  # Track trajectories
        self.object_persistence = defaultdict(int)  # How long objects have been detected
        
        # Visual enhancement settings
        self.ui_theme = {
            'primary': (0, 150, 255),      # Modern blue
            'secondary': (255, 100, 0),     # Orange accent
            'success': (0, 255, 100),       # Green
            'warning': (255, 255, 0),       # Yellow
            'danger': (255, 50, 50),        # Red
            'background': (20, 20, 30),     # Dark background
            'surface': (40, 40, 60),        # Card surface
            'text': (255, 255, 255),        # White text
            'text_secondary': (180, 180, 180) # Gray text
        }
        
        # Animation and effects
        self.pulse_animation = 0
        self.glow_effect = 0
        self.notification_queue = queue.Queue()
        self.alerts_enabled = True
        
        # Performance monitoring
        self.fps_counter = deque(maxlen=30)
        self.detection_stats = {
            'total_detections': 0,
            'humans_detected': 0,
            'animals_detected': 0,
            'objects_detected': 0,
            'session_start': time.time()
        }
        
        # Heat map for detection zones
        self.heat_map = np.zeros((480, 640), dtype=np.float32)
        self.heat_map_alpha = 0.7
        
        # Recording capabilities
        self.recording = False
        self.video_writer = None
        self.screenshots_dir = Path("screenshots")
        self.recordings_dir = Path("recordings")
        self.screenshots_dir.mkdir(exist_ok=True)
        self.recordings_dir.mkdir(exist_ok=True)
        
        # Sound system
        try:
            pygame.mixer.init()
            self.sound_enabled = True
            print("🔊 Audio system initialized")
        except:
            self.sound_enabled = False
            print("⚠️  Audio system not available")
        
        # Classification system
        self.setup_enhanced_classification()
        
        print("✅ Enhanced detection system ready!")
    
    def setup_enhanced_classification(self):
        """Setup enhanced classification with confidence-based categories"""
        self.category_confidence_thresholds = {
            'person': 0.7,
            'car': 0.6,
            'dog': 0.65,
            'cat': 0.65,
            'bicycle': 0.5,
            'motorcycle': 0.6,
            'bus': 0.7,
            'truck': 0.7
        }
        
        self.category_colors = {
            'person': (0, 255, 100),        # Bright green
            'vehicle': (255, 100, 0),       # Orange
            'animal': (255, 200, 0),        # Yellow-orange
            'object': (100, 150, 255),      # Light blue
            'unknown': (150, 150, 150)      # Gray
        }
        
        self.category_icons = {
            'person': '👤',
            'vehicle': '🚗',
            'animal': '🐾',
            'object': '📦',
            'unknown': '❓'
        }
    
    def create_gradient_background(self, width, height):
        """Create a modern gradient background"""
        gradient = np.zeros((height, width, 3), dtype=np.uint8)
        
        for y in range(height):
            # Create vertical gradient from dark blue to darker
            factor = y / height
            r = int(self.ui_theme['background'][0] * (1 - factor * 0.3))
            g = int(self.ui_theme['background'][1] * (1 - factor * 0.3))
            b = int(self.ui_theme['background'][2] + 20 * factor)
            gradient[y, :] = [b, g, r]  # BGR for OpenCV
        
        return gradient
    
    def draw_glassmorphism_panel(self, frame, x, y, width, height, alpha=0.3):
        """Draw a modern glassmorphism-style panel"""
        # Create panel background with blur effect
        panel = frame[y:y+height, x:x+width].copy()
        
        # Apply blur
        panel = cv2.GaussianBlur(panel, (15, 15), 0)
        
        # Create overlay with gradient
        overlay = np.ones_like(panel, dtype=np.uint8)
        overlay[:] = self.ui_theme['surface']
        
        # Blend with original
        blended = cv2.addWeighted(panel, 1-alpha, overlay, alpha, 0)
        
        # Add subtle border
        cv2.rectangle(blended, (2, 2), (width-2, height-2), self.ui_theme['primary'], 1)
        
        frame[y:y+height, x:x+width] = blended
        
        return frame
    
    def draw_enhanced_detection_box(self, frame, detection, track_id=None):
        """Draw enhanced detection box with modern styling"""
        box = detection['box']
        x1, y1, x2, y2 = map(int, box)
        class_name = detection['class']
        confidence = detection['confidence']
        
        # Determine category and color
        category = self.get_object_category(class_name)
        base_color = self.category_colors.get(category, self.category_colors['unknown'])
        
        # Add pulsing effect for high-confidence detections
        if confidence > 0.8:
            pulse = abs(math.sin(self.pulse_animation * 0.1)) * 0.3 + 0.7
            color = tuple(int(c * pulse) for c in base_color)
        else:
            color = base_color
        
        # Dynamic thickness based on confidence
        thickness = int(2 + confidence * 3)
        
        # Draw main bounding box with rounded corners effect
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw corner accents
        corner_length = 20
        corner_thickness = 3
        
        # Top-left corner
        cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, corner_thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, corner_thickness)
        
        # Top-right corner
        cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, corner_thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, corner_thickness)
        
        # Bottom-left corner
        cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, corner_thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, corner_thickness)
        
        # Bottom-right corner
        cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, corner_thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, corner_thickness)
        
        # Enhanced label with icon
        icon = self.category_icons.get(category, '❓')
        label = f"{icon} {class_name}"
        confidence_text = f"{confidence:.1%}"
        
        # Calculate label dimensions
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        
        (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        (conf_w, conf_h), _ = cv2.getTextSize(confidence_text, font, 0.5, 1)
        
        # Draw modern label background
        label_bg_height = max(label_h, conf_h) + 16
        label_bg_width = max(label_w, conf_w) + 20
        
        # Glassmorphism label background
        if y1 - label_bg_height > 0:
            label_y = y1 - label_bg_height
        else:
            label_y = y2
        
        # Create rounded rectangle background
        self.draw_rounded_rectangle(frame, (x1, label_y), (x1 + label_bg_width, label_y + label_bg_height), 
                                  color, -1, radius=8)
        
        # Add semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, label_y), (x1 + label_bg_width, label_y + label_bg_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Draw text with shadow effect
        text_y = label_y + label_h + 8
        
        # Shadow
        cv2.putText(frame, label, (x1 + 11, text_y + 1), font, font_scale, (0, 0, 0), font_thickness + 1)
        # Main text
        cv2.putText(frame, label, (x1 + 10, text_y), font, font_scale, (255, 255, 255), font_thickness)
        
        # Confidence text
        cv2.putText(frame, confidence_text, (x1 + 10, text_y + 20), font, 0.5, (200, 200, 200), 1)
        
        # Track ID if available
        if track_id:
            track_text = f"ID: {track_id}"
            cv2.putText(frame, track_text, (x2 - 60, y1 - 5), font, 0.4, color, 1)
        
        # Add glow effect for important objects
        if category in ['person', 'vehicle'] and confidence > 0.8:
            glow_intensity = abs(math.sin(self.glow_effect * 0.05)) * 100 + 50
            glow_color = tuple(int(c * 0.3) for c in color)
            cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), glow_color, 6)
    
    def draw_rounded_rectangle(self, img, pt1, pt2, color, thickness, radius=10):
        """Draw a rounded rectangle"""
        x1, y1 = pt1
        x2, y2 = pt2
        
        if thickness == -1:  # Filled rectangle
            # Draw main rectangle
            cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
            cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
            
            # Draw corners
            cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
            cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
            cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
            cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
        else:
            # Draw outline only
            cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
            cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
            cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
            cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
            
            cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
            cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
            cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
            cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
    
    def get_object_category(self, class_name):
        """Categorize objects into main groups"""
        vehicles = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'boat', 'airplane']
        animals = ['dog', 'cat', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']
        
        if class_name == 'person':
            return 'person'
        elif class_name in vehicles:
            return 'vehicle'
        elif class_name in animals:
            return 'animal'
        else:
            return 'object'
    
    def update_heat_map(self, detections):
        """Update heat map based on detection locations"""
        # Decay existing heat map
        self.heat_map *= 0.95
        
        # Add heat for current detections
        for detection in detections:
            box = detection['box']
            x1, y1, x2, y2 = map(int, box)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Add heat around detection center
            cv2.circle(self.heat_map, (center_x, center_y), 30, 1.0, -1)
    
    def draw_heat_map_overlay(self, frame):
        """Draw heat map overlay on frame"""
        # Normalize heat map
        normalized_heat = cv2.normalize(self.heat_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Apply color map
        heat_colored = cv2.applyColorMap(normalized_heat, cv2.COLORMAP_JET)
        
        # Blend with frame
        mask = normalized_heat > 10  # Only show significant heat
        heat_colored[~mask] = 0
        
        cv2.addWeighted(frame, 1 - self.heat_map_alpha, heat_colored, self.heat_map_alpha, 0, frame)
    
    def create_modern_dashboard(self, frame, detections, fps):
        """Create a modern dashboard overlay"""
        height, width = frame.shape[:2]
        
        # Main dashboard panel
        panel_width = 350
        panel_height = 200
        panel_x = width - panel_width - 20
        panel_y = 20
        
        # Draw glassmorphism panel
        self.draw_glassmorphism_panel(frame, panel_x, panel_y, panel_width, panel_height)
        
        # Dashboard title
        title_y = panel_y + 30
        cv2.putText(frame, "🎯 DETECTION DASHBOARD", (panel_x + 20, title_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.ui_theme['primary'], 2)
        
        # Statistics
        stats_y = title_y + 40
        line_height = 25
        
        # Count detections by category
        categories = {'person': 0, 'vehicle': 0, 'animal': 0, 'object': 0}
        for detection in detections:
            category = self.get_object_category(detection['class'])
            if category in categories:
                categories[category] += 1
        
        # Display counts with icons
        stats = [
            (f"👤 Humans: {categories['person']}", self.ui_theme['success']),
            (f"🚗 Vehicles: {categories['vehicle']}", self.ui_theme['warning']),
            (f"🐾 Animals: {categories['animal']}", self.ui_theme['secondary']),
            (f"📦 Objects: {categories['object']}", self.ui_theme['primary'])
        ]
        
        for i, (stat_text, color) in enumerate(stats):
            y_pos = stats_y + i * line_height
            cv2.putText(frame, stat_text, (panel_x + 20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # FPS and performance
        fps_y = stats_y + len(stats) * line_height + 10
        fps_color = self.ui_theme['success'] if fps > 20 else self.ui_theme['warning']
        cv2.putText(frame, f"⚡ FPS: {fps:.1f}", (panel_x + 20, fps_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 1)
        
        # Session time
        session_time = time.time() - self.detection_stats['session_start']
        time_str = f"⏱️ Time: {int(session_time//60):02d}:{int(session_time%60):02d}"
        cv2.putText(frame, time_str, (panel_x + 20, fps_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.ui_theme['text_secondary'], 1)
    
    def draw_trajectory_trails(self, frame):
        """Draw trajectory trails for tracked objects"""
        for track_id, history in self.track_history.items():
            if len(history) > 1:
                # Get color based on track ID
                color_idx = track_id % 6
                colors = [
                    (255, 100, 100), (100, 255, 100), (100, 100, 255),
                    (255, 255, 100), (255, 100, 255), (100, 255, 255)
                ]
                trail_color = colors[color_idx]
                
                # Draw trail
                points = list(history)
                for i in range(1, len(points)):
                    # Fade older points
                    alpha = i / len(points)
                    thickness = max(1, int(alpha * 3))
                    
                    cv2.line(frame, 
                            tuple(map(int, points[i-1])), 
                            tuple(map(int, points[i])), 
                            trail_color, thickness)
    
    def process_frame_enhanced(self, frame):
        """Enhanced frame processing with advanced features"""
        start_time = time.time()
        
        # Run YOLO detection
        results = self.model(frame, verbose=False)
        
        detections = []
        
        # Process detections
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy()
                    
                    # Dynamic confidence threshold
                    threshold = self.category_confidence_thresholds.get(class_name, 0.5)
                    
                    if confidence > threshold:
                        detection = {
                            'class': class_name,
                            'confidence': confidence,
                            'box': bbox,
                            'timestamp': time.time()
                        }
                        detections.append(detection)
        
        # Update tracking and trajectories
        self.update_object_tracking(detections)
        
        # Update heat map
        self.update_heat_map(detections)
        
        # Update statistics
        self.update_detection_stats(detections)
        
        # Calculate FPS
        processing_time = time.time() - start_time
        fps = 1.0 / max(processing_time, 0.001)
        self.fps_counter.append(fps)
        avg_fps = sum(self.fps_counter) / len(self.fps_counter)
        
        return detections, avg_fps
    
    def update_object_tracking(self, detections):
        """Enhanced object tracking with trajectory recording"""
        current_tracks = {}
        
        for detection in detections:
            box = detection['box']
            center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            
            # Find best matching track
            best_track_id = None
            min_distance = float('inf')
            
            for track_id, track_info in self.object_tracks.items():
                if track_info['class'] == detection['class']:
                    last_center = track_info['center']
                    distance = np.sqrt((center[0] - last_center[0])**2 + 
                                     (center[1] - last_center[1])**2)
                    
                    if distance < 100 and distance < min_distance:
                        min_distance = distance
                        best_track_id = track_id
            
            # Assign or create track
            if best_track_id is not None:
                track_id = best_track_id
                self.object_tracks[track_id]['center'] = center
                self.object_tracks[track_id]['last_seen'] = time.time()
                self.object_persistence[track_id] += 1
            else:
                track_id = self.next_track_id
                self.next_track_id += 1
                self.object_tracks[track_id] = {
                    'class': detection['class'],
                    'center': center,
                    'first_seen': time.time(),
                    'last_seen': time.time()
                }
                self.object_persistence[track_id] = 1
            
            # Record trajectory
            self.track_history[track_id].append(center)
            detection['track_id'] = track_id
            current_tracks[track_id] = detection
        
        # Clean up old tracks
        current_time = time.time()
        tracks_to_remove = []
        for track_id, track_info in self.object_tracks.items():
            if current_time - track_info['last_seen'] > 2.0:  # 2 second timeout
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.object_tracks[track_id]
            del self.object_persistence[track_id]
            if track_id in self.track_history:
                del self.track_history[track_id]
    
    def update_detection_stats(self, detections):
        """Update detection statistics"""
        self.detection_stats['total_detections'] += len(detections)
        
        for detection in detections:
            category = self.get_object_category(detection['class'])
            if category == 'person':
                self.detection_stats['humans_detected'] += 1
            elif category == 'animal':
                self.detection_stats['animals_detected'] += 1
            else:
                self.detection_stats['objects_detected'] += 1
    
    def play_detection_sound(self, category):
        """Play sound for detection"""
        if not self.sound_enabled:
            return
        
        try:
            # Create simple beep sounds for different categories
            duration = 200  # milliseconds
            frequency_map = {
                'person': 800,
                'vehicle': 400,
                'animal': 1000,
                'object': 600
            }
            
            frequency = frequency_map.get(category, 500)
            
            # Generate sine wave
            sample_rate = 22050
            frames = int(duration * sample_rate / 1000)
            arr = np.zeros(frames)
            
            for i in range(frames):
                arr[i] = np.sin(2 * np.pi * frequency * i / sample_rate)
            
            arr = (arr * 32767).astype(np.int16)
            
            # Convert to stereo
            stereo_arr = np.zeros((frames, 2), dtype=np.int16)
            stereo_arr[:, 0] = arr
            stereo_arr[:, 1] = arr
            
            sound = pygame.sndarray.make_sound(stereo_arr)
            sound.play()
            
        except Exception as e:
            pass  # Silently fail if sound doesn't work
    
    def save_screenshot(self, frame):
        """Save screenshot with timestamp"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.screenshots_dir / f"detection_{timestamp}.jpg"
        cv2.imwrite(str(filename), frame)
        print(f"📸 Screenshot saved: {filename}")
    
    def toggle_recording(self, frame):
        """Toggle video recording"""
        if not self.recording:
            # Start recording
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.recordings_dir / f"detection_{timestamp}.mp4"
            
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(str(filename), fourcc, 20.0, (width, height))
            self.recording = True
            print(f"🎥 Recording started: {filename}")
        else:
            # Stop recording
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.recording = False
            print("⏹️ Recording stopped")
    
    def run_enhanced_detection(self):
        """Main enhanced detection loop"""
        print("🎥 Starting Enhanced Detection System...")
        
        # Try different camera indices
        cap = None
        for i in range(3):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"✅ Camera {i} working!")
                    break
                else:
                    cap.release()
            else:
                cap.release() if cap else None
        
        if not cap or not cap.isOpened():
            print("❌ No camera available, running demo mode")
            self.run_demo_mode()
            return
        
        # Set optimal camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Enhanced detection system running!")
        print("🎮 Controls:")
        print("  Q - Quit")
        print("  S - Save screenshot")
        print("  R - Toggle recording")
        print("  H - Toggle heat map")
        print("  T - Toggle trajectory trails")
        print("  A - Toggle audio alerts\n")
        
        show_heat_map = False
        show_trails = True
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Enhanced processing
                detections, fps = self.process_frame_enhanced(frame)
                
                # Apply visual enhancements
                if show_heat_map:
                    self.draw_heat_map_overlay(frame)
                
                if show_trails:
                    self.draw_trajectory_trails(frame)
                
                # Draw enhanced detection boxes
                for detection in detections:
                    track_id = detection.get('track_id')
                    self.draw_enhanced_detection_box(frame, detection, track_id)
                    
                    # Play sound for new detections
                    if self.alerts_enabled and self.object_persistence.get(track_id, 0) == 1:
                        category = self.get_object_category(detection['class'])
                        self.play_detection_sound(category)
                
                # Create modern dashboard
                self.create_modern_dashboard(frame, detections, fps)
                
                # Update animations
                self.pulse_animation += 1
                self.glow_effect += 1
                
                # Record frame if recording
                if self.recording and self.video_writer:
                    self.video_writer.write(frame)
                
                # Add recording indicator
                if self.recording:
                    cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
                    cv2.putText(frame, "REC", (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Display enhanced frame
                cv2.imshow('🚀 Enhanced Object Detection System', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_screenshot(frame)
                elif key == ord('r'):
                    self.toggle_recording(frame)
                elif key == ord('h'):
                    show_heat_map = not show_heat_map
                    print(f"🔥 Heat map: {'ON' if show_heat_map else 'OFF'}")
                elif key == ord('t'):
                    show_trails = not show_trails
                    print(f"🛤️ Trajectory trails: {'ON' if show_trails else 'OFF'}")
                elif key == ord('a'):
                    self.alerts_enabled = not self.alerts_enabled
                    print(f"🔊 Audio alerts: {'ON' if self.alerts_enabled else 'OFF'}")
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\n👋 Enhanced detection stopped by user")
        
        finally:
            if self.recording and self.video_writer:
                self.video_writer.release()
            cap.release()
            cv2.destroyAllWindows()
            if self.sound_enabled:
                pygame.mixer.quit()
            print("✅ Enhanced detection system cleanup completed")
    
    def run_demo_mode(self):
        """Demo mode with enhanced features"""
        print("🖼️ Running Enhanced Demo Mode...")
        
        # Create enhanced demo image
        demo_frame = self.create_gradient_background(1280, 720)
        
        # Add demo content
        cv2.putText(demo_frame, "ENHANCED DETECTION DEMO", (400, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, self.ui_theme['primary'], 3)
        cv2.putText(demo_frame, "Advanced AI-Powered Object Detection", (420, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.ui_theme['text'], 2)
        
        # Add demo shapes and objects
        shapes = [
            {'type': 'rectangle', 'pos': (200, 300, 400, 500), 'label': 'Demo Object 1'},
            {'type': 'circle', 'pos': (600, 400, 80), 'label': 'Demo Object 2'},
            {'type': 'rectangle', 'pos': (800, 200, 1000, 350), 'label': 'Demo Object 3'}
        ]
        
        for shape in shapes:
            if shape['type'] == 'rectangle':
                x1, y1, x2, y2 = shape['pos']
                cv2.rectangle(demo_frame, (x1, y1), (x2, y2), self.ui_theme['secondary'], 2)
            elif shape['type'] == 'circle':
                x, y, r = shape['pos']
                cv2.circle(demo_frame, (x, y), r, self.ui_theme['success'], 2)
        
        frame_count = 0
        
        try:
            while True:
                # Create animated demo
                current_frame = demo_frame.copy()
                
                # Add animated elements
                self.pulse_animation += 1
                self.glow_effect += 1
                
                # Simulate detections
                demo_detections = [
                    {'class': 'person', 'confidence': 0.95, 'box': np.array([250, 350, 350, 450])},
                    {'class': 'car', 'confidence': 0.88, 'box': np.array([650, 450, 750, 500])},
                    {'class': 'dog', 'confidence': 0.76, 'box': np.array([850, 250, 950, 300])}
                ]
                
                # Draw enhanced boxes
                for detection in demo_detections:
                    self.draw_enhanced_detection_box(current_frame, detection)
                
                # Create dashboard
                self.create_modern_dashboard(current_frame, demo_detections, 30.0)
                
                cv2.imshow('🚀 Enhanced Object Detection Demo', current_frame)
                
                if cv2.waitKey(33) & 0xFF == ord('q'):
                    break
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\n👋 Demo stopped by user")
        
        finally:
            cv2.destroyAllWindows()
            print("✅ Demo completed")

def main():
    """Main function for enhanced detection"""
    print("🚀 Enhanced Object Detection System")
    print("=" * 60)
    
    try:
        detector = EnhancedObjectDetector()
        detector.run_enhanced_detection()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have all dependencies installed:")
        print("   pip install ultralytics opencv-python pygame matplotlib")

if __name__ == "__main__":
    main() 