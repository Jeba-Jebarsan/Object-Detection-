#!/usr/bin/env python3
"""
Human Activity Detection System
Detects and tracks human activities in real-time using computer vision
"""

import cv2
import numpy as np
import time
import json
import sqlite3
from collections import deque, defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import math
from datetime import datetime, timedelta

# Try to import advanced pose detection (fallback to basic if not available)
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️  MediaPipe not available, using basic movement detection")

from ultralytics import YOLO

class ActivityType(Enum):
    """Defined activity types for office environment"""
    SITTING = "sitting"
    STANDING = "standing"
    WALKING = "walking"
    PRESENTING = "presenting"
    TYPING = "typing"
    TALKING = "talking"
    READING = "reading"
    MEETING = "meeting"
    BREAK_TIME = "break"
    EXERCISING = "exercising"
    UNKNOWN = "unknown"

@dataclass
class ActivityData:
    """Store activity information"""
    person_id: int
    activity: ActivityType
    confidence: float
    start_time: float
    duration: float
    position: Tuple[int, int]
    pose_landmarks: Optional[List] = None

@dataclass
class PersonTracker:
    """Track individual person's activities"""
    person_id: int
    current_activity: ActivityType
    activity_history: deque
    total_time: Dict[ActivityType, float]
    last_position: Tuple[int, int]
    pose_history: deque
    movement_speed: float

class HumanActivityDetector:
    def __init__(self):
        """Initialize human activity detection system"""
        print("🏃‍♂️ Initializing Human Activity Detection System...")
        
        # Load detection models
        self.yolo_model = YOLO('yolov8n.pt')
        
        # Initialize pose detection if available
        self.pose_detector = None
        if MEDIAPIPE_AVAILABLE:
            mp_pose = mp.solutions.pose
            self.pose_detector = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing = mp_drawing
            self.mp_pose = mp_pose
            print("✅ MediaPipe pose detection enabled")
        else:
            print("📱 Using basic movement-based activity detection")
        
        # Activity tracking
        self.people_trackers: Dict[int, PersonTracker] = {}
        self.next_person_id = 1
        self.activity_history = deque(maxlen=1000)
        
        # Activity classification parameters
        self.activity_thresholds = {
            'movement_sitting': 5.0,      # pixels/frame for sitting
            'movement_standing': 15.0,    # pixels/frame for standing
            'movement_walking': 30.0,     # pixels/frame for walking
            'pose_confidence': 0.7,       # minimum pose confidence
            'activity_duration': 3.0,     # seconds to confirm activity
        }
        
        # Office zones for context-aware detection
        self.office_zones = {
            'presentation_area': (100, 50, 300, 200),
            'meeting_room': (350, 50, 600, 250),
            'desk_area': (50, 300, 400, 550),
            'break_area': (450, 300, 650, 550),
        }
        
        # Activity patterns and rules
        self.activity_rules = {
            'typing': {'zone': 'desk_area', 'pose': 'sitting', 'movement': 'low'},
            'presenting': {'zone': 'presentation_area', 'pose': 'standing', 'movement': 'medium'},
            'meeting': {'zone': 'meeting_room', 'pose': 'sitting', 'movement': 'low'},
            'break': {'zone': 'break_area', 'pose': 'any', 'movement': 'varied'},
        }
        
        # Database for activity logging
        self.init_database()
        
        # Visual theme
        self.theme = {
            'sitting': (0, 255, 0),        # Green
            'standing': (255, 255, 0),     # Yellow
            'walking': (255, 165, 0),      # Orange
            'presenting': (255, 0, 255),   # Magenta
            'typing': (0, 255, 255),       # Cyan
            'talking': (255, 100, 100),    # Light Red
            'meeting': (100, 100, 255),    # Light Blue
            'break_time': (200, 200, 200), # Gray
            'unknown': (128, 128, 128),    # Dark Gray
            'text': (255, 255, 255),       # White
            'background': (40, 40, 50),    # Dark background
        }
        
        print("✅ Human Activity Detection System initialized!")
    
    def init_database(self):
        """Initialize SQLite database for activity logging"""
        self.db_path = "human_activities.db"
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                person_id INTEGER,
                activity TEXT,
                confidence REAL,
                duration REAL,
                position_x INTEGER,
                position_y INTEGER,
                zone TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                person_id INTEGER,
                total_sitting_time REAL,
                total_standing_time REAL,
                total_walking_time REAL,
                total_typing_time REAL,
                total_meeting_time REAL,
                productivity_score REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        print("📊 Activity database initialized")
    
    def detect_people(self, frame):
        """Detect people in the frame using YOLO"""
        results = self.yolo_model(frame, verbose=False)
        people_detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = self.yolo_model.names[class_id]
                    confidence = float(box.conf[0])
                    
                    if class_name == 'person' and confidence > 0.6:
                        bbox = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, bbox)
                        center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        
                        people_detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'center': center,
                            'confidence': confidence,
                            'area': (x2 - x1) * (y2 - y1)
                        })
        
        return people_detections
    
    def get_pose_landmarks(self, frame, bbox):
        """Extract pose landmarks from person bounding box"""
        if not self.pose_detector:
            return None
        
        x1, y1, x2, y2 = bbox
        person_roi = frame[y1:y2, x1:x2]
        
        if person_roi.size == 0:
            return None
        
        try:
            rgb_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
            results = self.pose_detector.process(rgb_roi)
            
            if results.pose_landmarks:
                # Convert landmarks to original frame coordinates
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    x = int(landmark.x * (x2 - x1)) + x1
                    y = int(landmark.y * (y2 - y1)) + y1
                    landmarks.append({
                        'x': x, 'y': y, 
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                return landmarks
        except Exception as e:
            pass
        
        return None
    
    def classify_activity_from_pose(self, landmarks, movement_speed, zone):
        """Classify activity based on pose landmarks and context"""
        if not landmarks:
            return self.classify_activity_from_movement(movement_speed, zone)
        
        # Key pose landmarks indices (MediaPipe pose)
        key_points = {
            'nose': 0, 'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28
        }
        
        try:
            # Extract key points
            nose = landmarks[key_points['nose']]
            left_shoulder = landmarks[key_points['left_shoulder']]
            right_shoulder = landmarks[key_points['right_shoulder']]
            left_hip = landmarks[key_points['left_hip']]
            right_hip = landmarks[key_points['right_hip']]
            left_wrist = landmarks[key_points['left_wrist']]
            right_wrist = landmarks[key_points['right_wrist']]
            
            # Calculate pose metrics
            shoulder_hip_distance = abs(left_shoulder['y'] - left_hip['y'])
            wrist_shoulder_distance = min(
                abs(left_wrist['y'] - left_shoulder['y']),
                abs(right_wrist['y'] - right_shoulder['y'])
            )
            
            # Activity classification logic
            if movement_speed > 25:
                return ActivityType.WALKING
            elif zone == 'presentation_area' and shoulder_hip_distance > 80:
                return ActivityType.PRESENTING
            elif zone == 'desk_area' and wrist_shoulder_distance < 30:
                return ActivityType.TYPING
            elif zone == 'meeting_room':
                return ActivityType.MEETING
            elif shoulder_hip_distance < 60:  # Person appears to be sitting
                return ActivityType.SITTING
            else:
                return ActivityType.STANDING
                
        except (IndexError, KeyError):
            return self.classify_activity_from_movement(movement_speed, zone)
    
    def classify_activity_from_movement(self, movement_speed, zone):
        """Fallback activity classification based on movement and zone"""
        if movement_speed > self.activity_thresholds['movement_walking']:
            return ActivityType.WALKING
        elif movement_speed > self.activity_thresholds['movement_standing']:
            if zone == 'presentation_area':
                return ActivityType.PRESENTING
            else:
                return ActivityType.STANDING
        else:
            if zone == 'desk_area':
                return ActivityType.TYPING
            elif zone == 'meeting_room':
                return ActivityType.MEETING
            elif zone == 'break_area':
                return ActivityType.BREAK_TIME
            else:
                return ActivityType.SITTING
    
    def get_zone_for_position(self, position):
        """Determine which office zone a position belongs to"""
        x, y = position
        for zone_name, (x1, y1, x2, y2) in self.office_zones.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                return zone_name
        return 'general_area'
    
    def track_person_activity(self, person_detection, frame):
        """Track activity for a specific person"""
        center = person_detection['center']
        bbox = person_detection['bbox']
        
        # Find or assign person ID
        person_id = self.assign_person_id(center)
        
        # Get pose landmarks
        landmarks = self.get_pose_landmarks(frame, bbox)
        
        # Calculate movement speed
        movement_speed = self.calculate_movement_speed(person_id, center)
        
        # Determine zone
        zone = self.get_zone_for_position(center)
        
        # Classify activity
        activity = self.classify_activity_from_pose(landmarks, movement_speed, zone)
        
        # Update person tracker
        if person_id not in self.people_trackers:
            self.people_trackers[person_id] = PersonTracker(
                person_id=person_id,
                current_activity=activity,
                activity_history=deque(maxlen=50),
                total_time=defaultdict(float),
                last_position=center,
                pose_history=deque(maxlen=10),
                movement_speed=movement_speed
            )
        
        tracker = self.people_trackers[person_id]
        
        # Update tracker
        tracker.last_position = center
        tracker.movement_speed = movement_speed
        tracker.activity_history.append({
            'activity': activity,
            'timestamp': time.time(),
            'confidence': person_detection['confidence'],
            'zone': zone
        })
        
        if landmarks:
            tracker.pose_history.append(landmarks)
        
        # Determine stable activity (activity confirmed over time)
        stable_activity = self.get_stable_activity(tracker)
        if stable_activity != tracker.current_activity:
            # Log activity change
            self.log_activity_change(person_id, tracker.current_activity, stable_activity, center, zone)
            tracker.current_activity = stable_activity
        
        return {
            'person_id': person_id,
            'activity': stable_activity,
            'confidence': person_detection['confidence'],
            'position': center,
            'zone': zone,
            'movement_speed': movement_speed,
            'landmarks': landmarks
        }
    
    def assign_person_id(self, center):
        """Assign consistent person ID based on position tracking"""
        min_distance = float('inf')
        closest_id = None
        
        for person_id, tracker in self.people_trackers.items():
            distance = math.sqrt(
                (center[0] - tracker.last_position[0])**2 + 
                (center[1] - tracker.last_position[1])**2
            )
            if distance < min_distance and distance < 100:  # Maximum tracking distance
                min_distance = distance
                closest_id = person_id
        
        if closest_id is None:
            closest_id = self.next_person_id
            self.next_person_id += 1
        
        return closest_id
    
    def calculate_movement_speed(self, person_id, current_position):
        """Calculate movement speed for person"""
        if person_id not in self.people_trackers:
            return 0.0
        
        tracker = self.people_trackers[person_id]
        last_pos = tracker.last_position
        
        distance = math.sqrt(
            (current_position[0] - last_pos[0])**2 + 
            (current_position[1] - last_pos[1])**2
        )
        
        return distance  # pixels per frame
    
    def get_stable_activity(self, tracker):
        """Determine stable activity from recent history"""
        if len(tracker.activity_history) < 5:
            return tracker.current_activity
        
        # Get recent activities
        recent_activities = [entry['activity'] for entry in list(tracker.activity_history)[-10:]]
        activity_counts = Counter(recent_activities)
        
        # Return most common activity
        most_common = activity_counts.most_common(1)
        if most_common:
            return most_common[0][0]
        
        return tracker.current_activity
    
    def log_activity_change(self, person_id, old_activity, new_activity, position, zone):
        """Log activity change to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO activity_log (timestamp, person_id, activity, confidence, duration, position_x, position_y, zone)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                time.time(), person_id, new_activity.value, 0.8, 0.0, 
                position[0], position[1], zone
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Database logging error: {e}")
    
    def draw_activity_visualization(self, frame, activity_data):
        """Draw activity visualization on frame"""
        person_id = activity_data['person_id']
        activity = activity_data['activity']
        position = activity_data['position']
        zone = activity_data['zone']
        movement_speed = activity_data['movement_speed']
        landmarks = activity_data['landmarks']
        
        # Get activity color
        color = self.theme.get(activity.value, self.theme['unknown'])
        
        # Draw person bounding box (if available)
        x, y = position
        cv2.circle(frame, (x, y), 20, color, 3)
        
        # Draw pose landmarks if available
        if landmarks and self.pose_detector:
            self.draw_pose_landmarks(frame, landmarks)
        
        # Activity label
        label = f"P{person_id}: {activity.value.title()}"
        speed_label = f"Speed: {movement_speed:.1f}"
        zone_label = f"Zone: {zone.replace('_', ' ').title()}"
        
        # Label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x - 60, y - 80), (x + label_size[0] + 10, y - 20), (0, 0, 0), -1)
        
        # Text
        cv2.putText(frame, label, (x - 55, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, speed_label, (x - 55, y - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.theme['text'], 1)
        cv2.putText(frame, zone_label, (x - 55, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.theme['text'], 1)
    
    def draw_pose_landmarks(self, frame, landmarks):
        """Draw pose landmarks on frame"""
        if not landmarks:
            return
        
        # Draw key landmarks
        key_points = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]  # Important body points
        
        for i in key_points:
            if i < len(landmarks):
                landmark = landmarks[i]
                if landmark['visibility'] > 0.5:
                    cv2.circle(frame, (landmark['x'], landmark['y']), 3, (0, 255, 0), -1)
        
        # Draw key connections
        connections = [
            (11, 12), (11, 13), (12, 14), (13, 15), (14, 16),  # Arms
            (11, 23), (12, 24), (23, 24),  # Torso
            (23, 25), (24, 26), (25, 27), (26, 28)  # Legs
        ]
        
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                if start['visibility'] > 0.5 and end['visibility'] > 0.5:
                    cv2.line(frame, (start['x'], start['y']), (end['x'], end['y']), (255, 255, 0), 2)
    
    def draw_office_zones(self, frame):
        """Draw office zones on frame"""
        for zone_name, (x1, y1, x2, y2) in self.office_zones.items():
            # Zone boundary
            cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 2)
            
            # Zone label
            cv2.putText(frame, zone_name.replace('_', ' ').title(), (x1 + 5, y1 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def draw_activity_dashboard(self, frame):
        """Draw activity analytics dashboard"""
        height, width = frame.shape[:2]
        
        # Dashboard panel
        panel_width = 400
        panel_height = 350
        panel_x = width - panel_width - 20
        panel_y = 20
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                     self.theme['background'], -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Border
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                     (255, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "🏃‍♂️ ACTIVITY DASHBOARD", (panel_x + 20, panel_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.theme['text'], 2)
        
        # Current time
        current_time = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, f"Time: {current_time}", (panel_x + 20, panel_y + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.theme['text'], 1)
        
        # Active people count
        active_people = len(self.people_trackers)
        cv2.putText(frame, f"Active People: {active_people}", (panel_x + 20, panel_y + 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.theme['text'], 1)
        
        # Activity breakdown
        y_offset = panel_y + 120
        cv2.putText(frame, "Current Activities:", (panel_x + 20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.theme['text'], 2)
        
        # Count activities
        activity_counts = defaultdict(int)
        for tracker in self.people_trackers.values():
            activity_counts[tracker.current_activity] += 1
        
        y_offset += 25
        for activity, count in activity_counts.items():
            color = self.theme.get(activity.value, self.theme['unknown'])
            text = f"• {activity.value.title()}: {count}"
            cv2.putText(frame, text, (panel_x + 30, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 20
        
        # Zone occupancy
        y_offset += 20
        cv2.putText(frame, "Zone Occupancy:", (panel_x + 20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.theme['text'], 2)
        
        zone_counts = defaultdict(int)
        for tracker in self.people_trackers.values():
            if tracker.activity_history:
                recent_zone = tracker.activity_history[-1]['zone']
                zone_counts[recent_zone] += 1
        
        y_offset += 25
        for zone, count in zone_counts.items():
            text = f"• {zone.replace('_', ' ').title()}: {count}"
            cv2.putText(frame, text, (panel_x + 30, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.theme['text'], 1)
            y_offset += 20
        
        # Productivity insights
        y_offset += 20
        cv2.putText(frame, "Insights:", (panel_x + 20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.theme['text'], 2)
        
        productive_activities = sum(1 for a in activity_counts.keys() 
                                  if a in [ActivityType.TYPING, ActivityType.MEETING, ActivityType.PRESENTING])
        total_activities = sum(activity_counts.values())
        
        if total_activities > 0:
            productivity = (productive_activities / total_activities) * 100
            y_offset += 25
            cv2.putText(frame, f"• Productivity: {productivity:.1f}%", (panel_x + 30, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.theme['text'], 1)
    
    def generate_activity_report(self):
        """Generate comprehensive activity report"""
        print("\n🏃‍♂️ HUMAN ACTIVITY REPORT")
        print("=" * 60)
        
        current_time = datetime.now()
        print(f"Report Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Active People: {len(self.people_trackers)}")
        print()
        
        # Individual person reports
        for person_id, tracker in self.people_trackers.items():
            print(f"👤 Person {person_id}:")
            print(f"   Current Activity: {tracker.current_activity.value.title()}")
            print(f"   Movement Speed: {tracker.movement_speed:.1f} px/frame")
            
            if tracker.activity_history:
                recent_activities = [entry['activity'].value for entry in list(tracker.activity_history)[-10:]]
                activity_summary = Counter(recent_activities)
                print(f"   Recent Activities: {dict(activity_summary)}")
            print()
        
        # Overall statistics
        all_activities = []
        for tracker in self.people_trackers.values():
            if tracker.activity_history:
                all_activities.extend([entry['activity'].value for entry in tracker.activity_history])
        
        if all_activities:
            overall_stats = Counter(all_activities)
            print("📊 Overall Activity Statistics:")
            for activity, count in overall_stats.most_common():
                percentage = (count / len(all_activities)) * 100
                print(f"   {activity.title()}: {count} ({percentage:.1f}%)")
        
        print("\n" + "=" * 60)
    
    def run_activity_detection(self):
        """Main activity detection loop"""
        print("🏃‍♂️ Starting Human Activity Detection System...")
        
        # Try to open camera
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
            self.run_activity_demo()
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Human Activity Detection System running!")
        print("🎮 Controls:")
        print("  Q - Quit")
        print("  R - Generate activity report")
        print("  S - Save activity snapshot")
        print("  Z - Toggle zone visibility")
        print("  A - Toggle activity dashboard")
        print("  P - Toggle pose landmarks\n")
        
        show_zones = True
        show_dashboard = True
        show_poses = True
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect people
                people = self.detect_people(frame)
                
                # Track activities for each person
                activity_data_list = []
                for person in people:
                    activity_data = self.track_person_activity(person, frame)
                    activity_data_list.append(activity_data)
                
                # Draw visualizations
                if show_zones:
                    self.draw_office_zones(frame)
                
                for activity_data in activity_data_list:
                    self.draw_activity_visualization(frame, activity_data)
                
                if show_dashboard:
                    self.draw_activity_dashboard(frame)
                
                # Display frame
                cv2.imshow('🏃‍♂️ Human Activity Detection', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.generate_activity_report()
                elif key == ord('s'):
                    self.save_activity_snapshot(frame)
                elif key == ord('z'):
                    show_zones = not show_zones
                    print(f"Zone visibility: {'ON' if show_zones else 'OFF'}")
                elif key == ord('a'):
                    show_dashboard = not show_dashboard
                    print(f"Activity dashboard: {'ON' if show_dashboard else 'OFF'}")
                elif key == ord('p'):
                    show_poses = not show_poses
                    print(f"Pose landmarks: {'ON' if show_poses else 'OFF'}")
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\n👋 Activity detection stopped by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Activity detection cleanup completed")
    
    def save_activity_snapshot(self, frame):
        """Save activity snapshot with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"activity_snapshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 Activity snapshot saved: {filename}")
    
    def run_activity_demo(self):
        """Demo mode for activity detection"""
        print("🖼️ Running Activity Detection Demo...")
        
        # Create demo frame
        demo_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 50
        
        # Add title
        cv2.putText(demo_frame, "HUMAN ACTIVITY DETECTION DEMO", (300, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, self.theme['text'], 3)
        
        frame_count = 0
        
        try:
            while True:
                current_frame = demo_frame.copy()
                
                # Draw zones
                self.draw_office_zones(current_frame)
                
                # Simulate some activities
                import random
                simulated_activities = [
                    {'person_id': 1, 'activity': ActivityType.TYPING, 'position': (200, 400), 'zone': 'desk_area', 'movement_speed': 5},
                    {'person_id': 2, 'activity': ActivityType.PRESENTING, 'position': (200, 150), 'zone': 'presentation_area', 'movement_speed': 15},
                    {'person_id': 3, 'activity': ActivityType.MEETING, 'position': (500, 150), 'zone': 'meeting_room', 'movement_speed': 3},
                ]
                
                for activity_data in simulated_activities:
                    # Simulate some movement
                    x, y = activity_data['position']
                    x += random.randint(-10, 10)
                    y += random.randint(-5, 5)
                    activity_data['position'] = (x, y)
                    activity_data['landmarks'] = None
                    
                    self.draw_activity_visualization(current_frame, activity_data)
                
                # Update people trackers for dashboard
                for activity_data in simulated_activities:
                    person_id = activity_data['person_id']
                    if person_id not in self.people_trackers:
                        self.people_trackers[person_id] = PersonTracker(
                            person_id=person_id,
                            current_activity=activity_data['activity'],
                            activity_history=deque(maxlen=50),
                            total_time=defaultdict(float),
                            last_position=activity_data['position'],
                            pose_history=deque(maxlen=10),
                            movement_speed=activity_data['movement_speed']
                        )
                    else:
                        self.people_trackers[person_id].current_activity = activity_data['activity']
                
                self.draw_activity_dashboard(current_frame)
                
                cv2.imshow('🏃‍♂️ Human Activity Detection Demo', current_frame)
                
                if cv2.waitKey(33) & 0xFF == ord('q'):
                    break
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\n👋 Demo stopped by user")
        
        finally:
            cv2.destroyAllWindows()
            print("✅ Demo completed")

def main():
    """Main function for human activity detection"""
    print("🏃‍♂️ Human Activity Detection System")
    print("=" * 60)
    
    try:
        detector = HumanActivityDetector()
        detector.run_activity_detection()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have ultralytics and opencv-python installed:")
        print("   pip install ultralytics opencv-python")
        if not MEDIAPIPE_AVAILABLE:
            print("💡 For advanced pose detection, install mediapipe:")
            print("   pip install mediapipe")

if __name__ == "__main__":
    main() 