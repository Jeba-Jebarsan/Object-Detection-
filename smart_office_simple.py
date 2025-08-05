import cv2
from ultralytics import YOLO
import numpy as np
from collections import Counter, defaultdict, deque
import time
import json
import os
from pathlib import Path
import datetime
import math
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

@dataclass
class OfficeZone:
    """Define office zones for monitoring"""
    name: str
    coordinates: Tuple[int, int, int, int]  # x1, y1, x2, y2
    max_capacity: int
    zone_type: str  # 'desk', 'meeting_room', 'entrance', 'break_area'

@dataclass
class OccupancyData:
    """Store occupancy information"""
    zone_name: str
    current_count: int
    max_capacity: int
    utilization: float
    timestamp: float
    people_ids: List[int]

class SmartOfficeSimple:
    def __init__(self):
        """Initialize simplified smart office detection system"""
        print("🏢 Initializing Smart Office Detection System (Simplified)...")
        
        # Load detection model
        self.model = YOLO('yolov8n.pt')
        
        # Occupancy data (initialize before setup_office_zones)
        self.zone_occupancy = {}
        self.occupancy_history = deque(maxlen=1000)
        
        # Office zones configuration (customizable)
        self.office_zones = self.setup_office_zones()
        
        # Tracking system
        self.person_tracker = {}
        self.next_person_id = 1
        self.tracking_history = defaultdict(lambda: deque(maxlen=50))
        
        # Smart office features
        self.desk_availability = {}
        self.meeting_room_status = {}
        
        # Database for logging
        self.init_database()
        
        # Office hours and scheduling
        self.office_hours = {'start': 8, 'end': 18}  # 8 AM to 6 PM
        self.peak_hours = {'start': 9, 'end': 17}    # 9 AM to 5 PM
        
        # Alert system
        self.alerts = []
        self.alert_thresholds = {
            'overcrowding': 0.9,  # 90% capacity
        }
        
        # Visual theme for office monitoring
        self.office_theme = {
            'occupied': (0, 100, 255),      # Blue for occupied
            'available': (0, 255, 0),       # Green for available
            'warning': (0, 255, 255),       # Yellow for warnings
            'alert': (0, 0, 255),           # Red for alerts
            'neutral': (128, 128, 128),     # Gray for neutral
            'text': (255, 255, 255),        # White text
            'background': (40, 40, 50)      # Dark background
        }
        
        print("✅ Smart Office Detection System initialized!")
    
    def setup_office_zones(self):
        """Configure office zones - customize for your office layout"""
        zones = [
            OfficeZone("Entrance", (50, 50, 200, 300), 5, "entrance"),
            OfficeZone("Meeting Room A", (220, 50, 450, 250), 8, "meeting_room"),
            OfficeZone("Meeting Room B", (470, 50, 700, 250), 6, "meeting_room"),
            OfficeZone("Desk Area 1", (50, 320, 300, 500), 12, "desk"),
            OfficeZone("Desk Area 2", (320, 320, 570, 500), 12, "desk"),
            OfficeZone("Break Area", (590, 320, 750, 500), 10, "break_area"),
            OfficeZone("Reception", (50, 520, 200, 650), 3, "reception")
        ]
        
        # Initialize zone data
        for zone in zones:
            self.zone_occupancy[zone.name] = OccupancyData(
                zone_name=zone.name,
                current_count=0,
                max_capacity=zone.max_capacity,
                utilization=0.0,
                timestamp=time.time(),
                people_ids=[]
            )
        
        return zones
    
    def init_database(self):
        """Initialize SQLite database for logging office data"""
        self.db_path = "smart_office_simple.db"
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS occupancy_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                zone_name TEXT,
                occupant_count INTEGER,
                utilization REAL,
                event_type TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        print("📊 Office database initialized")
    
    def log_to_database(self, table: str, data: dict):
        """Log data to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if table == "occupancy_log":
                cursor.execute('''
                    INSERT INTO occupancy_log (timestamp, zone_name, occupant_count, utilization, event_type)
                    VALUES (?, ?, ?, ?, ?)
                ''', (data['timestamp'], data['zone_name'], data['count'], data['utilization'], data['event']))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Database logging error: {e}")
    
    def detect_people_in_zones(self, frame, detections):
        """Detect people in specific office zones"""
        zone_counts = {}
        
        for zone in self.office_zones:
            people_in_zone = []
            x1, y1, x2, y2 = zone.coordinates
            
            # Check each detected person
            for detection in detections:
                if detection['class'] == 'person' and detection['confidence'] > 0.7:
                    person_box = detection['box']
                    person_center_x = (person_box[0] + person_box[2]) / 2
                    person_center_y = (person_box[1] + person_box[3]) / 2
                    
                    # Check if person is in this zone
                    if x1 <= person_center_x <= x2 and y1 <= person_center_y <= y2:
                        people_in_zone.append(detection)
            
            # Update zone occupancy
            current_count = len(people_in_zone)
            utilization = (current_count / zone.max_capacity) * 100
            
            self.zone_occupancy[zone.name] = OccupancyData(
                zone_name=zone.name,
                current_count=current_count,
                max_capacity=zone.max_capacity,
                utilization=utilization,
                timestamp=time.time(),
                people_ids=[p.get('track_id', 0) for p in people_in_zone]
            )
            
            zone_counts[zone.name] = current_count
            
            # Log significant changes
            if current_count > 0:
                self.log_to_database("occupancy_log", {
                    'timestamp': time.time(),
                    'zone_name': zone.name,
                    'count': current_count,
                    'utilization': utilization,
                    'event': 'occupancy_update'
                })
            
            # Check for alerts
            self.check_zone_alerts(zone, current_count, utilization)
        
        return zone_counts
    
    def check_zone_alerts(self, zone: OfficeZone, current_count: int, utilization: float):
        """Check for zone-specific alerts"""
        # Overcrowding alert
        if utilization > self.alert_thresholds['overcrowding'] * 100:
            alert = {
                'type': 'overcrowding',
                'zone': zone.name,
                'message': f"⚠️ {zone.name} is {utilization:.1f}% full ({current_count}/{zone.max_capacity})",
                'severity': 'high',
                'timestamp': time.time()
            }
            self.alerts.append(alert)
            # Keep only last 10 alerts
            if len(self.alerts) > 10:
                self.alerts.pop(0)
        
        # Meeting room specific alerts
        if zone.zone_type == 'meeting_room':
            if current_count == 0:
                self.meeting_room_status[zone.name] = {
                    'status': 'available',
                    'since': time.time(),
                    'next_meeting': None  # Could integrate with calendar
                }
            else:
                self.meeting_room_status[zone.name] = {
                    'status': 'occupied',
                    'occupants': current_count,
                    'since': time.time()
                }
    
    def draw_office_zones(self, frame):
        """Draw office zones with occupancy status"""
        for zone in self.office_zones:
            x1, y1, x2, y2 = zone.coordinates
            occupancy = self.zone_occupancy[zone.name]
            
            # Choose color based on occupancy
            if occupancy.current_count == 0:
                color = self.office_theme['available']
                status = "Available"
            elif occupancy.utilization > 90:
                color = self.office_theme['alert']
                status = "Full"
            elif occupancy.utilization > 70:
                color = self.office_theme['warning']
                status = "Busy"
            else:
                color = self.office_theme['occupied']
                status = "Occupied"
            
            # Draw zone boundary
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Zone label background
            label_text = f"{zone.name}: {occupancy.current_count}/{zone.max_capacity}"
            status_text = f"{status} ({occupancy.utilization:.1f}%)"
            
            # Label background
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            bg_width = max(label_size[0], status_size[0]) + 10
            bg_height = label_size[1] + status_size[1] + 20
            
            # Semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1 - bg_height), (x1 + bg_width, y1), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Zone text
            cv2.putText(frame, label_text, (x1 + 5, y1 - bg_height + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.office_theme['text'], 2)
            cv2.putText(frame, status_text, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.office_theme['text'], 1)
    
    def draw_office_dashboard(self, frame):
        """Draw comprehensive office dashboard"""
        height, width = frame.shape[:2]
        
        # Dashboard panel
        panel_width = 400
        panel_height = 300
        panel_x = width - panel_width - 20
        panel_y = 20
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                     self.office_theme['background'], -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Border
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                     self.office_theme['occupied'], 2)
        
        # Title
        cv2.putText(frame, "🏢 SMART OFFICE DASHBOARD", (panel_x + 20, panel_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.office_theme['text'], 2)
        
        # Current time and office status
        current_hour = datetime.datetime.now().hour
        office_status = "Open" if self.office_hours['start'] <= current_hour <= self.office_hours['end'] else "Closed"
        time_str = datetime.datetime.now().strftime("%H:%M:%S")
        
        cv2.putText(frame, f"Time: {time_str} | Status: {office_status}", 
                   (panel_x + 20, panel_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                   self.office_theme['text'], 1)
        
        # Zone summary
        y_offset = panel_y + 90
        cv2.putText(frame, "Zone Occupancy:", (panel_x + 20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.office_theme['text'], 2)
        
        y_offset += 25
        total_people = 0
        total_capacity = 0
        
        for zone_name, occupancy in self.zone_occupancy.items():
            total_people += occupancy.current_count
            total_capacity += occupancy.max_capacity
            
            # Zone status
            status_color = self.office_theme['available'] if occupancy.current_count == 0 else self.office_theme['occupied']
            zone_text = f"{zone_name}: {occupancy.current_count}/{occupancy.max_capacity}"
            
            cv2.putText(frame, zone_text, (panel_x + 30, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
            y_offset += 20
        
        # Overall occupancy
        overall_utilization = (total_people / total_capacity * 100) if total_capacity > 0 else 0
        y_offset += 10
        cv2.putText(frame, f"Overall: {total_people}/{total_capacity} ({overall_utilization:.1f}%)", 
                   (panel_x + 20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                   self.office_theme['text'], 2)
        
        # Alerts section
        y_offset += 40
        cv2.putText(frame, "Recent Alerts:", (panel_x + 20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.office_theme['alert'], 2)
        
        # Show recent alerts
        alert_count = 0
        for alert in self.alerts[-3:]:  # Show last 3 alerts
            y_offset += 20
            alert_text = f"• {alert['message'][:35]}..."
            cv2.putText(frame, alert_text, (panel_x + 30, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.office_theme['alert'], 1)
            alert_count += 1
        
        if alert_count == 0:
            y_offset += 20
            cv2.putText(frame, "• No active alerts", (panel_x + 30, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.office_theme['available'], 1)
    
    def process_office_frame(self, frame):
        """Process frame for smart office features"""
        start_time = time.time()
        
        # Run object detection
        results = self.model(frame, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy()
                    
                    if confidence > 0.5:
                        detections.append({
                            'class': class_name,
                            'confidence': confidence,
                            'box': bbox,
                            'timestamp': time.time()
                        })
        
        # Filter for people
        people_detections = [d for d in detections if d['class'] == 'person']
        
        # Detect people in zones
        zone_counts = self.detect_people_in_zones(frame, people_detections)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        fps = 1.0 / max(processing_time, 0.001)
        
        return detections, zone_counts, fps
    
    def run_smart_office_monitoring(self):
        """Main smart office monitoring loop"""
        print("🏢 Starting Smart Office Monitoring System...")
        
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
            self.run_office_demo()
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Smart Office System running!")
        print("🎮 Controls:")
        print("  Q - Quit")
        print("  S - Save office snapshot")
        print("  R - Generate occupancy report")
        print("  Z - Toggle zone visibility")
        print("  D - Export office data\n")
        
        show_zones = True
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame for office features
                detections, zone_counts, fps = self.process_office_frame(frame)
                
                # Draw enhanced detection boxes
                for detection in detections:
                    if detection['class'] == 'person':
                        box = detection['box']
                        x1, y1, x2, y2 = map(int, box)
                        
                        person_color = self.office_theme['occupied']
                        cv2.rectangle(frame, (x1, y1), (x2, y2), person_color, 2)
                        cv2.putText(frame, f"Person {detection['confidence']:.2f}", 
                                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, person_color, 2)
                
                # Draw office zones
                if show_zones:
                    self.draw_office_zones(frame)
                
                # Draw office dashboard
                self.draw_office_dashboard(frame)
                
                # Display frame
                cv2.imshow('🏢 Smart Office Monitoring System', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_office_snapshot(frame)
                elif key == ord('r'):
                    self.generate_occupancy_report()
                elif key == ord('z'):
                    show_zones = not show_zones
                    print(f"Zone visibility: {'ON' if show_zones else 'OFF'}")
                elif key == ord('d'):
                    self.export_office_data()
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\n👋 Smart office monitoring stopped by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Smart office system cleanup completed")
    
    def save_office_snapshot(self, frame):
        """Save office snapshot with timestamp"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"office_snapshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 Office snapshot saved: {filename}")
    
    def generate_occupancy_report(self):
        """Generate occupancy report"""
        print("\n📊 OCCUPANCY REPORT")
        print("=" * 50)
        
        current_time = datetime.datetime.now()
        print(f"Report Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        total_people = 0
        total_capacity = 0
        
        for zone_name, occupancy in self.zone_occupancy.items():
            total_people += occupancy.current_count
            total_capacity += occupancy.max_capacity
            
            print(f"{zone_name:15} | {occupancy.current_count:2}/{occupancy.max_capacity:2} | {occupancy.utilization:5.1f}%")
        
        overall_utilization = (total_people / total_capacity * 100) if total_capacity > 0 else 0
        print("-" * 50)
        print(f"{'TOTAL':15} | {total_people:2}/{total_capacity:2} | {overall_utilization:5.1f}%")
        print()
    
    def export_office_data(self):
        """Export office data to JSON"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"office_data_{timestamp}.json"
        
        export_data = {
            'timestamp': time.time(),
            'zone_occupancy': {name: {
                'current_count': occ.current_count,
                'max_capacity': occ.max_capacity,
                'utilization': occ.utilization,
                'people_ids': occ.people_ids
            } for name, occ in self.zone_occupancy.items()},
            'meeting_room_status': self.meeting_room_status,
            'office_hours': self.office_hours,
            'alert_thresholds': self.alert_thresholds,
            'recent_alerts': self.alerts[-5:]  # Last 5 alerts
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"💾 Office data exported: {filename}")
    
    def run_office_demo(self):
        """Demo mode for smart office system"""
        print("🖼️ Running Smart Office Demo...")
        
        # Create demo frame
        demo_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 50
        
        # Add office layout
        cv2.putText(demo_frame, "SMART OFFICE DEMO", (400, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, self.office_theme['text'], 3)
        
        frame_count = 0
        
        try:
            while True:
                current_frame = demo_frame.copy()
                
                # Simulate people in zones
                import random
                for zone in self.office_zones:
                    simulated_count = random.randint(0, min(zone.max_capacity, 5))
                    self.zone_occupancy[zone.name].current_count = simulated_count
                    self.zone_occupancy[zone.name].utilization = (simulated_count / zone.max_capacity) * 100
                
                # Draw zones and dashboard
                self.draw_office_zones(current_frame)
                self.draw_office_dashboard(current_frame)
                
                cv2.imshow('🏢 Smart Office Demo', current_frame)
                
                if cv2.waitKey(33) & 0xFF == ord('q'):
                    break
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\n👋 Demo stopped by user")
        
        finally:
            cv2.destroyAllWindows()
            print("✅ Demo completed")

def main():
    """Main function for smart office detection"""
    print("🏢 Smart Office Detection System (Simplified)")
    print("=" * 60)
    
    try:
        detector = SmartOfficeSimple()
        detector.run_smart_office_monitoring()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have ultralytics and opencv-python installed:")
        print("   pip install ultralytics opencv-python")

if __name__ == "__main__":
    main() 