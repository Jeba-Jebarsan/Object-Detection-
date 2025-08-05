import cv2
import face_recognition
import numpy as np
import pandas as pd
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import threading
import queue
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
import os

@dataclass
class Employee:
    """Employee data structure"""
    id: str
    name: str
    department: str
    role: str
    email: str
    face_encoding: np.ndarray
    access_level: int  # 1-5 (1=basic, 5=admin)

@dataclass
class AttendanceRecord:
    """Attendance record structure"""
    employee_id: str
    name: str
    timestamp: datetime
    event_type: str  # 'check_in', 'check_out', 'break_start', 'break_end'
    location: str
    confidence: float

class SmartAttendanceSystem:
    def __init__(self):
        """Initialize smart attendance system"""
        print("👥 Initializing Smart Attendance System...")
        
        # Database setup
        self.db_path = "attendance.db"
        self.init_database()
        
        # Employee database
        self.employees: Dict[str, Employee] = {}
        self.face_encodings = []
        self.employee_names = []
        
        # Load existing employees
        self.load_employees()
        
        # Attendance tracking
        self.daily_attendance = {}
        self.active_sessions = {}  # Track who's currently in office
        
        # Recognition settings
        self.recognition_threshold = 0.6  # Lower = stricter
        self.min_confidence = 0.4
        self.frame_skip = 3  # Process every 3rd frame for performance
        
        # Office settings
        self.office_hours = {
            'start': 8,   # 8 AM
            'end': 18,    # 6 PM
            'break_start': 12,  # 12 PM
            'break_end': 13     # 1 PM
        }
        
        # Alert system
        self.alert_queue = queue.Queue()
        self.late_threshold = 30  # Minutes late before alert
        
        # UI theme
        self.theme = {
            'recognized': (0, 255, 0),      # Green
            'unknown': (0, 0, 255),         # Red
            'processing': (255, 255, 0),    # Yellow
            'text': (255, 255, 255),        # White
            'background': (40, 40, 50),     # Dark gray
            'success': (0, 255, 100),       # Light green
            'warning': (0, 200, 255),       # Orange
            'error': (0, 100, 255)          # Light red
        }
        
        print("✅ Smart Attendance System initialized!")
    
    def init_database(self):
        """Initialize SQLite database for attendance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Employees table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS employees (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                department TEXT,
                role TEXT,
                email TEXT,
                face_encoding BLOB,
                access_level INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Attendance records table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT,
                employee_name TEXT,
                timestamp TIMESTAMP,
                event_type TEXT,
                location TEXT,
                confidence REAL,
                FOREIGN KEY (employee_id) REFERENCES employees (id)
            )
        ''')
        
        # Daily summary table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT,
                date DATE,
                check_in_time TIMESTAMP,
                check_out_time TIMESTAMP,
                total_hours REAL,
                break_duration REAL,
                overtime_hours REAL,
                status TEXT,
                FOREIGN KEY (employee_id) REFERENCES employees (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        print("📊 Attendance database initialized")
    
    def add_employee(self, employee_id: str, name: str, department: str, role: str, 
                    email: str, face_image_path: str, access_level: int = 1):
        """Add new employee to the system"""
        try:
            # Load and encode face
            image = face_recognition.load_image_file(face_image_path)
            face_encodings = face_recognition.face_encodings(image)
            
            if not face_encodings:
                print(f"❌ No face found in image for {name}")
                return False
            
            face_encoding = face_encodings[0]
            
            # Create employee object
            employee = Employee(
                id=employee_id,
                name=name,
                department=department,
                role=role,
                email=email,
                face_encoding=face_encoding,
                access_level=access_level
            )
            
            # Add to memory
            self.employees[employee_id] = employee
            self.face_encodings.append(face_encoding)
            self.employee_names.append(employee_id)
            
            # Save to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO employees 
                (id, name, department, role, email, face_encoding, access_level)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (employee_id, name, department, role, email, 
                  pickle.dumps(face_encoding), access_level))
            
            conn.commit()
            conn.close()
            
            print(f"✅ Employee {name} added successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error adding employee {name}: {e}")
            return False
    
    def load_employees(self):
        """Load employees from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM employees')
            rows = cursor.fetchall()
            
            for row in rows:
                employee_id, name, department, role, email, face_encoding_blob, access_level, _ = row
                
                face_encoding = pickle.loads(face_encoding_blob)
                
                employee = Employee(
                    id=employee_id,
                    name=name,
                    department=department,
                    role=role,
                    email=email,
                    face_encoding=face_encoding,
                    access_level=access_level
                )
                
                self.employees[employee_id] = employee
                self.face_encodings.append(face_encoding)
                self.employee_names.append(employee_id)
            
            conn.close()
            print(f"📥 Loaded {len(self.employees)} employees from database")
            
        except Exception as e:
            print(f"⚠️ Error loading employees: {e}")
    
    def recognize_faces(self, frame):
        """Recognize faces in frame"""
        # Find face locations
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        
        recognized_faces = []
        
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Compare with known faces
            matches = face_recognition.compare_faces(self.face_encodings, face_encoding, 
                                                   tolerance=self.recognition_threshold)
            face_distances = face_recognition.face_distance(self.face_encodings, face_encoding)
            
            name = "Unknown"
            employee_id = "unknown"
            confidence = 0.0
            
            if matches and len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index] and face_distances[best_match_index] < self.recognition_threshold:
                    employee_id = self.employee_names[best_match_index]
                    name = self.employees[employee_id].name
                    confidence = 1.0 - face_distances[best_match_index]
            
            recognized_faces.append({
                'employee_id': employee_id,
                'name': name,
                'confidence': confidence,
                'location': face_location,
                'known': employee_id != "unknown"
            })
        
        return recognized_faces
    
    def record_attendance(self, employee_id: str, event_type: str, confidence: float):
        """Record attendance event"""
        if employee_id == "unknown":
            # Handle unknown person
            self.handle_unknown_person(confidence)
            return
        
        employee = self.employees[employee_id]
        current_time = datetime.now()
        
        # Create attendance record
        record = AttendanceRecord(
            employee_id=employee_id,
            name=employee.name,
            timestamp=current_time,
            event_type=event_type,
            location="Main Entrance",  # Could be configured
            confidence=confidence
        )
        
        # Determine event type based on current status
        if employee_id not in self.active_sessions:
            # First time today or after check out
            event_type = "check_in"
            self.active_sessions[employee_id] = {
                'check_in': current_time,
                'status': 'present',
                'break_start': None,
                'total_break_time': 0
            }
        else:
            # Already checked in
            session = self.active_sessions[employee_id]
            if session['status'] == 'present':
                # Could be going on break or checking out
                if self.is_break_time():
                    event_type = "break_start"
                    session['break_start'] = current_time
                    session['status'] = 'on_break'
                else:
                    event_type = "check_out"
                    self.calculate_daily_summary(employee_id)
                    del self.active_sessions[employee_id]
            elif session['status'] == 'on_break':
                event_type = "break_end"
                if session['break_start']:
                    break_duration = (current_time - session['break_start']).total_seconds() / 60
                    session['total_break_time'] += break_duration
                session['status'] = 'present'
                session['break_start'] = None
        
        # Update record with determined event type
        record.event_type = event_type
        
        # Save to database
        self.save_attendance_record(record)
        
        # Check for alerts
        self.check_attendance_alerts(employee_id, record)
        
        # Display notification
        self.show_attendance_notification(record)
        
        print(f"📝 {employee.name} - {event_type} at {current_time.strftime('%H:%M:%S')}")
    
    def save_attendance_record(self, record: AttendanceRecord):
        """Save attendance record to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO attendance 
                (employee_id, employee_name, timestamp, event_type, location, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (record.employee_id, record.name, record.timestamp, 
                  record.event_type, record.location, record.confidence))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"❌ Error saving attendance record: {e}")
    
    def calculate_daily_summary(self, employee_id: str):
        """Calculate daily summary when employee checks out"""
        if employee_id not in self.active_sessions:
            return
        
        session = self.active_sessions[employee_id]
        check_in = session['check_in']
        check_out = datetime.now()
        
        # Calculate total hours
        total_seconds = (check_out - check_in).total_seconds()
        total_hours = total_seconds / 3600
        
        # Calculate break time
        break_duration = session['total_break_time']  # Already in minutes
        if session['status'] == 'on_break' and session['break_start']:
            # Add current break if still on break
            current_break = (check_out - session['break_start']).total_seconds() / 60
            break_duration += current_break
        
        # Calculate working hours (total - break)
        working_hours = total_hours - (break_duration / 60)
        
        # Calculate overtime (assuming 8 hour workday)
        standard_hours = 8
        overtime_hours = max(0, working_hours - standard_hours)
        
        # Determine status
        if working_hours < 4:
            status = "Half Day"
        elif working_hours < 7:
            status = "Short Day"
        elif working_hours >= 8:
            status = "Full Day"
        else:
            status = "Regular"
        
        # Save daily summary
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            today = check_in.date()
            
            cursor.execute('''
                INSERT OR REPLACE INTO daily_summary 
                (employee_id, date, check_in_time, check_out_time, total_hours, 
                 break_duration, overtime_hours, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (employee_id, today, check_in, check_out, working_hours,
                  break_duration, overtime_hours, status))
            
            conn.commit()
            conn.close()
            
            print(f"📊 Daily summary calculated for {self.employees[employee_id].name}: {working_hours:.2f} hours")
            
        except Exception as e:
            print(f"❌ Error saving daily summary: {e}")
    
    def is_break_time(self):
        """Check if current time is break time"""
        current_hour = datetime.now().hour
        return self.office_hours['break_start'] <= current_hour < self.office_hours['break_end']
    
    def check_attendance_alerts(self, employee_id: str, record: AttendanceRecord):
        """Check for attendance-related alerts"""
        current_time = record.timestamp
        
        # Late arrival alert
        if record.event_type == "check_in":
            expected_start = current_time.replace(hour=self.office_hours['start'], minute=0, second=0)
            if current_time > expected_start + timedelta(minutes=self.late_threshold):
                minutes_late = (current_time - expected_start).total_seconds() / 60
                alert = {
                    'type': 'late_arrival',
                    'employee': record.name,
                    'message': f"⏰ {record.name} is {minutes_late:.0f} minutes late",
                    'severity': 'medium',
                    'timestamp': current_time
                }
                self.alert_queue.put(alert)
        
        # Early departure alert
        elif record.event_type == "check_out":
            expected_end = current_time.replace(hour=self.office_hours['end'], minute=0, second=0)
            if current_time < expected_end - timedelta(minutes=30):  # 30 min early
                alert = {
                    'type': 'early_departure',
                    'employee': record.name,
                    'message': f"🚪 {record.name} left early",
                    'severity': 'low',
                    'timestamp': current_time
                }
                self.alert_queue.put(alert)
        
        # Long break alert
        elif record.event_type == "break_start":
            # Could set up timer to alert if break is too long
            pass
    
    def handle_unknown_person(self, confidence: float):
        """Handle unknown person detection"""
        current_time = datetime.now()
        
        alert = {
            'type': 'unknown_person',
            'message': f"⚠️ Unknown person detected at entrance",
            'severity': 'high',
            'timestamp': current_time,
            'confidence': confidence
        }
        self.alert_queue.put(alert)
        
        # Log unknown access attempt
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO attendance 
            (employee_id, employee_name, timestamp, event_type, location, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', ("unknown", "Unknown Person", current_time, "unauthorized_access", 
              "Main Entrance", confidence))
        
        conn.commit()
        conn.close()
    
    def show_attendance_notification(self, record: AttendanceRecord):
        """Show attendance notification (would integrate with office display)"""
        # This could be integrated with office displays, mobile apps, etc.
        print(f"🔔 {record.name} - {record.event_type.replace('_', ' ').title()}")
    
    def draw_face_recognition(self, frame, recognized_faces):
        """Draw face recognition results on frame"""
        for face in recognized_faces:
            top, right, bottom, left = face['location']
            
            # Choose color based on recognition
            if face['known']:
                color = self.theme['recognized']
                label = f"{face['name']} ({face['confidence']:.2f})"
            else:
                color = self.theme['unknown']
                label = f"Unknown ({face['confidence']:.2f})"
            
            # Draw rectangle around face
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (left, bottom + 5), (left + label_size[0] + 10, bottom + 35), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (left + 5, bottom + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.theme['text'], 2)
            
            # Add status indicator
            if face['known']:
                employee_id = face['employee_id']
                if employee_id in self.active_sessions:
                    status = self.active_sessions[employee_id]['status']
                    status_color = self.theme['success'] if status == 'present' else self.theme['warning']
                    cv2.circle(frame, (right - 10, top + 10), 8, status_color, -1)
    
    def draw_attendance_dashboard(self, frame):
        """Draw attendance dashboard"""
        height, width = frame.shape[:2]
        
        # Dashboard panel
        panel_width = 350
        panel_height = 400
        panel_x = width - panel_width - 20
        panel_y = 20
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                     self.theme['background'], -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Border
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                     self.theme['recognized'], 2)
        
        # Title
        cv2.putText(frame, "👥 ATTENDANCE DASHBOARD", (panel_x + 20, panel_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.theme['text'], 2)
        
        # Current time
        current_time = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, f"Time: {current_time}", (panel_x + 20, panel_y + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.theme['text'], 1)
        
        # Present employees
        y_offset = panel_y + 90
        cv2.putText(frame, "Currently Present:", (panel_x + 20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.theme['text'], 2)
        
        y_offset += 25
        present_count = 0
        
        for employee_id, session in self.active_sessions.items():
            if present_count >= 8:  # Limit display
                break
                
            employee = self.employees[employee_id]
            status_text = session['status'].replace('_', ' ').title()
            status_color = self.theme['success'] if session['status'] == 'present' else self.theme['warning']
            
            # Employee name
            cv2.putText(frame, f"• {employee.name}", (panel_x + 30, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.theme['text'], 1)
            
            # Status
            cv2.putText(frame, status_text, (panel_x + 250, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
            
            y_offset += 20
            present_count += 1
        
        if present_count == 0:
            cv2.putText(frame, "• No one present", (panel_x + 30, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.theme['text'], 1)
        
        # Total count
        y_offset += 30
        cv2.putText(frame, f"Total Present: {len(self.active_sessions)}", (panel_x + 20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.theme['success'], 2)
        
        # Recent alerts
        y_offset += 40
        cv2.putText(frame, "Recent Alerts:", (panel_x + 20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.theme['error'], 2)
        
        alert_count = 0
        try:
            while not self.alert_queue.empty() and alert_count < 3:
                alert = self.alert_queue.get_nowait()
                y_offset += 20
                alert_text = f"• {alert['message'][:35]}..."
                cv2.putText(frame, alert_text, (panel_x + 30, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.theme['error'], 1)
                alert_count += 1
        except:
            pass
        
        if alert_count == 0:
            y_offset += 20
            cv2.putText(frame, "• No recent alerts", (panel_x + 30, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.theme['text'], 1)
    
    def run_attendance_system(self):
        """Main attendance system loop"""
        print("👥 Starting Smart Attendance System...")
        
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
            print("❌ No camera available")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("✅ Attendance system running!")
        print("🎮 Controls:")
        print("  Q - Quit")
        print("  A - Add new employee (requires setup)")
        print("  R - Generate attendance report")
        print("  E - Export attendance data")
        print("  C - Clear daily records\n")
        
        frame_count = 0
        last_recognition_time = {}
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every nth frame for performance
                if frame_count % self.frame_skip == 0:
                    # Recognize faces
                    recognized_faces = self.recognize_faces(frame)
                    
                    # Process each recognized face
                    for face in recognized_faces:
                        if face['known'] and face['confidence'] > self.min_confidence:
                            employee_id = face['employee_id']
                            current_time = time.time()
                            
                            # Avoid duplicate records (min 10 seconds between records)
                            if (employee_id not in last_recognition_time or 
                                current_time - last_recognition_time[employee_id] > 10):
                                
                                self.record_attendance(employee_id, "auto", face['confidence'])
                                last_recognition_time[employee_id] = current_time
                        
                        elif not face['known']:
                            # Handle unknown person
                            self.handle_unknown_person(face['confidence'])
                
                # Draw recognition results
                if frame_count % self.frame_skip == 0:
                    self.draw_face_recognition(frame, recognized_faces)
                
                # Draw dashboard
                self.draw_attendance_dashboard(frame)
                
                # Display frame
                cv2.imshow('👥 Smart Attendance System', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.generate_attendance_report()
                elif key == ord('e'):
                    self.export_attendance_data()
                elif key == ord('c'):
                    self.clear_daily_records()
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\n👋 Attendance system stopped by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Attendance system cleanup completed")
    
    def generate_attendance_report(self):
        """Generate attendance report for today"""
        today = datetime.now().date()
        
        print("\n📊 DAILY ATTENDANCE REPORT")
        print("=" * 60)
        print(f"Date: {today}")
        print()
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get today's attendance summary
            df = pd.read_sql_query('''
                SELECT employee_id, 
                       MIN(CASE WHEN event_type = 'check_in' THEN timestamp END) as check_in,
                       MAX(CASE WHEN event_type = 'check_out' THEN timestamp END) as check_out,
                       COUNT(*) as total_events
                FROM attendance 
                WHERE DATE(timestamp) = ?
                GROUP BY employee_id
            ''', conn, params=(today,))
            
            if df.empty:
                print("No attendance records for today.")
            else:
                print(f"{'Employee':<20} | {'Check In':<10} | {'Check Out':<10} | {'Events':<8}")
                print("-" * 60)
                
                for _, row in df.iterrows():
                    employee_name = self.employees.get(row['employee_id'], {}).name or row['employee_id']
                    check_in = pd.to_datetime(row['check_in']).strftime('%H:%M:%S') if row['check_in'] else '--:--:--'
                    check_out = pd.to_datetime(row['check_out']).strftime('%H:%M:%S') if row['check_out'] else '--:--:--'
                    
                    print(f"{employee_name:<20} | {check_in:<10} | {check_out:<10} | {row['total_events']:<8}")
            
            conn.close()
            
        except Exception as e:
            print(f"❌ Error generating report: {e}")
        
        print()
    
    def export_attendance_data(self):
        """Export attendance data to CSV"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"attendance_export_{timestamp}.csv"
            
            conn = sqlite3.connect(self.db_path)
            
            # Export all attendance data
            df = pd.read_sql_query('''
                SELECT * FROM attendance 
                ORDER BY timestamp DESC
            ''', conn)
            
            df.to_csv(filename, index=False)
            conn.close()
            
            print(f"💾 Attendance data exported: {filename}")
            
        except Exception as e:
            print(f"❌ Error exporting data: {e}")
    
    def clear_daily_records(self):
        """Clear today's attendance records (admin function)"""
        today = datetime.now().date()
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM attendance WHERE DATE(timestamp) = ?', (today,))
            deleted_count = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            # Clear active sessions
            self.active_sessions.clear()
            
            print(f"🗑️ Cleared {deleted_count} attendance records for today")
            
        except Exception as e:
            print(f"❌ Error clearing records: {e}")

def main():
    """Main function for attendance system"""
    print("👥 Smart Attendance System")
    print("=" * 50)
    
    # Create employees directory for face images
    Path("employee_photos").mkdir(exist_ok=True)
    
    try:
        system = SmartAttendanceSystem()
        
        # Check if employees exist
        if not system.employees:
            print("\n⚠️ No employees found in database")
            print("💡 To add employees, place their photos in 'employee_photos/' folder")
            print("   and use the add_employee() method")
            print("   Example: system.add_employee('E001', 'John Doe', 'IT', 'Developer', 'john@company.com', 'employee_photos/john.jpg')")
            
        system.run_attendance_system()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have installed: pip install face-recognition pandas")

if __name__ == "__main__":
    main() 