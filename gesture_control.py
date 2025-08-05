import cv2
import mediapipe as mp
import numpy as np
import time
import json
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable
import threading
import queue
import subprocess
import os
import pyautogui
from enum import Enum

class GestureType(Enum):
    PEACE = "peace"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    OPEN_PALM = "open_palm"
    CLOSED_FIST = "closed_fist"
    POINTING = "pointing"
    OK_SIGN = "ok_sign"
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    SWIPE_UP = "swipe_up"
    SWIPE_DOWN = "swipe_down"

@dataclass
class GestureCommand:
    """Define gesture-to-command mapping"""
    gesture: GestureType
    action: str
    description: str
    command: Callable
    cooldown: float = 2.0  # Seconds between activations

@dataclass
class HandLandmark:
    """Hand landmark data"""
    x: float
    y: float
    z: float

class SmartOfficeGestureControl:
    def __init__(self):
        """Initialize gesture control system"""
        print("🤚 Initializing Smart Office Gesture Control...")
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Gesture recognition
        self.gesture_history = deque(maxlen=10)
        self.last_gesture_time = defaultdict(float)
        self.gesture_threshold = 0.8
        
        # Hand tracking
        self.hand_positions = deque(maxlen=30)  # For movement detection
        self.swipe_threshold = 0.3  # Minimum distance for swipe
        self.swipe_time_window = 1.0  # Seconds
        
        # Office control commands
        self.setup_gesture_commands()
        
        # System state
        self.lights_on = True
        self.presentation_mode = False
        self.slide_number = 1
        self.volume_level = 50
        
        # UI theme
        self.theme = {
            'detected': (0, 255, 0),        # Green
            'processing': (255, 255, 0),    # Yellow
            'activated': (255, 100, 255),   # Magenta
            'text': (255, 255, 255),        # White
            'background': (40, 40, 50),     # Dark gray
            'landmark': (0, 100, 255),      # Orange
            'connection': (0, 255, 255)     # Cyan
        }
        
        # Debug mode
        self.debug_mode = True
        self.show_landmarks = True
        
        print("✅ Gesture control system initialized!")
    
    def setup_gesture_commands(self):
        """Setup gesture-to-command mappings"""
        self.gesture_commands = {
            GestureType.OPEN_PALM: GestureCommand(
                gesture=GestureType.OPEN_PALM,
                action="toggle_lights",
                description="Toggle office lights",
                command=self.toggle_lights
            ),
            GestureType.CLOSED_FIST: GestureCommand(
                gesture=GestureType.CLOSED_FIST,
                action="emergency_stop",
                description="Emergency stop all systems",
                command=self.emergency_stop
            ),
            GestureType.PEACE: GestureCommand(
                gesture=GestureType.PEACE,
                action="start_presentation",
                description="Start presentation mode",
                command=self.start_presentation
            ),
            GestureType.THUMBS_UP: GestureCommand(
                gesture=GestureType.THUMBS_UP,
                action="volume_up",
                description="Increase volume",
                command=self.volume_up
            ),
            GestureType.THUMBS_DOWN: GestureCommand(
                gesture=GestureType.THUMBS_DOWN,
                action="volume_down",
                description="Decrease volume",
                command=self.volume_down
            ),
            GestureType.POINTING: GestureCommand(
                gesture=GestureType.POINTING,
                action="laser_pointer",
                description="Activate laser pointer",
                command=self.activate_laser_pointer
            ),
            GestureType.OK_SIGN: GestureCommand(
                gesture=GestureType.OK_SIGN,
                action="confirm_action",
                description="Confirm current action",
                command=self.confirm_action
            ),
            GestureType.SWIPE_LEFT: GestureCommand(
                gesture=GestureType.SWIPE_LEFT,
                action="previous_slide",
                description="Previous slide",
                command=self.previous_slide
            ),
            GestureType.SWIPE_RIGHT: GestureCommand(
                gesture=GestureType.SWIPE_RIGHT,
                action="next_slide",
                description="Next slide",
                command=self.next_slide
            ),
            GestureType.SWIPE_UP: GestureCommand(
                gesture=GestureType.SWIPE_UP,
                action="brightness_up",
                description="Increase brightness",
                command=self.brightness_up
            ),
            GestureType.SWIPE_DOWN: GestureCommand(
                gesture=GestureType.SWIPE_DOWN,
                action="brightness_down",
                description="Decrease brightness",
                command=self.brightness_down
            )
        }
    
    def detect_hand_gesture(self, landmarks):
        """Detect hand gesture from landmarks"""
        if not landmarks:
            return None
        
        # Get key landmark positions
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        middle_tip = landmarks[12]
        middle_pip = landmarks[10]
        ring_tip = landmarks[16]
        ring_pip = landmarks[14]
        pinky_tip = landmarks[20]
        pinky_pip = landmarks[18]
        
        # Helper function to check if finger is extended
        def is_finger_extended(tip, pip):
            return tip.y < pip.y
        
        def is_thumb_extended(landmarks):
            return landmarks[4].x > landmarks[3].x  # Assuming right hand
        
        # Count extended fingers
        fingers_up = [
            is_thumb_extended(landmarks),
            is_finger_extended(index_tip, index_pip),
            is_finger_extended(middle_tip, middle_pip),
            is_finger_extended(ring_tip, ring_pip),
            is_finger_extended(pinky_tip, pinky_pip)
        ]
        
        total_fingers = sum(fingers_up)
        
        # Gesture recognition logic
        if total_fingers == 0:
            return GestureType.CLOSED_FIST
        
        elif total_fingers == 5:
            return GestureType.OPEN_PALM
        
        elif total_fingers == 1 and fingers_up[1]:  # Only index finger
            return GestureType.POINTING
        
        elif total_fingers == 2 and fingers_up[1] and fingers_up[2]:  # Index and middle
            return GestureType.PEACE
        
        elif total_fingers == 1 and fingers_up[0]:  # Only thumb
            # Check direction for thumbs up/down
            if thumb_tip.y < landmarks[2].y:  # Thumb above wrist
                return GestureType.THUMBS_UP
            else:
                return GestureType.THUMBS_DOWN
        
        elif (total_fingers == 3 and fingers_up[0] and fingers_up[1] and 
              not fingers_up[2] and not fingers_up[3] and not fingers_up[4]):
            # Check if it's OK sign (thumb and index forming circle)
            thumb_index_distance = np.sqrt(
                (thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2
            )
            if thumb_index_distance < 0.05:  # Close together
                return GestureType.OK_SIGN
        
        return None
    
    def detect_swipe_gesture(self, current_position):
        """Detect swipe gestures based on hand movement"""
        if len(self.hand_positions) < 10:
            self.hand_positions.append(current_position)
            return None
        
        self.hand_positions.append(current_position)
        
        # Calculate movement vector
        start_pos = self.hand_positions[0]
        end_pos = self.hand_positions[-1]
        
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance > self.swipe_threshold:
            # Determine swipe direction
            if abs(dx) > abs(dy):  # Horizontal swipe
                if dx > 0:
                    self.hand_positions.clear()
                    return GestureType.SWIPE_RIGHT
                else:
                    self.hand_positions.clear()
                    return GestureType.SWIPE_LEFT
            else:  # Vertical swipe
                if dy < 0:  # Up (y decreases going up)
                    self.hand_positions.clear()
                    return GestureType.SWIPE_UP
                else:
                    self.hand_positions.clear()
                    return GestureType.SWIPE_DOWN
        
        return None
    
    def process_gesture(self, gesture_type: GestureType):
        """Process detected gesture and execute command"""
        if not gesture_type or gesture_type not in self.gesture_commands:
            return
        
        current_time = time.time()
        command = self.gesture_commands[gesture_type]
        
        # Check cooldown
        if current_time - self.last_gesture_time[gesture_type] < command.cooldown:
            return
        
        # Execute command
        try:
            command.command()
            self.last_gesture_time[gesture_type] = current_time
            print(f"🤚 Gesture executed: {command.description}")
            
        except Exception as e:
            print(f"❌ Error executing gesture command: {e}")
    
    # Office control commands
    def toggle_lights(self):
        """Toggle office lights"""
        self.lights_on = not self.lights_on
        status = "ON" if self.lights_on else "OFF"
        print(f"💡 Lights turned {status}")
        
        # Here you would integrate with actual smart lighting system
        # Example: send HTTP request to smart light controller
        # requests.post(f"http://office-controller/lights", json={"state": self.lights_on})
    
    def emergency_stop(self):
        """Emergency stop all systems"""
        print("🚨 EMERGENCY STOP - All systems halted")
        self.lights_on = False
        self.presentation_mode = False
        # Stop all automated systems
    
    def start_presentation(self):
        """Start presentation mode"""
        self.presentation_mode = True
        self.slide_number = 1
        print("📽️ Presentation mode activated")
        
        # Could launch presentation software
        # subprocess.run(["powerpnt", "/s", "presentation.pptx"])
    
    def volume_up(self):
        """Increase system volume"""
        self.volume_level = min(100, self.volume_level + 10)
        print(f"🔊 Volume increased to {self.volume_level}%")
        
        # Control system volume
        try:
            pyautogui.press('volumeup')
        except:
            pass
    
    def volume_down(self):
        """Decrease system volume"""
        self.volume_level = max(0, self.volume_level - 10)
        print(f"🔉 Volume decreased to {self.volume_level}%")
        
        # Control system volume
        try:
            pyautogui.press('volumedown')
        except:
            pass
    
    def activate_laser_pointer(self):
        """Activate laser pointer mode"""
        print("🔴 Laser pointer activated")
        # Could control actual laser pointer or on-screen cursor
    
    def confirm_action(self):
        """Confirm current action"""
        print("✅ Action confirmed")
        # Send Enter key or confirmation
        try:
            pyautogui.press('enter')
        except:
            pass
    
    def next_slide(self):
        """Go to next slide"""
        if self.presentation_mode:
            self.slide_number += 1
            print(f"➡️ Next slide ({self.slide_number})")
            try:
                pyautogui.press('right')
            except:
                pass
    
    def previous_slide(self):
        """Go to previous slide"""
        if self.presentation_mode:
            self.slide_number = max(1, self.slide_number - 1)
            print(f"⬅️ Previous slide ({self.slide_number})")
            try:
                pyautogui.press('left')
            except:
                pass
    
    def brightness_up(self):
        """Increase screen brightness"""
        print("☀️ Brightness increased")
        # Could control monitor brightness or room lighting
    
    def brightness_down(self):
        """Decrease screen brightness"""
        print("🌙 Brightness decreased")
        # Could control monitor brightness or room lighting
    
    def draw_hand_landmarks(self, frame, landmarks, hand_label):
        """Draw hand landmarks and connections"""
        if not self.show_landmarks:
            return
        
        h, w, c = frame.shape
        
        # Draw landmarks
        for i, landmark in enumerate(landmarks):
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            
            # Different colors for different landmark types
            if i in [4, 8, 12, 16, 20]:  # Fingertips
                color = self.theme['detected']
                radius = 6
            else:
                color = self.theme['landmark']
                radius = 4
            
            cv2.circle(frame, (x, y), radius, color, -1)
            
            # Draw landmark number in debug mode
            if self.debug_mode:
                cv2.putText(frame, str(i), (x+5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 
                           self.theme['text'], 1)
        
        # Draw connections
        connections = self.mp_hands.HAND_CONNECTIONS
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            
            start_point = landmarks[start_idx]
            end_point = landmarks[end_idx]
            
            start_x = int(start_point.x * w)
            start_y = int(start_point.y * h)
            end_x = int(end_point.x * w)
            end_y = int(end_point.y * h)
            
            cv2.line(frame, (start_x, start_y), (end_x, end_y), 
                    self.theme['connection'], 2)
    
    def draw_gesture_info(self, frame, detected_gesture, hand_center):
        """Draw gesture information on frame"""
        if detected_gesture:
            gesture_name = detected_gesture.value.replace('_', ' ').title()
            command = self.gesture_commands.get(detected_gesture)
            
            if command:
                # Gesture name
                cv2.putText(frame, f"Gesture: {gesture_name}", 
                           (hand_center[0] + 50, hand_center[1] - 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.theme['detected'], 2)
                
                # Action description
                cv2.putText(frame, f"Action: {command.description}", 
                           (hand_center[0] + 50, hand_center[1] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.theme['text'], 1)
                
                # Draw gesture indicator circle
                cv2.circle(frame, hand_center, 30, self.theme['activated'], 3)
    
    def draw_control_dashboard(self, frame):
        """Draw gesture control dashboard"""
        height, width = frame.shape[:2]
        
        # Dashboard panel
        panel_width = 400
        panel_height = 350
        panel_x = 20
        panel_y = 20
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                     self.theme['background'], -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Border
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                     self.theme['detected'], 2)
        
        # Title
        cv2.putText(frame, "🤚 GESTURE CONTROL", (panel_x + 20, panel_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.theme['text'], 2)
        
        # System status
        y_offset = panel_y + 60
        status_items = [
            f"Lights: {'ON' if self.lights_on else 'OFF'}",
            f"Presentation: {'ACTIVE' if self.presentation_mode else 'INACTIVE'}",
            f"Volume: {self.volume_level}%",
            f"Slide: {self.slide_number}" if self.presentation_mode else "Slide: --"
        ]
        
        for item in status_items:
            color = self.theme['detected'] if 'ON' in item or 'ACTIVE' in item else self.theme['text']
            cv2.putText(frame, item, (panel_x + 20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 25
        
        # Available gestures
        y_offset += 20
        cv2.putText(frame, "Available Gestures:", (panel_x + 20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.theme['text'], 2)
        
        y_offset += 25
        gesture_list = [
            "✋ Open Palm - Toggle Lights",
            "✊ Closed Fist - Emergency Stop",
            "✌️ Peace - Start Presentation",
            "👍 Thumbs Up - Volume Up",
            "👎 Thumbs Down - Volume Down",
            "👆 Pointing - Laser Pointer",
            "👌 OK Sign - Confirm",
            "➡️ Swipe Right - Next Slide",
            "⬅️ Swipe Left - Previous Slide"
        ]
        
        for i, gesture in enumerate(gesture_list[:6]):  # Show first 6
            cv2.putText(frame, gesture, (panel_x + 30, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.theme['text'], 1)
            y_offset += 18
    
    def run_gesture_control(self):
        """Main gesture control loop"""
        print("🤚 Starting Gesture Control System...")
        
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
        
        print("✅ Gesture control system running!")
        print("🎮 Controls:")
        print("  Q - Quit")
        print("  D - Toggle debug mode")
        print("  L - Toggle landmark display")
        print("  R - Reset system state\n")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                detected_gesture = None
                
                if results.multi_hand_landmarks:
                    for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        # Convert landmarks to our format
                        landmarks = []
                        for landmark in hand_landmarks.landmark:
                            landmarks.append(HandLandmark(landmark.x, landmark.y, landmark.z))
                        
                        # Get hand center for display
                        h, w, c = frame.shape
                        hand_center = (
                            int(landmarks[9].x * w),  # Middle finger MCP
                            int(landmarks[9].y * h)
                        )
                        
                        # Draw hand landmarks
                        self.draw_hand_landmarks(frame, landmarks, f"Hand_{hand_idx}")
                        
                        # Detect static gestures
                        static_gesture = self.detect_hand_gesture(landmarks)
                        
                        # Detect swipe gestures
                        swipe_gesture = self.detect_swipe_gesture(hand_center)
                        
                        # Process gestures
                        if swipe_gesture:
                            detected_gesture = swipe_gesture
                            self.process_gesture(swipe_gesture)
                        elif static_gesture:
                            detected_gesture = static_gesture
                            self.process_gesture(static_gesture)
                        
                        # Draw gesture info
                        if detected_gesture:
                            self.draw_gesture_info(frame, detected_gesture, hand_center)
                
                # Draw control dashboard
                self.draw_control_dashboard(frame)
                
                # Display frame
                cv2.imshow('🤚 Smart Office Gesture Control', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode
                    print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
                elif key == ord('l'):
                    self.show_landmarks = not self.show_landmarks
                    print(f"Landmark display: {'ON' if self.show_landmarks else 'OFF'}")
                elif key == ord('r'):
                    self.reset_system_state()
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\n👋 Gesture control stopped by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()
            print("✅ Gesture control cleanup completed")
    
    def reset_system_state(self):
        """Reset system state to defaults"""
        self.lights_on = True
        self.presentation_mode = False
        self.slide_number = 1
        self.volume_level = 50
        print("🔄 System state reset to defaults")
    
    def save_gesture_config(self, filename="gesture_config.json"):
        """Save gesture configuration to file"""
        config = {
            'gesture_commands': {
                gesture.value: {
                    'action': cmd.action,
                    'description': cmd.description,
                    'cooldown': cmd.cooldown
                }
                for gesture, cmd in self.gesture_commands.items()
            },
            'settings': {
                'gesture_threshold': self.gesture_threshold,
                'swipe_threshold': self.swipe_threshold,
                'swipe_time_window': self.swipe_time_window
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"💾 Gesture configuration saved: {filename}")

def main():
    """Main function for gesture control"""
    print("🤚 Smart Office Gesture Control System")
    print("=" * 60)
    
    try:
        controller = SmartOfficeGestureControl()
        controller.run_gesture_control()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have installed: pip install mediapipe pyautogui")

if __name__ == "__main__":
    main() 