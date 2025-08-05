import cv2
import numpy as np
import time
import os
from collections import deque
from enum import Enum

class SimpleGestureType(Enum):
    HAND_DETECTED = "hand_detected"
    NO_HAND = "no_hand"
    MOVEMENT_LEFT = "movement_left" 
    MOVEMENT_RIGHT = "movement_right"
    MOVEMENT_UP = "movement_up"
    MOVEMENT_DOWN = "movement_down"

class SimpleGestureControl:
    def __init__(self):
        """Initialize simplified gesture control system"""
        print("🤚 Initializing Simple Gesture Control...")
        
        # Hand detection using color and motion
        self.hand_cascade = None
        try:
            # Try to load hand cascade if available
            cascade_path = cv2.data.haarcascades + 'haarcascade_hand.xml'
            if os.path.exists(cascade_path):
                self.hand_cascade = cv2.CascadeClassifier(cascade_path)
        except:
            pass
        
        # Motion detection for gestures
        self.previous_frame = None
        self.hand_positions = deque(maxlen=20)
        self.gesture_history = deque(maxlen=10)
        
        # Gesture recognition parameters
        self.movement_threshold = 30  # Minimum movement for gesture
        self.gesture_cooldown = 2.0   # Seconds between gestures
        self.last_gesture_time = 0
        
        # Office automation states
        self.lights_on = True
        self.presentation_mode = False
        self.slide_number = 1
        self.volume_level = 50
        
        # UI theme
        self.theme = {
            'detected': (0, 255, 0),        # Green
            'movement': (255, 255, 0),      # Yellow
            'activated': (255, 100, 255),   # Magenta
            'text': (255, 255, 255),        # White
            'background': (40, 40, 50),     # Dark gray
        }
        
        print("✅ Simple gesture control initialized!")
    
    def detect_hand_region(self, frame):
        """Detect hand region using color detection and motion"""
        # Convert to HSV for skin color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define skin color range (adjust these values as needed)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create mask for skin color
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3,3), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (likely to be hand)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Only consider if area is significant
            if cv2.contourArea(largest_contour) > 1000:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(largest_contour)
                center = (x + w//2, y + h//2)
                
                return {
                    'detected': True,
                    'center': center,
                    'bbox': (x, y, w, h),
                    'contour': largest_contour
                }
        
        return {'detected': False}
    
    def detect_motion_gesture(self, current_center):
        """Detect gesture based on hand movement"""
        if len(self.hand_positions) < 5:
            self.hand_positions.append(current_center)
            return None
        
        self.hand_positions.append(current_center)
        
        # Calculate movement vector from first to last position
        start_pos = self.hand_positions[0]
        end_pos = self.hand_positions[-1]
        
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance > self.movement_threshold:
            # Determine direction
            if abs(dx) > abs(dy):  # Horizontal movement
                if dx > 0:
                    self.hand_positions.clear()
                    return SimpleGestureType.MOVEMENT_RIGHT
                else:
                    self.hand_positions.clear()
                    return SimpleGestureType.MOVEMENT_LEFT
            else:  # Vertical movement
                if dy < 0:  # Up (y decreases going up)
                    self.hand_positions.clear()
                    return SimpleGestureType.MOVEMENT_UP
                else:
                    self.hand_positions.clear()
                    return SimpleGestureType.MOVEMENT_DOWN
        
        return None
    
    def process_gesture(self, gesture_type):
        """Process detected gesture and execute command"""
        if not gesture_type:
            return
        
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_gesture_time < self.gesture_cooldown:
            return
        
        # Execute command based on gesture
        try:
            if gesture_type == SimpleGestureType.MOVEMENT_LEFT:
                self.previous_slide()
                self.last_gesture_time = current_time
            elif gesture_type == SimpleGestureType.MOVEMENT_RIGHT:
                self.next_slide()
                self.last_gesture_time = current_time
            elif gesture_type == SimpleGestureType.MOVEMENT_UP:
                self.volume_up()
                self.last_gesture_time = current_time
            elif gesture_type == SimpleGestureType.MOVEMENT_DOWN:
                self.volume_down()
                self.last_gesture_time = current_time
                
        except Exception as e:
            print(f"❌ Error executing gesture command: {e}")
    
    # Office control commands (simplified)
    def toggle_lights(self):
        """Toggle office lights"""
        self.lights_on = not self.lights_on
        status = "ON" if self.lights_on else "OFF"
        print(f"💡 Lights turned {status}")
    
    def volume_up(self):
        """Increase system volume"""
        self.volume_level = min(100, self.volume_level + 10)
        print(f"🔊 Volume increased to {self.volume_level}%")
    
    def volume_down(self):
        """Decrease system volume"""
        self.volume_level = max(0, self.volume_level - 10)
        print(f"🔉 Volume decreased to {self.volume_level}%")
    
    def next_slide(self):
        """Go to next slide"""
        self.slide_number += 1
        print(f"➡️ Next slide ({self.slide_number})")
    
    def previous_slide(self):
        """Go to previous slide"""
        self.slide_number = max(1, self.slide_number - 1)
        print(f"⬅️ Previous slide ({self.slide_number})")
    
    def draw_hand_detection(self, frame, hand_info):
        """Draw hand detection results"""
        if hand_info['detected']:
            x, y, w, h = hand_info['bbox']
            center = hand_info['center']
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), self.theme['detected'], 2)
            
            # Draw center point
            cv2.circle(frame, center, 10, self.theme['detected'], -1)
            
            # Draw hand label
            cv2.putText(frame, "Hand Detected", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.theme['detected'], 2)
            
            # Draw contour
            if 'contour' in hand_info:
                cv2.drawContours(frame, [hand_info['contour']], -1, self.theme['movement'], 2)
    
    def draw_gesture_trail(self, frame):
        """Draw movement trail"""
        if len(self.hand_positions) > 1:
            points = list(self.hand_positions)
            for i in range(1, len(points)):
                # Fade trail from newest to oldest
                alpha = i / len(points)
                thickness = max(1, int(alpha * 5))
                
                cv2.line(frame, points[i-1], points[i], self.theme['movement'], thickness)
    
    def draw_control_dashboard(self, frame):
        """Draw simple control dashboard"""
        height, width = frame.shape[:2]
        
        # Dashboard panel
        panel_width = 350
        panel_height = 250
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
        cv2.putText(frame, "🤚 SIMPLE GESTURE CONTROL", (panel_x + 20, panel_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.theme['text'], 2)
        
        # System status
        y_offset = panel_y + 60
        status_items = [
            f"Lights: {'ON' if self.lights_on else 'OFF'}",
            f"Volume: {self.volume_level}%",
            f"Slide: {self.slide_number}",
        ]
        
        for item in status_items:
            color = self.theme['detected'] if 'ON' in item else self.theme['text']
            cv2.putText(frame, item, (panel_x + 20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 25
        
        # Available gestures
        y_offset += 20
        cv2.putText(frame, "Available Gestures:", (panel_x + 20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.theme['text'], 2)
        
        y_offset += 25
        gesture_list = [
            "➡️ Move Hand Right - Next Slide",
            "⬅️ Move Hand Left - Previous Slide", 
            "⬆️ Move Hand Up - Volume Up",
            "⬇️ Move Hand Down - Volume Down",
        ]
        
        for gesture in gesture_list:
            cv2.putText(frame, gesture, (panel_x + 30, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.theme['text'], 1)
            y_offset += 18
    
    def run_gesture_control(self):
        """Main gesture control loop"""
        print("🤚 Starting Simple Gesture Control System...")
        
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
        
        print("✅ Simple gesture control system running!")
        print("🎮 Controls:")
        print("  Q - Quit")
        print("  R - Reset system state")
        print("  L - Toggle lights manually")
        print("\n🤚 Gesture Instructions:")
        print("  - Show your hand to the camera")
        print("  - Move hand left/right to change slides")
        print("  - Move hand up/down to control volume\n")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Detect hand region
                hand_info = self.detect_hand_region(frame)
                
                detected_gesture = None
                
                if hand_info['detected']:
                    # Detect motion gestures
                    motion_gesture = self.detect_motion_gesture(hand_info['center'])
                    
                    if motion_gesture:
                        detected_gesture = motion_gesture
                        self.process_gesture(motion_gesture)
                    
                    # Draw hand detection
                    self.draw_hand_detection(frame, hand_info)
                else:
                    # Clear hand positions if no hand detected
                    self.hand_positions.clear()
                
                # Draw gesture trail
                self.draw_gesture_trail(frame)
                
                # Draw control dashboard
                self.draw_control_dashboard(frame)
                
                # Show last gesture
                if detected_gesture:
                    gesture_text = f"Gesture: {detected_gesture.value.replace('_', ' ').title()}"
                    cv2.putText(frame, gesture_text, (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, self.theme['activated'], 2)
                
                # Display frame
                cv2.imshow('🤚 Simple Gesture Control', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.reset_system_state()
                elif key == ord('l'):
                    self.toggle_lights()
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\n👋 Gesture control stopped by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Gesture control cleanup completed")
    
    def reset_system_state(self):
        """Reset system state to defaults"""
        self.lights_on = True
        self.slide_number = 1
        self.volume_level = 50
        print("🔄 System state reset to defaults")

def main():
    """Main function for simple gesture control"""
    print("🤚 Simple Gesture Control System")
    print("=" * 50)
    
    try:
        controller = SimpleGestureControl()
        controller.run_gesture_control()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have OpenCV installed: pip install opencv-python")

if __name__ == "__main__":
    main() 