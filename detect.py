import cv2
from ultralytics import YOLO
import numpy as np
from collections import Counter
import time

class ObjectDetector:
    def __init__(self):
        """Initialize the detector with YOLOv8 model"""
        print("🚀 Loading YOLOv8 model...")
        self.model = YOLO('yolov8n.pt')  # Downloads automatically on first run
        
        # Define categories for better organization
        self.human_classes = ['person']
        self.animal_classes = [
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 
            'bear', 'zebra', 'giraffe'
        ]
        
        # Colors for different categories (BGR format)
        self.colors = {
            'human': (0, 255, 0),    # Green
            'animal': (0, 165, 255), # Orange
            'object': (255, 0, 0)    # Blue
        }
        
        print("✅ Model loaded successfully!")
        
    def categorize_detection(self, class_name):
        """Categorize detection into human, animal, or object"""
        if class_name in self.human_classes:
            return 'human'
        elif class_name in self.animal_classes:
            return 'animal'
        else:
            return 'object'
    
    def draw_detection(self, frame, box, class_name, confidence, category):
        """Draw bounding box and label on frame"""
        x1, y1, x2, y2 = map(int, box)
        color = self.colors[category]
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # Draw label background
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def process_frame(self, frame):
        """Process a single frame and return annotated frame with detections"""
        # Run YOLO detection
        results = self.model(frame, verbose=False)
        
        detections = {
            'humans': [],
            'animals': [],
            'objects': []
        }
        
        # Process each detection
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get detection info
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy()
                    
                    # Filter by confidence threshold
                    if confidence > 0.5:
                        category = self.categorize_detection(class_name)
                        
                        # Store detection
                        detection_info = {
                            'class': class_name,
                            'confidence': confidence,
                            'box': bbox
                        }
                        
                        if category == 'human':
                            detections['humans'].append(detection_info)
                        elif category == 'animal':
                            detections['animals'].append(detection_info)
                        else:
                            detections['objects'].append(detection_info)
                        
                        # Draw on frame
                        self.draw_detection(frame, bbox, class_name, confidence, category)
        
        return frame, detections
    
    def print_detections(self, detections):
        """Print detection summary to console"""
        if detections['humans']:
            print(f"👤 Humans detected: {len(detections['humans'])}")
            
        if detections['animals']:
            animal_types = [d['class'] for d in detections['animals']]
            animal_counts = Counter(animal_types)
            for animal, count in animal_counts.items():
                print(f"🐾 Animal detected: {animal} (x{count})")
                
        if detections['objects']:
            object_types = [d['class'] for d in detections['objects']]
            object_counts = Counter(object_types)
            for obj, count in object_counts.items():
                print(f"📦 Object detected: {obj} (x{count})")
    
    def add_info_overlay(self, frame, detections):
        """Add information overlay to the frame"""
        height, width = frame.shape[:2]
        
        # Create info text
        info_lines = []
        if detections['humans']:
            info_lines.append(f"Humans: {len(detections['humans'])}")
        if detections['animals']:
            info_lines.append(f"Animals: {len(detections['animals'])}")
        if detections['objects']:
            info_lines.append(f"Objects: {len(detections['objects'])}")
        
        # Draw semi-transparent background for info
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 30 + len(info_lines) * 25), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw info text
        cv2.putText(frame, "Real-time Detection", (15, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (15, 55 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run_detection(self):
        """Main detection loop using webcam"""
        print("🎥 Starting webcam...")
        
        # Try different camera indices and backends
        cap = None
        camera_index = None
        
        for i in range(3):
            print(f"Testing camera index {i}...")
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # DirectShow backend for Windows
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"✅ Camera {i} working!")
                    camera_index = i
                    break
                else:
                    cap.release()
            else:
                cap.release() if cap else None
        
        if camera_index is None:
            print("❌ Error: Could not open any webcam")
            print("💡 Trying alternative: Using demo with test image")
            self.run_demo_with_image()
            return
        
        print("✅ Webcam opened successfully!")
        print("🔍 Starting real-time detection...")
        print("📝 Detection results will appear below:")
        print("❌ Press 'Q' to quit\n")
        
        # Set webcam properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        frame_count = 0
        last_print_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Could not read frame from webcam")
                    break
                
                # Process frame
                annotated_frame, detections = self.process_frame(frame)
                
                # Add info overlay
                self.add_info_overlay(annotated_frame, detections)
                
                # Print detections every 30 frames (roughly every second)
                current_time = time.time()
                if current_time - last_print_time > 1.0 and (
                    detections['humans'] or detections['animals'] or detections['objects']
                ):
                    print(f"\n--- Frame {frame_count} ---")
                    self.print_detections(detections)
                    last_print_time = current_time
                
                # Display frame
                cv2.imshow('Object + Human + Animal Detection', annotated_frame)
                
                # Check for quit command
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n👋 Stopping detection...")
                    break
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\n👋 Detection stopped by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Cleanup completed")
    
    def run_demo_with_image(self):
        """Demo detection using a sample image when webcam is not available"""
        print("🖼️  Running demo with sample image...")
        
        # Create a simple test image
        import numpy as np
        
        # Create a test image with text
        demo_image = np.ones((480, 640, 3), dtype=np.uint8) * 50
        
        # Add some text to the image
        cv2.putText(demo_image, "DEMO MODE - No Webcam Detected", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(demo_image, "This is a test image for detection", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(demo_image, "Press 'Q' to quit", (50, 400), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add some basic shapes that might be detected
        cv2.rectangle(demo_image, (200, 150), (400, 300), (100, 100, 255), 2)
        cv2.circle(demo_image, (500, 200), 50, (255, 100, 100), 2)
        
        print("✅ Demo image created")
        print("🔍 Running detection on demo image...")
        print("❌ Press 'Q' to quit demo\n")
        
        frame_count = 0
        
        try:
            while True:
                # Process the demo image
                annotated_frame, detections = self.process_frame(demo_image.copy())
                
                # Add info overlay
                self.add_info_overlay(annotated_frame, detections)
                
                # Add frame counter
                cv2.putText(annotated_frame, f"Demo Frame: {frame_count}", (10, 450), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Print detections occasionally
                if frame_count % 60 == 0 and detections:  # Every 2 seconds at 30fps
                    print(f"\n--- Demo Frame {frame_count} ---")
                    self.print_detections(detections)
                
                # Display frame
                cv2.imshow('Object Detection Demo (No Webcam)', annotated_frame)
                
                # Check for quit command
                if cv2.waitKey(33) & 0xFF == ord('q'):  # ~30fps
                    print("\n👋 Stopping demo...")
                    break
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\n👋 Demo stopped by user")
        
        finally:
            cv2.destroyAllWindows()
            print("✅ Demo completed")

def main():
    """Main function"""
    print("🤖 Object + Human + Animal Detection System")
    print("=" * 50)
    
    try:
        detector = ObjectDetector()
        detector.run_detection()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have installed all requirements:")
        print("   pip install ultralytics opencv-python")

if __name__ == "__main__":
    main() 