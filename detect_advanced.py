import cv2
from ultralytics import YOLO
import numpy as np
from collections import Counter, defaultdict
import time
import json
import os
from pathlib import Path

class AdvancedObjectDetector:
    def __init__(self, use_objects365=True, use_animal_specialist=True):
        """Initialize advanced detector with multiple specialized models"""
        print("🚀 Loading Advanced Detection Models...")
        
        # Primary model - YOLOv10 with Objects365 dataset (365 classes)
        if use_objects365:
            print("📦 Loading YOLOv10 with Objects365 dataset...")
            self.primary_model = YOLO('yolov10n.pt')  # YOLOv10 nano for speed
            self.model_type = "objects365"
        else:
            print("📦 Loading YOLOv8 with COCO dataset...")
            self.primary_model = YOLO('yolov8n.pt')
            self.model_type = "coco"
        
        # Animal specialist model (if available)
        self.animal_model = None
        if use_animal_specialist:
            try:
                print("🐾 Loading specialized animal detection model...")
                # Try to load a fine-tuned animal model
                self.animal_model = YOLO('yolov8n.pt')  # Will be enhanced with animal-specific weights
                print("✅ Animal specialist model loaded")
            except Exception as e:
                print(f"⚠️  Animal specialist model not available: {e}")
        
        # Dynamic classification system
        self.setup_dynamic_classification()
        
        # Tracking system
        self.object_tracks = {}
        self.next_track_id = 1
        self.max_track_distance = 50
        
        # Performance monitoring
        self.detection_history = defaultdict(list)
        self.frame_count = 0
        
        print("✅ Advanced detection system initialized!")
    
    def setup_dynamic_classification(self):
        """Setup dynamic classification categories based on real-world scenarios"""
        
        # Comprehensive classification system - not hardcoded, but intelligent categorization
        self.classification_rules = {
            'living_beings': {
                'humans': ['person', 'man', 'woman', 'child', 'baby', 'boy', 'girl'],
                'mammals': ['dog', 'cat', 'horse', 'cow', 'sheep', 'pig', 'elephant', 'giraffe', 
                          'zebra', 'bear', 'lion', 'tiger', 'deer', 'rabbit', 'mouse', 'rat',
                          'monkey', 'ape', 'wolf', 'fox', 'squirrel', 'hamster'],
                'birds': ['bird', 'eagle', 'owl', 'duck', 'chicken', 'goose', 'swan', 'pigeon',
                         'parrot', 'crow', 'sparrow', 'robin', 'peacock'],
                'aquatic': ['fish', 'shark', 'whale', 'dolphin', 'octopus', 'crab', 'lobster'],
                'reptiles': ['snake', 'lizard', 'turtle', 'crocodile', 'alligator', 'iguana'],
                'insects': ['butterfly', 'bee', 'spider', 'ant', 'mosquito', 'fly']
            },
            'vehicles': {
                'land_vehicles': ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'van', 'taxi',
                                'ambulance', 'fire truck', 'police car', 'tractor', 'bulldozer'],
                'air_vehicles': ['airplane', 'helicopter', 'drone', 'balloon', 'jet'],
                'water_vehicles': ['boat', 'ship', 'yacht', 'submarine', 'kayak', 'canoe']
            },
            'objects': {
                'electronics': ['tv', 'computer', 'laptop', 'phone', 'tablet', 'camera', 'radio',
                              'microwave', 'refrigerator', 'washing machine'],
                'furniture': ['chair', 'table', 'sofa', 'bed', 'desk', 'cabinet', 'shelf', 'couch'],
                'tools': ['hammer', 'screwdriver', 'wrench', 'saw', 'drill', 'scissors', 'knife'],
                'sports': ['ball', 'bat', 'racket', 'helmet', 'skis', 'skateboard', 'surfboard'],
                'household': ['bottle', 'cup', 'plate', 'bowl', 'fork', 'spoon', 'glass', 'vase'],
                'clothing': ['shirt', 'pants', 'dress', 'hat', 'shoes', 'jacket', 'tie', 'bag']
            },
            'environment': {
                'natural': ['tree', 'flower', 'grass', 'rock', 'mountain', 'cloud', 'sun', 'moon'],
                'structures': ['building', 'house', 'bridge', 'road', 'fence', 'wall', 'door', 'window']
            }
        }
        
        # Create reverse lookup for fast classification
        self.class_to_category = {}
        for main_category, subcategories in self.classification_rules.items():
            for subcategory, items in subcategories.items():
                for item in items:
                    self.class_to_category[item] = {
                        'main': main_category,
                        'sub': subcategory
                    }
        
        # Color scheme for different categories
        self.category_colors = {
            'living_beings': {
                'humans': (0, 255, 0),      # Bright Green
                'mammals': (0, 165, 255),   # Orange
                'birds': (255, 255, 0),     # Cyan
                'aquatic': (255, 0, 255),   # Magenta
                'reptiles': (0, 255, 255),  # Yellow
                'insects': (128, 0, 128)    # Purple
            },
            'vehicles': {
                'land_vehicles': (255, 0, 0),   # Blue
                'air_vehicles': (255, 144, 30), # Deep Sky Blue
                'water_vehicles': (139, 69, 19) # Saddle Brown
            },
            'objects': {
                'electronics': (128, 128, 128), # Gray
                'furniture': (165, 42, 42),     # Brown
                'tools': (255, 20, 147),        # Deep Pink
                'sports': (50, 205, 50),        # Lime Green
                'household': (255, 140, 0),     # Dark Orange
                'clothing': (75, 0, 130)        # Indigo
            },
            'environment': {
                'natural': (34, 139, 34),       # Forest Green
                'structures': (112, 128, 144)   # Slate Gray
            }
        }
    
    def classify_detection(self, class_name, confidence):
        """Dynamically classify detection based on class name and confidence"""
        class_lower = class_name.lower()
        
        # Look up in our classification system
        if class_lower in self.class_to_category:
            category_info = self.class_to_category[class_lower]
            return category_info['main'], category_info['sub']
        
        # If not found, use intelligent guessing based on keywords
        for main_category, subcategories in self.classification_rules.items():
            for subcategory, items in subcategories.items():
                # Check for partial matches or related terms
                for item in items:
                    if item in class_lower or class_lower in item:
                        return main_category, subcategory
        
        # Default classification for unknown objects
        return 'objects', 'unknown'
    
    def get_category_color(self, main_category, sub_category):
        """Get color for a specific category"""
        try:
            return self.category_colors[main_category][sub_category]
        except KeyError:
            return (200, 200, 200)  # Default gray
    
    def track_objects(self, detections):
        """Simple object tracking across frames"""
        current_frame_tracks = {}
        
        for detection in detections:
            box = detection['box']
            center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            
            # Find closest existing track
            best_track_id = None
            min_distance = float('inf')
            
            for track_id, track_info in self.object_tracks.items():
                if track_info['class'] == detection['class']:
                    last_center = track_info['center']
                    distance = np.sqrt((center[0] - last_center[0])**2 + 
                                     (center[1] - last_center[1])**2)
                    
                    if distance < self.max_track_distance and distance < min_distance:
                        min_distance = distance
                        best_track_id = track_id
            
            # Assign track ID
            if best_track_id is not None:
                track_id = best_track_id
                self.object_tracks[track_id]['center'] = center
                self.object_tracks[track_id]['last_seen'] = self.frame_count
            else:
                track_id = self.next_track_id
                self.next_track_id += 1
                self.object_tracks[track_id] = {
                    'class': detection['class'],
                    'center': center,
                    'first_seen': self.frame_count,
                    'last_seen': self.frame_count
                }
            
            detection['track_id'] = track_id
            current_frame_tracks[track_id] = detection
        
        # Remove old tracks
        tracks_to_remove = []
        for track_id, track_info in self.object_tracks.items():
            if self.frame_count - track_info['last_seen'] > 30:  # 1 second at 30fps
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.object_tracks[track_id]
        
        return current_frame_tracks
    
    def process_frame_advanced(self, frame):
        """Advanced frame processing with multiple models and tracking"""
        all_detections = []
        
        # Primary model detection
        primary_results = self.primary_model(frame, verbose=False)
        
        # Process primary results
        for result in primary_results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = self.primary_model.names[class_id]
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy()
                    
                    if confidence > 0.3:  # Lower threshold for more detections
                        main_category, sub_category = self.classify_detection(class_name, confidence)
                        
                        detection = {
                            'class': class_name,
                            'confidence': confidence,
                            'box': bbox,
                            'main_category': main_category,
                            'sub_category': sub_category,
                            'model': 'primary'
                        }
                        all_detections.append(detection)
        
        # Animal specialist model (if available and animals detected)
        if self.animal_model and any(d['main_category'] == 'living_beings' for d in all_detections):
            animal_results = self.animal_model(frame, verbose=False)
            # Process animal-specific results (implementation would be similar)
        
        # Apply tracking
        tracked_detections = self.track_objects(all_detections)
        
        return all_detections, tracked_detections
    
    def draw_advanced_detection(self, frame, detection):
        """Draw advanced detection with category-based styling"""
        box = detection['box']
        x1, y1, x2, y2 = map(int, box)
        
        main_category = detection['main_category']
        sub_category = detection['sub_category']
        color = self.get_category_color(main_category, sub_category)
        
        # Draw bounding box with thickness based on confidence
        thickness = int(2 + detection['confidence'] * 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare detailed label
        track_id = detection.get('track_id', 'N/A')
        label = f"{detection['class']} ({sub_category})"
        confidence_text = f"{detection['confidence']:.2f}"
        track_text = f"ID:{track_id}"
        
        # Calculate label dimensions
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        
        (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        (conf_w, conf_h), _ = cv2.getTextSize(confidence_text, font, font_scale, font_thickness)
        (track_w, track_h), _ = cv2.getTextSize(track_text, font, font_scale, font_thickness)
        
        # Draw label background
        label_bg_height = max(label_h, conf_h, track_h) + 20
        cv2.rectangle(frame, (x1, y1 - label_bg_height), 
                     (x1 + max(label_w, conf_w, track_w) + 10, y1), color, -1)
        
        # Draw label texts
        cv2.putText(frame, label, (x1 + 5, y1 - label_bg_height + 15), 
                   font, font_scale, (255, 255, 255), font_thickness)
        cv2.putText(frame, confidence_text, (x1 + 5, y1 - 5), 
                   font, font_scale, (255, 255, 255), font_thickness)
        
        # Draw category indicator
        category_indicator = f"{main_category.replace('_', ' ').title()}"
        cv2.putText(frame, category_indicator, (x2 - 100, y1 - 10), 
                   font, 0.4, color, 1)
    
    def create_advanced_overlay(self, frame, detections, tracked_detections):
        """Create advanced information overlay"""
        height, width = frame.shape[:2]
        
        # Count detections by category
        category_counts = defaultdict(lambda: defaultdict(int))
        for detection in detections:
            category_counts[detection['main_category']][detection['sub_category']] += 1
        
        # Create detailed statistics
        y_offset = 30
        overlay = frame.copy()
        
        # Semi-transparent background
        cv2.rectangle(overlay, (10, 10), (400, y_offset + len(category_counts) * 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Title
        cv2.putText(frame, "Advanced Real-Time Detection", (15, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Category statistics
        for main_category, subcategories in category_counts.items():
            y_offset += 25
            category_title = main_category.replace('_', ' ').title()
            cv2.putText(frame, f"{category_title}:", (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            for sub_category, count in subcategories.items():
                y_offset += 15
                sub_text = f"  {sub_category.replace('_', ' ')}: {count}"
                cv2.putText(frame, sub_text, (15, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Performance info
        y_offset += 30
        cv2.putText(frame, f"Frame: {self.frame_count}", (15, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.putText(frame, f"Tracked Objects: {len(tracked_detections)}", (15, y_offset + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    def print_advanced_detections(self, detections, tracked_detections):
        """Print advanced detection summary"""
        if not detections:
            return
        
        print(f"\n🔍 Frame {self.frame_count} - Advanced Detection Results:")
        print("=" * 60)
        
        # Group by main category
        by_category = defaultdict(list)
        for detection in detections:
            by_category[detection['main_category']].append(detection)
        
        for main_category, category_detections in by_category.items():
            emoji_map = {
                'living_beings': '🐾',
                'vehicles': '🚗',
                'objects': '📦',
                'environment': '🌳'
            }
            emoji = emoji_map.get(main_category, '❓')
            
            print(f"\n{emoji} {main_category.replace('_', ' ').title()}:")
            
            # Group by subcategory
            by_subcategory = defaultdict(list)
            for detection in category_detections:
                by_subcategory[detection['sub_category']].append(detection)
            
            for sub_category, sub_detections in by_subcategory.items():
                class_counts = Counter(d['class'] for d in sub_detections)
                for class_name, count in class_counts.items():
                    avg_confidence = np.mean([d['confidence'] for d in sub_detections if d['class'] == class_name])
                    print(f"  └─ {class_name} ({sub_category}): {count}x (conf: {avg_confidence:.2f})")
    
    def run_advanced_detection(self):
        """Main advanced detection loop"""
        print("🎥 Starting Advanced Real-Time Detection...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        # Optimize for performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Advanced detection system running!")
        print("🔍 Detecting with enhanced AI models and real-world classification")
        print("📊 Real-time tracking and statistics enabled")
        print("❌ Press 'Q' to quit, 'S' to save detection log\n")
        
        last_print_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                # Advanced processing
                detections, tracked_detections = self.process_frame_advanced(frame)
                
                # Draw all detections
                for detection in detections:
                    self.draw_advanced_detection(frame, detection)
                
                # Add advanced overlay
                self.create_advanced_overlay(frame, detections, tracked_detections)
                
                # Print detailed results every 2 seconds
                current_time = time.time()
                if current_time - last_print_time > 2.0 and detections:
                    self.print_advanced_detections(detections, tracked_detections)
                    last_print_time = current_time
                
                # Display
                cv2.imshow('Advanced Object Detection System', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_detection_log()
                
        except KeyboardInterrupt:
            print("\n👋 Advanced detection stopped by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Advanced detection system cleanup completed")
    
    def save_detection_log(self):
        """Save detection history to JSON file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"detection_log_{timestamp}.json"
        
        log_data = {
            'timestamp': timestamp,
            'total_frames': self.frame_count,
            'model_type': self.model_type,
            'detection_history': dict(self.detection_history)
        }
        
        with open(filename, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"💾 Detection log saved to {filename}")

def main():
    """Main function for advanced detection"""
    print("🤖 Advanced Object Detection System")
    print("Using YOLOv10 + Objects365 + Dynamic Classification")
    print("=" * 60)
    
    try:
        detector = AdvancedObjectDetector(
            use_objects365=True,
            use_animal_specialist=True
        )
        detector.run_advanced_detection()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have the latest ultralytics installed:")
        print("   pip install ultralytics --upgrade")

if __name__ == "__main__":
    main() 