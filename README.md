# 🤖 Object + Human + Animal Detection System

A real-time computer vision system that detects **humans**, **animals**, and **objects** using your webcam with YOLOv8.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)

## 🎯 Features

- **Real-time detection** via webcam
- **Color-coded bounding boxes**:
  - 🟢 Green for humans
  - 🟠 Orange for animals  
  - 🔵 Blue for objects
- **Live console output** with detection counts
- **Optimized performance** with configurable confidence thresholds
- **Easy setup** with automatic model downloading

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** installed
- **Webcam** connected to your computer
- **Internet connection** (for first-time model download)

### Installation

1. **Clone or download this project**
   ```bash
   # If using git
   git clone <repository-url>
   cd object-detection
   
   # Or download and extract the files to a folder
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the detection system**
   ```bash
   python detect.py
   ```

That's it! The YOLOv8 model will download automatically on the first run.

## 🎮 Usage

### Running the Program

```bash
python detect.py
```

### Controls

- **Q key**: Quit the program
- **Ctrl+C**: Emergency stop

### What You'll See

1. **Video window** with:
   - Live webcam feed
   - Colored bounding boxes around detected objects
   - Confidence scores for each detection
   - Real-time count overlay

2. **Console output** with:
   - Detection summaries every second
   - Object counts and types
   - System status messages

### Example Output

```
🤖 Object + Human + Animal Detection System
==================================================
🚀 Loading YOLOv8 model...
✅ Model loaded successfully!
🎥 Starting webcam...
✅ Webcam opened successfully!
🔍 Starting real-time detection...

--- Frame 156 ---
👤 Humans detected: 1
📦 Object detected: bottle (x1)
📦 Object detected: cell phone (x1)

--- Frame 187 ---
👤 Humans detected: 2
🐾 Animal detected: cat (x1)
📦 Object detected: laptop (x1)
```

## 🛠️ Technical Details

### Detected Categories

**Humans:**
- person

**Animals:**
- bird, cat, dog, horse, sheep, cow
- elephant, bear, zebra, giraffe

**Objects:**
- 70+ everyday objects (bottles, phones, cars, furniture, etc.)

### Performance Settings

- **Resolution**: 640x480 (optimized for real-time)
- **FPS**: 30 (configurable)
- **Confidence threshold**: 0.5 (50% minimum confidence)
- **Model**: YOLOv8n (nano - fastest version)

### File Structure

```
object-detection/
├── detect.py           # Main detection script
├── requirements.txt    # Python dependencies
├── README.md          # This documentation
└── yolov8n.pt         # Downloaded automatically
```

## 🔧 Customization

### Adjust Confidence Threshold

In `detect.py`, line 75:
```python
if confidence > 0.5:  # Change 0.5 to your preferred threshold
```

### Change Video Resolution

In `detect.py`, lines 135-136:
```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Change width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Change height
```

### Add New Animal Categories

In `detect.py`, lines 16-19:
```python
self.animal_classes = [
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 
    'bear', 'zebra', 'giraffe'
    # Add more animals here
]
```

## 🐛 Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| **"Could not open webcam"** | • Check webcam connection<br>• Close other apps using camera<br>• Try different camera index (change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`) |
| **"No module named 'ultralytics'"** | Run: `pip install ultralytics` |
| **"No module named 'cv2'"** | Run: `pip install opencv-python` |
| **Slow performance** | • Lower video resolution<br>• Increase confidence threshold<br>• Use GPU if available |
| **Model download fails** | • Check internet connection<br>• Try running again (sometimes temporary) |
| **Poor detection accuracy** | • Ensure good lighting<br>• Position objects clearly in frame<br>• Lower confidence threshold |

### System Requirements

- **RAM**: 4GB minimum, 8GB recommended
- **CPU**: Any modern processor (Intel i3 or equivalent)
- **GPU**: Optional (CUDA-compatible for faster processing)
- **Webcam**: Any USB or built-in camera

### Performance Tips

1. **Close other applications** using your webcam
2. **Ensure good lighting** for better detection
3. **Position camera at eye level** for optimal human detection
4. **Use a stable surface** to avoid shaky footage

## 🚀 Advanced Features

### Adding Sound Alerts

Add this to the detection loop:
```python
import pygame
pygame.mixer.init()

# In the detection loop
if detections['humans']:
    pygame.mixer.Sound('alert.wav').play()
```

### Saving Detection Screenshots

Add this to save images when objects are detected:
```python
import datetime

if detections['humans'] or detections['animals']:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(f"detection_{timestamp}.jpg", annotated_frame)
```

### Web Interface

For a web-based interface, check out the advanced examples using **Gradio** or **Streamlit**.

## 📦 Dependencies

- **ultralytics**: YOLOv8 implementation
- **opencv-python**: Computer vision and webcam handling
- **numpy**: Numerical computations
- **pillow**: Image processing
- **torch**: Deep learning framework
- **torchvision**: Computer vision utilities

## 🤝 Contributing

Found a bug or want to add a feature? Feel free to:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open source. YOLOv8 is provided by Ultralytics under the GPL-3.0 license.

## 🙏 Acknowledgments

- **Ultralytics** for the amazing YOLOv8 model
- **OpenCV** community for computer vision tools
- **YOLO creators** for revolutionizing object detection

---

**Happy detecting!** 🎯

For questions or issues, please check the troubleshooting section above or create an issue in the repository. 