#!/usr/bin/env python3
"""
Setup script for Object Detection System
Automatically installs dependencies and tests the system
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}")
        print(f"Command: {command}")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_webcam():
    """Test if webcam is accessible"""
    print("📷 Testing webcam access...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            cv2.destroyAllWindows()
            if ret:
                print("✅ Webcam is working correctly!")
                return True
            else:
                print("⚠️  Webcam detected but couldn't read frames")
                return False
        else:
            print("❌ Could not access webcam")
            print("💡 Make sure your webcam is connected and not used by other apps")
            return False
    except ImportError:
        print("⚠️  OpenCV not installed yet, will test webcam after installation")
        return True
    except Exception as e:
        print(f"❌ Error testing webcam: {e}")
        return False

def main():
    """Main setup function"""
    print("🤖 Object Detection System Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Upgrade pip
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        print("⚠️  Failed to upgrade pip, continuing anyway...")
    
    # Install requirements
    if os.path.exists("requirements.txt"):
        if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing dependencies"):
            print("❌ Failed to install dependencies")
            sys.exit(1)
    else:
        print("❌ requirements.txt not found")
        print("📝 Installing dependencies manually...")
        packages = [
            "ultralytics>=8.0.0",
            "opencv-python>=4.8.0", 
            "numpy>=1.21.0",
            "pillow>=8.0.0",
            "torch>=1.9.0",
            "torchvision>=0.10.0"
        ]
        for package in packages:
            if not run_command(f"{sys.executable} -m pip install {package}", f"Installing {package}"):
                print(f"❌ Failed to install {package}")
                sys.exit(1)
    
    # Test webcam
    check_webcam()
    
    # Test import of main dependencies
    print("🧪 Testing imports...")
    try:
        import cv2
        import ultralytics
        import numpy as np
        print("✅ All dependencies imported successfully!")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\n🚀 To start detection, run:")
    print("   python detect.py")
    print("\n📖 For more information, check README.md")

if __name__ == "__main__":
    main() 