@echo off
echo 🏃‍♂️ Human Activity Detection System Launcher
echo ================================================

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    echo 💡 Please install Python from https://python.org/downloads
    pause
    exit /b 1
)

echo ✅ Python detected

echo.
echo 🏃‍♂️ Choose Activity Detection Mode:
echo 1. 🏃‍♂️ Full Activity Detection (Recommended - with pose estimation)
echo 2. 🏃‍♂️ Basic Activity Detection (Works without MediaPipe)
echo 3. 🔧 Install Activity Detection Dependencies
echo 4. 🔧 Test Activity Dependencies
echo 5. 📊 Generate Activity Report (from database)
echo 6. ❌ Exit

set /p choice="🔤 Enter your choice (1-6): "

if "%choice%"=="1" (
    echo.
    echo 🏃‍♂️ Starting Full Activity Detection System...
    echo 📦 This mode uses advanced pose estimation for accurate activity recognition
    echo.
    echo 🎮 Activity Detection Controls:
    echo   Q - Quit
    echo   R - Generate activity report
    echo   S - Save activity snapshot
    echo   Z - Toggle zone visibility
    echo   A - Toggle activity dashboard
    echo   P - Toggle pose landmarks
    echo.
    echo 🏃‍♂️ Detected Activities:
    echo   • Sitting (green) - Person at desk/chair
    echo   • Standing (yellow) - Person standing still
    echo   • Walking (orange) - Person moving around
    echo   • Presenting (magenta) - Person in presentation area
    echo   • Typing (cyan) - Person actively typing
    echo   • Meeting (light blue) - Person in meeting room
    echo   • Break Time (gray) - Person in break area
    echo.
    echo 💡 Instructions:
    echo   - Position yourself in different office zones
    echo   - Try different activities (sitting, standing, walking)
    echo   - System will track and classify your activities
    echo   - Check the dashboard for real-time analytics
    echo.
    
    python human_activity_detector.py
    
) else if "%choice%"=="2" (
    echo.
    echo 🏃‍♂️ Starting Basic Activity Detection...
    echo 📦 This mode uses movement-based detection (no pose estimation required)
    echo.
    echo 🎯 Basic Detection Features:
    echo   - Movement speed analysis
    echo   - Zone-based activity classification
    echo   - Real-time activity tracking
    echo   - Activity analytics dashboard
    echo.
    
    echo Installing basic requirements...
    python -m pip install ultralytics opencv-python numpy pandas --quiet
    if %errorlevel% equ 0 (
        echo ✅ Basic requirements installed
        python human_activity_detector.py
    ) else (
        echo ❌ Failed to install basic requirements
        pause
    )
    
) else if "%choice%"=="3" (
    echo.
    echo 🔧 Installing Activity Detection Dependencies...
    echo 📦 Installing comprehensive activity detection packages...
    echo.
    
    echo Installing core requirements...
    python -m pip install --upgrade pip
    python -m pip install -r requirements_activity.txt
    
    if %errorlevel% equ 0 (
        echo ✅ Activity detection dependencies installed successfully!
        echo.
        echo 🎯 Installed Features:
        echo   ✅ Core object detection (YOLOv8)
        echo   ✅ Advanced pose estimation (MediaPipe)
        echo   ✅ Activity analytics (Pandas, Matplotlib)
        echo   ✅ Database logging (SQLite)
        echo   ✅ Enhanced visualizations
        echo.
        echo 💡 You can now run full activity detection (option 1)
        echo.
        echo 🎯 Quick Test:
        python -c "import cv2, numpy, ultralytics, mediapipe; print('✅ All imports successful!')"
    ) else (
        echo ⚠️  Some dependencies may have failed to install
        echo 💡 You can still use basic activity detection (option 2)
    )
    
    pause
    
) else if "%choice%"=="4" (
    echo.
    echo 🔍 Testing Activity Detection Dependencies...
    echo.
    
    echo Testing core detection...
    python -c "try: import cv2; print('✅ OpenCV: OK'); except: print('❌ OpenCV: MISSING')"
    python -c "try: from ultralytics import YOLO; print('✅ YOLO: OK'); except: print('❌ YOLO: MISSING')"
    python -c "try: import numpy; print('✅ NumPy: OK'); except: print('❌ NumPy: MISSING')"
    
    echo.
    echo Testing advanced pose detection...
    python -c "try: import mediapipe; print('✅ MediaPipe: OK - Full pose detection available'); except: print('❌ MediaPipe: MISSING - Basic detection only')"
    
    echo.
    echo Testing analytics libraries...
    python -c "try: import pandas; print('✅ Pandas: OK'); except: print('❌ Pandas: MISSING')"
    python -c "try: import matplotlib; print('✅ Matplotlib: OK'); except: print('❌ Matplotlib: MISSING')"
    
    echo.
    echo Testing camera access...
    python -c "import cv2; cap = cv2.VideoCapture(0); ret, frame = cap.read(); print('✅ Camera: OK' if ret else '❌ Camera: No frame'); cap.release()"
    
    echo.
    echo 📊 Activity Detection Capabilities:
    python -c "
import sys
try:
    import mediapipe
    print('🎯 FULL ACTIVITY DETECTION READY')
    print('   • Advanced pose estimation')
    print('   • Accurate activity classification')
    print('   • Detailed body landmark tracking')
except:
    print('🎯 BASIC ACTIVITY DETECTION READY')
    print('   • Movement-based classification')
    print('   • Zone-aware activity detection')
    print('   • Real-time activity tracking')
"
    
    pause
    
) else if "%choice%"=="5" (
    echo.
    echo 📊 Generating Activity Report from Database...
    echo.
    
    python -c "
import sqlite3
import os
from datetime import datetime

db_path = 'human_activities.db'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print('📊 ACTIVITY DATABASE REPORT')
    print('=' * 50)
    
    # Total activities logged
    cursor.execute('SELECT COUNT(*) FROM activity_log')
    total_activities = cursor.fetchone()[0]
    print(f'Total Activities Logged: {total_activities}')
    
    if total_activities > 0:
        # Activity breakdown
        cursor.execute('SELECT activity, COUNT(*) FROM activity_log GROUP BY activity ORDER BY COUNT(*) DESC')
        activities = cursor.fetchall()
        
        print('\nActivity Breakdown:')
        for activity, count in activities:
            percentage = (count / total_activities) * 100
            print(f'  {activity.title()}: {count} ({percentage:.1f}%)')
        
        # Recent activities
        cursor.execute('SELECT activity, timestamp FROM activity_log ORDER BY timestamp DESC LIMIT 10')
        recent = cursor.fetchall()
        
        print('\nRecent Activities:')
        for activity, timestamp in recent:
            dt = datetime.fromtimestamp(timestamp)
            print(f'  {dt.strftime(\"%H:%M:%S\")} - {activity.title()}')
    else:
        print('No activities logged yet. Run activity detection first!')
    
    conn.close()
else:
    print('❌ No activity database found.')
    print('💡 Run activity detection first to generate data.')
"
    
    pause
    
) else if "%choice%"=="6" (
    echo.
    echo 👋 Goodbye!
    exit /b 0
    
) else (
    echo.
    echo ❌ Invalid choice. Please run the script again.
    pause
    exit /b 1
)

:: Keep window open if there was an error
if %errorlevel% neq 0 (
    echo.
    echo ❌ Activity detection encountered an error
    echo 💡 Try installing dependencies first (option 3)
    echo 💡 Or test what works with option 4
    pause
)

echo.
echo 👋 Human activity detection stopped
pause 