@echo off
echo 🏢 Smart Office Computer Vision Suite - FIXED Launcher
echo =====================================================

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    echo 💡 Please install Python from https://python.org/downloads
    echo    Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo ✅ Python detected

echo.
echo 🏢 Choose Working Smart Office System:
echo 1. 🏢 Smart Office Monitoring (WORKING - Basic dependencies only)
echo 2. 🤚 Simple Gesture Control (WORKING - Hand movement detection)
echo 3. 🏃‍♂️ Human Activity Detection (NEW - Track office activities)
echo 4. 🎮 Enhanced Desktop Detection (WORKING - All visual features)
echo 5. 🌐 Web Interface (WORKING - Browser dashboard)
echo 6. 🔧 Test Dependencies (Check what works)
echo 7. 🔧 Install Basic Dependencies
echo 8. ❌ Exit

set /p choice="🔤 Enter your choice (1-8): "

if "%choice%"=="1" (
    echo.
    echo 🏢 Starting Smart Office Monitoring (Simplified)...
    echo 📦 This version works with basic dependencies only
    echo.
    echo 🎮 Smart Office Controls:
    echo   Q - Quit
    echo   S - Save office snapshot
    echo   R - Generate occupancy report
    echo   Z - Toggle zone visibility
    echo   D - Export office data
    echo.
    
    python smart_office_simple.py
    
) else if "%choice%"=="2" (
    echo.
    echo 🤚 Starting Simple Gesture Control...
    echo 📦 This version uses basic hand detection
    echo.
    echo 🎮 Gesture Controls:
    echo   Q - Quit
    echo   R - Reset system state
    echo   L - Toggle lights manually
    echo.
    echo 🤚 Available Gestures:
    echo   ➡️ Move Hand Right - Next Slide
    echo   ⬅️ Move Hand Left - Previous Slide
    echo   ⬆️ Move Hand Up - Volume Up
    echo   ⬇️ Move Hand Down - Volume Down
    echo.
    echo 💡 Instructions:
    echo   - Show your hand clearly to the camera
    echo   - Move hand in clear directions for gestures
    echo   - Works best with good lighting
    echo.
    
    python gesture_control_simple.py
    
) else if "%choice%"=="3" (
    echo.
    echo 🏃‍♂️ Starting Human Activity Detection System...
    echo 📦 This system tracks and analyzes human activities in real-time
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
    
) else if "%choice%"=="4" (
    echo.
    echo 🎮 Starting Enhanced Desktop Detection...
    echo 📦 Installing enhanced requirements...
    
    python -m pip install -r requirements_enhanced.txt --quiet
    if %errorlevel% neq 0 (
        echo ⚠️  Some packages may have failed to install, continuing anyway...
    )
    
    echo 🎮 Launching enhanced detection...
    python detect_enhanced.py
    
) else if "%choice%"=="5" (
    echo.
    echo 🌐 Starting Web Interface...
    echo 📦 Installing web requirements...
    
    python -m pip install streamlit plotly --quiet
    if %errorlevel% neq 0 (
        echo ⚠️  Some packages may have failed to install, continuing anyway...
    )
    
    echo 🌐 Launching web interface...
    echo 💡 Your browser will open automatically
    echo 💡 If not, go to: http://localhost:8501
    echo.
    
    streamlit run web_interface.py
    
) else if "%choice%"=="6" (
    echo.
    echo 🔍 Testing Dependencies...
    echo.
    
    echo 📊 Basic dependencies test:
    python -c "try: import cv2, numpy, ultralytics; print('✅ Core systems: WORKING'); except: print('❌ Core systems: NEEDS INSTALLATION')"
    echo 🏃‍♂️ Activity detection test:
    python -c "try: import mediapipe; print('✅ Advanced pose detection: AVAILABLE'); except: print('⚠️ Advanced pose: BASIC MODE ONLY')"
    pause
    
) else if "%choice%"=="8" (
    echo.
    echo 🔧 Installing Basic Dependencies...
    echo 📦 Installing only working packages...
    echo.
    
    python -m pip install --upgrade pip
    echo Installing core dependencies...
    python -m pip install ultralytics opencv-python numpy pandas
    echo Installing enhanced UI packages...
    python -m pip install pygame matplotlib
    echo Installing activity detection packages...
    python -m pip install mediapipe pandas
    
    if %errorlevel% equ 0 (
        echo ✅ All dependencies installed successfully!
        echo 💡 You can now run all options (1-5)
        echo.
        echo 🎯 Quick Test:
        python -c "import cv2, numpy, ultralytics; print('✅ Core imports successful!')"
        python -c "try: import mediapipe; print('✅ Activity detection ready!'); except: print('⚠️ Activity detection: basic mode only')"
    ) else (
        echo ⚠️  Some dependencies may have failed to install
        echo 💡 Try running as administrator or check your internet connection
    )
    
    pause
    
) else (
    echo.
    echo ❌ Invalid choice. Please run the script again.
    pause
    exit /b 1
)

:: Keep window open if there was an error
if %errorlevel% neq 0 (
    echo.
    echo ❌ System encountered an error
    echo 💡 Try installing basic dependencies first (option 6)
    echo 💡 Or test what works with option 5
    pause
)

echo.
echo 👋 Smart office system stopped
pause 