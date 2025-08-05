@echo off
echo 🏢 Smart Office Computer Vision Suite - Ultimate Launcher
echo ============================================================

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
echo 🏢 Choose Smart Office System:
echo 1. 🏢 Smart Office Monitoring (Occupancy + Zone tracking)
echo 2. 👥 Smart Attendance System (Face recognition)
echo 3. 🤚 Gesture Control System (Hand gesture automation)
echo 4. 🎮 Enhanced Desktop Detection (All features)
echo 5. 🌐 Web Interface (Browser dashboard)
echo 6. 🔧 Install Smart Office Dependencies
echo 7. 📊 Generate Office Reports
echo 8. ❌ Exit

set /p choice="🔤 Enter your choice (1-8): "

if "%choice%"=="1" (
    echo.
    echo 🏢 Starting Smart Office Monitoring...
    echo 📦 Installing requirements...
    
    python -m pip install -r requirements_smart_office.txt --quiet
    if %errorlevel% neq 0 (
        echo ⚠️  Some packages may have failed to install, continuing anyway...
    )
    
    echo 🏢 Launching smart office monitoring...
    echo.
    echo 🎮 Smart Office Controls:
    echo   Q - Quit
    echo   S - Save office snapshot
    echo   R - Generate occupancy report
    echo   Z - Toggle zone visibility
    echo   P - Toggle posture monitoring
    echo   D - Export office data
    echo.
    
    python smart_office_detector.py
    
) else if "%choice%"=="2" (
    echo.
    echo 👥 Starting Smart Attendance System...
    echo 📦 Installing face recognition requirements...
    
    python -m pip install face-recognition dlib pandas --quiet
    python -m pip install -r requirements_smart_office.txt --quiet
    if %errorlevel% neq 0 (
        echo ⚠️  Some packages may have failed to install, continuing anyway...
    )
    
    echo 👥 Launching attendance system...
    echo.
    echo 🎮 Attendance Controls:
    echo   Q - Quit
    echo   A - Add new employee (requires setup)
    echo   R - Generate attendance report
    echo   E - Export attendance data
    echo   C - Clear daily records
    echo.
    echo 💡 Note: Place employee photos in 'employee_photos/' folder
    echo    to add employees to the system
    echo.
    
    python smart_attendance.py
    
) else if "%choice%"=="3" (
    echo.
    echo 🤚 Starting Gesture Control System...
    echo 📦 Installing gesture control requirements...
    
    python -m pip install mediapipe pyautogui --quiet
    python -m pip install -r requirements_smart_office.txt --quiet
    if %errorlevel% neq 0 (
        echo ⚠️  Some packages may have failed to install, continuing anyway...
    )
    
    echo 🤚 Launching gesture control...
    echo.
    echo 🎮 Gesture Controls:
    echo   Q - Quit
    echo   D - Toggle debug mode
    echo   L - Toggle landmark display
    echo   R - Reset system state
    echo.
    echo 🤚 Available Gestures:
    echo   ✋ Open Palm - Toggle Lights
    echo   ✊ Closed Fist - Emergency Stop
    echo   ✌️ Peace - Start Presentation
    echo   👍 Thumbs Up - Volume Up
    echo   👎 Thumbs Down - Volume Down
    echo   👆 Pointing - Laser Pointer
    echo   👌 OK Sign - Confirm
    echo   ➡️ Swipe Right - Next Slide
    echo   ⬅️ Swipe Left - Previous Slide
    echo.
    
    python gesture_control.py
    
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
    python -m pip install -r requirements_smart_office.txt --quiet
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
    echo 🔧 Installing Smart Office Dependencies...
    echo 📦 This may take several minutes...
    echo.
    
    python -m pip install --upgrade pip
    echo Installing core dependencies...
    python -m pip install ultralytics opencv-python numpy
    echo Installing face recognition...
    python -m pip install face-recognition dlib
    echo Installing gesture control...
    python -m pip install mediapipe pyautogui
    echo Installing all smart office requirements...
    python -m pip install -r requirements_smart_office.txt
    
    if %errorlevel% equ 0 (
        echo ✅ All smart office dependencies installed successfully!
        echo 💡 You can now run any smart office system
        echo.
        echo 🎯 Quick Test:
        python -c "import cv2, face_recognition, mediapipe, ultralytics; print('✅ All imports successful!')"
    ) else (
        echo ⚠️  Some dependencies may have failed to install
        echo 💡 Try running as administrator or check your internet connection
        echo 💡 For face recognition issues on Windows, install Visual C++ Build Tools
    )
    
    pause
    
) else if "%choice%"=="7" (
    echo.
    echo 📊 Generating Office Reports...
    echo.
    
    echo 📈 Available Reports:
    echo 1. Occupancy Report
    echo 2. Attendance Report  
    echo 3. Gesture Usage Report
    echo 4. Combined Office Analytics
    echo.
    
    set /p report_choice="🔤 Choose report type (1-4): "
    
    if "!report_choice!"=="1" (
        echo 📊 Generating occupancy report...
        python -c "from smart_office_detector import SmartOfficeDetector; detector = SmartOfficeDetector(); detector.generate_occupancy_report()"
    ) else if "!report_choice!"=="2" (
        echo 📊 Generating attendance report...
        python -c "from smart_attendance import SmartAttendanceSystem; system = SmartAttendanceSystem(); system.generate_attendance_report()"
    ) else (
        echo 📊 Generating combined analytics...
        echo 💡 Feature coming soon - individual reports available
    )
    
    pause
    
) else if "%choice%"=="8" (
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
    echo ❌ System encountered an error
    echo 💡 Try installing dependencies first (option 6)
    pause
)

echo.
echo 👋 Smart office system stopped
pause 