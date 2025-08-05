@echo off
echo 🤖 Object Detection System - Windows Launcher
echo ================================================

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

:: Check if this is first run (no yolov8n.pt file)
if not exist "yolov8n.pt" (
    echo 🔧 First-time setup detected
    echo 📦 Installing dependencies...
    
    :: Run setup script
    python setup.py
    if %errorlevel% neq 0 (
        echo ❌ Setup failed
        pause
        exit /b 1
    )
) else (
    echo 🚀 System already set up
)

:: Run the detection system
echo 🎥 Starting Object Detection...
echo ❌ Press 'Q' in the video window to quit
echo.
python detect.py

:: Keep window open if there was an error
if %errorlevel% neq 0 (
    echo.
    echo ❌ Detection system encountered an error
    pause
)

echo.
echo 👋 Detection system stopped
pause 