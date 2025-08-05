@echo off
echo 🤖 Advanced Object Detection System - Windows Launcher
echo =======================================================

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
echo 🎯 Choose Detection Mode:
echo 1. Basic Detection (Original system)
echo 2. Advanced Detection (YOLOv10 + Dynamic Classification)
echo 3. Model Manager (Download and manage AI models)
echo 4. Exit

set /p choice="🔤 Enter your choice (1-4): "

if "%choice%"=="1" (
    echo.
    echo 🚀 Starting Basic Detection System...
    
    :: Check if basic requirements are installed
    if not exist "yolov8n.pt" (
        echo 🔧 Installing basic requirements...
        python -m pip install -r requirements.txt
        if %errorlevel% neq 0 (
            echo ❌ Failed to install basic requirements
            pause
            exit /b 1
        )
    )
    
    echo 🎥 Starting basic detection...
    python detect.py
    
) else if "%choice%"=="2" (
    echo.
    echo 🚀 Starting Advanced Detection System...
    
    :: Check if advanced requirements are installed
    echo 📦 Installing/updating advanced requirements...
    python -m pip install -r requirements_advanced.txt
    if %errorlevel% neq 0 (
        echo ❌ Failed to install advanced requirements
        pause
        exit /b 1
    )
    
    echo 🎥 Starting advanced detection...
    python detect_advanced.py
    
) else if "%choice%"=="3" (
    echo.
    echo 🤖 Starting Model Manager...
    
    :: Install advanced requirements for model manager
    python -m pip install -r requirements_advanced.txt >nul 2>&1
    
    python model_manager.py
    
) else if "%choice%"=="4" (
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
    echo ❌ Detection system encountered an error
    pause
)

echo.
echo 👋 Detection system stopped
pause 