@echo off
echo 🚀 Enhanced Object Detection System - Ultimate Launcher
echo ==========================================================

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
echo 🎯 Choose Enhanced Detection Mode:
echo 1. 🎮 Enhanced Desktop App (Advanced features + Modern UI)
echo 2. 🌐 Web Interface (Browser-based with analytics)
echo 3. 📊 Analytics Dashboard (Data visualization)
echo 4. 🔧 Install Enhanced Dependencies
echo 5. 🗑️ Basic Detection (Original system)
echo 6. ❌ Exit

set /p choice="🔤 Enter your choice (1-6): "

if "%choice%"=="1" (
    echo.
    echo 🚀 Starting Enhanced Desktop Detection...
    echo 📦 Installing enhanced requirements...
    
    python -m pip install -r requirements_enhanced.txt --quiet
    if %errorlevel% neq 0 (
        echo ⚠️  Some packages may have failed to install, continuing anyway...
    )
    
    echo 🎮 Launching enhanced desktop app...
    echo.
    echo 🎮 Enhanced Controls:
    echo   Q - Quit
    echo   S - Save screenshot
    echo   R - Toggle recording
    echo   H - Toggle heat map
    echo   T - Toggle trajectory trails
    echo   A - Toggle audio alerts
    echo.
    
    python detect_enhanced.py
    
) else if "%choice%"=="2" (
    echo.
    echo 🌐 Starting Web Interface...
    echo 📦 Installing web requirements...
    
    python -m pip install streamlit plotly --quiet
    python -m pip install -r requirements_enhanced.txt --quiet
    if %errorlevel% neq 0 (
        echo ⚠️  Some packages may have failed to install, continuing anyway...
    )
    
    echo 🌐 Launching web interface...
    echo 💡 Your browser will open automatically
    echo 💡 If not, go to: http://localhost:8501
    echo.
    
    streamlit run web_interface.py
    
) else if "%choice%"=="3" (
    echo.
    echo 📊 Starting Analytics Dashboard...
    echo 📦 Installing analytics requirements...
    
    python -m pip install streamlit plotly pandas --quiet
    if %errorlevel% neq 0 (
        echo ⚠️  Some packages may have failed to install, continuing anyway...
    )
    
    echo 📊 Launching analytics dashboard...
    streamlit run web_interface.py --server.headless=true
    
) else if "%choice%"=="4" (
    echo.
    echo 🔧 Installing Enhanced Dependencies...
    echo 📦 This may take a few minutes...
    
    python -m pip install --upgrade pip
    python -m pip install -r requirements_enhanced.txt
    
    if %errorlevel% equ 0 (
        echo ✅ All enhanced dependencies installed successfully!
        echo 💡 You can now run any enhanced detection mode
    ) else (
        echo ⚠️  Some dependencies may have failed to install
        echo 💡 Try running as administrator or check your internet connection
    )
    
    pause
    
) else if "%choice%"=="5" (
    echo.
    echo 🎮 Starting Basic Detection System...
    
    python -m pip install ultralytics opencv-python --quiet
    if %errorlevel% neq 0 (
        echo ❌ Failed to install basic requirements
        pause
        exit /b 1
    )
    
    echo 🎥 Starting basic detection...
    python detect.py
    
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
    echo ❌ System encountered an error
    pause
)

echo.
echo 👋 Enhanced detection system stopped
pause 