@echo off
REM Batch script to run the Viking DSR Streamlit application with automatic port handling
REM This script handles the "Port 8501 is already in use" error automatically

echo Starting Viking DSR Application...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python and try again.
    pause
    exit /b 1
)

REM Run the application with automatic port handling
python run_app.py

REM If there's an error, show troubleshooting information
if errorlevel 1 (
    echo.
    echo ===========================================
    echo Error occurred while starting the application
    echo ===========================================
    echo.
    echo Troubleshooting tips:
    echo 1. Make sure all dependencies are installed: pip install -r requirements.txt
    echo 2. Check if another Streamlit app is running on port 8501
    echo 3. Try running: python run_app.py 8502
    echo.
    pause
)