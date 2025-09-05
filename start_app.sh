#!/bin/bash
# Shell script to run the Viking DSR Streamlit application with automatic port handling
# This script handles the "Port 8501 is already in use" error automatically

echo "Starting Viking DSR Application..."
echo

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    echo "Please install Python and try again."
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

# Run the application with automatic port handling
$PYTHON_CMD run_app.py

# Check exit status
if [ $? -ne 0 ]; then
    echo
    echo "==========================================="
    echo "Error occurred while starting the application"
    echo "==========================================="
    echo
    echo "Troubleshooting tips:"
    echo "1. Make sure all dependencies are installed: pip install -r requirements.txt"
    echo "2. Check if another Streamlit app is running on port 8501"
    echo "3. Try running: $PYTHON_CMD run_app.py 8502"
    echo
    read -p "Press Enter to continue..."
fi