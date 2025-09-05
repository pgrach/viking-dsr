#!/usr/bin/env python3
"""
Streamlit Application Launcher with Port Conflict Resolution

This script automatically finds an available port when the default port (8501) is busy
and launches the Streamlit application.
"""

import socket
import subprocess
import sys
import os
from pathlib import Path

def is_port_available(port):
    """Check if a port is available for use."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
            return True
    except OSError:
        return False

def find_available_port(start_port=8501, max_port=8600):
    """Find the first available port starting from start_port."""
    for port in range(start_port, max_port):
        if is_port_available(port):
            return port
    return None

def launch_streamlit(port=None):
    """Launch the Streamlit application on the specified or available port."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    app_path = script_dir / "app.py"
    
    if not app_path.exists():
        print(f"Error: app.py not found in {script_dir}")
        sys.exit(1)
    
    # If no port specified, try to find an available one
    if port is None:
        port = find_available_port()
        if port is None:
            print("Error: No available ports found in range 8501-8600")
            print("Please close other Streamlit applications or use a different port range.")
            sys.exit(1)
    
    # Check if the specified port is available
    if not is_port_available(port):
        print(f"Port {port} is already in use. Finding alternative...")
        port = find_available_port(port + 1)
        if port is None:
            print("Error: No available ports found.")
            sys.exit(1)
    
    print(f"Starting Streamlit application on port {port}")
    print(f"You can access the application at: http://localhost:{port}")
    
    # Launch streamlit with the available port
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        str(app_path), 
        "--server.port", str(port),
        "--server.headless", "true"
    ]
    
    try:
        subprocess.run(cmd, cwd=script_dir)
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
    except Exception as e:
        print(f"Error running Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Parse command line arguments for custom port
    port = None
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
            print(f"Attempting to use custom port: {port}")
        except ValueError:
            print("Invalid port number provided. Using automatic port detection.")
    
    launch_streamlit(port)