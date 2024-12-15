#!/bin/bash

# Get the PID of the Python process
PYTHON_PID=$(pgrep -f "python3.*app.py")

if [ -n "$PYTHON_PID" ]; then
    # Get the app bundle path
    BUNDLE_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    ICON_PATH="$BUNDLE_PATH/AppIcon.icns"
    
    # Use AppleScript to set the dock icon
    osascript <<EOF
    tell application "System Events"
        set processID to $PYTHON_PID
        set frontmost of process id processID to true
        set icon of process id processID to "$ICON_PATH"
    end tell
EOF
fi 