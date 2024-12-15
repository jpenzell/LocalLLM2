#!/bin/bash

# Get script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Create app bundle structure
APP_NAME="LocalGPT.app"
CHAT_APP_NAME="LocalGPTChat.app"  # New chat interface app
CONTENTS_DIR="$DIR/$APP_NAME/Contents"
CHAT_CONTENTS_DIR="$DIR/$CHAT_APP_NAME/Contents"  # New chat app contents
MACOS_DIR="$CONTENTS_DIR/MacOS"
CHAT_MACOS_DIR="$CHAT_CONTENTS_DIR/MacOS"  # New chat app MacOS dir
RESOURCES_DIR="$CONTENTS_DIR/Resources"
CHAT_RESOURCES_DIR="$CHAT_CONTENTS_DIR/Resources"  # New chat app resources

# Clean up any existing bundles
rm -rf "$DIR/$APP_NAME"
rm -rf "$DIR/$CHAT_APP_NAME"

# Create directories
mkdir -p "$MACOS_DIR"
mkdir -p "$CHAT_MACOS_DIR"
mkdir -p "$RESOURCES_DIR"
mkdir -p "$CHAT_RESOURCES_DIR"

# Create chat app launch script
cat > "$CHAT_MACOS_DIR/launch" << 'EOF'
#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../Resources" && pwd )"
cd "$DIR"

# Activate virtual environment
source venv/bin/activate

# Set environment variables for proper app behavior
export PYTHONPATH="$DIR"
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export LSUIElement=1  # Run as agent (no dock icon)

# Launch the app with proper process name
exec python3 app.py
EOF

chmod +x "$CHAT_MACOS_DIR/launch"

# Copy necessary files to chat app
cp -R "$DIR/venv" "$CHAT_RESOURCES_DIR/"
cp "$DIR/app.py" "$CHAT_RESOURCES_DIR/"
cp "$DIR/requirements.txt" "$CHAT_RESOURCES_DIR/"
cp "$DIR/icon.icns" "$CHAT_RESOURCES_DIR/AppIcon.icns"

# Create Info.plist for chat app
cat > "$CHAT_CONTENTS_DIR/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>launch</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundleIdentifier</key>
    <string>com.localgpt.chat</string>
    <key>CFBundleName</key>
    <string>LocalGPT Chat</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.1</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.15</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>CFBundleDisplayName</key>
    <string>LocalGPT Chat</string>
    <key>CFBundleVersion</key>
    <string>1.1.0</string>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.productivity</string>
    <key>LSBackgroundOnly</key>
    <false/>
    <key>LSUIElement</key>
    <true/>
    <key>NSSupportsAutomaticGraphicsSwitching</key>
    <true/>
</dict>
</plist>
EOF

# Create launch script
cat > "$MACOS_DIR/launch" << 'EOF'
#!/bin/bash

# Get the app bundle's Resources directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../Resources" && pwd )"

# Change to the Resources directory
cd "$DIR"

# Create log directory
LOG_DIR="$HOME/.localgpt/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/localgpt.log"

# Function to show error notification
show_error() {
    osascript -e "display notification \"$1\" with title \"LocalGPT Error\""
    echo "[ERROR] $1" >> "$LOG_FILE"
    exit 1
}

# Function to show success notification
show_success() {
    osascript -e "display notification \"$1\" with title \"LocalGPT\""
    echo "[INFO] $1" >> "$LOG_FILE"
}

echo "Starting LocalGPT..." > "$LOG_FILE"
echo "Current directory: $DIR" >> "$LOG_FILE"

# Create chat history directory in user's home
CHAT_HISTORY_DIR="$HOME/.localgpt/chat_history"
mkdir -p "$CHAT_HISTORY_DIR"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    show_error "Python 3 is not installed. Please install Python 3 and try again."
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    show_error "Virtual environment not found. Please reinstall the application."
fi

# Activate virtual environment and start the app
source venv/bin/activate || show_error "Failed to activate virtual environment"

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
    show_error "Ollama is not running. Please start Ollama first."
fi

# Start the menubar app with explicit Python process name and icon
export PYTHON_PROCESS_NAME="LocalGPT"
export PYTHONPATH="$DIR"
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
show_success "LocalGPT is starting... Look for the icon in your menu bar"
exec python3 -c "import os; os.environ['PYTHON_PROCESS_NAME'] = 'LocalGPT'; os.environ['APP_BUNDLE_PATH'] = '$DIR/../..'; import menubar; menubar.main()" >> "$LOG_FILE" 2>&1 || show_error "Failed to start LocalGPT. Check logs at ~/.localgpt/logs/localgpt.log"
EOF

# Make launch script executable
chmod +x "$MACOS_DIR/launch"

# Create Python wrapper for menubar.py
cat > "$RESOURCES_DIR/menubar.py" << 'EOF'
import os
import sys
import rumps
import subprocess
import webbrowser
import logging
import traceback
from datetime import datetime
import time
import requests
import psutil
import socket

# Hide the Python app from dock for the menu bar app
try:
    from Foundation import NSBundle
    from AppKit import NSApplication
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(1)  # NSApplicationActivationPolicyAccessory
except Exception as e:
    pass

# Configure logging with both file and console output
log_dir = os.path.expanduser("~/.localgpt/logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "localgpt.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

def show_notification(title, message):
    """Show a macOS notification"""
    os.system(f"""
        osascript -e 'display notification "{message}" with title "{title}"'
    """)

def start_chat_interface():
    """Start the Gradio interface and open browser"""
    try:
        # Start the Gradio interface
        logging.info("Starting chat interface process...")
        subprocess.Popen([sys.executable, "app.py"])
        
        # Wait longer for the server to start
        logging.info("Waiting for server to start...")
        time.sleep(5)  # Increased from 2 to 5 seconds
        
        # Try to connect to the server before opening browser
        max_retries = 5
        for i in range(max_retries):
            try:
                requests.get('http://127.0.0.1:7860')
                logging.info("Server is ready")
                break
            except requests.exceptions.ConnectionError:
                if i < max_retries - 1:
                    logging.info(f"Server not ready, retrying... ({i+1}/{max_retries})")
                    time.sleep(2)
                else:
                    logging.error("Server failed to start")
                    show_notification("LocalGPT Error", "Failed to start chat interface")
                    return False
        
        # Open browser
        logging.info("Opening browser...")
        webbrowser.open('http://127.0.0.1:7860')
        return True
    except Exception as e:
        logging.error(f"Error starting chat interface: {e}")
        return False

class LocalGPTApp(rumps.App):
    def __init__(self):
        try:
            logging.info("Initializing LocalGPT menu bar app")
            super().__init__(
                "LocalGPT",
                icon="icon.png",
                quit_button=None  # Disable default quit button
            )
            
            # Create menu items
            self.menu = [
                rumps.MenuItem("Open Chat Interface", callback=self.open_chat),
                rumps.MenuItem("View Logs", callback=self.view_logs),
                None,  # Separator
                rumps.MenuItem("Quit LocalGPT", callback=self.quit_app)
            ]
            
            logging.info("Menu bar app initialized successfully")
            show_notification("LocalGPT", "LocalGPT is running in your menu bar")
        except Exception as e:
            error_msg = f"Error initializing app: {str(e)}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())
            show_notification("LocalGPT Error", error_msg)
            sys.exit(1)

    def open_chat(self, _):
        try:
            logging.info("Attempting to open chat interface")
            if start_chat_interface():
                logging.info("Chat interface launched successfully")
                show_notification("LocalGPT", "Opening chat interface in your browser")
            else:
                show_notification("LocalGPT Error", "Failed to start chat interface")
        except Exception as e:
            error_msg = f"Error opening chat interface: {str(e)}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())
            show_notification("LocalGPT Error", "Failed to open chat interface. Check logs.")

    def view_logs(self, _):
        try:
            subprocess.run(["open", log_file])
            logging.info("Opened log file")
        except Exception as e:
            error_msg = f"Error opening logs: {str(e)}"
            logging.error(error_msg)
            show_notification("LocalGPT Error", "Failed to open logs")

    def quit_app(self, _):
        try:
            logging.info("LocalGPT shutting down")
            
            # Try graceful shutdown first
            try:
                requests.post('http://127.0.0.1:7860/shutdown', timeout=2)
            except:
                pass

            # Get all Python processes
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    # Kill any Python process running app.py
                    if proc.info['cmdline'] and any('app.py' in cmd for cmd in proc.info['cmdline']):
                        logging.info(f"Killing process {proc.info['pid']}")
                        psutil.Process(proc.info['pid']).kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass

            # Kill any process using port 7860
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(('127.0.0.1', 7860))
                sock.close()
            except socket.error:
                # Port is in use, find and kill the process
                for proc in psutil.process_iter(['pid', 'connections']):
                    try:
                        for conn in proc.connections():
                            if conn.laddr.port == 7860:
                                logging.info(f"Killing process using port 7860: {proc.pid}")
                                psutil.Process(proc.pid).kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

            # Final verification with timeout
            try:
                requests.get('http://127.0.0.1:7860', timeout=1)
                logging.warning("Server still running after cleanup attempts")
            except requests.exceptions.ConnectionError:
                logging.info("Server successfully shut down")
            except:
                pass

            logging.info("Quitting application")
            rumps.quit_application()
        except Exception as e:
            logging.error(f"Error during shutdown: {e}")
            # Force quit even if there was an error
            rumps.quit_application()

def main():
    try:
        logging.info("Starting LocalGPT menu bar app")
        LocalGPTApp().run()
    except Exception as e:
        error_msg = f"Fatal error: {str(e)}"
        logging.error(error_msg)
        logging.error(traceback.format_exc())
        show_notification("LocalGPT Error", "Application crashed. Check logs.")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

# Copy application files
cp -R "$DIR/venv" "$RESOURCES_DIR/"
cp "$DIR/app.py" "$RESOURCES_DIR/"
cp "$DIR/requirements.txt" "$RESOURCES_DIR/"
cp "$DIR/icon.png" "$RESOURCES_DIR/"
cp "$DIR/set_icon.sh" "$RESOURCES_DIR/"

# Copy the icon files
if [ -f "$DIR/icon.icns" ]; then
    cp "$DIR/icon.icns" "$RESOURCES_DIR/AppIcon.icns"
else
    echo "Warning: icon.icns not found. Run Create Icons.command first."
    exit 1
fi

# Create Info.plist with additional metadata
cat > "$CONTENTS_DIR/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>launch</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundleIdentifier</key>
    <string>com.localgpt.app</string>
    <key>CFBundleName</key>
    <string>LocalGPT</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.1</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.15</string>
    <key>LSUIElement</key>
    <true/>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>CFBundleDisplayName</key>
    <string>LocalGPT</string>
    <key>CFBundleVersion</key>
    <string>1.1.0</string>
    <key>CFBundleDocumentTypes</key>
    <array>
        <dict>
            <key>CFBundleTypeName</key>
            <string>LocalGPT Document</string>
            <key>CFBundleTypeRole</key>
            <string>Editor</string>
            <key>LSHandlerRank</key>
            <string>Owner</string>
        </dict>
    </array>
</dict>
</plist>
EOF

echo "✅ Created $APP_NAME and $CHAT_APP_NAME successfully!"