#!/bin/bash

# Get script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Create template icon for menu bar (white, transparent background)
cat > "$DIR/icon_template.svg" << 'EOF'
<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="64" height="64" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path fill-rule="evenodd" clip-rule="evenodd" d="M32 8C18.7452 8 8 18.7452 8 32C8 45.2548 18.7452 56 32 56C45.2548 56 56 45.2548 56 32C56 18.7452 45.2548 8 32 8ZM28 24C28 22.8954 28.8954 22 30 22H34C35.1046 22 36 22.8954 36 24V30H42C43.1046 30 44 30.8954 44 32V36C44 37.1046 43.1046 38 42 38H36V44C36 45.1046 35.1046 46 34 46H30C28.8954 46 28 45.1046 28 44V38H22C20.8954 38 20 37.1046 20 36V32C20 30.8954 20.8954 30 22 30H28V24Z" fill="white"/>
</svg>
EOF

# Create menu bar icon (PNG with transparency)
sips -s format png "$DIR/icon_template.svg" --out "$DIR/icon.png"

# Create high-resolution app icon
cat > "$DIR/icon_app.svg" << 'EOF'
<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="1024" height="1024" viewBox="0 0 1024 1024" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect width="1024" height="1024" rx="224" fill="#000000"/>
    <path fill-rule="evenodd" clip-rule="evenodd" d="M512 192C334.904 192 192 334.904 192 512C192 689.096 334.904 832 512 832C689.096 832 832 689.096 832 512C832 334.904 689.096 192 512 192ZM448 384C448 366.327 462.327 352 480 352H544C561.673 352 576 366.327 576 384V480H672C689.673 480 704 494.327 704 512V576C704 593.673 689.673 608 672 608H576V704C576 721.673 561.673 736 544 736H480C462.327 736 448 721.673 448 704V608H352C334.327 608 320 593.673 320 576V512C320 494.327 334.327 480 352 480H448V384Z" fill="white"/>
</svg>
EOF

# Convert SVG to PNG
sips -s format png "$DIR/icon_app.svg" --out "$DIR/icon_1024x1024.png"

# Create icns file
mkdir -p "$DIR/icon.iconset"
sips -z 16 16   "$DIR/icon_1024x1024.png" --out "$DIR/icon.iconset/icon_16x16.png"
sips -z 32 32   "$DIR/icon_1024x1024.png" --out "$DIR/icon.iconset/icon_16x16@2x.png"
sips -z 32 32   "$DIR/icon_1024x1024.png" --out "$DIR/icon.iconset/icon_32x32.png"
sips -z 64 64   "$DIR/icon_1024x1024.png" --out "$DIR/icon.iconset/icon_32x32@2x.png"
sips -z 128 128 "$DIR/icon_1024x1024.png" --out "$DIR/icon.iconset/icon_128x128.png"
sips -z 256 256 "$DIR/icon_1024x1024.png" --out "$DIR/icon.iconset/icon_128x128@2x.png"
sips -z 256 256 "$DIR/icon_1024x1024.png" --out "$DIR/icon.iconset/icon_256x256.png"
sips -z 512 512 "$DIR/icon_1024x1024.png" --out "$DIR/icon.iconset/icon_256x256@2x.png"
sips -z 512 512 "$DIR/icon_1024x1024.png" --out "$DIR/icon.iconset/icon_512x512.png"
cp "$DIR/icon_1024x1024.png" "$DIR/icon.iconset/icon_512x512@2x.png"

iconutil -c icns "$DIR/icon.iconset"
rm -rf "$DIR/icon.iconset"

echo "✅ Created icons successfully!"
 