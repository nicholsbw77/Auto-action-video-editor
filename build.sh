#!/usr/bin/env bash
set -euo pipefail

echo "============================================"
echo "  Building Auto Video Editor (Linux)"
echo "============================================"
echo

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="$SCRIPT_DIR/tools"
FFMPEG_URL="https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"

# Ensure PyInstaller is installed
if ! python3 -m PyInstaller --version &>/dev/null; then
    echo "Installing PyInstaller..."
    pip install pyinstaller
fi

# Download FFmpeg static binaries if not present
if [ ! -f "$TOOLS_DIR/ffmpeg" ] || [ ! -f "$TOOLS_DIR/ffprobe" ]; then
    echo "Downloading FFmpeg static binaries..."
    mkdir -p "$TOOLS_DIR"
    TMPFILE=$(mktemp /tmp/ffmpeg-XXXXXX.tar.xz)
    curl -L -o "$TMPFILE" "$FFMPEG_URL"
    TMPDIR=$(mktemp -d)
    tar xf "$TMPFILE" -C "$TMPDIR"
    FFDIR=$(ls -d "$TMPDIR"/ffmpeg-*-static 2>/dev/null | head -1)
    cp "$FFDIR/ffmpeg" "$TOOLS_DIR/ffmpeg"
    cp "$FFDIR/ffprobe" "$TOOLS_DIR/ffprobe"
    chmod +x "$TOOLS_DIR/ffmpeg" "$TOOLS_DIR/ffprobe"
    rm -rf "$TMPFILE" "$TMPDIR"
    echo "FFmpeg downloaded to $TOOLS_DIR/"
else
    echo "FFmpeg binaries found in $TOOLS_DIR/"
fi

# Verify FFmpeg works
"$TOOLS_DIR/ffmpeg" -version | head -1

# Clean previous builds
if [ -d "$SCRIPT_DIR/dist/AutoVideoEditor" ]; then
    echo "Cleaning previous build..."
    rm -rf "$SCRIPT_DIR/dist/AutoVideoEditor"
fi
if [ -d "$SCRIPT_DIR/build/AutoVideoEditor" ]; then
    rm -rf "$SCRIPT_DIR/build/AutoVideoEditor"
fi

# Run PyInstaller
echo "Running PyInstaller..."
cd "$SCRIPT_DIR"
python3 -m PyInstaller autovideoeditor.spec --noconfirm

if [ $? -ne 0 ]; then
    echo
    echo "BUILD FAILED"
    exit 1
fi

echo
echo "============================================"
echo "  Build complete!"
echo "  Output: dist/AutoVideoEditor/"
echo "  Run:    dist/AutoVideoEditor/AutoVideoEditor"
echo "  FFmpeg: BUNDLED in dist/AutoVideoEditor/tools/"
echo "============================================"
