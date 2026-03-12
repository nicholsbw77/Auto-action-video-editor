@echo off
echo ============================================
echo  Building Auto Video Editor
echo ============================================
echo.

REM Ensure PyInstaller is installed
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

REM Clean previous builds
if exist dist\AutoVideoEditor (
    echo Cleaning previous build...
    rmdir /s /q dist\AutoVideoEditor
)
if exist build\AutoVideoEditor (
    rmdir /s /q build\AutoVideoEditor
)

REM Run PyInstaller
echo Running PyInstaller...
pyinstaller autovideoeditor.spec --noconfirm

if errorlevel 1 (
    echo.
    echo BUILD FAILED
    exit /b 1
)

echo.
echo ============================================
echo  Build complete!
echo  Output: dist\AutoVideoEditor\
echo  Run:    dist\AutoVideoEditor\AutoVideoEditor.exe
echo ============================================
echo.
echo NOTE: FFmpeg is NOT bundled.
echo       Users must install FFmpeg separately.
echo       Download from: https://ffmpeg.org/download.html
