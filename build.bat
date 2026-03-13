@echo off
echo ============================================
echo  Building Auto Video Editor (Windows)
echo ============================================
echo.

REM Ensure PyInstaller is installed
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

REM Download FFmpeg if not present
if not exist "tools\ffmpeg.exe" (
    echo Downloading FFmpeg...
    mkdir tools 2>nul
    powershell -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip' -OutFile 'tools\ffmpeg.zip' }"
    echo Extracting FFmpeg...
    powershell -Command "& { $zip = [System.IO.Compression.ZipFile]::OpenRead('tools\ffmpeg.zip'); foreach ($entry in $zip.Entries) { if ($entry.Name -eq 'ffmpeg.exe' -or $entry.Name -eq 'ffprobe.exe') { [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, ('tools\' + $entry.Name), $true) } }; $zip.Dispose() }"
    del tools\ffmpeg.zip 2>nul
    echo FFmpeg downloaded to tools\
)

if not exist "tools\ffmpeg.exe" (
    echo ERROR: FFmpeg download failed. Please manually place ffmpeg.exe and ffprobe.exe in the tools\ folder.
    exit /b 1
)

echo FFmpeg found:
tools\ffmpeg.exe -version 2>&1 | findstr /C:"ffmpeg version"

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
echo  FFmpeg: BUNDLED in dist\AutoVideoEditor\tools\
echo ============================================
