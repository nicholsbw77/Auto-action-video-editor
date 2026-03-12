# Auto Video Editor

A PyQt6 desktop application that automatically edits videos by analyzing audio tracks — detecting beats, onsets, and energy changes — then generating beat-synchronized cuts and smooth transitions.

---

## Features

### Auto-Edit
- **Beat-synchronized cutting** — librosa analyzes tempo, beats, onsets, and energy to place cuts at musically meaningful moments
- **Aggressiveness slider (1–10)** — dial in cut frequency from a few long segments (Gentle) to rapid-fire edits (Maximum)
- **6 transition types** — individually enable/disable Hard Cut, Crossfade, Slow Fade, Fade to Black, Wipe Left, and Wipe Right
- **Single Video mode** — re-edit a single clip to a new audio track or its own audio
- **Multi-Clip Assembly mode** — combine 2+ video clips against one audio track; clips cycle sequentially with automatic looping and resolution normalization so different-sized clips blend seamlessly
- **Separate audio or embedded audio** — load an MP3/WAV/FLAC/AAC/OGG file, or extract audio directly from the video
- **GPU acceleration** — auto-detects NVIDIA NVENC; falls back to CPU (libx264) if unavailable or if you force it

### Trim
- Load any video and define multiple trim regions with frame-accurate timecode inputs (HH:MM:SS.CS)
- Export regions as separate files (stream copy = fast) or concatenate them into one output (re-encode for precision)

### Quality of Life
- **FFmpeg auto-download** — if FFmpeg isn't found on your system, the app offers to download it (~40 MB, no installer)
- No flashing console windows during processing
- Settings (aggressiveness, transitions, FFmpeg path, GPU preference) persist between sessions
- Recent files menu, live progress bar, real-time log output

---

## Requirements

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| PyQt6 | ≥ 6.5 |
| librosa | ≥ 0.10 |
| numpy | ≥ 1.23 |
| soundfile | ≥ 0.12 |
| FFmpeg | ≥ 4.3.0 (auto-detected or auto-downloaded) |

Optional: NVIDIA GPU for hardware-accelerated encoding.

---

## Installation

```bash
# 1. Clone
git clone https://github.com/nicholsbw77/Auto-action-video-editor.git
cd Auto-action-video-editor

# 2. Create and activate a virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python src\main.py
```

On first launch, the app will locate FFmpeg automatically (system PATH or common Windows install locations). If it isn't found, you'll be prompted to download it automatically, or you can point to it manually via **Settings → Set FFmpeg Path**.

---

## Building a Standalone Executable

Requires [PyInstaller](https://pyinstaller.org):

```bash
build.bat
```

Output: `dist\AutoVideoEditor\AutoVideoEditor.exe`

> FFmpeg is **not** bundled in the executable. Place `ffmpeg.exe` and `ffprobe.exe` in a `tools\` folder next to the executable, or let the app auto-download them on first run.

---

## Usage

### Auto-Edit workflow

1. **Add Video(s)** — pick one file (Single Video) or two or more (Multi-Clip Assembly)
2. **Set audio source** — load a separate audio file, or tick *Use video's own audio*
3. **Adjust Edit Settings**
   - Move the **Cut Aggressiveness** slider (1 = few long cuts, 10 = rapid cuts on every beat)
   - Check/uncheck transitions to control which effects are used
4. **Set output folder and filename**
5. Click **Go** — progress is shown live; click **Cancel** at any time

### Trim workflow

1. **Load Video** from the Trim tab
2. **Add Region** and type start/end timecodes
3. Choose *Separate files* or *Concatenate*
4. Click **Export Trim**

### Settings

| Setting | Where |
|---|---|
| FFmpeg path | Settings → Set FFmpeg Path |
| Force CPU encoding | Settings → Force CPU Encoding |
| Cut aggressiveness | Edit Settings slider on Auto-Edit tab |
| Allowed transitions | Edit Settings checkboxes on Auto-Edit tab |

---

## Aggressiveness Levels

| Level | Label | Min segment | Max segment | Candidate pool |
|---|---|---|---|---|
| 1 | Gentle | 6.0 s | 15.0 s | Strong beats only |
| 2 | Relaxed | 5.0 s | 12.0 s | Strong beats only |
| 3 | Calm | 4.0 s | 10.0 s | Strong beats + strong onsets |
| 4 | Easy | 3.0 s | 10.0 s | Strong beats + strong onsets |
| 5 | Moderate *(default)* | 2.0 s | 8.0 s | All beats + strong onsets |
| 6 | Active | 1.5 s | 6.0 s | All beats + strong onsets |
| 7 | Energetic | 1.2 s | 5.0 s | All beats + all onsets |
| 8 | Fast | 1.0 s | 4.0 s | All beats + all onsets |
| 9 | Intense | 0.7 s | 3.0 s | All beats + all onsets |
| 10 | Maximum | 0.5 s | 2.0 s | All beats + all onsets |

---

## Supported Formats

| | Formats |
|---|---|
| **Video input** | MP4, MOV, MKV, AVI, WebM, TS |
| **Audio input** | MP3, WAV, FLAC, AAC, OGG |
| **Output** | MP4 (H.264 / NVENC, AAC 192k) |
| **Max output resolution** | 1920 × 1080 |
| **Max output frame rate** | 60 fps |

---

## Project Structure

```
src/
├── main.py                  # Entry point, startup checks, FFmpeg detection
├── config.py                # JSON config persistence
├── paths.py                 # App directory resolution (dev vs packaged)
├── core/
│   ├── audio_analyzer.py    # librosa beat / onset / energy detection
│   ├── cut_generator.py     # Beat-synced cut algorithm with aggressiveness table
│   ├── export_manager.py    # Output resolution / codec resolution
│   ├── models.py            # Data classes (CutPoint, AnalysisResult, …)
│   ├── temp_manager.py      # Session temp files, auto-cleanup
│   └── logger.py            # Rotating file logger
├── ffmpeg/
│   ├── runner.py            # Subprocess wrapper, progress parsing, cancellation
│   ├── filter_graph.py      # xfade / acrossfade filter graph builder
│   ├── detector.py          # FFmpeg / FFprobe binary detection
│   ├── downloader.py        # Auto-download from gyan.dev
│   ├── gpu.py               # NVIDIA NVENC detection
│   └── __init__.py          # CREATE_NO_WINDOW helper
└── gui/
    ├── main_window.py       # QMainWindow, menu bar, tab host
    ├── auto_edit_tab.py     # Full auto-edit pipeline UI
    ├── trim_tab.py          # Trim regions UI
    ├── widgets.py           # FileListWidget, TimecodeInput
    └── workers.py           # QThread workers (analysis, export, trim, download)
```

---

## License

MIT
