# Auto Video Editor - Design Specification

## Overview

A Python-based desktop application for Windows 11 that automatically edits video to match audio characteristics. Targeted at action/sports highlight content. Uses audio analysis to determine cut points, transition types, and pacing. Built with PyQt6 for the GUI and FFmpeg for all video processing. Supports NVIDIA GPU acceleration when available.

## Goals

- Fully automatic video editing driven by audio analysis (beats, energy, onsets)
- Support single-video and multi-clip assembly workflows
- Integrated trim tool for pre-processing video before auto-editing
- Export to 1080p 60fps (or source maximum if lower)
- Leverage NVIDIA NVENC for hardware-accelerated encoding when available
- Clean, functional PyQt6 interface

## Non-Goals (Deferred to v2)

- Transparent overlay images
- Manual timeline editing / cut point adjustment
- Real-time preview playback
- AI/ML-based scene classification

## Architecture

```
+------------------------------------------+
|              PyQt6 GUI                   |
|  +------------+  +------------+          |
|  | Auto-Edit  |  |   Trim     |          |
|  |    Tab     |  |    Tab     |          |
|  +------------+  +------------+          |
+------------------------------------------+
|           Core Engine                    |
|  +------------+ +------------+ +-------+ |
|  |   Audio    | |    Cut     | |Export | |
|  |  Analyzer  | | Generator  | |Manager| |
|  +------------+ +------------+ +-------+ |
+------------------------------------------+
|         FFmpeg Interface                 |
|  Filter graph builder, GPU detection,    |
|  subprocess management                   |
+------------------------------------------+
|         System Layer                     |
|  FFmpeg binary detection, NVIDIA check,  |
|  file I/O, config persistence            |
+------------------------------------------+
```

### Key Modules

1. **GUI Layer** - PyQt6 with two tabs (Auto-Edit, Trim). Handles file selection, progress reporting, export settings display.
2. **Audio Analyzer** - Uses librosa to extract beat positions, onset strength, spectral energy, and tempo. Classifies segments by intensity. Returns an `AnalysisResult` dataclass.
3. **Cut Generator** - Takes `AnalysisResult` and produces a list of `CutPoint` objects with transition types based on energy mapping rules.
4. **FFmpeg Interface** - Builds FFmpeg filter graphs from the cut list, manages subprocess execution, handles NVENC detection.
5. **Export Manager** - Resolves `ExportSettings` (resolution/framerate), manages encoding presets, parses FFmpeg progress output. Owns the encoding pipeline; the GUI layer only displays export settings, it does not resolve them.

## Data Models

```python
@dataclass
class AnalysisResult:
    beats: list[float]           # Beat timestamps in seconds
    onsets: list[float]          # Onset timestamps in seconds
    tempo: float                 # Detected BPM
    energy_envelope: np.ndarray  # RMS energy per frame
    energy_times: np.ndarray     # Timestamps for energy values
    spectral_centroids: np.ndarray  # Spectral centroid per frame
    duration: float              # Total audio duration in seconds

@dataclass
class CutPoint:
    start: float                 # Start time in seconds
    end: float                   # End time in seconds
    source_index: int            # Index into the source clip list (0 for single-video mode)
    transition_type: str         # "hard_cut", "crossfade", "crossfade_slow", "fade_black", "wipe_left", "wipe_right"
    transition_duration: float   # Duration in seconds (0.0 for hard cuts)

@dataclass
class ExportSettings:
    output_path: str
    width: int                   # Output width (max 1920)
    height: int                  # Output height (max 1080)
    fps: float                   # Output framerate (max 60)
    video_codec: str             # "h264_nvenc" or "libx264"
    video_crf: int               # Quality level (see Video Quality section)
    audio_codec: str             # Always "aac" for MP4 compatibility
    audio_bitrate: str           # "192k"
    container: str               # "mp4"

@dataclass
class TrimRegion:
    start: float                 # Start time in seconds
    end: float                   # End time in seconds
    label: str                   # Optional user label (auto-generated: "Region 1", "Region 2", etc.)
```

## Supported Formats

**Video input** (any container/codec FFmpeg can decode, validated via ffprobe):
- Containers: MP4, MOV, MKV, AVI, WEBM, TS
- Codecs: H.264, H.265/HEVC, VP9, AV1, MPEG-4, ProRes

**Audio input** (for separate audio upload):
- Formats: MP3, WAV, FLAC, AAC, OGG

**Output**: MP4 container with H.264 video and AAC audio.

Validation: run `ffprobe` on input files. For video inputs, ffprobe must report at least one video stream (audio-only files in video containers are rejected with message "No video stream found"). For audio inputs, ffprobe must report at least one audio stream. If ffprobe returns an error, reject with a user-facing message.

## Audio Analysis Pipeline

### Processing Steps

1. **Load audio** - Extract audio from video via FFmpeg (`ffmpeg -i input.mp4 -vn -ac 1 -ar 22050 temp_audio.wav`), or load user-uploaded audio file. Convert to mono WAV at 22050 Hz for analysis.
2. **Beat detection** - `librosa.beat.beat_track()` identifies beat positions and tempo (BPM).
3. **Onset detection** - `librosa.onset.onset_detect()` finds attack transients (hits, impacts, drops).
4. **Energy envelope** - `librosa.feature.rms()` computes rolling energy levels. Segments classified as high/medium/low intensity.
5. **Spectral analysis** - `librosa.feature.spectral_centroid()` detects tonal shifts to differentiate transition types.

### Energy Classification

Energy levels are classified using percentile thresholds on the RMS envelope:
- **High**: above 75th percentile of RMS values
- **Medium**: between 25th and 75th percentile
- **Low**: below 25th percentile

### Energy Drop and Spike Detection

- **Energy drop**: RMS value decreases by more than 50% within a 200ms window compared to the preceding 500ms average. Triggers a fade-to-black transition before the drop point.
- **Energy spike**: RMS value increases by more than 100% within a 200ms window compared to the preceding 500ms average. Triggers a hard cut on the spike.

### Beat Strength Classification

Beat "strength" is determined by the onset strength envelope value at each beat position. Using `librosa.onset.onset_strength()`, the onset strength at each beat timestamp is sampled. Beats are classified as:
- **Strong**: onset strength at the beat position is above the median onset strength across all beats
- **Weak**: onset strength at the beat position is at or below the median

### Cut Point Decision Logic

| Audio Characteristic         | Cut Behavior             | Transition Type     |
|------------------------------|--------------------------|---------------------|
| Strong beat + high energy    | Fast cut (on beat)       | Hard cut or wipe    |
| Strong beat + medium energy  | Cut on beat              | Hard cut            |
| Weak beat + low energy       | Longer segment hold      | Crossfade           |
| Energy drop (sudden quiet)   | Cut before the drop      | Fade to black       |
| Energy spike (sudden loud)   | Cut on the spike         | Hard cut            |
| Sustained low energy         | Hold current segment     | Slow crossfade      |

**Wipe vs hard cut rule**: For "Strong beat + high energy" cuts, every 3rd qualifying cut uses a wipe transition instead of a hard cut. This keeps wipes as accents rather than overused. The cut generator tracks a counter for this.

**Wipe direction rule**: Wipe direction alternates: odd-numbered wipe cuts use `wipe_left`, even-numbered use `wipe_right`.

### Transition Durations

Default transition durations by type:
- **Hard cut**: 0.0 seconds (instant)
- **Crossfade**: 0.3 seconds
- **Slow crossfade / `crossfade_slow`** (sustained low energy): 0.8 seconds. Uses the same FFmpeg `xfade=transition=fade` as regular crossfade, just with longer duration.
- **Fade to black**: 0.5 seconds
- **Wipe (left/right)**: 0.2 seconds

These are fixed defaults. Transition duration is clamped so it never exceeds half the shorter of the two adjacent segment durations (prevents a transition from consuming an entire segment).

**Sustained low energy behavior**: When the energy envelope stays below the 25th percentile for a continuous stretch, the cut generator extends the current segment up to `max_cut_duration` (8s). The segment still ends at that limit, transitioning with a slow crossfade. The `max_cut_duration` is never overridden.

### Timing Rules

- Minimum cut duration: 0.5 seconds (prevents jarring micro-cuts)
- Maximum cut duration: 8.0 seconds (prevents stale segments)
- Beat-snap tolerance: 50ms (configurable in JSON config file, value in milliseconds). Cuts snap to nearest beat when within this tolerance.
- **Order of operations**: The cut generator first identifies candidate cut points from beats/onsets/energy events, then enforces the minimum cut duration by dropping any candidate that falls within 0.5s of the previous accepted cut point, then snaps the remaining accepted cuts to the nearest beat within tolerance. This ensures the minimum duration constraint is satisfied before beat-snapping.

### Multi-Clip Mode

When multiple videos are loaded, the cut generator round-robins through available clips. Each cut point advances to the next clip in the list (wrapping around). Segment length within each clip is determined by the audio analysis - each cut driven by beats/energy selects the next sequential segment from the next clip in rotation. Each clip maintains a playhead tracking how far into that clip has been used, so segments are pulled in order.

**Clip exhaustion**: When a clip's playhead reaches the end of that clip, it is removed from the rotation. The round-robin continues with the remaining clips. If all clips are exhausted before the audio ends, the output is truncated and the user is warned.

**Audio source in multi-clip mode**: Only the uploaded separate audio file can be used for analysis and output. The "use video's own audio" option is disabled in multi-clip mode (grayed out with tooltip explaining why). This avoids the complexity of mixing audio from multiple sources.

### Audio Duration Limits

Maximum input audio duration for analysis: 30 minutes. Librosa loads audio into memory; at 22050 Hz mono, 30 minutes is ~40MB of raw audio which is manageable. If input exceeds 30 minutes, warn the user and suggest trimming first. The trim tool can be used to reduce source material before auto-editing.

## Audio Track in Output

- **Separate audio uploaded**: The uploaded audio replaces the video's original audio entirely. The uploaded audio drives the cuts AND becomes the output audio track. The audio is laid down as-is; video cuts are placed on top of the continuous audio track.
- **"Use video's own audio" checked** (single-video mode only): The video's audio is extracted and used for analysis. In the output, audio segments are placed back-to-back corresponding to the video segments (non-contiguous source audio is expected - gaps in the source are simply skipped, producing continuous output audio from the selected segments). At cut boundaries, a short audio crossfade is applied to prevent pops/clicks. Audio `atrim` regions mirror video `trim` regions exactly (including the overlap for transitions). For video transitions with duration D, the audio `atrim` for the outgoing segment extends D seconds into the next segment's time range, and the incoming segment's `atrim` starts D seconds before its visual start, providing overlap material for `acrossfade` with duration D. For hard cuts (0 duration), a 30ms `acrossfade` is applied with 30ms of overlap. Audio and video filter graphs are built together in the same FFmpeg command to maintain sync.

## FFmpeg Interface

### Binary Detection

1. Check `PATH` for `ffmpeg` and `ffprobe`
2. Check common install locations (`C:\ffmpeg\bin`, `C:\Program Files\ffmpeg\bin`)
3. If not found, show dialog with download link (https://ffmpeg.org/download.html) and a "Browse..." button to locate the binary manually
4. Validate version supports required filters (`xfade` requires FFmpeg 4.3+)
5. Store validated FFmpeg path in config for future runs

### NVIDIA GPU Detection

1. Run `nvidia-smi` to check for NVIDIA GPU presence
2. Probe FFmpeg for `h264_nvenc` encoder support via `ffmpeg -encoders | grep nvenc`
3. If NVENC available, use for encoding; otherwise fall back to `libx264` (CPU)
4. GPU used for encoding only - audio analysis stays on CPU
5. User can override in Settings menu (force CPU encoding even with GPU available)

### VFR Handling

Variable framerate (VFR) video from action cameras and screen recordings can cause sync issues. On input, ffprobe checks for VFR by comparing `avg_frame_rate` and `r_frame_rate`; if they differ by more than 5%, the video is flagged as VFR. If detected, the pipeline inserts a preprocessing step to convert to constant framerate (CFR). The target CFR uses `r_frame_rate` (the container-declared framerate) as it represents the intended playback rate. Command: `ffmpeg -i input.mp4 -vsync cfr -r <r_frame_rate> -c:v libx264 -crf 16 -preset fast temp_cfr.mp4` (CRF 16 to preserve quality before the final re-encode, preset fast since this is preprocessing). The CFR temp file is created in the session temp directory (`<app_dir>/temp/<session_id>/`). The user is notified that VFR conversion is occurring.

### Multi-Clip Normalization

In multi-clip mode, source clips may have different resolutions, framerates, or pixel formats. Before processing, all clips are normalized to match the output settings:
- Scale all clips to the target resolution (e.g., 1920x1080) using `scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2` to handle aspect ratio differences (letterboxing rather than stretching)
- Convert all clips to the target framerate
- Convert pixel format to `yuv420p`
- Normalization is done as a preprocessing step, producing intermediate files in the session temp directory before the main edit pipeline runs
- After normalization, all filter graph construction uses the normalized intermediate files as inputs (one input per clip: `[0:v]`, `[1:v]`, etc. when segments in a transition group span multiple sources, or single-input trim when a group is within one source)

### Filter Graph Construction

Filter graphs are built programmatically from the cut list.

**Batch processing strategy**: To avoid FFmpeg filter graph complexity limits and Windows command-line length limits (~32K chars), segments are processed in batches of 20. Each batch produces an intermediate MP4 file. The final step concatenates all intermediate files using FFmpeg's concat demuxer. This keeps individual filter graphs manageable.

**Segment grouping within batches**: The cut list is scanned to identify "transition groups" - runs of consecutive segments connected by non-hard-cut transitions. Each transition group is rendered as a single FFmpeg filter graph using `xfade`. Hard cuts act as group boundaries. Groups are then concatenated via the concat demuxer. If a single transition group exceeds 20 segments, it is split at segment 20 with a forced hard cut. The xfade offset accumulator resets to zero for each new sub-group/batch (each sub-group is a separate FFmpeg invocation with its own independent filter graph). This minimizes the chance of a jarring forced cut in a low-energy passage.

**xfade offset calculation**: The `offset` parameter for the Nth xfade transition (0-indexed, where N is the transition index between segment N and segment N+1) is computed as:
```
offset_0 = duration_of_segment_0 - transition_duration_0
offset_N = offset_(N-1) + duration_of_segment_N - transition_duration_N
         (for N >= 1)
```
Where `duration_of_segment_K = segment[K].end - segment[K].start`, and `transition_duration_K` is the xfade duration for transition K. The offset represents the point in the output timeline where the Nth transition begins. Each xfade consumes `transition_duration` from the end of the outgoing segment, so the next segment's contribution to the timeline starts at `offset_(N-1) + duration_of_segment_N` minus the current transition's duration.

Worked example:
- Segment 0: 3s (trim 2-5), Segment 1: 4s (trim 12-16), Segment 2: 3s (trim 22-25)
- Transition 0 (fade): 0.3s, Transition 1 (wipe): 0.2s
- offset_0 = 3.0 - 0.3 = 2.7
- offset_1 = 2.7 + 4.0 - 0.2 = 6.5

Example filter graph for the above:

```
[0:v]trim=start=2:end=5,setpts=PTS-STARTPTS[v0];
[0:v]trim=start=12:end=16,setpts=PTS-STARTPTS[v1];
[0:v]trim=start=22:end=25,setpts=PTS-STARTPTS[v2];
[v0][v1]xfade=transition=fade:duration=0.3:offset=2.7[vt1];
[vt1][v2]xfade=transition=wipeleft:duration=0.2:offset=6.5[vout]
```

**Multi-source example** (multi-clip mode, segments from different normalized clips):

```
ffmpeg -i clip0_normalized.mp4 -i clip1_normalized.mp4 -filter_complex "
[0:v]trim=start=5:end=8,setpts=PTS-STARTPTS[v0];
[1:v]trim=start=2:end=6,setpts=PTS-STARTPTS[v1];
[0:v]trim=start=15:end=18,setpts=PTS-STARTPTS[v2];
[v0][v1]xfade=transition=fade:duration=0.3:offset=2.7[vt1];
[vt1][v2]xfade=transition=wipeleft:duration=0.2:offset=6.5[vout]
" -map "[vout]" output.mp4
```

Each FFmpeg invocation for a batch receives only the clips referenced by that batch as `-i` inputs. Clips are re-indexed sequentially (if a batch references original clips 0, 3, 5, they become `[0:v]`, `[1:v]`, `[2:v]` in that command). The filter graph builder maintains a mapping from original clip index to batch-local input index.

### Transition Mapping

- Hard cut: no xfade, segments concatenated via concat demuxer
- Crossfade: `xfade=transition=fade` (0.3s)
- Crossfade slow: `xfade=transition=fade` (0.8s, same filter, longer duration)
- Fade to black: `xfade=transition=fadeblack` (0.5s)
- Wipe left: `xfade=transition=wipeleft` (0.2s)
- Wipe right: `xfade=transition=wiperight` (0.2s)

### Video Quality

- **libx264 (CPU)**: CRF mode with CRF=18 (visually lossless). Preset `medium` for balance of speed and compression.
- **h264_nvenc (GPU)**: CQ mode with CQ=20 (comparable visual quality to CRF 18). Preset `p4` (medium quality/speed balance).
- Both produce high-quality output suitable for action/sports content.

### Export Settings

- Resolution: In single-video mode, use source resolution capped at 1080p. In multi-clip mode, use the maximum resolution among all source clips, capped at 1080p. Clips below the target are upscaled during normalization.
- Framerate: Use source fps capped at 60. In multi-clip mode, use the maximum fps among all source clips, capped at 60.
- Video codec: `h264_nvenc` (GPU) or `libx264` (CPU) with quality settings above
- Audio codec: always re-encode to `aac` at 192kbps (ensures compatibility with MP4 container regardless of uploaded audio format)
- Container: MP4

## GUI Design

### Main Window

- Menu bar: File (open, recent files, exit), Settings (FFmpeg path, GPU toggle)
- Tab widget: Auto-Edit tab, Trim tab
- Status bar: FFmpeg status, GPU availability, current operation

### Recent Files

- Track last 10 opened files (video and audio), stored in the JSON config file
- On load, check if file still exists on disk; remove stale entries silently
- Display as a submenu under File > Recent Files

### Auto-Edit Tab

- **Input section**: "Add Video(s)" button with file list, "Load Audio" button with checkbox to use video's own audio instead (disabled in multi-clip mode)
- **Drag-and-drop**: The file list accepts dropped files. Only files with recognized video extensions (mp4, mov, mkv, avi, webm, ts) are accepted. Non-video files are silently ignored. Dropping a folder is not supported (ignored). Duplicate files (same path already in list) are silently ignored.
- **Mode toggle**: Single Video / Multi-Clip assembly. Switching modes preserves the loaded file list. In single-video mode, only the first file in the list is used (others are grayed out but retained). Switching back to multi-clip mode re-enables all files.
- **Export settings**: Output folder picker, output filename field (default: `<first_video_name>_edited.mp4`), resolution/fps display (auto-detected from source with 1080p60 cap)
- **Go button**: Starts pipeline. Progress bar shows stages (Analyzing audio... Generating cuts... Encoding...)
- **Log area**: Collapsible text area showing FFmpeg output and cut decisions

### Trim Tab

- **Video loader**: Single file input
- **Range selector**: Start/end time inputs (HH:MM:SS.ms format) with manual entry
- **Multi-range support**: "Add Region" button to add additional trim ranges. Regions listed vertically with remove buttons. Regions are exported in list order.
- **Export mode toggle**: "Export as separate files" (named `<source>_trim_1.mp4`, `<source>_trim_2.mp4`, etc.) or "Concatenate into one file" (regions joined in list order)
- **Re-encode checkbox**: "Force re-encode for frame accuracy" - unchecked by default. When unchecked, uses stream copy (`-c copy`) which is fast but only cuts on keyframes (trims may be off by up to several seconds). A tooltip on the checkbox explains this trade-off. When checked, re-encodes for frame-accurate cuts. **Important**: When "Concatenate into one file" is selected, re-encode is forced on (checkbox checked and disabled) because concatenating non-keyframe-aligned stream-copy segments can produce corrupted frames at boundaries.

### Progress & Threading

- QProgressBar tied to FFmpeg progress output (parsed from stderr via time-based regex matching `time=HH:MM:SS.ms`)
- All long operations (audio analysis, FFmpeg encoding) run in QThread workers
- **Cancellation on Windows**: For FFmpeg subprocesses, cancellation is done by writing `q` to the process's stdin (FFmpeg's graceful quit mechanism). The subprocess is opened with `stdin=PIPE` to enable this. For librosa analysis, a cancel flag is checked between processing steps; if set, the worker raises an abort and returns.
- GUI remains responsive during all operations

## Error Handling & Edge Cases

### Minimum Input Requirements

- **Single-video mode**: At least one video file. Audio source: either a separate audio file OR the video must have an audio stream (if "use video's own audio" is checked).
- **Multi-clip mode**: At least two video files AND a separate audio file (since "use video's own audio" is disabled).
- **Trim tab**: Exactly one video file.
- The "Go" button is disabled until minimum requirements are met, with a tooltip explaining what's missing.

### Input Validation

- Validate files via ffprobe; reject with descriptive error if ffprobe fails. Files added to the video list that have no video stream (e.g., audio-only files) are rejected with message "No video stream found in [filename]"
- If "use video's own audio" is checked but the video has no audio stream (detected via ffprobe), disable the checkbox with tooltip "This video has no audio track" and require a separate audio file upload
- Warn if video < 5 seconds (auto-editing may not be meaningful)
- Warn if audio length differs from total video length by more than 20%. Default behavior: truncate the output to the shorter of the two durations. If video is shorter than audio, the cut generator stops when video segments are exhausted and the audio is truncated to match. If audio is shorter than video, cuts stop at the end of the audio and remaining video is unused.

### FFmpeg Failures

- Capture stderr, display meaningful errors in log area and write to log file
- If NVENC fails mid-encode, automatically retry with CPU libx264 and notify user

### Audio Analysis Edge Cases

- No beats detected (ambient/silence) or pure silence: fall back to fixed-interval cuts every 3 seconds with hard cut transitions (no crossfades or wipes, since there's no audio energy data to drive transition selection)
- Very short audio (< 2 seconds): bypass analysis entirely, produce a single segment covering the full duration
- Librosa exceptions (corrupt audio, unsupported format, `NoBackendError`, `LibsndfileError`): catch, log the full traceback, and surface a user-facing error: "Audio analysis failed: [exception message]. Please check the audio file." Abort the pipeline gracefully.

### Multi-Clip Mode

- Total video shorter than audio: truncate output to total available video duration with warning
- If one clip fails to load: skip it, continue with remaining clips, log warning

### Trim Tool

- Validate end time > start time
- Clamp values to video duration
- Warn if trim range < 0.5 seconds
- Overlapping regions are allowed (the user may intentionally want repeated sections). In "concatenate" mode, overlapping regions produce duplicate frames as expected. No automatic merging.

### Disk Space

Before starting export, estimate total disk usage and check available space on the output drive. Estimate includes:
- **Output file**: 20 Mbps for 1080p60 H.264 (150 MB per minute), scaled proportionally for lower resolutions/framerates
- **Intermediate files**: In multi-clip mode, add estimated size of normalized clips (roughly equal to total source clip duration at output bitrate) plus batch intermediates (roughly equal to output size). Also include VFR conversion intermediates if VFR was detected (roughly equal to the VFR source file size).
- **Safety margin**: Warn the user if free space is less than the total estimate. Total estimate = output + intermediates.

### Temporary File Cleanup

- All temp files are created in a dedicated directory: `<app_dir>/temp/` with session-based subdirectories (timestamp-named)
- After successful export, the session temp directory is deleted
- On application startup, scan `<app_dir>/temp/` for any stale session directories older than 24 hours and delete them (handles crash recovery)

### Logging

- All operations log to both the GUI log area and a rotating log file at `<app_dir>/logs/autoeditor.log`
- Log file rotation: max 5MB per file, keep 3 rotations
- Log includes timestamps, FFmpeg commands, cut decisions, errors, and warnings

## Configuration

Config file location: `<app_dir>/config.json` (same directory as `main.py`).

### Schema and Defaults

```json
{
  "ffmpeg_path": "",
  "ffprobe_path": "",
  "gpu_detected": true,
  "force_cpu": false,
  "last_output_folder": "",
  "last_input_folder": "",
  "recent_files": [],
  "beat_snap_tolerance_ms": 50,
  "min_cut_duration": 0.5,
  "max_cut_duration": 8.0
}
```

- `ffmpeg_path` / `ffprobe_path`: Empty string means auto-detect. Once found, stored here.
- `gpu_detected`: Whether an NVIDIA GPU with NVENC was detected at startup (informational, set by app, read-only).
- `force_cpu`: User override to disable GPU encoding even when available. This is the field controlled by the Settings > "GPU toggle" menu item. When `true`, CPU encoding is used regardless of GPU availability.
- `last_output_folder` / `last_input_folder`: Remembers the last used directories for file dialogs.
- `recent_files`: Array of up to 10 file path strings, most recent first.
- `beat_snap_tolerance_ms`: Beat-snap tolerance in milliseconds.
- `min_cut_duration` / `max_cut_duration`: Cut duration bounds in seconds.

If config file is missing or corrupt, the app creates a new one with defaults.

## Technology Stack

- **Python 3.10+**
- **PyQt6 >= 6.5** - GUI framework
- **librosa >= 0.10, < 1.0** - Audio analysis
- **numpy >= 1.23** - Numerical operations (librosa dependency)
- **soundfile >= 0.12** - Audio file I/O (librosa dependency)
- **FFmpeg >= 4.3 / FFprobe** - Video processing (external binary, not a pip package)
- **subprocess** - FFmpeg process management (stdlib)
- **json** - Config file persistence (stdlib)
- **logging** - File and console logging (stdlib)

## File Structure

```
Project2/
  src/
    main.py                  # Entry point, app initialization, startup checks
    gui/
      __init__.py
      main_window.py         # Main window, tab management, menu bar
      auto_edit_tab.py       # Auto-edit workflow UI
      trim_tab.py            # Trim tool UI
      widgets.py             # Shared widgets (progress bar, file pickers, time inputs)
      workers.py             # QThread workers for async operations
    core/
      __init__.py
      audio_analyzer.py      # librosa-based audio analysis, returns AnalysisResult
      cut_generator.py       # Cut point decision logic, returns list[CutPoint]
      export_manager.py      # ExportSettings resolution, encoding pipeline orchestration
      models.py              # Dataclass definitions (AnalysisResult, CutPoint, ExportSettings, TrimRegion)
    ffmpeg/
      __init__.py
      detector.py            # FFmpeg/ffprobe binary detection and validation
      gpu.py                 # NVIDIA GPU and NVENC detection
      filter_graph.py        # Filter graph construction, batch strategy, offset calculation
      runner.py              # Subprocess management, progress parsing, cancellation
    config.py                # Settings persistence (JSON config read/write)
  requirements.txt           # Pinned dependency versions
  logs/                      # Log file directory (created at runtime)
  temp/                      # Temporary files (created/cleaned at runtime)
```
