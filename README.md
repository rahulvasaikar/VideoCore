# VideoCore

**VideoCore** is a modular, high-performance video processing toolkit designed to host an extensible suite of video manipulation utilities.

The first tool in the suite is the **Video Joiner**, allowing you to merge multiple video clips into a single video file in any specified order, featuring automatic stream compatibility detection, instant lossless stream concatenation, and intelligent normalization for disparate resolutions, framerates, and audio layouts.

---

## Key Features

- **Multi-Video Sequencing**: Pass any number of videos via command line arguments or a playlist manifest file.
- **Custom Ordering**: Order videos directly on the CLI or use `--order 3,1,2` to dynamically rearrange clips without changing the input flags.
- **Zero-Loss Fast Mode**: Automatically uses FFmpeg's concat demuxer (`-c copy`) when clips have matching video/audio streams, joining files in seconds with zero quality degradation.
- **Smart Normalization**: When videos have mismatched resolutions, aspect ratios, or framerates, VideoCore scales them cleanly to a unified canvas (with smart letterboxing/pillarboxing) and resamples audio.
- **Missing Audio Handling**: If one clip has audio and another is silent, VideoCore automatically synthesizes synchronized silent audio tracks to prevent audio/video desynchronization.
- **Extensible Architecture**: Built around a clean `BaseTool` registry so additional utilities (trimmer, compressor, converter, audio extractor, watermarker) can be seamlessly plugged in.
- **Rich Terminal Feedback**: Formatted tables detailing every clip's resolution, FPS, duration, and audio track, accompanied by live progress bars and speed metrics.

---

## Installation & Setup

### 1. Virtual Environment & Dependencies

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment:
# If using Git Bash (MINGW64):
source .venv/Scripts/activate

# If using PowerShell:
# .\.venv\Scripts\Activate.ps1

# If using Command Prompt (cmd.exe):
# .venv\Scripts\activate.bat

# Install dependencies and VideoCore in editable mode
pip install -r requirements.txt
pip install -e .
```

### 2. FFmpeg & FFprobe

VideoCore automatically resolves `ffmpeg` and `ffprobe` by checking:
1. `VIDEOCORE_FFMPEG_PATH` and `VIDEOCORE_FFPROBE_PATH` environment variables.
2. System `PATH`.
3. Local `bin/` directory (`VideoCore/bin/ffmpeg.exe`).
4. Windows WinGet package locations (`Gyan.FFmpeg.Essentials`).
5. Python bundled static binaries (`imageio-ffmpeg`).

---

## Interactive App: VideoCore Studio

VideoCore includes an interactive graphical interface featuring native Windows File Explorer integration, video playback, sequence reordering, and side-by-side AI enhancement previews.

### Launch Methods:

#### 1. One-Click Desktop Launcher
Simply double-click **`Run_VideoCore_App.bat`** in the project folder! It will activate the virtual environment and automatically open the studio in your browser.

#### 2. CLI Command
```bash
videocore ui
# Or using aliases:
videocore app
```

---

## CLI Usage: Video Joiner

The `join` command is available directly through the `videocore` CLI:

```bash
videocore join [OPTIONS]
```

### 1. Joining an Entire Folder (Auto-Order & In-Place Output)

Simply pass the directory path. VideoCore will discover all video files, sort them naturally by filename (e.g. `MOV00021.AVI`, `MOV00022.AVI`), and save the combined output directly in that same folder:

```bash
# Pass the folder path directly:
videocore join "G:\Temp\New folder (3)"

# Or using the -d / --folder flag:
videocore join -d "G:\Temp\New folder (3)"
```
Output is automatically saved inside the folder as `G:\Temp\New folder (3)\New folder (3)_joined.avi` (or `.mp4`).

### 2. Basic Sequential Join (Specific Files)

Join specific clips in the order they are listed:

```bash
videocore join -i intro.mp4 clip1.mp4 clip2.mp4 outro.mp4 -o final_video.mp4
```

### 3. Reordering Clips with `--order`

Specify a custom sequence order using 1-based indices:

```bash
# Plays clip3 first, then clip1, then clip2:
videocore join -i clip1.mp4 clip2.mp4 clip3.mp4 --order 3,1,2 -o reordered.mp4
```

### 3. Using a Playlist / Manifest File

You can store a list of video paths in a `.txt` file (one video per line, comments starting with `#` supported):

```text
# playlist.txt
samples/part1.mp4
samples/part2.mp4
samples/part3.mp4
```

Then run:

```bash
videocore join -l playlist.txt -o joined.mp4
```

### 4. Normalization and Resolution Targeting

Standardize different video resolutions to 1080p, 720p, 4K, or custom dimensions:

```bash
# Preset resolutions (1080p, 720p, 4k, vertical_1080p, etc.)
videocore join -i clip_720p.mp4 clip_4k.mp4 -r 1080p -o standardized.mp4

# Custom dimensions
videocore join -i clip1.mp4 clip2.mp4 -r 1920x1080 --fps 60 -o output.mp4
```

### 5. Mode Selection

- `--mode auto` (default): Probes input videos. If they have matching resolution, framerate, and codecs, uses instant fast copy. Otherwise, automatically falls back to normalized re-encoding.
- `--mode fast`: Forces lossless stream copy (`-c copy`).
- `--mode normalize`: Forces canvas scaling and re-encoding.

### 6. Hardware GPU Acceleration (NVIDIA NVENC, Intel QSV, AMD AMF)

VideoCore automatically detects your GPU and offloads video encoding to dedicated hardware:

```bash
# Automatically detects and uses your NVIDIA RTX GPU (NVENC):
videocore join -i clip1.mp4 clip2.mp4 -m normalize -g auto -o output.mp4

# Explicitly select GPU acceleration:
videocore join -i clip1.mp4 clip2.mp4 -m normalize -g nvenc -o output.mp4

# Or force CPU-only encoding:
videocore join -i clip1.mp4 clip2.mp4 -m normalize -g cpu -o output.mp4
```

### 7. Encoding Quality & Presets

Fine-tune output size and visual quality when re-encoding:

```bash
videocore join -i clip1.mp4 clip2.mp4 --crf 18 --preset slow -o high_quality.mp4
```

---

## Programmatic Python API

You can also use VideoCore directly in your Python code:

```python
from pathlib import Path
from videocore.tools.joiner import VideoJoiner, JoinConfig, JoinMode
from videocore.core.probe import probe_video

# 1. Inspect metadata
info = probe_video("samples/part1.mp4")
print(f"Duration: {info.duration}s, Resolution: {info.primary_video.width}x{info.primary_video.height}")

# 2. Configure and run joiner
config = JoinConfig(
    inputs=[
        Path("samples/part1.mp4"),
        Path("samples/part2.mp4"),
        Path("samples/part3.mp4"),
    ],
    output=Path("output.mp4"),
    mode=JoinMode.AUTO,
    target_resolution=(1920, 1080),  # Optional
    target_fps=30.0,                 # Optional
    crf=20,
    preset="medium",
)

joiner = VideoJoiner(config)
result = joiner.join(
    on_progress=lambda frac, speed, fps, time: print(f"Progress: {frac*100:.1f}% ({speed})")
)

print(f"Done! Output saved to {result.output_path} ({result.output_size_bytes} bytes)")
```

---

## Running Tests

Run the comprehensive unit test suite:

```bash
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

---

---

## CLI Usage: AI Media Enhancer

The `enhance` command provides GPU-accelerated deep learning enhancement for pictures and videos:

```bash
videocore enhance [OPTIONS]
```

### Key Capabilities:
- **Spatial Super-Resolution & Deblurring**: 2x, 4x upscaling, artifact reduction, and edge sharpening using **Real-ESRGAN**.
- **Facial Reconstruction & Restoration**: Detects faces with **YuNet** and reconstructs crisp facial features (eyes, mouth, skin) with **GFPGAN v1.4**.
- **Motion Smoothing & Interpolation**: Synthesizes intermediate frames using **RIFE** to turn choppy footage into fluid 60 fps.
- **Hardware Acceleration**: Automatically accelerates models on the **NVIDIA RTX 3070 Ti** via Vulkan / DirectML and re-encodes video with **NVIDIA NVENC** (`h264_nvenc`).

### Examples:

#### 1. Image Super-Resolution & Deblur (4x)
```bash
videocore enhance -i blurry_photo.jpg -o clear_photo.png -s 4
```

#### 2. Portrait & Face Restoration
```bash
videocore enhance -i old_portrait.jpg -o restored.png --face-restore --face-fidelity 0.7
```

#### 3. Video Motion Smoothing (30fps -> 60fps)
```bash
videocore enhance -i clip.mp4 -o clip_60fps.mp4 --target-fps 60
```

#### 4. Video 2x Upscale & Deblur with NVENC
```bash
videocore enhance -i 720p_video.mp4 -o 1440p_video.mp4 -s 2 --codec h264_nvenc
```

#### 5. Full Enhancement Suite (Upscale + Face Restore + 60fps)
```bash
videocore enhance -i home_video.mp4 -o restored_video.mp4 -s 2 --face-restore --target-fps 60
```

---

## CLI Usage: Real ⇄ Animation Studio

The `animate` command (with aliases `stylize`, `anime`) transforms real video/photos into anime and converts animation back into photorealistic video:

```bash
videocore animate [OPTIONS]
```

### 1. Real Video ➔ Anime (Studio Ghibli / Hayao Style)
```bash
videocore animate -i real_video.mp4 -o anime_video.mp4 -s hayao --codec h264_nvenc
```

### 2. Real Video ➔ Makoto Shinkai Style (Luminous Lighting)
```bash
videocore animate -i scenery.mp4 -o shinkai_scenery.mp4 -s shinkai
```

### 3. Fast Cartoon & Comic Filters
```bash
# Cel-shaded cartoon
videocore animate -i video.mp4 -o cartoon.mp4 -s cartoon --line-weight 1.2

# Manga / Comic ink
videocore animate -i video.mp4 -o comic.mp4 -s comic
```

### 4. Animation ➔ Real Video (Photorealism & Film Grain)
```bash
# De-cartoonize, soften ink outlines, and inject 35mm film grain
videocore animate -i anime_clip.mp4 -o real_clip.mp4 -d anim_to_real --edge-soften 0.7 --film-grain 0.15

# With AI Face Realism (GFPGAN facial reconstruction)
videocore animate -i anime_portrait.png -o photoreal.png -d anim_to_real --face-realism
```

---

## CLI Usage: Face Studio & Replacement

The `faces` command (aliases `face-studio`, `faceswap`) scans videos to detect and cluster all unique faces, inspect appearance counts, and replace any target face with custom pictures:

```bash
videocore faces [OPTIONS]
```

### 1. Scan Video for All Faces & Cluster Individuals
```bash
videocore faces -i interview.mp4 --scan-only
```
Outputs a table listing all detected people (`person_1`, `person_2`), appearance frame counts, and first appearance timestamps.

### 2. Replace a Specific Face Across the Video
```bash
videocore faces -i video.mp4 -o swapped_video.mp4 --target person_1 --new-pic new_actor.jpg --codec h264_nvenc
```
Automatically aligns replacement face landmarks, matches skin tone and lighting with the scene, and enhances details with GFPGAN.

---

## Adding New Utilities to VideoCore

To add a new utility to VideoCore (e.g., `trim`, `compress`, `convert`):

1. Subclass `BaseTool` from `videocore.tools.base`:
   ```python
   from videocore.tools.base import BaseTool

   class TrimmerTool(BaseTool):
       @property
       def name(self) -> str:
           return "trim"

       @property
       def description(self) -> str:
           return "Trim start and end of a video."

       def register_subcommand(self, subparser):
           parser = subparser.add_parser(self.name, help=self.description)
           parser.add_argument("-i", "--input", required=True)
           # add args...

       def execute(self, args) -> int:
           # implement logic...
           return 0
   ```
2. Register the tool instance in `videocore/tools/__init__.py`:
   ```python
   from videocore.tools.joiner import JoinerTool
   from videocore.tools.enhancer import EnhancerTool
   from videocore.tools.trimmer import TrimmerTool

   TOOLS = [
       JoinerTool(),
       EnhancerTool(),
       TrimmerTool(),
   ]
   ```
The CLI automatically picks up the new command!

