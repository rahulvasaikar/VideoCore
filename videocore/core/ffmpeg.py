"""FFmpeg binary resolution, execution, and real-time progress parsing."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple


def _search_winget_packages(binary_name: str) -> Optional[str]:
    """Search for binary in WinGet package directories."""
    app_data = os.environ.get("LOCALAPPDATA")
    if not app_data:
        return None
    winget_dir = Path(app_data) / "Microsoft" / "WinGet"
    if not winget_dir.exists():
        return None

    # Check links directory first
    link_path = winget_dir / "Links" / f"{binary_name}.exe"
    if link_path.is_file():
        return str(link_path)

    # Search in Packages
    packages_dir = winget_dir / "Packages"
    if packages_dir.exists():
        for pkg in packages_dir.iterdir():
            if pkg.is_dir() and "ffmpeg" in pkg.name.lower():
                matches = list(pkg.rglob(f"{binary_name}.exe"))
                if matches:
                    return str(matches[0])
    return None


def get_ffmpeg_path() -> str:
    """Finds a usable ffmpeg executable path."""
    # 1. Environment variable override
    env_path = os.environ.get("VIDEOCORE_FFMPEG_PATH")
    if env_path and Path(env_path).is_file():
        return env_path

    # 2. PATH
    which_path = shutil.which("ffmpeg")
    if which_path:
        return which_path

    # 3. Local project bin directory
    project_bin = Path(__file__).resolve().parent.parent.parent / "bin"
    local_bin = project_bin / ("ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")
    if local_bin.is_file():
        return str(local_bin)

    # 4. WinGet links/packages
    winget_path = _search_winget_packages("ffmpeg")
    if winget_path:
        return winget_path

    # 5. imageio_ffmpeg fallback
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).is_file():
            return exe
    except Exception:
        pass

    raise FileNotFoundError(
        "FFmpeg binary not found. Please install FFmpeg or set VIDEOCORE_FFMPEG_PATH."
    )


def get_ffprobe_path() -> Optional[str]:
    """Finds a usable ffprobe executable path, or None if unavailable."""
    # 1. Environment variable override
    env_path = os.environ.get("VIDEOCORE_FFPROBE_PATH")
    if env_path and Path(env_path).is_file():
        return env_path

    # 2. PATH
    which_path = shutil.which("ffprobe")
    if which_path:
        return which_path

    # 3. Local project bin directory
    project_bin = Path(__file__).resolve().parent.parent.parent / "bin"
    local_bin = project_bin / ("ffprobe.exe" if sys.platform == "win32" else "ffprobe")
    if local_bin.is_file():
        return str(local_bin)

    # 4. Check sibling of resolved ffmpeg
    try:
        ffmpeg = Path(get_ffmpeg_path())
        sibling = ffmpeg.parent / ("ffprobe.exe" if sys.platform == "win32" else "ffprobe")
        if sibling.is_file():
            return str(sibling)
    except Exception:
        pass

    # 5. WinGet packages
    winget_path = _search_winget_packages("ffprobe")
    if winget_path:
        return winget_path

    return None


_GPU_ENCODERS_CACHE: Optional[dict[str, bool]] = None


def detect_available_gpu_encoders() -> dict[str, bool]:
    """Checks which hardware acceleration encoders are supported by FFmpeg."""
    global _GPU_ENCODERS_CACHE
    if _GPU_ENCODERS_CACHE is not None:
        return _GPU_ENCODERS_CACHE

    res_map = {"nvenc": False, "qsv": False, "amf": False}
    try:
        ffmpeg_exe = get_ffmpeg_path()
        res = subprocess.run(
            [ffmpeg_exe, "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        out = res.stdout
        if "h264_nvenc" in out:
            res_map["nvenc"] = True
        if "h264_qsv" in out:
            res_map["qsv"] = True
        if "h264_amf" in out:
            res_map["amf"] = True
    except Exception:
        pass

    _GPU_ENCODERS_CACHE = res_map
    return res_map


def parse_time_to_seconds(timestr: str) -> float:
    """Parses HH:MM:SS.micro to total seconds."""
    try:
        parts = timestr.strip().split(":")
        if len(parts) == 3:
            h, m, s = parts
            return float(h) * 3600 + float(m) * 60 + float(s)
        elif len(parts) == 2:
            m, s = parts
            return float(m) * 60 + float(s)
        return float(parts[0])
    except ValueError:
        return 0.0


def run_ffmpeg_command(
    cmd: List[str],
    total_duration_seconds: Optional[float] = None,
    on_progress: Optional[Callable[[float, str, str, str], None]] = None,
) -> str:
    """
    Executes an FFmpeg command and parses real-time progress.
    
    Args:
        cmd: FFmpeg command args (beginning with ffmpeg executable or flags)
        total_duration_seconds: If provided, used to calculate 0.0 - 1.0 progress fraction
        on_progress: Callback with signature (fraction, speed_str, fps_str, time_str)
        
    Returns:
        The full stderr output from the FFmpeg command.
    """
    ffmpeg_exe = get_ffmpeg_path()
    if cmd[0] in ("ffmpeg", "ffmpeg.exe"):
        cmd = [ffmpeg_exe] + cmd[1:]
    elif cmd[0] != ffmpeg_exe:
        cmd = [ffmpeg_exe] + cmd

    # Run subprocess redirecting stderr for progress parsing
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
    )

    time_regex = re.compile(r"time=(\d{2}:\d{2}:\d{2}\.\d+)")
    speed_regex = re.compile(r"speed=\s*([\d\.]+x|N/A)")
    fps_regex = re.compile(r"fps=\s*([\d\.]+)")

    stderr_lines = []
    assert process.stderr is not None

    for line in process.stderr:
        stderr_lines.append(line)
        time_match = time_regex.search(line)
        if time_match:
            time_str = time_match.group(1)
            speed_match = speed_regex.search(line)
            fps_match = fps_regex.search(line)
            speed_str = speed_match.group(1) if speed_match else ""
            fps_str = fps_match.group(1) if fps_match else ""

            fraction = 0.0
            if total_duration_seconds and total_duration_seconds > 0:
                current_secs = parse_time_to_seconds(time_str)
                fraction = min(1.0, max(0.0, current_secs / total_duration_seconds))

            if on_progress:
                on_progress(fraction, speed_str, fps_str, time_str)

    process.stderr.close()
    process.wait()
    output = "".join(stderr_lines)

    if process.returncode != 0:
        raise RuntimeError(
            f"FFmpeg command failed with exit code {process.returncode}:\n{output[-1500:]}"
        )

    if on_progress and total_duration_seconds:
        on_progress(1.0, "done", "", "")

    return output
