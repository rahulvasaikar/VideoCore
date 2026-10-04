"""RIFE backend for motion interpolation and video framerate smoothing."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Callable, Optional

from videocore.tools.enhancer.models import ensure_rife


class VideoInterpolator:
    """Wrapper around rife-ncnn-vulkan for high-performance AI frame interpolation."""

    def __init__(self, binary_path: Optional[Path] = None, gpu_id: int = 0):
        self.binary_path = binary_path or ensure_rife()
        self.gpu_id = gpu_id

    def interpolate(
        self,
        input_dir: Path,
        output_dir: Path,
        model_name: str = "rife-v4.6",
        target_frame_count: Optional[int] = None,
        multiplier: Optional[int] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Path:
        """Interpolates frames between consecutive images in input_dir and saves into output_dir.

        Args:
            input_dir: Directory containing ordered input frame images.
            output_dir: Destination directory for interpolated frames.
            model_name: RIFE model preset ('rife-v4.6', 'rife-v4', 'rife-anime', etc.)
            target_frame_count: Total target frame count.
            multiplier: Multiplier factor (e.g. 2 for 2x frames, 4 for 4x frames).
            progress_callback: Optional callback receiving float percentage (0.0 to 100.0).

        Returns:
            output_dir
        """
        if not input_dir.is_dir():
            raise NotADirectoryError(f"Input directory not found: {input_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(self.binary_path),
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-m", model_name,
            "-g", str(self.gpu_id),
        ]

        if target_frame_count:
            cmd.extend(["-n", str(target_frame_count)])
        elif multiplier and multiplier > 1:
            # Count input frames
            input_frames = len([f for f in input_dir.iterdir() if f.suffix.lower() in [".png", ".jpg", ".webp"]])
            cmd.extend(["-n", str(input_frames * multiplier)])

        cwd = self.binary_path.parent

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(cwd),
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        pct_regex = re.compile(r"(\d+(?:\.\d+)?)%")

        if process.stdout:
            for line in process.stdout:
                line_str = line.strip()
                match = pct_regex.search(line_str)
                if match and progress_callback:
                    try:
                        pct = float(match.group(1))
                        progress_callback(pct)
                    except ValueError:
                        pass

        ret = process.wait()
        if ret != 0:
            raise RuntimeError(f"RIFE frame interpolation failed with exit code {ret}.")

        return output_dir
