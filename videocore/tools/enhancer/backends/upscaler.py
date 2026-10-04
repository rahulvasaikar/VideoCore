"""Real-ESRGAN backend for super-resolution, deblurring, and sharpening."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Callable, Optional

from videocore.tools.enhancer.models import ensure_realesrgan


class RealESRGANUpscaler:
    """Wrapper around realesrgan-ncnn-vulkan for high-speed GPU upscaling."""

    def __init__(self, binary_path: Optional[Path] = None, gpu_id: int = 0):
        self.binary_path = binary_path or ensure_realesrgan()
        self.gpu_id = gpu_id

    def upscale(
        self,
        input_path: Path,
        output_path: Path,
        scale: int = 4,
        model_name: str = "realesrgan-x4plus",
        tile_size: int = 400,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Path:
        """Upscales an image file or a directory of frames.

        Args:
            input_path: Path to image file or directory containing image frames.
            output_path: Path to output image file or directory for upscaled frames.
            scale: Scale factor (usually 2, 3, or 4).
            model_name: Model name ('realesrgan-x4plus', 'realesrgan-x4plus-anime', etc.)
            tile_size: Tile size to prevent GPU out-of-memory errors (0 for auto, default 400).
            progress_callback: Optional callback receiving float percentage (0.0 to 100.0).

        Returns:
            output_path
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Input path not found: {input_path}")

        # If output is directory, make sure it exists
        if input_path.is_dir():
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(self.binary_path),
            "-i", str(input_path),
            "-o", str(output_path),
            "-s", str(scale),
            "-n", model_name,
            "-t", str(tile_size),
            "-g", str(self.gpu_id),
        ]

        # Use working directory of the binary so it can resolve its models/ folder
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
            raise RuntimeError(f"Real-ESRGAN failed with exit code {ret}.")

        return output_path
