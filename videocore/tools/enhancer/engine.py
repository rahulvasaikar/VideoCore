"""Enhancement engine orchestrating image and video pipelines."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

from videocore.core.constants import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from videocore.core.ffmpeg import get_ffmpeg_path
from videocore.core.probe import probe_video
from videocore.tools.enhancer.backends import FaceRestorer, RealESRGANUpscaler, VideoInterpolator


@dataclass
class EnhanceConfig:
    input_path: Path
    output_path: Optional[Path] = None
    scale: int = 2
    model_name: str = "realesrgan-x4plus"
    face_restore: bool = False
    face_fidelity: float = 0.7
    interpolate: bool = False
    target_fps: Optional[float] = None
    tile_size: int = 400
    gpu_id: int = 0
    video_codec: str = "h264_nvenc"
    temp_dir: Optional[Path] = None


@dataclass
class EnhanceResult:
    input_path: Path
    output_path: Path
    is_video: bool
    original_resolution: Tuple[int, int]
    enhanced_resolution: Tuple[int, int]
    original_fps: Optional[float] = None
    enhanced_fps: Optional[float] = None
    faces_restored: bool = False
    interpolated: bool = False


class EnhancerEngine:
    """Orchestrates multi-stage AI enhancement for images and videos."""

    def __init__(self, config: EnhanceConfig):
        self.config = config
        self.upscaler = RealESRGANUpscaler(gpu_id=config.gpu_id)
        self.interpolator = VideoInterpolator(gpu_id=config.gpu_id)
        self.face_restorer = FaceRestorer() if config.face_restore else None

    def execute(
        self,
        stage_callback: Optional[Callable[[str, float], None]] = None,
    ) -> EnhanceResult:
        """Executes the enhancement pipeline based on input type."""
        inp = self.config.input_path
        if not inp.exists():
            raise FileNotFoundError(f"Input path not found: {inp}")

        ext = inp.suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            return self._enhance_image(inp, stage_callback)
        elif ext in VIDEO_EXTENSIONS:
            return self._enhance_video(inp, stage_callback)
        else:
            raise ValueError(f"Unsupported file format '{ext}'. Supported formats: {IMAGE_EXTENSIONS | VIDEO_EXTENSIONS}")

    def _determine_image_output_path(self, inp: Path) -> Path:
        if self.config.output_path:
            return self.config.output_path
        return inp.parent / f"{inp.stem}_enhanced.png"

    def _determine_video_output_path(self, inp: Path) -> Path:
        if self.config.output_path:
            return self.config.output_path
        return inp.parent / f"{inp.stem}_enhanced.mp4"

    def _enhance_image(
        self,
        inp: Path,
        callback: Optional[Callable[[str, float], None]] = None,
    ) -> EnhanceResult:
        import cv2
        orig = cv2.imread(str(inp))
        if orig is None:
            raise ValueError(f"Could not read image: {inp}")
        orig_h, orig_w = orig.shape[:2]

        out_path = self._determine_image_output_path(inp)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Temporary file if chaining upscaling and face restoration
        needs_upscale = self.config.scale > 1
        needs_face = self.config.face_restore

        current_file = inp
        temp_upscaled: Optional[Path] = None

        try:
            if needs_upscale:
                if callback:
                    callback("Upscaling & Deblurring", 0.0)
                temp_upscaled = out_path.with_name(f"temp_{uuid.uuid4().hex[:8]}_upscaled.png")
                self.upscaler.upscale(
                    input_path=current_file,
                    output_path=temp_upscaled,
                    scale=self.config.scale,
                    model_name=self.config.model_name,
                    tile_size=self.config.tile_size,
                    progress_callback=lambda p: callback("Upscaling & Deblurring", p) if callback else None,
                )
                current_file = temp_upscaled

            if needs_face:
                if callback:
                    callback("Restoring Faces", 0.0)
                assert self.face_restorer is not None
                self.face_restorer.enhance_image_file(
                    input_file=current_file,
                    output_file=out_path,
                    fidelity=self.config.face_fidelity,
                )
                if callback:
                    callback("Restoring Faces", 100.0)
            else:
                # If only upscaled, copy or re-encode to final output format
                if current_file != out_path:
                    if current_file.suffix.lower() == out_path.suffix.lower():
                        shutil.copyfile(current_file, out_path)
                    else:
                        c_img = cv2.imread(str(current_file))
                        if c_img is not None:
                            cv2.imwrite(str(out_path), c_img)
                        else:
                            shutil.copyfile(current_file, out_path)

        finally:
            if temp_upscaled and temp_upscaled.exists():
                temp_upscaled.unlink()

        final_img = cv2.imread(str(out_path))
        final_h, final_w = (final_img.shape[:2]) if final_img is not None else (orig_h * self.config.scale, orig_w * self.config.scale)

        return EnhanceResult(
            input_path=inp,
            output_path=out_path,
            is_video=False,
            original_resolution=(orig_w, orig_h),
            enhanced_resolution=(final_w, final_h),
            faces_restored=needs_face,
            interpolated=False,
        )

    def _get_best_temp_dir(self, inp: Path, out_path: Optional[Path] = None) -> Path:
        """Determines the temporary folder location on the drive with the most free space."""
        if self.config.temp_dir and Path(self.config.temp_dir).exists():
            return Path(self.config.temp_dir)

        candidates = []
        if inp.parent.exists():
            candidates.append(inp.parent)
        if out_path and out_path.parent.exists():
            candidates.append(out_path.parent)
        candidates.append(Path(tempfile.gettempdir()))

        best_cand = candidates[0]
        max_free = 0
        for cand in candidates:
            try:
                free = shutil.disk_usage(cand).free
                if free > max_free:
                    max_free = free
                    best_cand = cand
            except Exception:
                pass
        return best_cand

    def _enhance_video(
        self,
        inp: Path,
        callback: Optional[Callable[[str, float], None]] = None,
    ) -> EnhanceResult:
        meta = probe_video(inp)
        if not meta.primary_video:
            raise ValueError(f"No video streams found in {inp}")

        v_stream = meta.primary_video
        orig_w, orig_h = v_stream.width, v_stream.height
        orig_fps = v_stream.fps

        out_path = self._determine_video_output_path(inp)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        ffmpeg_bin = get_ffmpeg_path()
        base_temp = self._get_best_temp_dir(inp, out_path)

        # Pre-flight disk space estimation
        total_frames = int(meta.duration * (orig_fps or 30.0))
        # Estimate: 1080p frame PNG ~ 2MB; 4K ~ 8MB
        mb_per_frame = (orig_w * orig_h * 3) / (1024 * 1024 * 3.5)
        est_gb = (total_frames * mb_per_frame * 1.5) / 1024
        try:
            free_gb = shutil.disk_usage(base_temp).free / (1024**3)
            if free_gb < 5.0 or (est_gb > 15.0 and free_gb < est_gb * 0.7):
                drive_name = str(base_temp.drive or base_temp)
                raise RuntimeError(
                    f"Insufficient disk space on {drive_name} ({free_gb:.1f} GB free). "
                    f"Enhancing this {meta.duration/60:.1f} min video (~{total_frames:,} frames) requires estimated ~{est_gb:.1f} GB temporary space. "
                    f"Please free up space or choose an output drive with more free space."
                )
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise

        work_dir = Path(tempfile.mkdtemp(prefix="videocore_enhance_", dir=base_temp))

        try:
            frames_dir = work_dir / "raw_frames"
            frames_dir.mkdir(parents=True, exist_ok=True)

            # 1. Extract frames
            if callback:
                callback("Extracting Video Frames", 0.0)
            extract_cmd = [
                ffmpeg_bin, "-y",
                "-i", str(inp),
                "-qscale:v", "1",
                "-qmin", "1",
                str(frames_dir / "%08d.png"),
            ]
            extract_proc = subprocess.run(extract_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if extract_proc.returncode != 0:
                err_text = extract_proc.stderr.decode("utf-8", errors="replace")[-1000:].strip()
                raise RuntimeError(f"FFmpeg frame extraction failed (exit code {extract_proc.returncode}): {err_text}")
            if callback:
                callback("Extracting Video Frames", 100.0)

            active_frames = frames_dir
            needs_upscale = self.config.scale > 1

            # 2. Upscaling / Deblurring
            if needs_upscale:
                if callback:
                    callback("AI Super-Resolution & Deblur", 0.0)
                upscaled_dir = work_dir / "upscaled_frames"
                self.upscaler.upscale(
                    input_path=active_frames,
                    output_path=upscaled_dir,
                    scale=self.config.scale,
                    model_name=self.config.model_name,
                    tile_size=self.config.tile_size,
                    progress_callback=lambda p: callback("AI Super-Resolution & Deblur", p) if callback else None,
                )
                # Cleanup raw frames to free space
                shutil.rmtree(active_frames, ignore_errors=True)
                active_frames = upscaled_dir

            # 3. Face Restoration
            if self.config.face_restore:
                if callback:
                    callback("AI Face Restoration", 0.0)
                assert self.face_restorer is not None

                def face_prog(current: int, total: int):
                    if callback and total > 0:
                        callback("AI Face Restoration", (current / total) * 100.0)

                self.face_restorer.enhance_frames_dir(
                    frames_dir=active_frames,
                    fidelity=self.config.face_fidelity,
                    progress_callback=face_prog,
                )

            # 4. Frame Interpolation
            out_fps = orig_fps
            is_interpolated = False
            if self.config.interpolate or self.config.target_fps:
                if callback:
                    callback("RIFE Frame Interpolation", 0.0)

                target_fps = self.config.target_fps or (orig_fps * 2.0)
                multiplier = max(2, int(round(target_fps / orig_fps)))
                interpolated_dir = work_dir / "interpolated_frames"

                self.interpolator.interpolate(
                    input_dir=active_frames,
                    output_dir=interpolated_dir,
                    multiplier=multiplier,
                    progress_callback=lambda p: callback("RIFE Frame Interpolation", p) if callback else None,
                )
                shutil.rmtree(active_frames, ignore_errors=True)
                active_frames = interpolated_dir
                out_fps = orig_fps * multiplier
                is_interpolated = True

            # 5. Video Re-encoding with Hardware Acceleration (NVENC)
            if callback:
                callback("Encoding Video (NVENC)", 0.0)

            encode_cmd = [
                ffmpeg_bin, "-y",
                "-framerate", str(out_fps),
                "-i", str(active_frames / "%08d.png"),
            ]

            # If input had audio, add audio stream
            if meta.has_audio:
                encode_cmd.extend(["-i", str(inp), "-map", "0:v:0", "-map", "1:a:0?"])
            else:
                encode_cmd.extend(["-map", "0:v:0"])

            # Use NVENC codec with graceful fallback to libx264
            codec = self.config.video_codec
            if "nvenc" in codec:
                encode_cmd.extend([
                    "-c:v", codec,
                    "-preset", "p7",
                    "-cq", "19",
                    "-pix_fmt", "yuv420p",
                ])
            else:
                encode_cmd.extend([
                    "-c:v", "libx264",
                    "-crf", "18",
                    "-preset", "slow",
                    "-pix_fmt", "yuv420p",
                ])

            if meta.has_audio:
                encode_cmd.extend(["-c:a", "copy"])

            encode_cmd.append(str(out_path))

            encode_res = subprocess.run(encode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if encode_res.returncode != 0 and "nvenc" in codec:
                # Fallback to libx264 if NVENC had an error
                encode_cmd[encode_cmd.index(codec)] = "libx264"
                if "-cq" in encode_cmd:
                    cq_idx = encode_cmd.index("-cq")
                    encode_cmd[cq_idx] = "-crf"
                    encode_cmd[cq_idx + 1] = "18"
                subprocess.run(encode_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            if callback:
                callback("Encoding Video (NVENC)", 100.0)

        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

        final_w = orig_w * self.config.scale if needs_upscale else orig_w
        final_h = orig_h * self.config.scale if needs_upscale else orig_h

        return EnhanceResult(
            input_path=inp,
            output_path=out_path,
            is_video=True,
            original_resolution=(orig_w, orig_h),
            enhanced_resolution=(final_w, final_h),
            original_fps=orig_fps,
            enhanced_fps=out_fps,
            faces_restored=self.config.face_restore,
            interpolated=is_interpolated,
        )
