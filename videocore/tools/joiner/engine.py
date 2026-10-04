"""Engine for joining multiple videos in a specified order."""

from __future__ import annotations

import enum
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from videocore.core.ffmpeg import detect_available_gpu_encoders, run_ffmpeg_command
from videocore.core.probe import VideoMetadata, are_streams_compatible, probe_video


class JoinMode(str, enum.Enum):
    AUTO = "auto"
    FAST = "fast"
    NORMALIZE = "normalize"


@dataclass
class JoinConfig:
    inputs: List[Path]
    output: Path
    mode: JoinMode = JoinMode.AUTO
    target_resolution: Optional[Tuple[int, int]] = None  # (width, height)
    target_fps: Optional[float] = None
    crf: int = 20
    preset: str = "medium"
    gpu: str = "auto"  # 'auto', 'nvenc', 'qsv', 'amf', or 'cpu'
    overwrite: bool = True


@dataclass
class JoinResult:
    output_path: Path
    mode_used: JoinMode
    total_duration: float
    output_size_bytes: int
    inputs_count: int
    encoder_used: str = "copy"


class VideoJoiner:
    """Handles ordering, probing, and concatenating multiple video files."""

    def __init__(self, config: JoinConfig):
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        if not self.config.inputs:
            raise ValueError("At least one input video must be provided.")
        for p in self.config.inputs:
            if not p.is_file():
                raise FileNotFoundError(f"Input video does not exist: {p}")

        if not self.config.output.parent.exists():
            self.config.output.parent.mkdir(parents=True, exist_ok=True)

        if self.config.output.exists() and not self.config.overwrite:
            raise FileExistsError(f"Output file already exists: {self.config.output}")

    def inspect_inputs(self) -> List[VideoMetadata]:
        """Probes all inputs and returns their metadata in order."""
        return [probe_video(p) for p in self.config.inputs]

    def determine_mode(
        self, metas: List[VideoMetadata]
    ) -> Tuple[JoinMode, str]:
        """Determines whether fast copy or normalize re-encode should be used."""
        if self.config.mode == JoinMode.FAST:
            compat, reason = are_streams_compatible(metas)
            return JoinMode.FAST, reason
        elif self.config.mode == JoinMode.NORMALIZE:
            return JoinMode.NORMALIZE, "Normalization explicitly requested."

        # AUTO mode
        compat, reason = are_streams_compatible(metas)
        if compat:
            return JoinMode.FAST, "All streams compatible for fast lossless copy."
        return JoinMode.NORMALIZE, f"Normalization required: {reason}"

    def join(
        self,
        on_progress: Optional[Callable[[float, str, str, str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> JoinResult:
        """Joins the videos according to configuration."""
        if on_status:
            on_status("Probing input files...")

        metas = self.inspect_inputs()
        total_duration = sum(m.duration for m in metas)

        mode_to_use, reason = self.determine_mode(metas)
        if on_status:
            on_status(f"Mode selected: {mode_to_use.value.upper()} ({reason})")

        if mode_to_use == JoinMode.FAST:
            self._join_fast(metas, total_duration, on_progress)
            encoder_used = "Stream Copy (no re-encoding)"
        else:
            encoder_used = self._join_normalized(metas, total_duration, on_progress)

        output_size = self.config.output.stat().st_size if self.config.output.exists() else 0

        return JoinResult(
            output_path=self.config.output,
            mode_used=mode_to_use,
            total_duration=total_duration,
            output_size_bytes=output_size,
            inputs_count=len(self.config.inputs),
            encoder_used=encoder_used,
        )

    def _join_fast(
        self,
        metas: List[VideoMetadata],
        total_duration: float,
        on_progress: Optional[Callable[[float, str, str, str], None]],
    ) -> None:
        """Fast lossless stream concatenation via concat demuxer."""
        # Create temp concat file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            temp_list_path = Path(f.name)
            for m in metas:
                # Escape single quotes and use forward slashes for ffmpeg concat file
                escaped = str(m.path.resolve()).replace("\\", "/").replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")

        try:
            cmd = [
                "ffmpeg",
                "-f", "concat",
                "-safe", "0",
                "-i", str(temp_list_path),
                "-c", "copy",
            ]
            if self.config.overwrite:
                cmd.append("-y")
            cmd.append(str(self.config.output))

            run_ffmpeg_command(cmd, total_duration_seconds=total_duration, on_progress=on_progress)
        finally:
            if temp_list_path.exists():
                try:
                    os.remove(temp_list_path)
                except OSError:
                    pass

    def _join_normalized(
        self,
        metas: List[VideoMetadata],
        total_duration: float,
        on_progress: Optional[Callable[[float, str, str, str], None]],
    ) -> str:
        """Normalized filter complex concatenation with canvas scaling and audio alignment."""
        # Target resolution
        if self.config.target_resolution:
            target_w, target_h = self.config.target_resolution
        else:
            # Default to first video's resolution
            v0 = metas[0].primary_video
            target_w = v0.width if v0 else 1920
            target_h = v0.height if v0 else 1080

        # Target FPS
        if self.config.target_fps:
            target_fps = self.config.target_fps
        else:
            v0 = metas[0].primary_video
            target_fps = v0.fps if (v0 and v0.fps > 0) else 30.0

        cmd: List[str] = ["ffmpeg"]

        # Add input files
        for m in metas:
            cmd.extend(["-i", str(m.path)])

        # Build filter_complex
        filter_parts: List[str] = []
        concat_inputs: List[str] = []

        for i, m in enumerate(metas):
            # Video filter: scale to fit canvas with black padding, set SAR, and set target FPS
            v_label = f"v{i}"
            v_filter = (
                f"[{i}:v]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
                f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:color=black,"
                f"setsar=1,fps={target_fps}[{v_label}]"
            )
            filter_parts.append(v_filter)

            # Audio filter: resample to 48kHz stereo, or generate silence if input lacks audio
            a_label = f"a{i}"
            if m.has_audio:
                a_filter = f"[{i}:a]aformat=sample_rates=48000:channel_layouts=stereo[{a_label}]"
            else:
                # Generate silent audio matching duration of this clip
                clip_dur = max(0.1, m.duration)
                a_filter = (
                    f"anullsrc=channel_layout=stereo:sample_rate=48000,atrim=duration={clip_dur:.3f}[{a_label}]"
                )
            filter_parts.append(a_filter)

            concat_inputs.append(f"[{v_label}][{a_label}]")

        n = len(metas)
        concat_filter = f"{''.join(concat_inputs)}concat=n={n}:v=1:a=1[outv][outa]"
        filter_parts.append(concat_filter)

        cmd.extend(["-filter_complex", ";".join(filter_parts)])
        cmd.extend(["-map", "[outv]", "-map", "[outa]"])

        # Determine video encoder (GPU acceleration or CPU fallback)
        available_gpus = detect_available_gpu_encoders()
        encoder_choice = "cpu"
        if self.config.gpu == "auto":
            if available_gpus.get("nvenc"):
                encoder_choice = "nvenc"
            elif available_gpus.get("qsv"):
                encoder_choice = "qsv"
            elif available_gpus.get("amf"):
                encoder_choice = "amf"
        elif self.config.gpu in ("nvenc", "qsv", "amf"):
            if available_gpus.get(self.config.gpu):
                encoder_choice = self.config.gpu
            else:
                encoder_choice = "cpu"

        if encoder_choice == "nvenc":
            video_encoding_args = [
                "-c:v", "h264_nvenc",
                "-preset", "p4",
                "-cq", str(self.config.crf),
                "-pix_fmt", "yuv420p",
            ]
            encoder_name = "NVIDIA NVENC (h264_nvenc)"
        elif encoder_choice == "qsv":
            video_encoding_args = [
                "-c:v", "h264_qsv",
                "-global_quality", str(self.config.crf),
                "-pix_fmt", "nv12",
            ]
            encoder_name = "Intel QuickSync (h264_qsv)"
        elif encoder_choice == "amf":
            video_encoding_args = [
                "-c:v", "h264_amf",
                "-rc", "cqp",
                "-qp_i", str(self.config.crf),
                "-qp_p", str(self.config.crf),
                "-pix_fmt", "yuv420p",
            ]
            encoder_name = "AMD AMF (h264_amf)"
        else:
            video_encoding_args = [
                "-c:v", "libx264",
                "-preset", self.config.preset,
                "-crf", str(self.config.crf),
                "-pix_fmt", "yuv420p",
            ]
            encoder_name = "CPU (libx264)"

        cmd.extend(video_encoding_args)
        cmd.extend([
            "-c:a", "aac",
            "-b:a", "192k",
        ])

        if self.config.overwrite:
            cmd.append("-y")
        cmd.append(str(self.config.output))

        run_ffmpeg_command(cmd, total_duration_seconds=total_duration, on_progress=on_progress)
        return encoder_name
