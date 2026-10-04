"""Media inspection and stream compatibility checking using ffprobe / ffmpeg."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from videocore.core.ffmpeg import get_ffmpeg_path, get_ffprobe_path, parse_time_to_seconds


@dataclass
class VideoStream:
    codec_name: str
    width: int
    height: int
    fps: float
    pix_fmt: str
    duration: float = 0.0
    bitrate: Optional[int] = None
    sar: Optional[str] = None
    dar: Optional[str] = None


@dataclass
class AudioStream:
    codec_name: str
    sample_rate: int
    channels: int
    channel_layout: Optional[str] = None
    duration: float = 0.0


@dataclass
class VideoMetadata:
    path: Path
    duration: float
    format_name: str
    video_streams: List[VideoStream] = field(default_factory=list)
    audio_streams: List[AudioStream] = field(default_factory=list)

    @property
    def primary_video(self) -> Optional[VideoStream]:
        return self.video_streams[0] if self.video_streams else None

    @property
    def primary_audio(self) -> Optional[AudioStream]:
        return self.audio_streams[0] if self.audio_streams else None

    @property
    def has_audio(self) -> bool:
        return len(self.audio_streams) > 0


def _parse_fps(fps_str: str) -> float:
    """Parses fraction strings like '30000/1001' or '30' to float."""
    try:
        if "/" in fps_str:
            num, den = fps_str.split("/")
            den_f = float(den)
            return float(num) / den_f if den_f != 0 else 0.0
        return float(fps_str)
    except Exception:
        return 0.0


def probe_with_ffprobe(file_path: Path, ffprobe_exe: str) -> VideoMetadata:
    """Probes video using ffprobe CLI."""
    cmd = [
        ffprobe_exe,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(file_path),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {file_path}:\n{res.stderr}")

    data = json.loads(res.stdout)
    fmt = data.get("format", {})
    duration = float(fmt.get("duration", 0.0))
    format_name = fmt.get("format_name", "")

    video_streams: List[VideoStream] = []
    audio_streams: List[AudioStream] = []

    for stream in data.get("streams", []):
        codec_type = stream.get("codec_type")
        if codec_type == "video":
            v_dur = float(stream.get("duration", duration))
            fps = _parse_fps(stream.get("r_frame_rate", "0/0")) or _parse_fps(stream.get("avg_frame_rate", "0/0"))
            video_streams.append(
                VideoStream(
                    codec_name=stream.get("codec_name", "unknown"),
                    width=int(stream.get("width", 0)),
                    height=int(stream.get("height", 0)),
                    fps=fps,
                    pix_fmt=stream.get("pix_fmt", ""),
                    duration=v_dur,
                    bitrate=int(stream.get("bit_rate")) if stream.get("bit_rate") else None,
                    sar=stream.get("sample_aspect_ratio"),
                    dar=stream.get("display_aspect_ratio"),
                )
            )
        elif codec_type == "audio":
            a_dur = float(stream.get("duration", duration))
            audio_streams.append(
                AudioStream(
                    codec_name=stream.get("codec_name", "unknown"),
                    sample_rate=int(stream.get("sample_rate", 44100)),
                    channels=int(stream.get("channels", 2)),
                    channel_layout=stream.get("channel_layout"),
                    duration=a_dur,
                )
            )

    return VideoMetadata(
        path=file_path,
        duration=duration,
        format_name=format_name,
        video_streams=video_streams,
        audio_streams=audio_streams,
    )


def probe_with_ffmpeg(file_path: Path, ffmpeg_exe: str) -> VideoMetadata:
    """Fallback probe using ffmpeg -i."""
    cmd = [ffmpeg_exe, "-i", str(file_path)]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
    output = res.stderr

    duration_match = re.search(r"Duration:\s*(\d{2}:\d{2}:\d{2}\.\d+)", output)
    duration = parse_time_to_seconds(duration_match.group(1)) if duration_match else 0.0

    video_match = re.search(
        r"Stream #\d+:\d+.*?: Video: ([^,\s]+).*?, ([^,\s]+).*?, (\d+)x(\d+).*?,\s*([\d\.]+)\s*fps",
        output,
    )
    video_streams = []
    if video_match:
        codec = video_match.group(1)
        pix_fmt = video_match.group(2)
        w = int(video_match.group(3))
        h = int(video_match.group(4))
        fps = float(video_match.group(5))
        video_streams.append(
            VideoStream(codec_name=codec, width=w, height=h, fps=fps, pix_fmt=pix_fmt, duration=duration)
        )

    audio_match = re.search(r"Stream #\d+:\d+.*?: Audio: ([^,\s]+).*?,\s*(\d+)\s*Hz,\s*([^,]+)", output)
    audio_streams = []
    if audio_match:
        a_codec = audio_match.group(1)
        sr = int(audio_match.group(2))
        channels = 2 if "stereo" in audio_match.group(3) else 1
        audio_streams.append(
            AudioStream(codec_name=a_codec, sample_rate=sr, channels=channels, duration=duration)
        )

    return VideoMetadata(
        path=file_path,
        duration=duration,
        format_name="ffmpeg_probe",
        video_streams=video_streams,
        audio_streams=audio_streams,
    )


def probe_video(file_path: str | Path) -> VideoMetadata:
    """Inspects a video file and returns its metadata."""
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {file_path}")

    ffprobe = get_ffprobe_path()
    if ffprobe:
        try:
            return probe_with_ffprobe(path, ffprobe)
        except Exception:
            pass

    ffmpeg = get_ffmpeg_path()
    return probe_with_ffmpeg(path, ffmpeg)


def are_streams_compatible(videos: List[VideoMetadata]) -> Tuple[bool, str]:
    """
    Checks if all videos are compatible for lossless direct stream concatenation (-c copy).
    Returns (compatible_bool, explanation_string).
    """
    if len(videos) < 2:
        return True, "Single or zero videos provided."

    first = videos[0]
    v0 = first.primary_video
    if not v0:
        return False, f"First input '{first.path.name}' has no video stream."

    a0 = first.primary_audio

    for i, meta in enumerate(videos[1:], start=2):
        vi = meta.primary_video
        if not vi:
            return False, f"Video #{i} ('{meta.path.name}') has no video stream."

        # Compare dimensions
        if (vi.width, vi.height) != (v0.width, v0.height):
            return False, (
                f"Resolution mismatch: Video #1 is {v0.width}x{v0.height}, "
                f"but Video #{i} ('{meta.path.name}') is {vi.width}x{vi.height}."
            )

        # Compare video codec
        if vi.codec_name.lower() != v0.codec_name.lower():
            return False, (
                f"Video codec mismatch: Video #1 is {v0.codec_name}, "
                f"but Video #{i} ('{meta.path.name}') is {vi.codec_name}."
            )

        # Compare fps (allow 0.1 delta)
        if abs(vi.fps - v0.fps) > 0.1:
            return False, (
                f"Framerate mismatch: Video #1 is {v0.fps:.2f} fps, "
                f"but Video #{i} ('{meta.path.name}') is {vi.fps:.2f} fps."
            )

        # Compare pixel format
        if vi.pix_fmt and v0.pix_fmt and vi.pix_fmt != v0.pix_fmt:
            return False, (
                f"Pixel format mismatch: Video #1 is {v0.pix_fmt}, "
                f"but Video #{i} ('{meta.path.name}') is {vi.pix_fmt}."
            )

        # Compare audio presence and parameters
        ai = meta.primary_audio
        if bool(a0) != bool(ai):
            return False, (
                f"Audio stream mismatch: Video #1 {'has' if a0 else 'lacks'} audio, "
                f"but Video #{i} ('{meta.path.name}') {'has' if ai else 'lacks'} audio."
            )

        if a0 and ai:
            if a0.codec_name.lower() != ai.codec_name.lower():
                return False, (
                    f"Audio codec mismatch: Video #1 is {a0.codec_name}, "
                    f"but Video #{i} ('{meta.path.name}') is {ai.codec_name}."
                )
            if a0.sample_rate != ai.sample_rate:
                return False, (
                    f"Audio sample rate mismatch: Video #1 is {a0.sample_rate}Hz, "
                    f"but Video #{i} ('{meta.path.name}') is {ai.sample_rate}Hz."
                )
            if a0.channels != ai.channels:
                return False, (
                    f"Audio channels mismatch: Video #1 has {a0.channels} channels, "
                    f"but Video #{i} ('{meta.path.name}') has {ai.channels} channels."
                )

    return True, "All video and audio streams match identical parameters."
