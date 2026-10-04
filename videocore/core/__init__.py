"""Core modules for VideoCore (FFmpeg execution, media probing, process management)."""

from videocore.core.ffmpeg import get_ffmpeg_path, get_ffprobe_path, run_ffmpeg_command
from videocore.core.probe import probe_video, are_streams_compatible, VideoMetadata

__all__ = [
    "get_ffmpeg_path",
    "get_ffprobe_path",
    "run_ffmpeg_command",
    "probe_video",
    "are_streams_compatible",
    "VideoMetadata",
]
