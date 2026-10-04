"""Face Studio Execution Engine: orchestrates analysis, preview, and video rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import time
from typing import Callable, Dict, List, Optional, Tuple

from videocore.core.constants import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
import cv2
import numpy as np

from videocore.core.ffmpeg import detect_available_gpu_encoders, get_ffmpeg_path
from videocore.core.probe import probe_video
from videocore.tools.face_studio.analyzer import DetectedFace, VideoFaceAnalyzer
from videocore.tools.face_studio.swapper import FaceSwapper

@dataclass
class FaceStudioConfig:
    input_path: Path
    output_path: Optional[Path] = None
    replacement_map: Optional[Dict[str, Path]] = None  # e.g. {"person_1": Path("new_actor.jpg")}
    blend_strength: float = 1.0
    enable_gfpgan: bool = True
    gfpgan_fidelity: float = 0.70
    similarity_threshold: float = 0.40
    video_codec: str = "h264_nvenc"


@dataclass
class FaceStudioResult:
    input_path: Path
    output_path: Path
    is_video: bool
    resolution: Tuple[int, int]
    duration: Optional[float] = None
    fps: Optional[float] = None
    swapped_faces_count: int = 0


class FaceStudioEngine:
    """Orchestrates face analysis, sample previews, and full video face swapping."""

    def __init__(self, config: FaceStudioConfig):
        self.config = config
        self.analyzer = VideoFaceAnalyzer(similarity_threshold=config.similarity_threshold)
        self.swapper = FaceSwapper(
            similarity_threshold=config.similarity_threshold,
            enable_gfpgan=config.enable_gfpgan,
            gfpgan_fidelity=config.gfpgan_fidelity,
        )

    def analyze_faces(
        self,
        sample_interval: float = 0.5,
        progress_cb: Optional[Callable[[str, float], None]] = None,
    ) -> List[DetectedFace]:
        """Analyzes input media to detect all distinct individuals."""
        inp = self.config.input_path
        if not inp.exists():
            raise FileNotFoundError(f"Input file not found: {inp}")

        ext = inp.suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            return self.analyzer.analyze_image(inp)
        elif ext in VIDEO_EXTENSIONS:
            return self.analyzer.analyze_video(inp, sample_interval_sec=sample_interval, progress_callback=progress_cb)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def preview_swap_frame(
        self,
        target_clusters: List[DetectedFace],
        timestamp: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generates a Before and After frame at the given timestamp with replacement applied."""
        inp = self.config.input_path
        if not inp.exists():
            raise FileNotFoundError(f"Input file not found: {inp}")

        # Pre-load replacement faces
        sources: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        if self.config.replacement_map:
            for p_id, pic_p in self.config.replacement_map.items():
                if Path(pic_p).is_file():
                    sources[p_id] = self.swapper.extract_source_face(Path(pic_p))

        ext = inp.suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            frame = cv2.imread(str(inp))
            if frame is None:
                raise ValueError(f"Could not load image: {inp}")
        else:
            cap = cv2.VideoCapture(str(inp))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_no = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                raise RuntimeError(f"Could not read frame from video: {inp}")

        # Swap
        swapped = self.swapper.swap_faces_in_frame(
            frame,
            target_clusters,
            sources,
            blend_factor=self.config.blend_strength,
        )
        return frame, swapped

    def execute_swap(
        self,
        target_clusters: List[DetectedFace],
        stage_callback: Optional[Callable[[str, float], None]] = None,
    ) -> FaceStudioResult:
        """Executes full face replacement on video or image."""
        inp = self.config.input_path
        if not inp.exists():
            raise FileNotFoundError(f"Input file not found: {inp}")

        # Load replacement faces
        sources: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        if self.config.replacement_map:
            for p_id, pic_p in self.config.replacement_map.items():
                if Path(pic_p).is_file():
                    sources[p_id] = self.swapper.extract_source_face(Path(pic_p))

        ext = inp.suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            return self._execute_image_swap(inp, target_clusters, sources, stage_callback)
        elif ext in VIDEO_EXTENSIONS:
            return self._execute_video_swap(inp, target_clusters, sources, stage_callback)
        else:
            raise ValueError(f"Unsupported format: {ext}")

    def _determine_output_path(self, inp: Path, is_video: bool) -> Path:
        if self.config.output_path:
            return self.config.output_path
        sfx = ".mp4" if is_video else ".png"
        return inp.parent / f"{inp.stem}_faceswapped{sfx}"

    def _execute_image_swap(
        self,
        inp: Path,
        target_clusters: List[DetectedFace],
        sources: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
        callback: Optional[Callable[[str, float], None]] = None,
    ) -> FaceStudioResult:
        if callback:
            callback("Loading Image", 10.0)

        img = cv2.imread(str(inp))
        if img is None:
            raise ValueError(f"Could not read image: {inp}")
        h, w = img.shape[:2]

        out_path = self._determine_output_path(inp, is_video=False)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if callback:
            callback("Replacing Faces", 40.0)

        res = self.swapper.swap_faces_in_frame(img, target_clusters, sources, self.config.blend_strength)

        if callback:
            callback("Saving Output", 90.0)

        cv2.imwrite(str(out_path), res)

        if callback:
            callback("Complete", 100.0)

        return FaceStudioResult(
            input_path=inp,
            output_path=out_path,
            is_video=False,
            resolution=(w, h),
            swapped_faces_count=len(sources),
        )

    def _execute_video_swap(
        self,
        inp: Path,
        target_clusters: List[DetectedFace],
        sources: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
        callback: Optional[Callable[[str, float], None]] = None,
    ) -> FaceStudioResult:
        meta = probe_video(inp)
        if not meta.primary_video:
            raise ValueError(f"No video streams found in: {inp}")

        v_stream = meta.primary_video
        vw, vh = v_stream.width, v_stream.height
        fps = v_stream.fps or 30.0
        total_frames = int(meta.duration * fps) if meta.duration else 0

        out_path = self._determine_output_path(inp, is_video=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        ffmpeg_bin = get_ffmpeg_path()

        def build_cmd(codec: str) -> list[str]:
            cmd = [
                ffmpeg_bin, "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{vw}x{vh}",
                "-pix_fmt", "bgr24",
                "-r", str(fps),
                "-i", "-",
            ]
            if meta.has_audio:
                cmd.extend(["-i", str(inp), "-map", "0:v:0", "-map", "1:a:0?"])
            else:
                cmd.extend(["-map", "0:v:0"])

            if "nvenc" in codec:
                cmd.extend(["-c:v", codec, "-preset", "p7", "-cq", "19", "-pix_fmt", "yuv420p"])
            else:
                cmd.extend(["-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p"])

            if meta.has_audio:
                cmd.extend(["-c:a", "copy"])

            cmd.append(str(out_path))
            return cmd

        codec = self.config.video_codec
        gpu_map = detect_available_gpu_encoders()
        if "nvenc" in codec and not gpu_map.get("nvenc", False):
            codec = "libx264"

        cmd = build_cmd(codec)
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        cap = cv2.VideoCapture(str(inp))
        frame_idx = 0

        if callback:
            callback("Updating Faces in Video", 0.0)

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                processed = self.swapper.swap_faces_in_frame(
                    frame,
                    target_clusters,
                    sources,
                    blend_factor=self.config.blend_strength,
                )
                if proc.stdin:
                    proc.stdin.write(processed.tobytes())
                frame_idx += 1

                if callback and total_frames > 0 and frame_idx % 5 == 0:
                    pct = min(99.0, (frame_idx / total_frames) * 100.0)
                    callback("Updating Faces in Video", pct)

        finally:
            cap.release()
            if proc.stdin:
                proc.stdin.close()
            proc.wait()

        if proc.returncode != 0 and codec != "libx264":
            # Fallback to libx264
            if callback:
                callback("Retrying with CPU Encoder (libx264)", 0.0)
            cmd = build_cmd("libx264")
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            cap = cv2.VideoCapture(str(inp))
            frame_idx = 0
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                    processed = self.swapper.swap_faces_in_frame(frame, target_clusters, sources, self.config.blend_strength)
                    if proc.stdin:
                        proc.stdin.write(processed.tobytes())
                    frame_idx += 1
                    if callback and total_frames > 0 and frame_idx % 5 == 0:
                        pct = min(99.0, (frame_idx / total_frames) * 100.0)
                        callback("Updating Faces (CPU Encoder)", pct)
            finally:
                cap.release()
                if proc.stdin:
                    proc.stdin.close()
                proc.wait()

        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg encoding failed with exit code: {proc.returncode}")

        if callback:
            callback("Face Swapping Complete", 100.0)

        return FaceStudioResult(
            input_path=inp,
            output_path=out_path,
            is_video=True,
            resolution=(vw, vh),
            duration=meta.duration,
            fps=fps,
            swapped_faces_count=len(sources),
        )
