"""Animation and Stylization Engine for VideoCore.

Provides bidirectional transformation:
- Real Video / Image -> Animation / Anime (AnimeGANv2 AI + Artistic Filters)
- Animation / Anime -> Real / Photorealistic (De-cartoonizing, Film Grain, Tone Mapping & GFPGAN Face Realism)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Callable, Optional, Tuple

from videocore.core.constants import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
import cv2
import numpy as np
import onnxruntime as ort

from videocore.core.ffmpeg import detect_available_gpu_encoders, get_ffmpeg_path
from videocore.core.probe import probe_video
from videocore.tools.animator.filters import (
    adjust_brightness_contrast,
    adjust_saturation,
    apply_cartoon_filter,
    apply_comic_filter,
    apply_de_cartoon_filter,
    apply_watercolor_filter,
    blend_with_original,
)
from videocore.tools.animator.models import (
    ANIMEGAN_MODELS,
    download_model,
    get_animegan_model_path,
)


@dataclass
class AnimationConfig:
    input_path: Path
    output_path: Optional[Path] = None
    direction: str = "real_to_anim"  # "real_to_anim" or "anim_to_real"
    style: str = "face_paint"  # "face_paint", "hayao", "shinkai", "paprika", "cartoon", "comic", "watercolor"
    strength: float = 1.0  # 0.0 to 1.0 (blend factor)
    line_weight: float = 1.0  # 0.5 to 2.5
    color_saturation: float = 1.15  # 0.5 to 2.0
    edge_soften: float = 0.8  # 0.0 to 1.0 (for anim_to_real)
    film_grain: float = 0.15  # 0.0 to 0.5 (for anim_to_real)
    face_realism: bool = True  # Use GFPGAN in anim_to_real
    face_fidelity: float = 0.85  # GFPGAN fidelity (higher for realistic transformation)
    video_codec: str = "h264_nvenc"  # "h264_nvenc", "hevc_nvenc", or "libx264"
    max_dimension: Optional[int] = None  # Optional max width/height constraint


@dataclass
class AnimationResult:
    input_path: Path
    output_path: Path
    is_video: bool
    original_resolution: Tuple[int, int]
    processed_resolution: Tuple[int, int]
    duration: Optional[float] = None
    original_fps: Optional[float] = None
    direction: str = "real_to_anim"
    style: str = "face_paint"


class AnimatorEngine:
    """Orchestrates AI neural and algorithmic style transformations for images and videos."""

    def __init__(self, config: AnimationConfig):
        self.config = config
        self._onnx_session: Optional[ort.InferenceSession] = None
        self._session_model: Optional[str] = None
        self._face_restorer = None

    def _get_onnx_session(self, style: str) -> ort.InferenceSession:
        """Retrieves or creates a cached ONNX inference session for AnimeGAN."""
        if self._onnx_session is not None and self._session_model == style:
            return self._onnx_session

        model_path = get_animegan_model_path(style)
        if not model_path or not model_path.is_file():
            model_path = download_model(style)

        # Providers: CUDA GPU if CUDA runtime is configured, DirectML, then CPU fallback
        available = ort.get_available_providers()
        providers = []
        import os
        if "CUDAExecutionProvider" in available and os.environ.get("CUDA_PATH"):
            providers.append("CUDAExecutionProvider")
        if "DmlExecutionProvider" in available:
            providers.append("DmlExecutionProvider")
        providers.append("CPUExecutionProvider")

        sess = ort.InferenceSession(str(model_path), providers=providers)
        self._onnx_session = sess
        self._session_model = style
        return sess

    def _get_face_restorer(self):
        """Lazy load face restorer for facial photorealism."""
        if self._face_restorer is None:
            try:
                from videocore.tools.enhancer.backends.face_restorer import FaceRestorer
                self._face_restorer = FaceRestorer()
            except Exception:
                self._face_restorer = None
        return self._face_restorer

    def transform_frame(self, frame: np.ndarray) -> np.ndarray:
        """Transforms a single BGR video frame or image array according to config."""
        orig_h, orig_w = frame.shape[:2]
        direction = self.config.direction
        style = self.config.style.lower()

        # Handle max dimension scaling if configured
        target_w, target_h = orig_w, orig_h
        if self.config.max_dimension and max(orig_w, orig_h) > self.config.max_dimension:
            scale = self.config.max_dimension / float(max(orig_w, orig_h))
            target_w = int(round(orig_w * scale))
            target_h = int(round(orig_h * scale))
            frame_resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else:
            frame_resized = frame

        if direction == "real_to_anim":
            processed = self._transform_real_to_anim(frame_resized, style)
        else:
            processed = self._transform_anim_to_real(frame_resized)

        # Resize back to original dimensions if resized
        if (target_w, target_h) != (orig_w, orig_h):
            processed = cv2.resize(processed, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Blend with original
        return blend_with_original(frame, processed, self.config.strength)

    def _transform_real_to_anim(self, frame: np.ndarray, style: str) -> np.ndarray:
        """Applies anime or cartoon stylization with strong visible transformations."""
        h, w = frame.shape[:2]

        if style == "face_paint":
            # Dedicated AnimeGANv2 character / portrait neural model (NCHW [1, 3, 512, 512])
            sess = self._get_onnx_session("face_paint")
            inp_512 = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(inp_512, cv2.COLOR_BGR2RGB).astype(np.float32)
            norm = (rgb / 127.5) - 1.0
            inp_tensor = np.expand_dims(np.transpose(norm, (2, 0, 1)), axis=0)

            input_name = sess.get_inputs()[0].name
            out = sess.run(None, {input_name: inp_tensor})[0]

            out_hwc = np.transpose(out[0], (1, 2, 0))
            out_denorm = np.clip((out_hwc + 1.0) * 127.5, 0, 255).astype(np.uint8)
            out_bgr = cv2.cvtColor(out_denorm, cv2.COLOR_RGB2BGR)

            if (w, h) != (512, 512):
                out_bgr = cv2.resize(out_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)

            # Clean line art overlay
            if self.config.line_weight > 0.6:
                gray = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2GRAY)
                blur = cv2.medianBlur(gray, 3)
                edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 2.5)
                if self.config.line_weight > 1.0:
                    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                    inv = cv2.dilate(255 - edges, k, iterations=int(round(self.config.line_weight - 1.0)))
                    edges = 255 - inv
                edge_mask = (edges.astype(np.float32) / 255.0)[:, :, np.newaxis]
                out_bgr = (out_bgr.astype(np.float32) * (1.0 - (1.0 - edge_mask) * 0.45 * self.config.line_weight)).clip(0, 255).astype(np.uint8)

            return adjust_saturation(out_bgr, self.config.color_saturation)

        elif style in ("hayao", "shinkai", "paprika"):
            sess = self._get_onnx_session(style)
            # Receptive field calibration: AnimeGANv2 convolutions are calibrated for 512-720px.
            # Downscale high-res images so the neural receptive field covers facial features & scenery properly.
            max_d = max(h, w)
            scale = 720.0 / max_d if max_d > 720 else 1.0
            target_w = int(round(w * scale / 8.0)) * 8
            target_h = int(round(h * scale / 8.0)) * 8

            if (target_w, target_h) != (w, h):
                inp_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            else:
                inp_frame = frame

            # Convert BGR to RGB and normalize [-1, 1] (NHWC)
            img_rgb = cv2.cvtColor(inp_frame, cv2.COLOR_BGR2RGB)
            img_norm = (img_rgb.astype(np.float32) / 127.5) - 1.0
            inp_tensor = np.expand_dims(img_norm, axis=0)

            input_name = sess.get_inputs()[0].name
            out = sess.run(None, {input_name: inp_tensor})[0]

            # Denormalize output
            out = (out[0] + 1.0) * 127.5
            out_clipped = np.clip(out, 0, 255).astype(np.uint8)
            out_bgr = cv2.cvtColor(out_clipped, cv2.COLOR_RGB2BGR)

            if (target_w, target_h) != (w, h):
                out_bgr = cv2.resize(out_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)

            # Adjust line weight
            if self.config.line_weight > 0.6:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2.0)
                inv = 255 - edges
                if self.config.line_weight > 1.0:
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                    inv = cv2.dilate(inv, kernel, iterations=1)
                edge_mask = ((255 - inv).astype(np.float32) / 255.0)[:, :, np.newaxis]
                out_bgr = (out_bgr.astype(np.float32) * (1.0 - (1.0 - edge_mask) * (self.config.line_weight - 1.0))).clip(0, 255).astype(np.uint8)

            # Adjust saturation
            return adjust_saturation(out_bgr, self.config.color_saturation)

        elif style == "cartoon":
            res = apply_cartoon_filter(frame, line_weight=self.config.line_weight, color_levels=7)
            return adjust_saturation(res, self.config.color_saturation)

        elif style == "comic":
            return apply_comic_filter(frame, line_weight=self.config.line_weight, color_mode="color")

        elif style == "watercolor":
            res = apply_watercolor_filter(frame, line_weight=self.config.line_weight)
            return adjust_saturation(res, self.config.color_saturation)

        else:
            # Default fallback to face_paint
            return self._transform_real_to_anim(frame, "face_paint")

    def _transform_anim_to_real(self, frame: np.ndarray) -> np.ndarray:
        """Applies de-animation, photographic optical grading, and facial photorealism."""
        # 1. Base photographic de-cartoonizing (delineation inpainting, unquantization, film grain)
        delineated = apply_de_cartoon_filter(
            frame,
            edge_soften=self.config.edge_soften,
            grain_amount=self.config.film_grain,
            natural_tone=True,
        )

        # 2. Facial photorealism via GFPGAN if enabled
        if self.config.face_realism:
            restorer = self._get_face_restorer()
            if restorer is not None:
                try:
                    delineated = restorer.enhance_image_array(
                        delineated,
                        fidelity=self.config.face_fidelity,
                    )
                except Exception:
                    pass

        # 3. Tone and saturation adjustment
        if abs(self.config.color_saturation - 1.0) > 0.02:
            delineated = adjust_saturation(delineated, self.config.color_saturation)

        return delineated

    def preview_frame(self, timestamp: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts a single frame from the input and generates a preview.

        Returns:
            Tuple of (original_frame_bgr, processed_frame_bgr).
        """
        inp = self.config.input_path
        if not inp.exists():
            raise FileNotFoundError(f"Input path not found: {inp}")

        ext = inp.suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            orig = cv2.imread(str(inp))
            if orig is None:
                raise ValueError(f"Could not read image: {inp}")
            processed = self.transform_frame(orig)
            return orig, processed

        elif ext in VIDEO_EXTENSIONS:
            cap = cv2.VideoCapture(str(inp))
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {inp}")

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            target_frame_idx = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                # Fallback to first frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            cap.release()

            if not ret or frame is None:
                raise RuntimeError(f"Could not extract frame from video: {inp}")

            processed = self.transform_frame(frame)
            return frame, processed

        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    def execute(
        self,
        stage_callback: Optional[Callable[[str, float], None]] = None,
    ) -> AnimationResult:
        """Executes the transformation pipeline."""
        inp = self.config.input_path
        if not inp.exists():
            raise FileNotFoundError(f"Input path not found: {inp}")

        ext = inp.suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            return self._process_image(inp, stage_callback)
        elif ext in VIDEO_EXTENSIONS:
            return self._process_video(inp, stage_callback)
        else:
            raise ValueError(f"Unsupported media type: {ext}")

    def _determine_output_path(self, inp: Path, is_video: bool) -> Path:
        if self.config.output_path:
            return self.config.output_path

        suffix = ".mp4" if is_video else ".png"
        tag = self.config.style if self.config.direction == "real_to_anim" else "photoreal"
        return inp.parent / f"{inp.stem}_{tag}{suffix}"

    def _process_image(
        self,
        inp: Path,
        callback: Optional[Callable[[str, float], None]] = None,
    ) -> AnimationResult:
        if callback:
            callback("Transforming Image", 10.0)

        img = cv2.imread(str(inp))
        if img is None:
            raise ValueError(f"Could not read image: {inp}")
        orig_h, orig_w = img.shape[:2]

        out_path = self._determine_output_path(inp, is_video=False)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if callback:
            callback("Applying Style Pipeline", 40.0)

        processed = self.transform_frame(img)

        if callback:
            callback("Saving Image", 90.0)

        cv2.imwrite(str(out_path), processed)

        if callback:
            callback("Complete", 100.0)

        return AnimationResult(
            input_path=inp,
            output_path=out_path,
            is_video=False,
            original_resolution=(orig_w, orig_h),
            processed_resolution=(orig_w, orig_h),
            direction=self.config.direction,
            style=self.config.style,
        )

    def _process_video(
        self,
        inp: Path,
        callback: Optional[Callable[[str, float], None]] = None,
    ) -> AnimationResult:
        meta = probe_video(inp)
        if not meta.primary_video:
            raise ValueError(f"No video streams found in {inp}")

        v_stream = meta.primary_video
        orig_w, orig_h = v_stream.width, v_stream.height
        orig_fps = v_stream.fps or 30.0
        total_frames = int(meta.duration * orig_fps) if meta.duration else 0

        out_path = self._determine_output_path(inp, is_video=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        ffmpeg_bin = get_ffmpeg_path()

        # Build FFmpeg command with rawvideo stdin pipe
        def build_encode_cmd(codec: str) -> list[str]:
            cmd = [
                ffmpeg_bin, "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{orig_w}x{orig_h}",
                "-pix_fmt", "bgr24",
                "-r", str(orig_fps),
                "-i", "-",
            ]
            if meta.has_audio:
                cmd.extend(["-i", str(inp), "-map", "0:v:0", "-map", "1:a:0?"])
            else:
                cmd.extend(["-map", "0:v:0"])

            if "nvenc" in codec:
                cmd.extend([
                    "-c:v", codec,
                    "-preset", "p7",
                    "-cq", "19",
                    "-pix_fmt", "yuv420p",
                ])
            else:
                cmd.extend([
                    "-c:v", "libx264",
                    "-crf", "18",
                    "-preset", "medium",
                    "-pix_fmt", "yuv420p",
                ])

            if meta.has_audio:
                cmd.extend(["-c:a", "copy"])

            cmd.append(str(out_path))
            return cmd

        # Try NVENC first if requested, fallback to libx264
        codec_to_use = self.config.video_codec
        gpu_encoders = detect_available_gpu_encoders()
        if "nvenc" in codec_to_use and not gpu_encoders.get("nvenc", False):
            codec_to_use = "libx264"

        cmd = build_encode_cmd(codec_to_use)
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        cap = cv2.VideoCapture(str(inp))
        if not cap.isOpened():
            proc.kill()
            raise RuntimeError(f"Could not open video: {inp}")

        # Pre-warm ONNX session or models
        if self.config.direction == "real_to_anim" and self.config.style in ("hayao", "shinkai", "paprika"):
            self._get_onnx_session(self.config.style)

        if callback:
            callback("Stylizing Video Frames", 0.0)

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                processed_frame = self.transform_frame(frame)
                if proc.stdin:
                    proc.stdin.write(processed_frame.tobytes())
                frame_idx += 1

                if callback and total_frames > 0 and frame_idx % 5 == 0:
                    pct = min(99.0, (frame_idx / total_frames) * 100.0)
                    callback("Stylizing Video Frames", pct)

        finally:
            cap.release()
            if proc.stdin:
                proc.stdin.close()
            proc.wait()

        # If NVENC failed, try fallback
        if proc.returncode != 0 and codec_to_use != "libx264":
            if callback:
                callback("Retrying with CPU Encoder (libx264)", 0.0)
            codec_to_use = "libx264"
            cmd = build_encode_cmd("libx264")
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            cap = cv2.VideoCapture(str(inp))
            frame_idx = 0
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                    processed_frame = self.transform_frame(frame)
                    if proc.stdin:
                        proc.stdin.write(processed_frame.tobytes())
                    frame_idx += 1
                    if callback and total_frames > 0 and frame_idx % 5 == 0:
                        pct = min(99.0, (frame_idx / total_frames) * 100.0)
                        callback("Stylizing Video Frames (CPU)", pct)
            finally:
                cap.release()
                if proc.stdin:
                    proc.stdin.close()
                proc.wait()

        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg encoding failed with exit code: {proc.returncode}")

        if callback:
            callback("Video Processing Complete", 100.0)

        return AnimationResult(
            input_path=inp,
            output_path=out_path,
            is_video=True,
            original_resolution=(orig_w, orig_h),
            processed_resolution=(orig_w, orig_h),
            duration=meta.duration,
            original_fps=orig_fps,
            direction=self.config.direction,
            style=self.config.style,
        )
