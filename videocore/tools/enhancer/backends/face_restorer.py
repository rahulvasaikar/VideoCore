"""Face restoration backend using YuNet detection and GFPGAN v1.4 via ONNX DirectML."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from videocore.tools.enhancer.models import ensure_face_models


# Standard 5-point face landmarks for canonical 512x512 alignment (FFHQ / GFPGAN space)
STD_POINTS_512 = np.array(
    [
        [192.98138, 239.94708],  # right eye (image left)
        [318.90277, 240.19360],  # left eye (image right)
        [256.03415, 314.01935],  # nose tip
        [201.26117, 371.41043],  # right mouth corner
        [313.08905, 371.15118],  # left mouth corner
    ],
    dtype=np.float32,
)


class FaceRestorer:
    """Restores blurry or degraded faces using YuNet detector and GFPGAN v1.4."""

    def __init__(
        self,
        yunet_path: Optional[Path] = None,
        gfpgan_path: Optional[Path] = None,
        score_threshold: float = 0.5,
    ):
        if not yunet_path or not gfpgan_path:
            yunet_p, gfpgan_p = ensure_face_models()
            self.yunet_path = yunet_path or yunet_p
            self.gfpgan_path = gfpgan_path or gfpgan_p
        else:
            self.yunet_path = yunet_path
            self.gfpgan_path = gfpgan_path

        self.score_threshold = score_threshold
        self._session: Optional[ort.InferenceSession] = None
        self._detector: Optional[cv2.FaceDetectorYN] = None
        self._detector_dims: Tuple[int, int] = (0, 0)
        self._feather_mask: Optional[np.ndarray] = None

    def _get_session(self) -> ort.InferenceSession:
        if self._session is None:
            available = ort.get_available_providers()
            providers = []
            import os
            if "CUDAExecutionProvider" in available and os.environ.get("CUDA_PATH"):
                providers.append("CUDAExecutionProvider")
            if "DmlExecutionProvider" in available:
                providers.append("DmlExecutionProvider")
            providers.append("CPUExecutionProvider")
            self._session = ort.InferenceSession(str(self.gfpgan_path), providers=providers)
        return self._session

    def _get_feather_mask(self) -> np.ndarray:
        if self._feather_mask is None:
            mask = np.zeros((512, 512), dtype=np.float32)
            cv2.ellipse(mask, (256, 270), (180, 240), 0, 0, 360, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (51, 51), 0)
            self._feather_mask = mask[:, :, None]
        return self._feather_mask

    def _get_detector(self, width: int, height: int) -> cv2.FaceDetectorYN:
        if self._detector is None or self._detector_dims != (width, height):
            self._detector = cv2.FaceDetectorYN.create(
                str(self.yunet_path),
                "",
                (width, height),
                score_threshold=self.score_threshold,
                nms_threshold=0.3,
                top_k=5000,
            )
            self._detector_dims = (width, height)
        return self._detector

    def enhance_image_array(self, image: np.ndarray, fidelity: float = 0.7) -> np.ndarray:
        """Restores all detected faces in a BGR image array.

        Args:
            image: BGR numpy image array.
            fidelity: Balance between restored sharpness and original identity (0.0 to 1.0).

        Returns:
            Enhanced BGR image array with restored faces.
        """
        h, w = image.shape[:2]
        detector = self._get_detector(w, h)
        _, faces = detector.detect(image)

        # Fallback to lower score threshold (crucial for anime and stylized drawings)
        if faces is None or len(faces) == 0:
            fallback_det = cv2.FaceDetectorYN.create(
                str(self.yunet_path),
                "",
                (w, h),
                score_threshold=0.25,
                nms_threshold=0.3,
                top_k=5000,
            )
            _, faces = fallback_det.detect(image)

        if faces is None or len(faces) == 0:
            return image

        session = self._get_session()
        feather_mask = self._get_feather_mask()
        result = image.copy()

        for face in faces:
            # Face landmark format:
            # [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
            landmarks = np.array(
                [
                    [face[4], face[5]],   # right eye (image left)
                    [face[6], face[7]],   # left eye (image right)
                    [face[8], face[9]],   # nose tip
                    [face[10], face[11]], # right mouth
                    [face[12], face[13]], # left mouth
                ],
                dtype=np.float32,
            )

            # Estimate similarity affine matrix to 512x512
            affine_matrix, _ = cv2.estimateAffinePartial2D(landmarks, STD_POINTS_512)
            if affine_matrix is None:
                continue

            aligned_face = cv2.warpAffine(
                result,
                affine_matrix,
                (512, 512),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

            # Normalize to [-1, 1] RGB
            rgb_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
            input_tensor = (rgb_face.astype(np.float32) / 255.0 - 0.5) / 0.5
            input_tensor = np.transpose(input_tensor, (2, 0, 1))[None, ...]

            # Run GFPGAN inference
            out = session.run(None, {"input": input_tensor})[0]
            restored_rgb = (out[0].transpose(1, 2, 0) * 0.5 + 0.5) * 255.0
            restored_rgb = np.clip(restored_rgb, 0, 255).astype(np.uint8)
            restored_bgr = cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR)

            # Blend with aligned face according to fidelity
            if fidelity < 1.0:
                restored_bgr = cv2.addWeighted(
                    restored_bgr,
                    float(fidelity),
                    aligned_face,
                    float(1.0 - fidelity),
                    0.0,
                )

            # Inverse warp back to original image coordinates
            inv_matrix = cv2.invertAffineTransform(affine_matrix)
            warped_face = cv2.warpAffine(
                restored_bgr,
                inv_matrix,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            warped_mask = cv2.warpAffine(
                feather_mask,
                inv_matrix,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )

            # Alpha blend back
            mask_3c = warped_mask[:, :, np.newaxis]
            blended = (
                result.astype(np.float32) * (1.0 - mask_3c)
                + warped_face.astype(np.float32) * mask_3c
            )
            result = np.clip(blended, 0, 255).astype(np.uint8)

        return result

    def enhance_image_file(
        self,
        input_file: Path,
        output_file: Path,
        fidelity: float = 0.7,
    ) -> Path:
        """Restores faces in an image file and saves the output."""
        img = cv2.imread(str(input_file))
        if img is None:
            raise ValueError(f"Could not read image: {input_file}")

        restored = self.enhance_image_array(img, fidelity=fidelity)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_file), restored)
        return output_file

    def enhance_frames_dir(
        self,
        frames_dir: Path,
        fidelity: float = 0.7,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """Restores faces across all frames in a directory in-place."""
        frame_files = sorted(
            [f for f in frames_dir.iterdir() if f.suffix.lower() in [".png", ".jpg", ".webp"]]
        )
        total = len(frame_files)

        for idx, frame_path in enumerate(frame_files):
            img = cv2.imread(str(frame_path))
            if img is not None:
                restored = self.enhance_image_array(img, fidelity=fidelity)
                cv2.imwrite(str(frame_path), restored)

            if progress_callback:
                progress_callback(idx + 1, total)
