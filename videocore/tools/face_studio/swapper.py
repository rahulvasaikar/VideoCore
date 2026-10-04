"""Face Replacement & Swapping engine using landmark alignment, color transfer, and GFPGAN."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from videocore.tools.enhancer.backends.face_restorer import FaceRestorer
from videocore.tools.face_studio.analyzer import DetectedFace
from videocore.tools.face_studio.models import ensure_face_models


class FaceSwapper:
    """Replaces designated faces in frames with source photos."""

    def __init__(
        self,
        similarity_threshold: float = 0.40,
        enable_gfpgan: bool = True,
        gfpgan_fidelity: float = 0.70,
    ):
        self.similarity_threshold = similarity_threshold
        self.enable_gfpgan = enable_gfpgan
        self.gfpgan_fidelity = gfpgan_fidelity
        self.yunet_path, self.sface_path, _ = ensure_face_models()

        self._detector: Optional[cv2.FaceDetectorYN] = None
        self._detector_size: Tuple[int, int] = (0, 0)
        self._recognizer: Optional[cv2.FaceRecognizerSF] = None
        self._restorer: Optional[FaceRestorer] = None

    def _get_recognizer(self) -> cv2.FaceRecognizerSF:
        if self._recognizer is None:
            self._recognizer = cv2.FaceRecognizerSF.create(str(self.sface_path), "")
        return self._recognizer

    def _get_detector(self, width: int, height: int) -> cv2.FaceDetectorYN:
        if self._detector is None or self._detector_size != (width, height):
            self._detector = cv2.FaceDetectorYN.create(
                str(self.yunet_path),
                "",
                (width, height),
                score_threshold=0.55,
                nms_threshold=0.3,
                top_k=500,
            )
            self._detector_size = (width, height)
        return self._detector

    def _get_restorer(self) -> Optional[FaceRestorer]:
        if self.enable_gfpgan and self._restorer is None:
            try:
                self._restorer = FaceRestorer()
            except Exception:
                self._restorer = None
        return self._restorer

    def extract_source_face(self, pic_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Loads replacement image, detects main face, and returns (face_img, landmarks_5, embedding)."""
        img = cv2.imread(str(pic_path))
        if img is None:
            raise ValueError(f"Could not load replacement face image: {pic_path}")

        h, w = img.shape[:2]
        detector = self._get_detector(w, h)
        _, faces = detector.detect(img)
        if faces is None or len(faces) == 0:
            raise ValueError(f"No human face detected in replacement image: {pic_path}")

        # Choose highest score face
        best_face = max(faces, key=lambda f: f[-1])
        landmarks = np.array([
            best_face[4:6],
            best_face[6:8],
            best_face[8:10],
            best_face[10:12],
            best_face[12:14],
        ], dtype=np.float32)

        recognizer = self._get_recognizer()
        aligned = recognizer.alignCrop(img, best_face)
        feat = recognizer.feature(aligned)

        return img, landmarks, feat

    @staticmethod
    def _transfer_color(source_bgr: np.ndarray, target_bgr: np.ndarray) -> np.ndarray:
        """Transfers lighting and color balance from target to source in LAB color space."""
        src_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        tgt_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        for i in range(3):
            s_mean, s_std = src_lab[:, :, i].mean(), src_lab[:, :, i].std() + 1e-5
            t_mean, t_std = tgt_lab[:, :, i].mean(), tgt_lab[:, :, i].std() + 1e-5
            src_lab[:, :, i] = ((src_lab[:, :, i] - s_mean) * (t_std / s_std)) + t_mean
        return cv2.cvtColor(np.clip(src_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    def swap_faces_in_frame(
        self,
        frame: np.ndarray,
        target_clusters: List[DetectedFace],
        replacement_sources: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
        blend_factor: float = 1.0,
    ) -> np.ndarray:
        """Replaces matching faces in a single frame.

        Args:
            frame: Input video/image frame (BGR).
            target_clusters: List of identified clusters in the video.
            replacement_sources: Mapping of person_id -> (source_img, source_landmarks, source_embedding).
            blend_factor: Strength of replacement (0.0 to 1.0).
        """
        if not replacement_sources or blend_factor <= 0.001:
            return frame

        fh, fw = frame.shape[:2]
        detector = self._get_detector(fw, fh)
        _, faces = detector.detect(frame)
        if faces is None or len(faces) == 0:
            return frame

        recognizer = self._get_recognizer()
        result_frame = frame.copy()
        swapped_any = False

        for f in faces:
            aligned = recognizer.alignCrop(frame, f)
            feat = recognizer.feature(aligned)

            # Check which cluster this face belongs to
            best_person_id: Optional[str] = None
            best_sim = -1.0
            for cluster in target_clusters:
                sim = recognizer.match(feat, cluster.embedding, cv2.FaceRecognizerSF_FR_COSINE)
                if sim > best_sim:
                    best_sim = sim
                    best_person_id = cluster.person_id

            if not best_person_id or best_sim < self.similarity_threshold:
                continue

            # If user specified a replacement for this person
            if best_person_id not in replacement_sources:
                continue

            src_img, s_landmarks, _ = replacement_sources[best_person_id]
            t_landmarks = np.array([f[4:6], f[6:8], f[8:10], f[10:12], f[12:14]], dtype=np.float32)

            # 1. Estimate partial affine transformation
            M, inliers = cv2.estimateAffinePartial2D(s_landmarks, t_landmarks)
            if M is None:
                continue

            # 2. Warp source image to match target face scale & orientation
            warped_src = cv2.warpAffine(src_img, M, (fw, fh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

            # 3. Create target face crop for lighting reference
            tx, ty, tfw, tfh = map(int, f[:4])
            x1, y1 = max(0, tx), max(0, ty)
            x2, y2 = min(fw, tx + tfw), min(fh, ty + tfh)
            target_face_crop = frame[y1:y2, x1:x2]
            warped_face_crop = warped_src[y1:y2, x1:x2]

            if target_face_crop.size > 0 and warped_face_crop.size > 0:
                color_matched = self._transfer_color(warped_src, frame)
            else:
                color_matched = warped_src

            # 4. Soft feathered elliptical blend mask
            center_x = int(np.mean(t_landmarks[:, 0]))
            center_y = int(np.mean(t_landmarks[:, 1]))
            eye_dist = np.linalg.norm(t_landmarks[0] - t_landmarks[1])
            dx = int(eye_dist * 1.35)
            dy = int(eye_dist * 1.70)

            mask = np.zeros((fh, fw), dtype=np.float32)
            cv2.ellipse(mask, (center_x, center_y), (dx, dy), 0, 0, 360, 1.0, -1)
            ksize = max(15, int(eye_dist * 0.4) | 1)
            mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
            mask_3c = mask[:, :, np.newaxis] * np.clip(blend_factor, 0.0, 1.0)

            result_frame = (color_matched.astype(np.float32) * mask_3c + result_frame.astype(np.float32) * (1.0 - mask_3c)).clip(0, 255).astype(np.uint8)
            swapped_any = True

        # 5. Photorealistic facial refinement with GFPGAN
        if swapped_any and self.enable_gfpgan:
            restorer = self._get_restorer()
            if restorer is not None:
                try:
                    result_frame = restorer.enhance_image_array(result_frame, fidelity=self.gfpgan_fidelity)
                except Exception:
                    pass

        return result_frame
