"""Video Face Analyzer: scans videos to detect and cluster distinct human faces."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from videocore.core.probe import probe_video
from videocore.tools.face_studio.models import ensure_face_models


@dataclass
class DetectedFace:
    person_id: str
    label: str
    thumbnail_bgr: np.ndarray
    embedding: np.ndarray
    appearance_count: int
    first_seen_sec: float
    best_score: float
    sample_timestamps: List[float] = field(default_factory=list)


class VideoFaceAnalyzer:
    """Scans video frames, detects human faces, and clusters unique individuals."""

    def __init__(
        self,
        similarity_threshold: float = 0.40,
        score_threshold: float = 0.60,
    ):
        self.similarity_threshold = similarity_threshold
        self.score_threshold = score_threshold
        self.yunet_path, self.sface_path, _ = ensure_face_models()
        self._recognizer: Optional[cv2.FaceRecognizerSF] = None
        self._detector: Optional[cv2.FaceDetectorYN] = None
        self._detector_size: Tuple[int, int] = (0, 0)

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
                score_threshold=self.score_threshold,
                nms_threshold=0.3,
                top_k=500,
            )
            self._detector_size = (width, height)
        return self._detector

    def analyze_image(self, img_path: Path) -> List[DetectedFace]:
        """Detects and extracts all faces in a single image."""
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")
        h, w = img.shape[:2]
        detector = self._get_detector(w, h)
        _, faces = detector.detect(img)
        if faces is None or len(faces) == 0:
            return []

        recognizer = self._get_recognizer()
        results: List[DetectedFace] = []

        for idx, f in enumerate(faces):
            x, y, fw, fh = map(int, f[:4])
            score = float(f[-1])
            # Bounding box clamp
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + fw), min(h, y + fh)
            crop = img[y1:y2, x1:x2].copy()
            if crop.size == 0:
                continue

            aligned = recognizer.alignCrop(img, f)
            feat = recognizer.feature(aligned)

            results.append(
                DetectedFace(
                    person_id=f"person_{idx + 1}",
                    label=f"Person {idx + 1}",
                    thumbnail_bgr=crop,
                    embedding=feat,
                    appearance_count=1,
                    first_seen_sec=0.0,
                    best_score=score,
                    sample_timestamps=[0.0],
                )
            )

        return results

    def analyze_video(
        self,
        video_path: Path,
        sample_interval_sec: float = 0.5,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> List[DetectedFace]:
        """Scans video frames at intervals, clustering detected faces into unique persons."""
        meta = probe_video(video_path)
        if not meta.primary_video:
            raise ValueError(f"No video streams found in: {video_path}")

        v_stream = meta.primary_video
        vw, vh = v_stream.width, v_stream.height
        fps = v_stream.fps or 30.0
        duration = meta.duration or 0.0

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        detector = self._get_detector(vw, vh)
        recognizer = self._get_recognizer()

        clusters: List[DetectedFace] = []
        step_frames = max(1, int(round(sample_interval_sec * fps)))
        total_steps = max(1, int((duration * fps) // step_frames)) if duration > 0 else 100

        frame_idx = 0
        step_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                if frame_idx % step_frames == 0:
                    step_count += 1
                    current_time_sec = frame_idx / fps

                    if progress_callback:
                        pct = min(99.0, (step_count / total_steps) * 100.0)
                        progress_callback(f"Scanning Frame {frame_idx:,} ({current_time_sec:.1f}s)", pct)

                    _, faces = detector.detect(frame)
                    if faces is not None and len(faces) > 0:
                        for f in faces:
                            score = float(f[-1])
                            if score < self.score_threshold:
                                continue

                            x, y, fw, fh = map(int, f[:4])
                            # Margin for thumbnail crop
                            margin = int(max(fw, fh) * 0.25)
                            x1, y1 = max(0, x - margin), max(0, y - margin)
                            x2, y2 = min(vw, x + fw + margin), min(vh, y + fh + margin)
                            crop = frame[y1:y2, x1:x2].copy()
                            if crop.size == 0:
                                continue

                            aligned = recognizer.alignCrop(frame, f)
                            feat = recognizer.feature(aligned)

                            # Compare with known clusters
                            best_match_idx = -1
                            best_sim = -1.0

                            for c_idx, cluster in enumerate(clusters):
                                sim = recognizer.match(feat, cluster.embedding, cv2.FaceRecognizerSF_FR_COSINE)
                                if sim > best_sim:
                                    best_sim = sim
                                    best_match_idx = c_idx

                            if best_match_idx >= 0 and best_sim >= self.similarity_threshold:
                                # Matched existing person
                                c = clusters[best_match_idx]
                                c.appearance_count += 1
                                c.sample_timestamps.append(current_time_sec)
                                # If this detection is higher quality / score, update thumbnail
                                if score > c.best_score or (score >= c.best_score - 0.05 and crop.shape[0] > c.thumbnail_bgr.shape[0]):
                                    c.thumbnail_bgr = crop
                                    c.best_score = score
                                    # Blend embedding slightly to adapt to pose variations
                                    c.embedding = (0.75 * c.embedding + 0.25 * feat)
                            else:
                                # New unique person
                                p_num = len(clusters) + 1
                                new_cluster = DetectedFace(
                                    person_id=f"person_{p_num}",
                                    label=f"Person {p_num}",
                                    thumbnail_bgr=crop,
                                    embedding=feat,
                                    appearance_count=1,
                                    first_seen_sec=current_time_sec,
                                    best_score=score,
                                    sample_timestamps=[current_time_sec],
                                )
                                clusters.append(new_cluster)

                frame_idx += 1

        finally:
            cap.release()

        # Sort clusters by frequency of appearance (most prominent person first)
        clusters.sort(key=lambda c: c.appearance_count, reverse=True)
        # Re-label cleanly: Person 1, Person 2, ...
        for idx, c in enumerate(clusters):
            c.person_id = f"person_{idx + 1}"
            c.label = f"Person {idx + 1}"

        if progress_callback:
            progress_callback(f"Analysis Complete: Found {len(clusters)} distinct individuals", 100.0)

        return clusters
