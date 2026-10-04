"""Model management for Video Face Analyzer & Swapper."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

from rich.console import Console
from videocore.core.download import download_file

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
LOCAL_BIN_DIR = PROJECT_ROOT / "bin"
MODELS_DIR = LOCAL_BIN_DIR / "models"

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
GFPGAN_URL = "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GFPGANv1.4.onnx"


def get_yunet_model() -> Optional[Path]:
    path = MODELS_DIR / "yunet.onnx"
    return path if path.is_file() else None


def get_sface_model() -> Optional[Path]:
    path = MODELS_DIR / "sface.onnx"
    return path if path.is_file() else None


def get_gfpgan_model() -> Optional[Path]:
    path = MODELS_DIR / "GFPGANv1.4.onnx"
    return path if path.is_file() else None



def ensure_face_models() -> Tuple[Path, Path, Optional[Path]]:
    """Ensures YuNet, SFace, and GFPGAN models are downloaded."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    yunet_p = get_yunet_model()
    if not yunet_p:
        yunet_p = download_file(YUNET_URL, MODELS_DIR / "yunet.onnx", "YuNet Face Detector")

    sface_p = get_sface_model()
    if not sface_p:
        sface_p = download_file(SFACE_URL, MODELS_DIR / "sface.onnx", "SFace Face Recognizer")

    gfpgan_p = get_gfpgan_model()
    if not gfpgan_p:
        try:
            gfpgan_p = download_file(GFPGAN_URL, MODELS_DIR / "GFPGANv1.4.onnx", "GFPGAN Face Restorer")
        except Exception:
            gfpgan_p = None

    return yunet_p, sface_p, gfpgan_p
