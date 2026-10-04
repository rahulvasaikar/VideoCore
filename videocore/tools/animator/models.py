"""Model management for VideoCore Animation & Stylization Tool."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional

from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from videocore.core.download import download_file

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
LOCAL_BIN_DIR = PROJECT_ROOT / "bin"
MODELS_DIR = LOCAL_BIN_DIR / "models"

ANIMEGAN_MODELS: Dict[str, Dict[str, str]] = {
    "face_paint": {
        "filename": "AnimeGANv2_Face.onnx",
        "url": "https://huggingface.co/vumichien/AnimeGANv2_Face/resolve/main/face_paint_512_v2_0.onnx",
        "name": "Anime Character & Portrait (Face Paint)",
        "description": "High-stylization anime portrait model designed specifically for real people, faces, and characters",
    },
    "hayao": {
        "filename": "AnimeGANv2_Hayao.onnx",
        "url": "https://huggingface.co/vumichien/AnimeGANv2_Hayao/resolve/main/AnimeGANv2_Hayao.onnx",
        "name": "Hayao (Studio Ghibli)",
        "description": "Soft pastel tones and rich hand-drawn anime aesthetic",
    },
    "shinkai": {
        "filename": "AnimeGANv2_Shinkai.onnx",
        "url": "https://huggingface.co/vumichien/AnimeGANv2_Shinkai/resolve/main/AnimeGANv2_Shinkai.onnx",
        "name": "Shinkai (Makoto Shinkai)",
        "description": "Luminous skies, vibrant atmospheric lighting, and high contrast",
    },
    "paprika": {
        "filename": "AnimeGANv2_Paprika.onnx",
        "url": "https://huggingface.co/vumichien/AnimeGANv2_Paprika/resolve/main/AnimeGANv2_Paprika.onnx",
        "name": "Paprika (Satoshi Kon)",
        "description": "Saturated palette with expressive, stylized dramatic colors",
    },
}


def get_animegan_model_path(style: str) -> Optional[Path]:
    """Returns local path to the specified AnimeGANv2 model if it exists."""
    style_key = style.lower().strip()
    info = ANIMEGAN_MODELS.get(style_key)
    if not info:
        return None
    model_path = MODELS_DIR / info["filename"]
    return model_path if model_path.is_file() else None


def download_model(style: str) -> Path:
    """Downloads an AnimeGANv2 model if missing and returns its path."""
    style_key = style.lower().strip()
    if style_key not in ANIMEGAN_MODELS:
        raise ValueError(f"Unknown anime model style: {style}. Available: {list(ANIMEGAN_MODELS.keys())}")

    info = ANIMEGAN_MODELS[style_key]
    dest_path = MODELS_DIR / info["filename"]
    if dest_path.is_file() and dest_path.stat().st_size > 1024 * 1024:
        return dest_path

    url = info["url"]
    desc = f"Downloading {info['name']}"
    download_file(url, dest_path, desc)
    return dest_path


def ensure_all_anime_models() -> None:
    """Ensures all AnimeGANv2 models are present locally."""
    for style in ANIMEGAN_MODELS:
        download_model(style)
