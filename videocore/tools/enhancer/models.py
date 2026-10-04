"""Model and binary management for VideoCore Enhancer."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from videocore.core.download import download_file

console = Console()

# Base directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
LOCAL_BIN_DIR = PROJECT_ROOT / "bin"
MODELS_DIR = LOCAL_BIN_DIR / "models"

# Remote resource URLs
REALESRGAN_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip"
RIFE_URL = "https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-windows.zip"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
GFPGAN_URL = "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GFPGANv1.4.onnx"


def get_realesrgan_bin() -> Optional[Path]:
    """Finds the Real-ESRGAN binary path."""
    candidates = [
        LOCAL_BIN_DIR / "realesrgan" / "realesrgan-ncnn-vulkan.exe",
        LOCAL_BIN_DIR / "realesrgan-ncnn-vulkan.exe",
    ]
    for c in candidates:
        if c.is_file():
            return c
    found = shutil.which("realesrgan-ncnn-vulkan")
    return Path(found) if found else None


def get_rife_bin() -> Optional[Path]:
    """Finds the RIFE binary path."""
    candidates = [
        LOCAL_BIN_DIR / "rife" / "rife-ncnn-vulkan.exe",
        LOCAL_BIN_DIR / "rife-ncnn-vulkan.exe",
    ]
    for c in candidates:
        if c.is_file():
            return c
    found = shutil.which("rife-ncnn-vulkan")
    return Path(found) if found else None


def get_yunet_model() -> Optional[Path]:
    """Finds the YuNet face detector model."""
    path = MODELS_DIR / "yunet.onnx"
    return path if path.is_file() else None


def get_gfpgan_model() -> Optional[Path]:
    """Finds the GFPGAN model."""
    path = MODELS_DIR / "GFPGANv1.4.onnx"
    return path if path.is_file() else None


def _download_file(url: str, dest_path: Path, description: str = "Downloading") -> None:
    """Downloads a file with rich progress reporting."""
    download_file(url, dest_path, description)


def ensure_realesrgan(auto_download: bool = True) -> Path:
    """Ensures Real-ESRGAN binary and models are installed."""
    bin_path = get_realesrgan_bin()
    if bin_path:
        return bin_path

    if not auto_download:
        raise FileNotFoundError("Real-ESRGAN binary not found. Run with --download-models to download.")

    LOCAL_BIN_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = LOCAL_BIN_DIR / "realesrgan.zip"
    extract_dir = LOCAL_BIN_DIR / "realesrgan"

    console.print("[cyan]Real-ESRGAN not found. Downloading standalone GPU package...[/cyan]")
    _download_file(REALESRGAN_URL, zip_path, "Downloading Real-ESRGAN")

    console.print("[cyan]Extracting Real-ESRGAN...[/cyan]")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    if zip_path.exists():
        zip_path.unlink()

    bin_path = get_realesrgan_bin()
    if not bin_path:
        raise RuntimeError("Failed to set up Real-ESRGAN binary after extraction.")
    return bin_path


def ensure_rife(auto_download: bool = True) -> Path:
    """Ensures RIFE binary and models are installed."""
    bin_path = get_rife_bin()
    if bin_path:
        return bin_path

    if not auto_download:
        raise FileNotFoundError("RIFE binary not found. Run with --download-models to download.")

    LOCAL_BIN_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = LOCAL_BIN_DIR / "rife.zip"
    extract_dir = LOCAL_BIN_DIR / "rife"

    console.print("[cyan]RIFE motion interpolation not found. Downloading GPU package (~400MB)...[/cyan]")
    _download_file(RIFE_URL, zip_path, "Downloading RIFE")

    console.print("[cyan]Extracting RIFE...[/cyan]")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    # Flatten nested folder if present
    for child in extract_dir.iterdir():
        if child.is_dir() and "rife-ncnn-vulkan" in child.name:
            for item in child.iterdir():
                shutil.move(str(item), str(extract_dir / item.name))
            shutil.rmtree(child, ignore_errors=True)

    if zip_path.exists():
        zip_path.unlink()

    bin_path = get_rife_bin()
    if not bin_path:
        raise RuntimeError("Failed to set up RIFE binary after extraction.")
    return bin_path


def ensure_face_models(auto_download: bool = True) -> tuple[Path, Path]:
    """Ensures YuNet and GFPGAN models are downloaded."""
    yunet = get_yunet_model()
    gfpgan = get_gfpgan_model()

    if yunet and gfpgan:
        return yunet, gfpgan

    if not auto_download:
        raise FileNotFoundError("Face restoration models missing. Run with --download-models.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not yunet:
        target_yunet = MODELS_DIR / "yunet.onnx"
        console.print("[cyan]Downloading YuNet Face Detector...[/cyan]")
        _download_file(YUNET_URL, target_yunet, "Downloading YuNet")
        yunet = target_yunet

    if not gfpgan:
        target_gfpgan = MODELS_DIR / "GFPGANv1.4.onnx"
        console.print("[cyan]Downloading GFPGAN v1.4 Restorer (~340MB)...[/cyan]")
        _download_file(GFPGAN_URL, target_gfpgan, "Downloading GFPGAN")
        gfpgan = target_gfpgan

    return yunet, gfpgan


def ensure_all_models() -> None:
    """Pre-downloads all models and binaries."""
    console.print("[bold green]Checking and downloading all AI enhancement models...[/bold green]")
    ensure_realesrgan(auto_download=True)
    ensure_rife(auto_download=True)
    ensure_face_models(auto_download=True)
    console.print("[bold green]All models and binaries are ready![/bold green]")
