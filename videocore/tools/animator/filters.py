"""Artistic and photorealistic image transformation filters."""

from __future__ import annotations

import cv2
import numpy as np


def apply_color_quantization(img: np.ndarray, levels: int = 8) -> np.ndarray:
    """Reduces the number of color levels for a cel-shaded animated look."""
    div = 256 // max(2, min(levels, 64))
    quantized = (img // div) * div + div // 2
    return quantized.astype(np.uint8)


def apply_cartoon_filter(
    img: np.ndarray,
    line_weight: float = 1.0,
    color_levels: int = 7,
    blur_iterations: int = 3,
) -> np.ndarray:
    """Converts a real image into a stylized cel-shaded cartoon.

    Applies spatial receptive downscaling, edge-preserving bilateral filtering,
    cel-shading color quantization, and adaptive contour ink edge extraction.
    """
    h, w = img.shape[:2]
    # Downscale large images for proper cel-shading receptive field and performance
    max_dim = max(h, w)
    scale = 640.0 / max_dim if max_dim > 640 else 1.0
    if scale < 1.0:
        sw, sh = int(round(w * scale)), int(round(h * scale))
        working = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        working = img.copy()

    # 1. Edge-preserving bilateral filter passes to flatten textures into cel planes
    smooth = working
    for _ in range(max(1, blur_iterations)):
        smooth = cv2.bilateralFilter(smooth, d=9, sigmaColor=80, sigmaSpace=80)

    # 2. Cel-shading color quantization
    quantized = apply_color_quantization(smooth, levels=color_levels)

    # 3. Bold contour ink extraction
    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray_blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        9,
        3.5,
    )

    # Line thickness adjustment
    if line_weight > 1.0:
        k_size = int(np.clip(round(line_weight * 1.5), 2, 5))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        inv = 255 - edges
        inv = cv2.dilate(inv, kernel, iterations=1)
        edges = 255 - inv
    elif line_weight < 0.8:
        inv = 255 - edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        inv = cv2.erode(inv, kernel, iterations=1)
        edges = 255 - inv

    # Combine edge mask with quantized colors
    edge_3c = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    result = (quantized.astype(np.float32) * edge_3c).clip(0, 255).astype(np.uint8)

    # Scale back to original resolution if scaled
    if scale < 1.0:
        result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)
    return result


def apply_comic_filter(
    img: np.ndarray,
    line_weight: float = 1.2,
    color_mode: str = "color",
) -> np.ndarray:
    """Converts an image into high-contrast comic book / manga ink art."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    # Strong ink outlines
    edges = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        9,
        3.0,
    )

    if line_weight > 1.0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        inv = 255 - edges
        inv = cv2.dilate(inv, kernel, iterations=int(round(line_weight)))
        edges = 255 - inv

    if color_mode == "bw":
        # Pure B&W Manga
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Stylized saturated color comic
    bilateral = cv2.bilateralFilter(img, d=9, sigmaColor=120, sigmaSpace=120)
    hsv = cv2.cvtColor(bilateral, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.35, 0, 255)
    boosted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    edge_mask = (edges.astype(np.float32) / 255.0)[:, :, np.newaxis]
    return (boosted * edge_mask).astype(np.uint8)


def apply_watercolor_filter(
    img: np.ndarray,
    line_weight: float = 0.8,
) -> np.ndarray:
    """Creates a soft watercolor / gouache painterly effect."""
    # Use stylization filter if supported, or multi-pass bilateral
    smooth = cv2.bilateralFilter(img, d=15, sigmaColor=90, sigmaSpace=90)
    smooth = cv2.bilateralFilter(smooth, d=9, sigmaColor=50, sigmaSpace=50)

    # Soft pencil outlines
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv_gray = 255 - gray
    blurred_inv = cv2.GaussianBlur(inv_gray, (21, 21), 0)
    dodge = cv2.divide(gray, 255 - blurred_inv, scale=256)

    # Blend pencil sketch subtly with watercolor wash
    sketch_3c = cv2.cvtColor(dodge, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    sketch_weight = np.clip(line_weight * 0.35, 0.0, 0.8)
    watercolor = (smooth.astype(np.float32) * (1.0 - sketch_weight + sketch_weight * sketch_3c)).astype(np.uint8)
    return watercolor


def apply_de_cartoon_filter(
    img: np.ndarray,
    edge_soften: float = 0.7,
    grain_amount: float = 0.15,
    natural_tone: bool = True,
) -> np.ndarray:
    """Transforms 2D animation / cartoon visuals into photorealistic camera footage.

    1. Delineation: Detects and inpaints harsh black drawn contour lines using neighbor color diffusion.
    2. De-posterization: Eliminates cel-shading banding and creates continuous optical gradients.
    3. Photographic tone mapping: Calibrates contrast S-curves, skin tones, and dynamic range.
    4. Film grain / Sensor noise: Injects 35mm physical film grain correlated with luminance.
    """
    h, w = img.shape[:2]

    # --- 1. Delineation (Detect and inpaint drawn ink outlines) ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        7,
        2.5,
    )
    inv_edges = 255 - edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated_edges = cv2.dilate(inv_edges, kernel, iterations=1)

    soften_factor = np.clip(edge_soften, 0.0, 1.0)
    if soften_factor > 0.05:
        # Inpaint dark drawn lines with neighbor color diffusion
        inpainted = cv2.inpaint(img, dilated_edges, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        delineated = cv2.addWeighted(inpainted, float(soften_factor), img, float(1.0 - soften_factor), 0.0)
    else:
        delineated = img.copy()

    # --- 2. De-posterization & Continuous Gradient Shading ---
    # Bilateral smoothing eliminates harsh cel boundaries into smooth optical transitions
    smooth_tones = cv2.bilateralFilter(delineated, d=7, sigmaColor=50, sigmaSpace=50)
    delineated = cv2.addWeighted(smooth_tones, 0.65, delineated, 0.35, 0.0)

    # --- 3. Photographic Tone Curve & Natural Skin Color Grading ---
    if natural_tone:
        # Convert to LAB for natural photographic color grading
        lab = cv2.cvtColor(delineated, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_channel, a_channel, b_channel = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

        # Apply photographic S-curve to Luminance (L) for camera sensor dynamic range
        norm_l = l_channel / 255.0
        s_curve_l = norm_l + 0.06 * np.sin(2.0 * np.pi * norm_l)
        lab[:, :, 0] = np.clip(s_curve_l * 255.0, 0, 255)

        # Modulate anime super-saturated dyes towards organic, natural skin and fabric tones
        a_channel = (a_channel - 128.0) * 0.88 + 128.0
        b_channel = (b_channel - 128.0) * 0.86 + 128.0
        lab[:, :, 1] = np.clip(a_channel, 0, 255)
        lab[:, :, 2] = np.clip(b_channel, 0, 255)

        delineated = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # --- 4. 35mm Film Grain / Sensor Noise Synthesis ---
    if grain_amount > 0.01:
        gray_for_luma = cv2.cvtColor(delineated, cv2.COLOR_BGR2GRAY)
        luma_norm = (gray_for_luma.astype(np.float32) / 255.0)[:, :, np.newaxis]
        # Midtone mask: highest at 0.5, lower at deep blacks and peak whites
        grain_mask = 4.0 * luma_norm * (1.0 - luma_norm)

        noise_std = grain_amount * 22.0
        noise = np.random.normal(0, noise_std, (h, w, 3)).astype(np.float32)
        delineated = (delineated.astype(np.float32) + (noise * grain_mask)).clip(0, 255).astype(np.uint8)

    return delineated


def adjust_saturation(img: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """Adjusts image saturation."""
    if abs(factor - 1.0) < 0.01:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def adjust_brightness_contrast(img: np.ndarray, brightness: float = 0.0, contrast: float = 1.0) -> np.ndarray:
    """Adjusts image brightness and contrast."""
    if abs(brightness) < 0.01 and abs(contrast - 1.0) < 0.01:
        return img
    res = img.astype(np.float32) * contrast + brightness * 255.0
    return np.clip(res, 0, 255).astype(np.uint8)


def blend_with_original(original: np.ndarray, transformed: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Blends transformed image with original based on effect strength (0.0 to 1.0)."""
    alpha = np.clip(strength, 0.0, 1.0)
    if alpha >= 0.999:
        return transformed
    if alpha <= 0.001:
        return original
    return cv2.addWeighted(transformed, alpha, original, 1.0 - alpha, 0.0)
