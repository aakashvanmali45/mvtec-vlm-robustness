"""Image corruption pipeline for robustness evaluation.

Implements five standard corruption types (Gaussian blur, Gaussian noise,
JPEG compression, brightness variation, contrast reduction) at three severity
levels each. Corruption is applied at inference time to test images; training
data remains clean.

Design follows the ImageNet-C convention (Hendrycks & Dietterich, 2019)
adapted for industrial images: severity levels are chosen to produce visible
degradation without pushing images out of the meaningful visual range.
"""

import io
from typing import Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


# Severity parameters, tuned per corruption type.
# Level 1 = light, 2 = moderate, 3 = severe.
_SEVERITY_PARAMS = {
    "gaussian_blur":   {1: 1.0, 2: 2.5, 3: 4.0},           # blur radius in pixels
    "gaussian_noise":  {1: 0.04, 2: 0.08, 3: 0.15},        # std as fraction of 255
    "jpeg_compression":{1: 50,  2: 25,  3: 10},            # JPEG quality (lower = worse)
    "brightness":      {1: 0.7, 2: 1.4, 3: 1.8},           # multiplier (< 1 dim, > 1 bright)
    "contrast":        {1: 0.7, 2: 0.5, 3: 0.3},           # contrast multiplier (< 1 = lower)
}


CORRUPTION_TYPES = tuple(_SEVERITY_PARAMS.keys())
SEVERITY_LEVELS = (1, 2, 3)


def apply_gaussian_blur(image: Image.Image, radius: float) -> Image.Image:
    """Gaussian blur with given radius in pixels."""
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_gaussian_noise(image: Image.Image, std_frac: float) -> Image.Image:
    """Additive Gaussian noise, std as fraction of 255."""
    arr = np.array(image, dtype=np.float32)
    noise = np.random.normal(loc=0.0, scale=std_frac * 255.0, size=arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def apply_jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    """Re-encode image as JPEG at given quality, then decode."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_brightness(image: Image.Image, factor: float) -> Image.Image:
    """Multiply brightness by factor (< 1 darker, > 1 brighter)."""
    return ImageEnhance.Brightness(image).enhance(factor)


def apply_contrast(image: Image.Image, factor: float) -> Image.Image:
    """Multiply contrast by factor (< 1 = flatter/hazier)."""
    return ImageEnhance.Contrast(image).enhance(factor)


_CORRUPTION_FUNCS: dict[str, Callable[[Image.Image, float], Image.Image]] = {
    "gaussian_blur":    apply_gaussian_blur,
    "gaussian_noise":   apply_gaussian_noise,
    "jpeg_compression": apply_jpeg_compression,
    "brightness":       apply_brightness,
    "contrast":         apply_contrast,
}


def corrupt_image(image: Image.Image, corruption_type: str, severity: int) -> Image.Image:
    """Apply a named corruption at a given severity level.

    Args:
        image: PIL RGB image.
        corruption_type: One of CORRUPTION_TYPES.
        severity: One of SEVERITY_LEVELS.

    Returns:
        New PIL Image with the corruption applied. Original is not modified.

    Raises:
        ValueError: If corruption_type or severity is unknown.
    """
    if corruption_type not in _CORRUPTION_FUNCS:
        raise ValueError(
            f"Unknown corruption '{corruption_type}'. Valid: {list(CORRUPTION_TYPES)}"
        )
    if severity not in SEVERITY_LEVELS:
        raise ValueError(f"Severity must be one of {SEVERITY_LEVELS}, got {severity}")

    param = _SEVERITY_PARAMS[corruption_type][severity]
    return _CORRUPTION_FUNCS[corruption_type](image, param)