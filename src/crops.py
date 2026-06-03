"""Turn segmented grains into per-grain image crops ready for a classifier.

Each crop is the grain's bounding box (with a little padding for context),
optionally with the area outside the grain mask replaced by a neutral
background so the classifier focuses on the grain itself.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .segmentation import Grain


def background_fraction(
    image_rgb: np.ndarray,
    grain: Grain,
    value_thresh: int = 55,
    sat_thresh: int = 45,
) -> float:
    """Fraction of a grain's pixels that look like isotropic background.

    Under cross-polarized light, true background / mounting epoxy / holes stay
    *dark and unsaturated* (they don't show interference colors), whereas mineral
    grains light up. We flag a pixel as background if it's both dark (HSV value <
    ``value_thresh``) and low-saturation (HSV saturation < ``sat_thresh``).
    """
    m = grain.mask.astype(bool)
    if not m.any():
        return 1.0
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    s = hsv[..., 1][m]
    v = hsv[..., 2][m]
    dark_unsat = (v < value_thresh) & (s < sat_thresh)
    return float(dark_unsat.mean())


def is_background(
    image_rgb: np.ndarray,
    grain: Grain,
    dark_frac: float = 0.6,
    value_thresh: int = 55,
    sat_thresh: int = 45,
) -> bool:
    """True if most of the grain reads as isotropic background (see above)."""
    return background_fraction(image_rgb, grain, value_thresh, sat_thresh) >= dark_frac


def _pad_bbox(
    x: int, y: int, w: int, h: int, pad: int, W: int, H: int
) -> tuple[int, int, int, int]:
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W, x + w + pad)
    y1 = min(H, y + h + pad)
    return x0, y0, x1, y1


def crop_grain(
    image_rgb: np.ndarray,
    grain: Grain,
    pad_frac: float = 0.15,
    mask_background: bool = True,
    background: tuple[int, int, int] = (127, 127, 127),
) -> Image.Image:
    """Extract one grain as an RGB :class:`PIL.Image`.

    Parameters
    ----------
    pad_frac:
        Padding around the bbox as a fraction of the larger bbox side. Gives the
        classifier a little surrounding context without swamping the grain.
    mask_background:
        If True, pixels outside the grain mask (within the cropped window) are
        set to ``background``. If False, the raw rectangular crop is returned.
    background:
        Fill colour (RGB) used when ``mask_background`` is True. A mid-grey
        default avoids biasing CLIP toward "black/white" descriptors.
    """
    H, W = image_rgb.shape[:2]
    x, y, w, h = grain.bbox_xywh
    pad = int(round(max(w, h) * pad_frac))
    x0, y0, x1, y1 = _pad_bbox(x, y, w, h, pad, W, H)

    window = image_rgb[y0:y1, x0:x1]
    if mask_background:
        sub_mask = grain.mask[y0:y1, x0:x1].astype(bool)
        out = np.empty_like(window)
        out[...] = np.array(background, dtype=window.dtype)
        out[sub_mask] = window[sub_mask]
        window = out

    return Image.fromarray(window)


def crop_grains(
    image_rgb: np.ndarray,
    grains: list[Grain],
    pad_frac: float = 0.15,
    mask_background: bool = True,
    background: tuple[int, int, int] = (127, 127, 127),
) -> list[Image.Image]:
    """Vectorised convenience wrapper: one crop per grain, in input order."""
    return [
        crop_grain(
            image_rgb,
            g,
            pad_frac=pad_frac,
            mask_background=mask_background,
            background=background,
        )
        for g in grains
    ]
