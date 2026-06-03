"""SAM-based grain segmentation for petrographic thin sections."""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

SAM_CHECKPOINTS = {
    "vit_b": (
        "sam_vit_b_01ec64.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    ),
    "vit_l": (
        "sam_vit_l_0b3195.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    ),
    "vit_h": (
        "sam_vit_h_4b8939.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    ),
}

# Heavy SAM weights (~375 MB–2.5 GB) can be kept off the (small) SSD / OneDrive
# by setting CS153_CHECKPOINT_DIR — e.g. to a large secondary drive.
CHECKPOINT_DIR = Path(
    os.environ.get(
        "CS153_CHECKPOINT_DIR", Path(__file__).resolve().parent.parent / "checkpoints"
    )
)


@dataclass
class Grain:
    """One segmented grain (mask at original image resolution)."""

    index: int
    mask: np.ndarray
    bbox_xywh: tuple[int, int, int, int]
    area_px: int
    score: float


def ensure_checkpoint(model_type: str) -> Path:
    if model_type not in SAM_CHECKPOINTS:
        raise ValueError(f"Unknown SAM model_type: {model_type!r}")
    filename, url = SAM_CHECKPOINTS[model_type]
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / filename
    if not path.exists():
        print(f"[segmentation] Downloading {filename} → {path}")
        urllib.request.urlretrieve(url, path)
    return path


def load_sam(
    model_type: str = "vit_b",
    points_per_side: int = 16,
    pred_iou_thresh: float = 0.85,
    stability_score_thresh: float = 0.90,
    min_mask_region_area: int = 200,
) -> SamAutomaticMaskGenerator:
    ckpt = ensure_checkpoint(model_type)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=str(ckpt))
    sam.to(device=device)
    return SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area,
    )


def read_image_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_to_long_edge(
    image_rgb: np.ndarray, long_edge: int
) -> tuple[np.ndarray, float]:
    """Resize so max(h, w) == long_edge. Returns (resized, scale) where
    scale = resized_dim / original_dim. If image already smaller, returns unchanged
    with scale=1.0."""
    h, w = image_rgb.shape[:2]
    cur_long = max(h, w)
    if cur_long <= long_edge:
        return image_rgb, 1.0
    scale = long_edge / cur_long
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def segment_grains(
    image_rgb: np.ndarray,
    model_type: str = "vit_b",
    min_area: int = 200,
    long_edge: int = 1024,
    points_per_side: int = 16,
    pred_iou_thresh: float = 0.85,
    stability_score_thresh: float = 0.90,
) -> list[Grain]:
    """Run SAM auto-mask-gen at a downsampled resolution, then rescale masks
    back to the original image size. min_area is enforced in ORIGINAL pixels."""
    h0, w0 = image_rgb.shape[:2]
    resized, scale = resize_to_long_edge(image_rgb, long_edge)
    if scale != 1.0:
        print(
            f"[segmentation] Resized {w0}x{h0} → {resized.shape[1]}x{resized.shape[0]} (scale={scale:.3f})"
        )

    mask_generator = load_sam(
        model_type=model_type,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=max(1, int(min_area * scale * scale)),
    )
    raw = mask_generator.generate(resized)
    raw.sort(key=lambda m: m["area"], reverse=True)

    grains: list[Grain] = []
    for i, m in enumerate(raw):
        small_mask = m["segmentation"].astype(np.uint8)
        if scale != 1.0:
            mask = cv2.resize(small_mask, (w0, h0), interpolation=cv2.INTER_NEAREST)
        else:
            mask = small_mask
        area = int(mask.sum())
        if area < min_area:
            continue
        ys, xs = np.where(mask > 0)
        if xs.size == 0:
            continue
        x, y = int(xs.min()), int(ys.min())
        w, h = int(xs.max() - x + 1), int(ys.max() - y + 1)
        grains.append(
            Grain(
                index=i,
                mask=mask,
                bbox_xywh=(x, y, w, h),
                area_px=area,
                score=float(m.get("predicted_iou", 0.0)),
            )
        )
    return grains


def draw_overlay(image_rgb: np.ndarray, grains: list[Grain]) -> np.ndarray:
    overlay = image_rgb.copy()
    rng = np.random.default_rng(42)
    fill = np.zeros_like(overlay)
    for g in grains:
        color = rng.integers(64, 255, size=3, dtype=np.uint8)
        fill[g.mask.astype(bool)] = color
    overlay = cv2.addWeighted(overlay, 0.6, fill, 0.4, 0)
    for g in grains:
        contours, _ = cv2.findContours(
            g.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)
    return overlay
