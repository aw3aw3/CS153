"""SAM-based grain segmentation for petrographic thin sections."""
from __future__ import annotations

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

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


@dataclass
class Grain:
    """One segmented grain."""

    index: int
    mask: np.ndarray
    bbox_xywh: tuple[int, int, int, int]
    area_px: int
    score: float


def ensure_checkpoint(model_type: str) -> Path:
    """Download the SAM checkpoint on first use."""
    if model_type not in SAM_CHECKPOINTS:
        raise ValueError(f"Unknown SAM model_type: {model_type!r}")
    filename, url = SAM_CHECKPOINTS[model_type]
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / filename
    if not path.exists():
        print(f"[segmentation] Downloading {filename} → {path}")
        urllib.request.urlretrieve(url, path)
    return path


def load_sam(model_type: str = "vit_b") -> SamAutomaticMaskGenerator:
    ckpt = ensure_checkpoint(model_type)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=str(ckpt))
    sam.to(device=device)
    # Defaults are tuned for natural images; we drop the IoU/stability floors a
    # touch since grain edges in XPL are often lower-contrast.
    return SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.85,
        stability_score_thresh=0.90,
        min_mask_region_area=200,
    )


def read_image_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def segment_grains(
    image_rgb: np.ndarray,
    model_type: str = "vit_b",
    min_area: int = 200,
) -> list[Grain]:
    """Run SAM automatic mask generation and wrap results as Grain objects."""
    mask_generator = load_sam(model_type)
    raw = mask_generator.generate(image_rgb)
    raw.sort(key=lambda m: m["area"], reverse=True)
    grains: list[Grain] = []
    for i, m in enumerate(raw):
        if m["area"] < min_area:
            continue
        x, y, w, h = (int(v) for v in m["bbox"])
        grains.append(
            Grain(
                index=i,
                mask=m["segmentation"].astype(np.uint8),
                bbox_xywh=(x, y, w, h),
                area_px=int(m["area"]),
                score=float(m.get("predicted_iou", 0.0)),
            )
        )
    return grains


def draw_overlay(image_rgb: np.ndarray, grains: list[Grain]) -> np.ndarray:
    """Return an RGB image with grain boundaries + random fills."""
    overlay = image_rgb.copy()
    rng = np.random.default_rng(42)
    fill = np.zeros_like(overlay)
    for g in grains:
        color = rng.integers(64, 255, size=3, dtype=np.uint8)
        fill[g.mask.astype(bool)] = color
    overlay = cv2.addWeighted(overlay, 0.6, fill, 0.4, 0)
    # Boundaries on top.
    for g in grains:
        contours, _ = cv2.findContours(
            g.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)
    return overlay
