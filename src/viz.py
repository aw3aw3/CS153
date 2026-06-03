"""Visual outputs: a mineral-colored overlay and a modal-distribution chart.

Colors are assigned deterministically per mineral name so the overlay legend
and the distribution chart stay consistent across runs and images.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .classify import Prediction
from .segmentation import Grain
from .stats import ModalSummary

# A fixed, perceptually distinct palette (RGB). Extra minerals wrap around.
_PALETTE: tuple[tuple[int, int, int], ...] = (
    (228, 26, 28), (55, 126, 184), (77, 175, 74), (152, 78, 163),
    (255, 127, 0), (255, 217, 47), (166, 86, 40), (247, 129, 191),
    (153, 153, 153), (23, 190, 207), (188, 189, 34), (140, 86, 75),
    (127, 127, 127), (44, 160, 44), (214, 39, 40), (148, 103, 189),
)


def mineral_color_map(labels: list[str]) -> dict[str, tuple[int, int, int]]:
    """Deterministic label -> RGB. ``uncertain`` is always grey."""
    cmap: dict[str, tuple[int, int, int]] = {}
    i = 0
    for label in labels:
        if label == "uncertain":
            cmap[label] = (110, 110, 110)
        else:
            cmap[label] = _PALETTE[i % len(_PALETTE)]
            i += 1
    return cmap


def draw_mineral_overlay(
    image_rgb: np.ndarray,
    grains: list[Grain],
    predictions: list[Prediction],
    cmap: dict[str, tuple[int, int, int]] | None = None,
    alpha: float = 0.45,
) -> np.ndarray:
    """Fill each grain with its mineral's color, draw white boundaries."""
    if cmap is None:
        cmap = mineral_color_map(sorted({p.label for p in predictions}))
    overlay = image_rgb.copy()
    fill = np.zeros_like(overlay)
    for g, p in zip(grains, predictions):
        color = np.array(cmap.get(p.label, (127, 127, 127)), dtype=np.uint8)
        fill[g.mask.astype(bool)] = color
    overlay = cv2.addWeighted(overlay, 1.0 - alpha, fill, alpha, 0)
    for g in grains:
        contours, _ = cv2.findContours(
            g.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)
    return overlay


def save_distribution_chart(
    summary: ModalSummary,
    path: Path,
    cmap: dict[str, tuple[int, int, int]] | None = None,
) -> None:
    """Horizontal bar chart of area% and count% per mineral."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    minerals = summary.minerals
    if cmap is None:
        cmap = mineral_color_map([m.mineral for m in minerals])

    names = [m.mineral for m in minerals]
    area = [m.area_frac * 100 for m in minerals]
    count = [m.count_frac * 100 for m in minerals]
    colors = [tuple(c / 255 for c in cmap.get(n, (127, 127, 127))) for n in names]
    y = np.arange(len(names))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, max(3, 0.5 * len(names) + 1)))
    ax1.barh(y, area, color=colors)
    ax1.set_yticks(y, names)
    ax1.invert_yaxis()
    ax1.set_xlabel("area %")
    ax1.set_title("Modal mineralogy (area %)")
    for i, v in enumerate(area):
        ax1.text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=8)

    ax2.barh(y, count, color=colors)
    ax2.set_yticks(y, [])
    ax2.invert_yaxis()
    ax2.set_xlabel("count %")
    ax2.set_title("Grain count %")
    for i, v in enumerate(count):
        ax2.text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
