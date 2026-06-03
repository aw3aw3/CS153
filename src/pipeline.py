"""End-to-end thin-section analysis pipeline.

    image  ->  SAM segmentation  ->  per-grain crops  ->  mineral classification
           ->  modal-mineralogy summary

``analyze_thin_section`` is the single importable entrypoint. A future upload UI
(or batch job) should build a classifier ONCE and pass it in, so the model
weights are loaded a single time and reused across many images.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .classify import MineralClassifier, Prediction, build_classifier
from .crops import crop_grains
from .minerals import DEFAULT_MINERALS, Mineral
from .segmentation import Grain, read_image_rgb, segment_grains
from .stats import ModalSummary, summarize

UNCERTAIN_LABEL = "uncertain"


@dataclass
class ThinSectionResult:
    image_rgb: np.ndarray
    grains: list[Grain]
    predictions: list[Prediction]
    summary: ModalSummary

    @property
    def n_grains(self) -> int:
        return len(self.grains)


def _apply_confidence_floor(
    predictions: list[Prediction], min_confidence: float
) -> list[Prediction]:
    if min_confidence <= 0:
        return predictions
    return [
        p if p.confidence >= min_confidence else replace(p, label=UNCERTAIN_LABEL)
        for p in predictions
    ]


def analyze_thin_section(
    image: str | Path | np.ndarray,
    classifier: MineralClassifier | None = None,
    *,
    minerals: tuple[Mineral, ...] = DEFAULT_MINERALS,
    backend: str = "clip",
    # segmentation knobs (forwarded to segment_grains)
    model_type: str = "vit_b",
    min_area: int = 200,
    long_edge: int = 1024,
    points_per_side: int = 16,
    pred_iou_thresh: float = 0.85,
    stability_score_thresh: float = 0.90,
    # crop / classification knobs
    crop_pad_frac: float = 0.15,
    mask_background: bool = True,
    min_confidence: float = 0.0,
    progress: bool = False,
) -> ThinSectionResult:
    """Run the full pipeline on one thin-section image.

    Parameters
    ----------
    image:
        Path to an image, or an already-loaded RGB ``np.ndarray``.
    classifier:
        A ready :class:`MineralClassifier`. If ``None``, one is built from
        ``backend`` + ``minerals`` (default: local CLIP zero-shot). Pass your own
        to reuse loaded weights across many images.
    min_confidence:
        Predictions below this confidence are relabeled ``"uncertain"`` before
        the summary is computed (0 disables — the default).
    """
    if isinstance(image, np.ndarray):
        image_rgb = image
    else:
        image_rgb = read_image_rgb(Path(image))

    def log(msg: str) -> None:
        if progress:
            print(f"[pipeline] {msg}")

    log(f"Segmenting ({model_type}, long_edge={long_edge})...")
    grains = segment_grains(
        image_rgb,
        model_type=model_type,
        min_area=min_area,
        long_edge=long_edge,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
    )
    log(f"{len(grains)} grains found.")

    if classifier is None:
        log(f"Loading '{backend}' classifier...")
        classifier = build_classifier(backend, minerals=minerals)

    crops = crop_grains(
        image_rgb,
        grains,
        pad_frac=crop_pad_frac,
        mask_background=mask_background,
    )
    log(f"Classifying {len(crops)} grain crops...")
    predictions = classifier.classify(crops)
    predictions = _apply_confidence_floor(predictions, min_confidence)

    summary = summarize(grains, predictions)
    log("Done.")
    return ThinSectionResult(
        image_rgb=image_rgb,
        grains=grains,
        predictions=predictions,
        summary=summary,
    )
