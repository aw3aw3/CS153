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
from .crops import crop_grains, is_background
from .minerals import DEFAULT_MINERALS, NON_GRAIN_LABEL, Mineral
from .segmentation import Grain, read_image_rgb, segment_grains
from .stats import ModalSummary, summarize
from .uncertainty import TTAClassifier, annotate, is_uncertain

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

    @property
    def n_uncertain(self) -> int:
        return sum(1 for p in self.predictions if p.label == UNCERTAIN_LABEL)

    @property
    def n_non_grain(self) -> int:
        return sum(1 for p in self.predictions if p.label == NON_GRAIN_LABEL)


def _flag_uncertain(
    predictions: list[Prediction],
    min_confidence: float,
    max_entropy: float,
    min_agreement: float,
) -> list[Prediction]:
    """Annotate entropy/margin, then relabel grains that trip any threshold.

    The original top call is preserved under ``scores`` so nothing is lost — only
    the headline ``label`` becomes ``"uncertain"``.
    """
    out: list[Prediction] = []
    for p in predictions:
        p = annotate(p)
        # A confident "non-grain" call takes precedence: it gets excluded from
        # the modal mineralogy, so don't dilute it into "uncertain".
        if p.label == NON_GRAIN_LABEL:
            out.append(p)
        elif is_uncertain(p, min_confidence, max_entropy, min_agreement):
            out.append(replace(p, label=UNCERTAIN_LABEL))
        else:
            out.append(p)
    return out


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
    # uncertainty knobs
    tta: bool = False,
    min_confidence: float = 0.0,
    max_entropy: float = 1.0,
    min_agreement: float = 0.0,
    # exclude background/epoxy/holes from the modal mineralogy
    exclude_non_grain: bool = True,
    bg_dark_frac: float = 0.6,
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
    tta:
        Test-time augmentation — classify several rotated/flipped views of each
        grain and average them. Improves robustness and yields an ``agreement``
        stability score per grain. Costs ~5x classifier inference.
    min_confidence, max_entropy, min_agreement:
        A grain is relabeled ``"uncertain"`` if its top probability is below
        ``min_confidence``, OR its normalized entropy exceeds ``max_entropy``, OR
        (with TTA) its view-agreement is below ``min_agreement``. Defaults
        disable relabeling, but entropy/margin are always reported.
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

    if tta:
        classifier = TTAClassifier(classifier)

    # Photometric background detection: isotropic (dark + unsaturated) regions
    # are background/epoxy/holes, not minerals. Flag them up front so we don't
    # waste classifier calls on them and they stay out of the modal mineralogy.
    if exclude_non_grain:
        bg_flags = [is_background(image_rgb, g, dark_frac=bg_dark_frac) for g in grains]
    else:
        bg_flags = [False] * len(grains)
    n_bg = sum(bg_flags)
    if n_bg:
        log(f"{n_bg} grains flagged as non-grain background (excluded).")

    fg_grains = [g for g, bg in zip(grains, bg_flags) if not bg]
    crops = crop_grains(
        image_rgb,
        fg_grains,
        pad_frac=crop_pad_frac,
        mask_background=mask_background,
    )
    log(f"Classifying {len(crops)} grain crops{' (TTA)' if tta else ''}...")
    fg_preds = classifier.classify(crops)
    fg_preds = _flag_uncertain(
        fg_preds, min_confidence, max_entropy, min_agreement
    )

    # Reassemble predictions aligned to the full grain list.
    bg_pred = lambda: Prediction(
        label=NON_GRAIN_LABEL, confidence=1.0, scores={NON_GRAIN_LABEL: 1.0},
        entropy=0.0, margin=1.0, agreement=1.0,
    )
    predictions: list[Prediction] = []
    it = iter(fg_preds)
    for bg in bg_flags:
        predictions.append(bg_pred() if bg else next(it))

    exclude = (NON_GRAIN_LABEL,) if exclude_non_grain else ()
    summary = summarize(grains, predictions, exclude_labels=exclude)
    log("Done.")
    return ThinSectionResult(
        image_rgb=image_rgb,
        grains=grains,
        predictions=predictions,
        summary=summary,
    )
