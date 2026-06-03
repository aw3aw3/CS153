"""Uncertainty quantification for grain classification.

Two complementary signals per grain:

1. **Distributional uncertainty** — from a single softmax distribution:
   - ``entropy``  normalized predictive entropy in [0, 1] (high = spread out).
   - ``margin``   top1 - top2 probability (low = the model is torn).

2. **Stability under test-time augmentation (TTA)** — minerals change appearance
   as the microscope stage rotates, so we classify several rotated/flipped views
   of each grain and measure how much the prediction moves:
   - averaged softmax across views (a more robust point estimate), and
   - ``agreement`` = fraction of views that pick the final label (low = unstable).

``TTAClassifier`` wraps ANY backend (it only calls ``classify`` and reads the
returned score distributions), so CLIP / fine-tuned / Claude all gain TTA for
free without changing the pipeline.
"""
from __future__ import annotations

import math
from typing import Callable

from PIL import Image

from .classify import MineralClassifier, Prediction


def predictive_entropy(scores: dict[str, float]) -> float:
    """Shannon entropy of ``scores``, normalized to [0, 1] by log(n_classes)."""
    probs = [p for p in scores.values() if p > 0]
    if len(probs) <= 1:
        return 0.0
    h = -sum(p * math.log(p) for p in probs)
    return h / math.log(len(scores))


def margin(scores: dict[str, float]) -> float:
    """Top1 - top2 probability. 1.0 if only one class."""
    vals = sorted(scores.values(), reverse=True)
    if len(vals) < 2:
        return 1.0
    return vals[0] - vals[1]


def annotate(pred: Prediction) -> Prediction:
    """Fill entropy/margin on a prediction from its score distribution."""
    if pred.scores:
        pred.entropy = predictive_entropy(pred.scores)
        pred.margin = margin(pred.scores)
    return pred


# --- test-time augmentations (lossless D4-style views) --------------------- #
def _identity(im: Image.Image) -> Image.Image:
    return im


def _rot90(im: Image.Image) -> Image.Image:
    return im.transpose(Image.Transpose.ROTATE_90)


def _rot180(im: Image.Image) -> Image.Image:
    return im.transpose(Image.Transpose.ROTATE_180)


def _rot270(im: Image.Image) -> Image.Image:
    return im.transpose(Image.Transpose.ROTATE_270)


def _hflip(im: Image.Image) -> Image.Image:
    return im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)


# Rotations are the petrographically meaningful augmentation (stage rotation).
DEFAULT_AUGMENTATIONS: tuple[Callable[[Image.Image], Image.Image], ...] = (
    _identity, _rot90, _rot180, _rot270, _hflip,
)


class TTAClassifier:
    """Wrap a base classifier and aggregate predictions over augmented views."""

    def __init__(
        self,
        base: MineralClassifier,
        augmentations: tuple[Callable[[Image.Image], Image.Image], ...] | None = None,
    ) -> None:
        self.base = base
        self.labels = base.labels
        self.augs = augmentations or DEFAULT_AUGMENTATIONS

    def classify(self, crops: list[Image.Image]) -> list[Prediction]:
        if not crops:
            return []
        k = len(self.augs)
        # Flatten N crops x K views into one batch for a single backend call.
        flat = [aug(im) for im in crops for aug in self.augs]
        flat_preds = self.base.classify(flat)

        out: list[Prediction] = []
        for i in range(len(crops)):
            group = flat_preds[i * k : (i + 1) * k]
            agg: dict[str, float] = {}
            for p in group:
                for label, v in p.scores.items():
                    agg[label] = agg.get(label, 0.0) + v
            n = len(group)
            agg = {lab: v / n for lab, v in agg.items()}
            agg = dict(sorted(agg.items(), key=lambda kv: kv[1], reverse=True))
            best = next(iter(agg))
            agreement = sum(1 for p in group if p.label == best) / n
            out.append(
                Prediction(
                    label=best,
                    confidence=agg[best],
                    scores=agg,
                    entropy=predictive_entropy(agg),
                    margin=margin(agg),
                    agreement=agreement,
                )
            )
        return out


def is_uncertain(
    pred: Prediction,
    min_confidence: float = 0.0,
    max_entropy: float = 1.0,
    min_agreement: float = 0.0,
) -> bool:
    """A grain is uncertain if it trips ANY enabled threshold."""
    return (
        pred.confidence < min_confidence
        or pred.entropy > max_entropy
        or pred.agreement < min_agreement
    )
