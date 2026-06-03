"""Pluggable mineral classifiers.

The pipeline depends only on the :class:`MineralClassifier` protocol:
given a list of grain crops, return one :class:`Prediction` per crop. This is
the "fan out N model queries" stage — swap the backend without touching the
rest of the pipeline.

Backends
--------
``ClipZeroShotClassifier``  Local, GPU-accelerated, no labels/training needed.
                            The default. Runs offline once weights are cached.
``ClaudeVisionClassifier``  Optional. Sends each crop to a Claude vision model.
                            Needs ANTHROPIC_API_KEY and the ``anthropic`` package.
"""
from __future__ import annotations

import base64
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Protocol

from PIL import Image

from .minerals import DEFAULT_MINERALS, Mineral, build_prompts


@dataclass
class Prediction:
    """One classifier result for one grain crop.

    Uncertainty fields (filled in by the pipeline / TTA wrapper):
      entropy   normalized predictive entropy of ``scores`` in [0, 1];
                0 = fully certain, 1 = uniform over all classes.
      margin    top1 - top2 probability; small = the model is torn.
      agreement fraction of test-time-augmented views that agreed with the
                final label (1.0 when TTA is off / single view).
    """

    label: str
    confidence: float
    # Full distribution over the label set (label -> probability), best first.
    scores: dict[str, float] = field(default_factory=dict)
    entropy: float = 0.0
    margin: float = 1.0
    agreement: float = 1.0


class MineralClassifier(Protocol):
    """Anything the pipeline can use to label grain crops."""

    labels: list[str]

    def classify(self, crops: list[Image.Image]) -> list[Prediction]: ...


# --------------------------------------------------------------------------- #
# Backend 1: local CLIP zero-shot
# --------------------------------------------------------------------------- #
class ClipZeroShotClassifier:
    """Zero-shot mineral classification with an open_clip model.

    No training and no labeled data: each mineral's classifier weight is the
    mean text embedding of several descriptive prompts (see ``minerals.py``),
    and a crop is labeled by cosine similarity to those weights.
    """

    def __init__(
        self,
        minerals: tuple[Mineral, ...] = DEFAULT_MINERALS,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        import open_clip  # lazy: keep import cost off the segmentation-only path
        import torch

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.eval().to(self.device)
        tokenizer = open_clip.get_tokenizer(model_name)

        names, prompts_per_name = build_prompts(minerals)
        self.labels = names

        # Build one averaged, L2-normalized text feature per mineral.
        feats = []
        with torch.no_grad():
            for prompts in prompts_per_name:
                tokens = tokenizer(prompts).to(self.device)
                t = self.model.encode_text(tokens)
                t = t / t.norm(dim=-1, keepdim=True)
                t = t.mean(dim=0)
                t = t / t.norm()
                feats.append(t)
        self.text_features = torch.stack(feats)  # (n_labels, dim)

    def classify(self, crops: list[Image.Image]) -> list[Prediction]:
        torch = self._torch
        if not crops:
            return []

        logit_scale = self.model.logit_scale.exp()
        preds: list[Prediction] = []
        for start in range(0, len(crops), self.batch_size):
            batch = crops[start : start + self.batch_size]
            imgs = torch.stack([self.preprocess(im) for im in batch]).to(self.device)
            with torch.no_grad():
                img_feats = self.model.encode_image(imgs)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                probs = (logit_scale * img_feats @ self.text_features.T).softmax(dim=-1)
            for row in probs:
                order = row.argsort(descending=True)
                scores = {self.labels[i]: float(row[i]) for i in order}
                best = order[0].item()
                preds.append(
                    Prediction(
                        label=self.labels[best],
                        confidence=float(row[best]),
                        scores=scores,
                    )
                )
        return preds


# --------------------------------------------------------------------------- #
# Backend 2: Claude vision (optional)
# --------------------------------------------------------------------------- #
_CLAUDE_SYSTEM = (
    "You are a petrographer identifying minerals in cross-polarized thin-section "
    "photomicrographs. You will be shown a single grain crop. Respond with the "
    "most likely mineral and a 0-1 confidence."
)


class ClaudeVisionClassifier:
    """Classify each crop by asking a Claude vision model (one call per crop).

    Fans out calls across a thread pool. Requires ANTHROPIC_API_KEY.
    """

    def __init__(
        self,
        minerals: tuple[Mineral, ...] = DEFAULT_MINERALS,
        model: str = "claude-opus-4-8",
        max_workers: int = 8,
        max_edge: int = 384,
    ) -> None:
        import anthropic  # lazy

        self.labels = [m.name for m in minerals]
        self.model = model
        self.max_workers = max_workers
        self.max_edge = max_edge
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def _encode(self, im: Image.Image) -> str:
        im = im.convert("RGB")
        im.thumbnail((self.max_edge, self.max_edge))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return base64.standard_b64encode(buf.getvalue()).decode()

    def _classify_one(self, im: Image.Image) -> Prediction:
        label_list = ", ".join(self.labels)
        prompt = (
            f"Identify the mineral. Choose exactly one from: {label_list}. "
            'Reply ONLY as JSON: {"mineral": "<name>", "confidence": <0-1>}.'
        )
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=100,
            system=[{"type": "text", "text": _CLAUDE_SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": self._encode(im),
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        text = "".join(b.text for b in msg.content if b.type == "text").strip()
        return self._parse(text)

    def _parse(self, text: str) -> Prediction:
        try:
            start, end = text.index("{"), text.rindex("}") + 1
            data = json.loads(text[start:end])
            label = str(data.get("mineral", "")).strip().lower()
            if label not in self.labels:
                # snap to closest known label by substring match, else first label
                label = next(
                    (l for l in self.labels if l in label or label in l),
                    self.labels[0],
                )
            conf = float(data.get("confidence", 0.0))
        except (ValueError, json.JSONDecodeError):
            label, conf = self.labels[0], 0.0
        return Prediction(label=label, confidence=conf, scores={label: conf})

    def classify(self, crops: list[Image.Image]) -> list[Prediction]:
        if not crops:
            return []
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            return list(ex.map(self._classify_one, crops))


# --------------------------------------------------------------------------- #
# Backend 3: fine-tuned CNN (trained on labeled thin-section crops)
# --------------------------------------------------------------------------- #
DEFAULT_CNN_CHECKPOINT = "checkpoints/mineral_cnn.pt"


class FineTunedClassifier:
    """Supervised CNN trained on labeled thin-section crops (see train_classifier.py).

    The checkpoint is self-describing: it carries its own label list, model name,
    and input size, so this backend ignores the ``minerals`` vocabulary used by
    the zero-shot backends — it reports exactly the classes it was trained on.
    """

    def __init__(
        self,
        checkpoint: str = DEFAULT_CNN_CHECKPOINT,
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        import torch

        from .cnn import build_model, eval_transform

        self._torch = torch
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(
                f"No fine-tuned checkpoint at {checkpoint!r}. "
                "Train one with: python train_classifier.py"
            )
        ckpt = torch.load(checkpoint, map_location="cpu")
        self.labels: list[str] = list(ckpt["labels"])
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.transform = eval_transform(ckpt.get("img_size", 224))

        self.model = build_model(
            ckpt.get("model_name", "resnet18"), len(self.labels), pretrained=False
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval().to(self.device)

    def classify(self, crops: list[Image.Image]) -> list[Prediction]:
        torch = self._torch
        if not crops:
            return []
        preds: list[Prediction] = []
        for start in range(0, len(crops), self.batch_size):
            batch = crops[start : start + self.batch_size]
            x = torch.stack([self.transform(im.convert("RGB")) for im in batch])
            x = x.to(self.device)
            with torch.no_grad():
                probs = self.model(x).softmax(dim=-1)
            for row in probs:
                order = row.argsort(descending=True)
                scores = {self.labels[i]: float(row[i]) for i in order}
                best = int(order[0])
                preds.append(
                    Prediction(
                        label=self.labels[best],
                        confidence=float(row[best]),
                        scores=scores,
                    )
                )
        return preds


def build_classifier(
    backend: str,
    minerals: tuple[Mineral, ...] = DEFAULT_MINERALS,
    **kwargs,
) -> MineralClassifier:
    """Factory: ``backend`` is ``"clip"`` (default), ``"claude"``, or ``"finetuned"``."""
    if backend == "clip":
        return ClipZeroShotClassifier(minerals=minerals, **kwargs)
    if backend == "claude":
        return ClaudeVisionClassifier(minerals=minerals, **kwargs)
    if backend == "finetuned":
        # Trained model is self-describing; mineral vocabulary does not apply.
        return FineTunedClassifier(**kwargs)
    raise ValueError(
        f"Unknown classifier backend: {backend!r} (use 'clip', 'claude', or 'finetuned')"
    )
