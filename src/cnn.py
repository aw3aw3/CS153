"""Shared CNN architecture + preprocessing for the fine-tuned mineral classifier.

Imported by both ``train_classifier.py`` (training) and ``classify.py``
(inference) so the model definition and image normalization stay in sync.
"""
from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Background grey used by the segmentation pipeline (crops.crop_grain).
PIPELINE_BG = (127, 127, 127)


class RandomGrainMask:
    """Grey out everything outside a random rotated ellipse.

    Closes the train/inference *domain gap*: at inference the pipeline feeds the
    model a grain on a grey background (its SAM mask, everything else set to grey),
    but the raw training crops are full rectangular single-mineral images. This
    augmentation makes training crops look like real segmented-grain crops — an
    irregular blob of mineral surrounded by ``PILPELINE_BG`` grey — so the model
    learns to ignore the grey and read the grain. (``PIPELINE_BG`` matches the
    grey used by crops.crop_grain so train and inference backgrounds are identical.)
    """

    def __init__(self, p: float = 0.6, fill: tuple[int, int, int] = PIPELINE_BG):
        self.p = p
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        arr = np.asarray(img.convert("RGB"))
        h, w = arr.shape[:2]
        yy, xx = np.ogrid[:h, :w]
        cy, cx = h * random.uniform(0.42, 0.58), w * random.uniform(0.42, 0.58)
        ry, rx = h * random.uniform(0.33, 0.5), w * random.uniform(0.33, 0.5)
        theta = random.uniform(0.0, math.pi)
        ct, stt = math.cos(theta), math.sin(theta)
        x, y = xx - cx, yy - cy
        xr, yr = x * ct + y * stt, -x * stt + y * ct
        inside = (xr / rx) ** 2 + (yr / ry) ** 2 <= 1.0
        out = arr.copy()
        out[~inside] = np.array(self.fill, dtype=arr.dtype)
        return Image.fromarray(out)

# torchvision model name -> (constructor attr, default weights enum attr)
SUPPORTED_MODELS = ("resnet18", "resnet34", "resnet50")


def build_model(model_name: str, n_classes: int, pretrained: bool = True):
    """Build a torchvision ResNet with its final layer resized to ``n_classes``."""
    import torch.nn as nn
    import torchvision.models as tvm

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model {model_name!r}; use one of {SUPPORTED_MODELS}")

    ctor = getattr(tvm, model_name)
    weights = "DEFAULT" if pretrained else None
    model = ctor(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def eval_transform(img_size: int = 224):
    """Deterministic preprocessing for inference / validation."""
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def train_transform(img_size: int = 224):
    """Augmented preprocessing for training.

    - ``RandomGrainMask`` makes crops look like the pipeline's segmented-grain
      crops (grain on grey) — the key domain-gap fix.
    - Rotation/flip: minerals look different at different stage angles (the
      dataset spans 360°), so this improves rotational robustness. Rotation fills
      new borders with the same pipeline grey, not black.
    - Lower RandomResizedCrop scale adds partial-grain / edge views."""
    from torchvision import transforms

    return transforms.Compose(
        [
            RandomGrainMask(p=0.6),
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180, fill=PIPELINE_BG),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
