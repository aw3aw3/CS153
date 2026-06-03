"""Shared CNN architecture + preprocessing for the fine-tuned mineral classifier.

Imported by both ``train_classifier.py`` (training) and ``classify.py``
(inference) so the model definition and image normalization stay in sync.
"""
from __future__ import annotations

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

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
    """Augmented preprocessing for training. Rotation is apt here — minerals
    look different at different microscope stage angles, and the dataset itself
    spans 360°, so random rotation/flip improves rotational robustness."""
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
