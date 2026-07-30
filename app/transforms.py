"""
Image preprocessing + augmentation shared across enrollment & inference.

Both the Kaggle training notebook and this CPU app use **torchvision transforms**
for augmentation (no extra dependency, stable across versions, numpy-2 safe).
Here we provide the deterministic eval transform plus light stochastic
augmentation for test-time augmentation (TTA) and enrollment prototype averaging.
"""

from __future__ import annotations

import random
from typing import List

import torch
import torchvision.transforms as T
from PIL import Image

from config import CONFIG


# ---------------------------------------------------------------------------
# Deterministic training/eval transform
# ---------------------------------------------------------------------------
def build_eval_transform() -> T.Compose:
    """Resize -> ToTensor -> normalize (ImageNet stats). Matches the notebook."""
    return T.Compose([
        T.Resize((CONFIG.img_size, CONFIG.img_size)),
        T.ToTensor(),
        T.Normalize(mean=CONFIG.mean, std=CONFIG.std),
    ])


def preprocess_face(face: Image.Image) -> torch.Tensor:
    """Single deterministic tensor for a PIL face crop."""
    return build_eval_transform()(face).unsqueeze(0)


# ---------------------------------------------------------------------------
# Stochastic augmentations (for TTA + enrollment prototype averaging)
# ---------------------------------------------------------------------------
_COLOR = (0.4, 0.4), (0.4, 0.4)  # color jitter magnitude


def build_aug_transform() -> T.Compose:
    """One draw of light augmentation. Resize kept deterministic at 112x112."""
    return T.Compose([
        T.Resize((CONFIG.img_size, CONFIG.img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        T.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        T.ToTensor(),
        T.Normalize(mean=CONFIG.mean, std=CONFIG.std),
    ])


def augment_face(face: Image.Image, n: int, seed: int | None = None) -> List[torch.Tensor]:
    """Return `n` augmented tensors for a single PIL face crop."""
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    tfm = build_aug_transform()
    return [tfm(face) for _ in range(n)]
