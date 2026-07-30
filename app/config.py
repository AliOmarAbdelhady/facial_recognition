"""
Central configuration for the Siamese Facial Recognition app.

Edit paths / thresholds here rather than scattering magic values across files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# Resolve project root as: <repo>/app/../  => <repo>
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default locations (override via env or direct edit).
MODEL_DIR = PROJECT_ROOT / "models"
DATASET_DIR = PROJECT_ROOT / "Photos_dataset"
APP_DIR = PROJECT_ROOT / "app"


@dataclass
class AppConfig:
    # ---- Model / image -------------------------------------------------------
    checkpoint_path: Path = MODEL_DIR / "siamese_backbone.pth"
    threshold_path: Path = MODEL_DIR / "threshold.json"
    embedding_db_path: Path = MODEL_DIR / "embeddings.npz"
    dataset_dir: Path = DATASET_DIR

    img_size: int = 112                # input face crop size (H==W)
    embedding_dim: int = 512

    # Normalization stats (matches Kaggle notebook: ImageNet stats on [0,1] input).
    # If you change the notebook's normalization, update these too.
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)

    # ---- Decision thresholds -------------------------------------------------
    # cosine distance = 1 - cos_sim. Lower => more similar.
    # These are sensible defaults; the notebook emits a calibrated value too.
    default_cosine_distance_threshold: float = 0.55   # 1 - cos_sim
    default_verify_threshold: float = 0.55
    default_identify_threshold: float = 0.50          # stricter for 1:N

    # ---- Face detection (MTCNN) ---------------------------------------------
    min_face_size: int = 40
    face_detect_thresholds: tuple = (0.6, 0.7, 0.7)

    # ---- Inference performance ----------------------------------------------
    device: str = "cpu"               # no GPU locally
    tta_augments: int = 5             # test-time augmentations (0 = single pass)
    batch_size: int = 8

    # ---- Enrollment ----------------------------------------------------------
    enroll_augments: int = 10         # augmentations averaged into prototype


CONFIG = AppConfig()


def ensure_dirs() -> None:
    """Create model/output directories on startup if missing."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
