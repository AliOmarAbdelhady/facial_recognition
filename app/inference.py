"""
Inference: verification (1:1) and identification (1:N) with threshold gating.

Distances are computed in cosine-distance space:  d = 1 - cos_sim.
Because embeddings are L2-normalized, this is equivalent to 0.5*||a-b||^2 and
ranges over [0, 2]. Smaller => more similar.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from config import CONFIG
from enrollment import embed_face, load_db
from face_detector import FaceDetector, load_image
from model import EmbeddingNet


# ---------------------------------------------------------------------------
# Threshold loading (prefer the calibrated value shipped from Kaggle)
# ---------------------------------------------------------------------------
def load_threshold(default: Optional[float] = None) -> float:
    default = CONFIG.default_identify_threshold if default is None else default
    path = CONFIG.threshold_path
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
            # Prefer cosine-distance threshold if present.
            for key in ("cosine_distance_threshold", "identify_threshold",
                        "threshold", "distance_threshold"):
                if key in data:
                    return float(data[key])
        except Exception:
            pass
    return default


# ---------------------------------------------------------------------------
# Core geometry
# ---------------------------------------------------------------------------
def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(1.0 - np.dot(a, b))   # in [0, 2], smaller == more similar


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - cosine_distance(a, b))


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class VerifyResult:
    same_person: bool
    cosine_distance: float
    cosine_similarity: float
    threshold: float
    face1_found: bool
    face2_found: bool
    message: str


@dataclass
class IdentifyResult:
    identity: Optional[str]
    cosine_distance: float
    cosine_similarity: float
    threshold: float
    face_found: bool
    ranked: List[Tuple[str, float]]   # (name, cosine_distance) sorted ascending
    message: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def verify(
    image1,
    image2,
    net: EmbeddingNet,
    detector: FaceDetector,
    threshold: Optional[float] = None,
    tta: int = CONFIG.tta_augments,
) -> VerifyResult:
    """1:1 verification: are these two images the same person?"""
    threshold = CONFIG.default_verify_threshold if threshold is None else threshold

    p1 = load_image(image1)
    p2 = load_image(image2)
    f1 = detector.get_largest_face(p1)
    f2 = detector.get_largest_face(p2)

    if f1 is None or f2 is None:
        return VerifyResult(
            same_person=False, cosine_distance=2.0, cosine_similarity=-1.0,
            threshold=threshold, face1_found=f1 is not None,
            face2_found=f2 is not None,
            message="No face detected in one or both images.",
        )

    e1 = embed_face(f1, net, tta=tta).numpy()
    e2 = embed_face(f2, net, tta=tta).numpy()
    d = cosine_distance(e1, e2)
    sim = 1.0 - d
    same = d <= threshold

    return VerifyResult(
        same_person=same, cosine_distance=d, cosine_similarity=sim,
        threshold=threshold, face1_found=True, face2_found=True,
        message=(f"SAME person (cos_dist={d:.3f} <= {threshold:.3f})"
                 if same else
                 f"DIFFERENT people (cos_dist={d:.3f} > {threshold:.3f})"),
    )


def identify(
    image,
    net: EmbeddingNet,
    detector: FaceDetector,
    threshold: Optional[float] = None,
    tta: int = CONFIG.tta_augments,
    db: Optional[Dict[str, np.ndarray]] = None,
) -> IdentifyResult:
    """1:N identification: who is this, or Unknown?"""
    threshold = load_threshold() if threshold is None else threshold
    db = load_db() if db is None else db

    pil = load_image(image)
    face = detector.get_largest_face(pil)
    if face is None:
        return IdentifyResult(
            identity=None, cosine_distance=2.0, cosine_similarity=-1.0,
            threshold=threshold, face_found=False, ranked=[],
            message="No face detected.",
        )
    if not db:
        return IdentifyResult(
            identity=None, cosine_distance=2.0, cosine_similarity=-1.0,
            threshold=threshold, face_found=True, ranked=[],
            message="Database is empty. Enroll people first.",
        )

    query = embed_face(face, net, tta=tta).numpy()
    ranked = sorted(
        ((name, cosine_distance(query, emb)) for name, emb in db.items()),
        key=lambda x: x[1],
    )

    best_name, best_dist = ranked[0]
    best_sim = 1.0 - best_dist
    identity = best_name if best_dist <= threshold else None

    if identity is None:
        msg = (f"UNKNOWN (closest='{best_name}', cos_dist={best_dist:.3f} "
               f"> {threshold:.3f})")
    else:
        msg = (f"IDENTIFIED as '{best_name}' (cos_dist={best_dist:.3f} "
               f"<= {threshold:.3f})")

    return IdentifyResult(
        identity=identity, cosine_distance=best_dist, cosine_similarity=best_sim,
        threshold=threshold, face_found=True, ranked=ranked, message=msg,
    )
