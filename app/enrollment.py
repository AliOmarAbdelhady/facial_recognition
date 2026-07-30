"""
Embedding database (enrollment).

Each enrolled person is represented by a *prototype* = the L2-normalized mean of
several augmented embeddings of their available face crops. Prototypical
averaging gives much more stable 1:N recognition than a single embedding when
only 2-3 photos exist per person.

The database is persisted as a single .npz file:
    {
        "names": np.ndarray[str],          # (N,)
        "embeddings": np.ndarray[float32], # (N, embedding_dim)
    }
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from config import CONFIG
from face_detector import FaceDetector, load_image
from model import EmbeddingNet
from transforms import augment_face, preprocess_face


# ---------------------------------------------------------------------------
# Database IO
# ---------------------------------------------------------------------------
def load_db(path: Optional[Path] = None) -> Dict[str, np.ndarray]:
    """Load {name: prototype_embedding} from the .npz DB. Empty if missing."""
    path = path or CONFIG.embedding_db_path
    if not path.exists():
        return {}
    data = np.load(path)
    names = data["names"]
    emb = data["embeddings"]
    return {str(n): e for n, e in zip(names, emb)}


def save_db(db: Dict[str, np.ndarray], path: Optional[Path] = None) -> None:
    path = path or CONFIG.embedding_db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if not db:
        # Persist an empty-but-valid file.
        np.savez(path, names=np.array([], dtype=object),
                 embeddings=np.zeros((0, CONFIG.embedding_dim), dtype=np.float32))
        return
    names = np.array(list(db.keys()), dtype=object)
    emb = np.stack([db[n] for n in db]).astype(np.float32)
    np.savez(path, names=names, embeddings=emb)


# ---------------------------------------------------------------------------
# Embedding computation (with TTA + prototype averaging)
# ---------------------------------------------------------------------------
@torch.no_grad()
def embed_face(face: Image.Image, net: EmbeddingNet, tta: int = 0) -> torch.Tensor:
    """
    Compute a single L2-normalized embedding for a face crop.
    If `tta>0`, average `tta` augmented embeddings (+ the clean pass) for stability.
    Returns shape (embedding_dim,) on CPU.
    """
    tensors = [preprocess_face(face)]
    if tta > 0:
        tensors.extend(augment_face(face, tta))

    batch = torch.cat(tensors, dim=0).to(CONFIG.device)
    emb = net(batch)                          # (1+tta, D), already L2-normalized
    proto = emb.mean(dim=0)                   # centroid
    proto = torch.nn.functional.normalize(proto, p=2, dim=0)
    return proto.detach().cpu()


@torch.no_grad()
def embed_person(
    crops: List[Image.Image],
    net: EmbeddingNet,
    augments_per_crop: int = CONFIG.enroll_augments,
) -> torch.Tensor:
    """
    Build one prototype embedding for a person from multiple face crops.

    For each crop we produce (1 clean + augments_per_crop augmented) embeddings,
    then average ALL of them and L2-normalize -> a stable prototype.
    """
    all_embs = []
    for crop in crops:
        tensors = [preprocess_face(crop)]
        if augments_per_crop > 0:
            tensors.extend(augment_face(crop, augments_per_crop))
        batch = torch.cat(tensors, dim=0).to(CONFIG.device)
        all_embs.append(net(batch))

    stacked = torch.cat(all_embs, dim=0)      # (num_crops*(1+aug), D)
    proto = stacked.mean(dim=0)
    proto = torch.nn.functional.normalize(proto, p=2, dim=0)
    return proto.detach().cpu()


# ---------------------------------------------------------------------------
# High-level enrollment API
# ---------------------------------------------------------------------------
def enroll_image(
    name: str,
    image,
    net: EmbeddingNet,
    detector: FaceDetector,
    db: Optional[Dict[str, np.ndarray]] = None,
    augments_per_crop: int = CONFIG.enroll_augments,
) -> Dict:
    """
    Enroll one person from a single uploaded image (largest face).

    Returns a status dict with: ok, name, faces_found, preview_face (PIL or None),
    message.
    """
    db = db if db is not None else load_db()
    pil = load_image(image)
    crops = detector.get_largest_face(pil, return_all=True) or []
    if not crops:
        return {"ok": False, "name": name, "faces_found": 0, "preview_face": None,
                "message": "No face detected in the uploaded image."}

    proto = embed_person(crops, net, augments_per_crop=augments_per_crop)
    db[name] = proto.numpy()
    save_db(db)
    return {
        "ok": True,
        "name": name,
        "faces_found": len(crops),
        "preview_face": crops[0],
        "message": f"Enrolled '{name}' ({len(crops)} face crop, "
                   f"{1 + augments_per_crop} augmented embeddings averaged).",
    }


def enroll_directory(
    root_dir,
    net: EmbeddingNet,
    detector: FaceDetector,
    extensions=(".jpg", ".jpeg", ".png", ".bmp", ".webp"),
) -> Dict:
    """
    Batch-enroll a dataset directory structured as:
        root_dir/<person_name>/*.jpg
    Each person gets a single prototype from ALL their images.
    """
    root = Path(root_dir)
    if not root.exists():
        return {"ok": False, "enrolled": {}, "message": f"{root} does not exist."}

    db = load_db()
    report = {}
    for person_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        name = person_dir.name
        crops = []
        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() not in extensions:
                continue
            try:
                pil = load_image(img_path)
            except Exception:
                continue
            faces = detector.get_largest_face(pil, return_all=True) or []
            crops.extend(faces)
        if not crops:
            report[name] = {"ok": False, "message": "no faces found"}
            continue
        proto = embed_person(crops, net)
        db[name] = proto.numpy()
        report[name] = {"ok": True, "crops": len(crops),
                        "message": f"enrolled from {len(crops)} crop(s)"}

    save_db(db)
    return {"ok": True, "enrolled": report,
            "message": f"Enrolled {sum(1 for r in report.values() if r['ok'])} people."}


def remove_person(name: str) -> Dict:
    db = load_db()
    if name not in db:
        return {"ok": False, "message": f"'{name}' is not in the database."}
    del db[name]
    save_db(db)
    return {"ok": True, "message": f"Removed '{name}'."}


def list_people() -> List[str]:
    return sorted(load_db().keys())
