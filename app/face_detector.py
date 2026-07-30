"""
Face detection + 5-point landmark alignment using MTCNN (facenet-pytorch).

This is the single most important accuracy lever: we never feed raw images to
the embedding network. We detect the face, then warp the 5 facial landmarks
(eyes, nose, mouth corners) into a canonical 112x112 template using an affine
transform. This removes most pose/scale variation and dramatically tightens
the embedding clusters for a few-shot Siamese network.

Runs entirely on CPU (MTCNN is fast enough for live webcam use).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from PIL import Image

# facenet-pytorch's MTCNN bundles detection + alignment in one call.
try:
    from facenet_pytorch import MTCNN  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "facenet-pytorch is required. Install with: pip install facenet-pytorch"
    ) from exc


# Canonical 112x112 arcface-style template (where each landmark should land).
# These coordinates match the standard ArcFace alignment used widely.
ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],   # left eye
        [73.5318, 51.5014],   # right eye
        [56.0252, 71.7366],   # nose
        [41.5493, 92.3655],   # left mouth corner
        [70.7299, 92.2041],   # right mouth corner
    ],
    dtype=np.float32,
)


class FaceDetector:
    """Wraps MTCNN for detection + aligned 112x112 face cropping."""

    def __init__(
        self,
        image_size: int = 112,
        min_face_size: int = 40,
        thresholds: tuple = (0.6, 0.7, 0.7),
        device: str = "cpu",
        keep_all: bool = False,
        post_process: bool = False,  # we do our own normalization later
    ):
        self.image_size = image_size
        self.mtcnn = MTCNN(
            image_size=image_size,
            margin=0,
            min_face_size=min_face_size,
            thresholds=list(thresholds),
            factor=0.709,
            post_process=post_process,
            keep_all=keep_all,
            device=device,
        )

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def detect(self, image: Image.Image):
        """
        Detect faces + landmarks.

        Returns (boxes, probs, landmarks):
          boxes      : (N, 4) float array  (x1,y1,x2,y2) or None
          probs      : (N,)   float array of confidences or None
          landmarks  : (N, 5, 2) float array of (x,y) per face or None
        """
        boxes, probs, landmarks = self.mtcnn.detect(image, landmarks=True)
        return boxes, probs, landmarks

    def align_crop(self, image: Image.Image, landmarks: np.ndarray) -> Image.Image:
        """
        Affine-warp `landmarks` (5x2) onto ARCFACE_TEMPLATE -> 112x112 PIL.

        Uses cv2.warpAffine (FaceNet/InsightFace standard). skimage.warp is NOT
        used: current skimage misreads a 2x3 matrix as a coordinate array
        ('invalid shape for coordinate array'). Falls back to a plain MTCNN crop
        only if OpenCV is unavailable or the warp fails.
        """
        import cv2  # OpenCV is a dependency of facenet-pytorch (always present)

        lm = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
        img_arr = np.array(image.convert("RGB"))
        try:
            M, _ = cv2.estimateAffinePartial2D(lm, ARCFACE_TEMPLATE)
            if M is None:
                return self._fallback_crop(image, landmarks)
            warped = cv2.warpAffine(img_arr, M, (112, 112),
                                    flags=cv2.INTER_LINEAR, borderValue=0)
            return Image.fromarray(warped)
        except Exception:
            return self._fallback_crop(image, landmarks)

    def _fallback_crop(self, image: Image.Image, landmarks: np.ndarray) -> Image.Image:
        """Crude fallback: let MTCNN do its default (non-landmark) crop."""
        face = self.mtcnn(image)
        if face is None:
            return image.convert("RGB").resize((112, 112))
        arr = (face.permute(1, 2, 0).numpy() * 255).astype(np.uint8) \
            if torch.is_tensor(face) else np.array(face)
        return Image.fromarray(arr)

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------
    def get_largest_face(
        self, image: Image.Image, return_all: bool = False
    ) -> Optional[list] | Optional[Image.Image]:
        """
        Detect faces and return aligned crops.

        - return_all=False: returns the largest single aligned PIL face, or None.
        - return_all=True : returns list[Image.Image] of all aligned faces (maybe empty).
        """
        boxes, probs, landmarks = self.detect(image)
        if boxes is None or len(boxes) == 0:
            return [] if return_all else None

        # Sort by area descending so index 0 = largest face.
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        order = np.argsort(-areas)

        crops = []
        for idx in order:
            lm = landmarks[idx]
            crops.append(self.align_crop(image, lm))

        if return_all:
            return crops
        return crops[0]


def load_image(path_or_array) -> Image.Image:
    """Accept a filesystem path, URL bytes, numpy array, or PIL image."""
    if isinstance(path_or_array, Image.Image):
        return path_or_array.convert("RGB")
    if isinstance(path_or_array, np.ndarray):
        return Image.fromarray(path_or_array.astype(np.uint8)).convert("RGB")
    return Image.open(path_or_array).convert("RGB")
