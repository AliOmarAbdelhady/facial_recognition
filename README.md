# Siamese Facial Recognition

End-to-end **Siamese facial recognition** with a polished Gradio UI for
**picture upload** and **live webcam detection** — designed to give **maximum
accuracy** while running on a machine **without a GPU**.

Training happens on Kaggle's free GPU; **inference runs locally on CPU**.

> Built around a **from-scratch ResNet-18-style Siamese network** with
> **Triplet loss** (semi-hard negative mining), trained for **few-shot**
> conditions (only 2–3 photos per person), and pushed to high accuracy with
> MTCNN face alignment + heavy augmentation.

---

## Highlights

- **Architecture**: from-scratch ResNet-18-style CNN + Squeeze-and-Excitation
  attention → 512-d L2-normalized embeddings.
- **Loss**: Triplet loss with **semi-hard negative mining** (refreshed each
  epoch from current model embeddings).
- **Accuracy boosters**:
  1. **MTCNN** face detection + **5-point landmark alignment** to a canonical
     112×112 ArcFace template.
  2. **Heavy augmentation** via torchvision (flip, affine, brightness/contrast,
     blur, Gaussian noise, cutout/random-erasing) — no extra deps, stable.
  3. **Mixed-precision (AMP)**, **cosine LR + warmup**, **EMA weights**,
     gradient clipping.
  4. **ROC-calibrated decision threshold** (`threshold.json`).
  5. **Prototypical enrollment** (mean of multiple augmented embeddings/person)
     + **Test-Time Augmentation** at inference.
- **UI** (Gradio, 4 tabs, pure CPU): **1:1 Verify**, **1:N Identify**,
  **Enroll**, and **Live Webcam**.

---

## Project structure

```
facial_recognition/
├── README.md
├── Photos_dataset/                 # YOUR data: <person_name>/{1,2,3}.jpg
├── kaggle/
│   ├── train_siamese.ipynb         # one-click Kaggle GPU training notebook
│   ├── build_notebook.py           # (dev) regenerates the .ipynb
│   └── kaggle_setup.md             # full step-by-step Kaggle guide
├── app/
│   ├── app.py                      # Gradio UI (4 tabs)
│   ├── model.py                    # EmbeddingNet (matches the notebook)
│   ├── face_detector.py            # MTCNN detect + align
│   ├── transforms.py               # eval + TTA augmentation
│   ├── enrollment.py               # embedding database (.npz)
│   ├── inference.py                # verify (1:1) + identify (1:N)
│   ├── config.py                   # all paths/thresholds
│   └── requirements.txt
└── models/                         # siamese_backbone.pth + threshold.json land here
```

---

## Quick start

### 1. Organize your dataset
```
Photos_dataset/
├── alice/{1.jpg, 2.jpg}
├── bob/{1.jpg, 2.jpg, 3.jpg}
└── ...      (5-20 people, 2-3 clear face photos each)
```

### 2. Train on Kaggle (no local GPU needed)
Follow [`kaggle/kaggle_setup.md`](kaggle/kaggle_setup.md). Summary:
1. Zip `Photos_dataset/`, create a **Kaggle Dataset** from it.
2. Upload `kaggle/train_siamese.ipynb` → enable **GPU T4 x2** + **Internet**.
3. Attach your dataset, **Run All** (~15 min).
4. Download `siamese_backbone.pth` and `threshold.json` into `models/`.

### 3. Run the app locally (CPU)
```bash
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r app/requirements.txt
python app/app.py
```
Open http://127.0.0.1:7860.

**First run:** open the **Enroll** tab → **Batch enroll folder**
(path defaults to `Photos_dataset`) to build the known-people database, then
use **Verify**, **Identify**, and **Live Webcam**.

---

## How recognition works

Every face is detected + aligned to 112×112, then mapped to a **512-d unit
embedding**. Two faces are compared by **cosine distance** `d = 1 − cos_sim`
(smaller = more similar, range [0, 2]).

- **Verify (1:1)**: `d ≤ threshold` ⇒ *same person*.
- **Identify (1:N)**: nearest enrolled person; if its `d ≤ threshold` ⇒
  identified, else ⇒ **Unknown**.
- The threshold is **automatically calibrated** on the validation genuine/
  impostor pairs (Youden's J) and shipped in `threshold.json`. You can also
  nudge it live with the UI sliders — lower = stricter.

---

## Adding a new person later (no retraining)

Two ways, both in the **Enroll** tab:
- **Single image**: type a name, upload a photo → **Enroll person**.
- **Folder**: drop more `Photos_dataset/<name>/` folders and click
  **Batch enroll folder**.

Enrollment averages several augmented embeddings into a stable **prototype**
per person and appends it to `models/embeddings.npz`.

---

## Config

All paths, thresholds, and runtime options live in [`app/config.py`](app/config.py).
Notable knobs:
- `tta_augments` — number of test-time augmentations at inference (0 = fastest).
- `enroll_augments` — augmentations averaged into each person's prototype.
- `default_*_threshold` — fallbacks if `threshold.json` is absent.

---

## Expected accuracy (honest note)

With only 2–3 photos/person, a **from-scratch** Siamese network can't match a
pretrained FaceNet — but the alignment + augmentation + triplet-mining pipeline
above typically yields clean genuine/impostor separation for the trained
identities and reliable unknown-rejection once the threshold is calibrated.
For better/lighting-robust results, use clear, front-facing, well-lit enrollment
photos, and prefer ≥3 photos/person where possible.

---

## Requirements
- Python 3.10+
- See [`app/requirements.txt`](app/requirements.txt) (torch, torchvision,
  facenet-pytorch, scikit-image, gradio, Pillow, numpy)
- Kaggle account with GPU + Internet enabled (for training only)

---

## License
MIT — use freely for study and projects. Face data is your responsibility;
enroll only people who consent.
