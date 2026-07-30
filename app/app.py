"""
Siamese Facial Recognition - Gradio UI (CPU inference).

Four tabs:
  1. Verify (1:1)   - are these two images the same person?
  2. Identify (1:N) - who is this? (or Unknown)
  3. Enroll         - add a new person to the database
  4. Live Webcam    - real-time identification from the camera

Run from the project root:
    python app/app.py
Then open the printed local URL (default http://127.0.0.1:7860).
"""

from __future__ import annotations

import os
import sys
import threading

# Ensure sibling modules import cleanly whether run as `python app/app.py`
# or `python -m app.app`.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import numpy as np
import gradio as gr

from config import CONFIG, ensure_dirs
from face_detector import FaceDetector, load_image
from inference import identify, load_threshold, verify
from model import EmbeddingNet, load_embedding_net
import enrollment as enr


# ---------------------------------------------------------------------------
# Singleton resources (loaded once at startup)
# ---------------------------------------------------------------------------
class AppState:
    net: EmbeddingNet | None = None
    detector: FaceDetector | None = None
    model_loaded: bool = False
    model_message: str = ""


STATE = AppState()
_LOCK = threading.Lock()  # Gradio can call handlers concurrently


def _ensure_model() -> tuple[EmbeddingNet, FaceDetector]:
    """Lazy-load the embedding network + MTCNN detector (cached in STATE)."""
    with _LOCK:
        if STATE.detector is None:
            STATE.detector = FaceDetector(
                image_size=CONFIG.img_size,
                min_face_size=CONFIG.min_face_size,
                thresholds=CONFIG.face_detect_thresholds,
                device=CONFIG.device,
            )
        if STATE.net is None:
            if CONFIG.checkpoint_path.exists():
                try:
                    STATE.net = load_embedding_net(
                        str(CONFIG.checkpoint_path),
                        map_location=CONFIG.device,
                        embedding_dim=CONFIG.embedding_dim,
                    )
                    STATE.model_loaded = True
                    STATE.model_message = (
                        f"Loaded trained model from {CONFIG.checkpoint_path}"
                    )
                except Exception as exc:  # pragma: no cover
                    STATE.model_loaded = False
                    STATE.model_message = (
                        f"Failed to load checkpoint: {exc}\n"
                        "Place the trained 'siamese_backbone.pth' (from Kaggle) "
                        f"in {CONFIG.checkpoint_path.parent}/ and restart."
                    )
            else:
                STATE.model_loaded = False
                STATE.model_message = (
                    f"No checkpoint found at {CONFIG.checkpoint_path}.\n"
                    "1) Train on Kaggle (see kaggle/train_siamese.ipynb), "
                    "2) download 'siamese_backbone.pth', "
                    f"3) drop it into {CONFIG.checkpoint_path.parent}/, "
                    "4) restart this app."
                )
        return STATE.net, STATE.detector


def model_status_md() -> str:
    return f"**Model:** {STATE.model_message or '(not initialized)'}"


def db_status_md() -> str:
    people = enr.list_people()
    return f"**Database:** {len(people)} people enrolled " \
           f"({', '.join(people) if people else 'empty'})."


# ---------------------------------------------------------------------------
# Tab 1 - Verify (1:1)
# ---------------------------------------------------------------------------
def verify_handler(img1, img2, threshold, tta):
    net, det = _ensure_model()
    if net is None:
        return None, None, 0.0, 0.0, STATE.model_message, "## \u26a0\ufe0f Model not loaded"
    if img1 is None or img2 is None:
        return None, None, 0.0, 0.0, "Upload both images first.", "## \u274c Missing input"

    res = verify(img1, img2, net, det, threshold=threshold, tta=int(tta))
    face1 = det.get_largest_face(load_image(img1)) if res.face1_found else None
    face2 = det.get_largest_face(load_image(img2)) if res.face2_found else None

    emoji = "\u2705" if res.same_person else "\u274c"
    title = (f"## {emoji} SAME PERSON" if res.same_person
             else f"## {emoji} DIFFERENT PEOPLE")
    detail = (
        f"{res.message}\n\n"
        f"- Cosine similarity: **{res.cosine_similarity:.3f}**\n"
        f"- Cosine distance: **{res.cosine_distance:.3f}**\n"
        f"- Threshold: **{res.threshold:.3f}**"
    )
    return (np.array(face1) if face1 else None,
            np.array(face2) if face2 else None,
            res.cosine_similarity, res.cosine_distance, detail, title)


# ---------------------------------------------------------------------------
# Tab 2 - Identify (1:N)
# ---------------------------------------------------------------------------
def identify_handler(img, threshold, tta):
    net, det = _ensure_model()
    if net is None:
        return None, STATE.model_message, "", "## \u26a0\ufe0f Model not loaded"
    if img is None:
        return None, "Upload an image first.", "", "## \u274c Missing input"

    res = identify(img, net, det, threshold=threshold, tta=int(tta))
    face = det.get_largest_face(load_image(img)) if res.face_found else None

    if res.identity:
        title = f"## \U0001f464 IDENTIFIED: **{res.identity}**"
    elif not res.face_found:
        title = "## \u26d4 No face detected"
    elif not res.ranked:
        title = "## \U0001f4cb Database empty"
    else:
        title = "## \u2753 UNKNOWN"

    rows = "\n".join(
        f"| {i+1} | {name} | {dist:.3f} | {1.0 - dist:.3f} |"
        for i, (name, dist) in enumerate(res.ranked[:10])
    )
    table = ("| Rank | Name | Cos Dist | Cos Sim |\n|---|---|---|---|\n" + rows
             if rows else "")
    detail = (
        f"{res.message}\n\n"
        f"- Best cosine similarity: **{res.cosine_similarity:.3f}**\n"
        f"- Best cosine distance: **{res.cosine_distance:.3f}**\n"
        f"- Threshold: **{res.threshold:.3f}**"
    )
    detail = detail + "\n\n### Top matches\n" + table if table else detail
    return np.array(face) if face else None, detail, table, title


# ---------------------------------------------------------------------------
# Tab 3 - Enroll
# ---------------------------------------------------------------------------
def enroll_handler(name, img, augments):
    net, det = _ensure_model()
    if net is None:
        return STATE.model_message, db_status_md()
    name = (name or "").strip()
    if not name:
        return "Enter the person's name.", db_status_md()
    if img is None:
        return "Upload an image containing the face.", db_status_md()

    result = enr.enroll_image(name, img, net, det, augments_per_crop=int(augments))
    face_preview = result.get("preview_face")
    msg = result["message"]
    full = msg + f"\n\nNow enrolled: {', '.join(enr.list_people())}"
    return full, db_status_md()


def enroll_dataset_handler(dataset_path, augments):
    net, det = _ensure_model()
    if net is None:
        return STATE.model_message, db_status_md()
    path = dataset_path or str(CONFIG.dataset_dir)
    report = enr.enroll_directory(path, net, det)
    lines = [report["message"]]
    for name, r in report["enrolled"].items():
        status = "\u2705" if r["ok"] else "\u274c"
        lines.append(f"{status} {name}: {r['message']}")
    return "\n".join(lines), db_status_md()


def remove_handler(name):
    name = (name or "").strip()
    if not name:
        return "Enter a name to remove.", db_status_md()
    res = enr.remove_person(name)
    return res["message"], db_status_md()


# ---------------------------------------------------------------------------
# Tab 4 - Live webcam
# ---------------------------------------------------------------------------
def webcam_identify(frame, threshold, tta):
    """Called for each streamed webcam frame."""
    net, det = _ensure_model()
    if net is None or frame is None:
        return "", "## \u26a0\ufe0f Model not loaded / no frame"
    res = identify(frame, net, det, threshold=threshold, tta=int(tta))
    if res.identity:
        return f"{res.cosine_similarity:.2f}", f"## \U0001f464 {res.identity}"
    if not res.ranked:
        return "", "## \U0001f4cb DB empty"
    return f"{res.cosine_similarity:.2f}", f"## \u2753 UNKNOWN ({res.ranked[0][0]})"


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    ensure_dirs()
    default_threshold = load_threshold()

    with gr.Blocks(title="Siamese Facial Recognition") as demo:
        gr.Markdown("# \U0001f464 Siamese Facial Recognition\n"
                    "From-scratch ResNet-18 Siamese network (triplet loss) with "
                    "MTCNN face alignment. Runs on CPU.")
        with gr.Accordion("System status", open=False):
            gr.Markdown(model_status_md)
            db_box = gr.Markdown(db_status_md)

        # ------------------- Tab 1: Verify -------------------
        with gr.Tab("1:1 Verify"):
            with gr.Row():
                with gr.Column():
                    v1 = gr.Image(label="Image A", sources=["upload"], type="numpy")
                    v2 = gr.Image(label="Image B", sources=["upload"], type="numpy")
                    v_thr = gr.Slider(0.0, 2.0, value=CONFIG.default_verify_threshold,
                                      step=0.01, label="Cosine distance threshold "
                                      "(lower = stricter)")
                    v_tta = gr.Slider(0, 10, value=CONFIG.tta_augments, step=1,
                                      label="Test-time augmentations (TTA)")
                    v_btn = gr.Button("Compare", variant="primary")
                with gr.Column():
                    v_title = gr.Markdown("## Awaiting input")
                    v_f1 = gr.Image(label="Aligned face A")
                    v_f2 = gr.Image(label="Aligned face B")
                    v_sim = gr.Slider(-1, 1, value=0, step=0.001,
                                      label="Cosine similarity", interactive=False)
                    v_dist = gr.Slider(0, 2, value=0, step=0.001,
                                       label="Cosine distance", interactive=False)
                    v_detail = gr.Markdown("")
            v_btn.click(verify_handler,
                        inputs=[v1, v2, v_thr, v_tta],
                        outputs=[v_f1, v_f2, v_sim, v_dist, v_detail, v_title])

        # ------------------- Tab 2: Identify -------------------
        with gr.Tab("1:N Identify"):
            with gr.Row():
                with gr.Column():
                    i_img = gr.Image(label="Query image", sources=["upload"],
                                     type="numpy")
                    i_thr = gr.Slider(0.0, 2.0, value=default_threshold, step=0.01,
                                      label="Cosine distance threshold "
                                      "(calibrated from Kaggle if available)")
                    i_tta = gr.Slider(0, 10, value=CONFIG.tta_augments, step=1,
                                      label="Test-time augmentations (TTA)")
                    i_btn = gr.Button("Identify", variant="primary")
                with gr.Column():
                    i_title = gr.Markdown("## Awaiting input")
                    i_face = gr.Image(label="Aligned query face")
                    i_detail = gr.Markdown("")
                    i_table = gr.Markdown("")
            i_btn.click(identify_handler,
                        inputs=[i_img, i_thr, i_tta],
                        outputs=[i_face, i_detail, i_table, i_title])

        # ------------------- Tab 3: Enroll -------------------
        with gr.Tab("Enroll"):
            with gr.Row():
                with gr.Column():
                    e_name = gr.Textbox(label="Person name")
                    e_img = gr.Image(label="Face photo", sources=["upload"],
                                     type="numpy")
                    e_aug = gr.Slider(0, 30, value=CONFIG.enroll_augments, step=1,
                                      label="Augmented embeddings per crop")
                    e_btn = gr.Button("Enroll person", variant="primary")
                    gr.Markdown("---\n**Batch enroll** a whole dataset folder "
                                "(structure: `<folder>/<name>/*.jpg`):")
                    e_path = gr.Textbox(value=str(CONFIG.dataset_dir),
                                        label="Dataset root")
                    e_batch_btn = gr.Button("Batch enroll folder")
                    gr.Markdown("**Remove** a person:")
                    e_rm_name = gr.Textbox(label="Name to remove")
                    e_rm_btn = gr.Button("Remove person")
                with gr.Column():
                    e_status = gr.Textbox(label="Result", lines=6)
                    e_db = gr.Markdown(db_status_md())
            e_btn.click(enroll_handler, inputs=[e_name, e_img, e_aug],
                        outputs=[e_status, e_db])
            e_batch_btn.click(enroll_dataset_handler, inputs=[e_path, e_aug],
                              outputs=[e_status, e_db])
            e_rm_btn.click(remove_handler, inputs=[e_rm_name],
                           outputs=[e_status, e_db])

        # ------------------- Tab 4: Live webcam -------------------
        with gr.Tab("Live Webcam"):
            gr.Markdown("Grant camera access. Each frame is detected, aligned, "
                        "embedded, and matched against the database in real time.")
            with gr.Row():
                with gr.Column():
                    cam = gr.Image(label="Webcam", sources=["webcam"],
                                   streaming=True, type="numpy")
                    w_thr = gr.Slider(0.0, 2.0, value=default_threshold, step=0.01,
                                      label="Cosine distance threshold")
                    w_tta = gr.Slider(0, 10, value=0, step=1,
                                      label="TTA (0 recommended for live speed)")
                with gr.Column():
                    w_title = gr.Markdown("## Waiting for camera\u2026")
                    w_sim = gr.Slider(-1, 1, value=0, step=0.001,
                                      label="Best cosine similarity",
                                      interactive=False)
            try:
                cam.stream(webcam_identify, inputs=[cam, w_thr, w_tta],
                           outputs=[w_sim, w_title])
            except Exception:
                # Older Gradio without Component.stream: fall back to a button.
                gr.Button("Identify current frame").click(
                    webcam_identify, inputs=[cam, w_thr, w_tta],
                    outputs=[w_sim, w_title])

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    ensure_dirs()
    _ensure_model()  # warm up + surface load errors early
    demo = build_ui()
    demo.queue(default_concurrency_limit=2).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        share=bool(int(os.environ.get("GRADIO_SHARE", "0"))),
        inbrowser=False,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
