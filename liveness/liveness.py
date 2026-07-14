"""
liveness/liveness.py  —  NEW

Shared anti-spoofing module used by ml-lbph/webcam.py, dl-resnetv2/webcam.py
(dl-facenet), and dl-arcface/webcam.py. One implementation, three callers —
none of them should duplicate this logic.

Model: MiniFASNet-V2 (from minivision-ai/Silent-Face-Anti-Spoofing), ONNX
export via https://github.com/yakhyo/face-anti-spoofing (Apache-2.0).
Chosen over the other two options discussed with the user:
  - Passive texture/moire heuristics: skipped, meaningfully weaker than a
    trained classifier and redundant once a real model is available.
  - Active blink/head-turn challenge: skipped as the primary mechanism —
    adds friction to one-person-at-a-time attendance scanning and doesn't
    defend against video replay attacks anyway. Temporal consistency
    (below) gives a passive substitute for some of that robustness.

IMPORTANT — measured limitation, read before trusting this blindly:
Tested against this project's own split_dataset/val photos (real faces,
not spoofs): well-lit captures scored a wide 0.02-0.99 on the "real"
class, but darker captures scored 0.001-0.03 almost across the board —
this model is sensitive to lighting conditions unlike its CelebA-Spoof
training data. RAW_LIVE_THRESHOLD is set deliberately low (0.02) and
LivenessGate averages over a rolling window rather than gating on any
single frame, specifically to avoid blocking real faces in imperfect
lighting. It has NOT been validated against a genuine spoof (printed
photo / phone-screen replay) — that requires a live test with a real
camera, which isn't available in the environment this was built in.
Test with your own camera before relying on it, and tighten
RAW_LIVE_THRESHOLD / PASS_RATIO if spoofs get through, or loosen further
if real faces keep failing.

Usage (see any of the three webcam.py files for the full wiring):
    from liveness.liveness import load_model, check_liveness, LivenessGate

    session = load_model()
    gate    = LivenessGate()
    ...
    live_prob        = check_liveness(session, frame, box)
    state, sub_label = gate.update(slot_id, live_prob)
    # state is one of: "live", "checking", "spoof"
"""

import os
import urllib.request
import collections
import cv2
import numpy as np
import onnxruntime

MODEL_URL  = "https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV2.onnx"
CACHE_DIR  = os.path.join(os.path.expanduser("~"), ".face_liveness")
MODEL_PATH = os.path.join(CACHE_DIR, "MiniFASNetV2_yakhyo.onnx")

CROP_SCALE = 2.7      # MiniFASNetV2's trained crop scale (MiniFASNetV1SE uses 4.0 instead)
INPUT_SIZE = 80

# Per-frame floor on the "real" class softmax probability. Deliberately
# lenient (see module docstring) — LivenessGate's rolling window is what
# actually decides whether a face is treated as live, not this alone.
RAW_LIVE_THRESHOLD = 0.02

# ── rolling window (tolerates per-frame noise) ───────────────────────────────
WINDOW              = 5     # recent liveness checks considered
PASS_RATIO           = 0.3   # fraction of WINDOW that must clear RAW_LIVE_THRESHOLD
CONFIRM_SPOOF_AFTER  = 40    # consecutive failing checks before showing "Spoof detected"
                             # (~2-3s at 15-20fps) — avoids alarming a real face
                             # having a rough patch; only fires on sustained failure.


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(CACHE_DIR, exist_ok=True)
        print(f"[INFO] Downloading MiniFASNetV2 anti-spoofing model (~1.7MB, one-time)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def load_model():
    ensure_model()
    return onnxruntime.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])


def _crop_face(frame, box):
    """Official MiniFASNet crop: expand the box by CROP_SCALE around its
    center, clamped to stay inside the frame. Must match this exactly —
    these models are sensitive to crop convention, not just face content."""
    src_h, src_w = frame.shape[:2]
    x1, y1, x2, y2 = box
    box_w, box_h = x2 - x1, y2 - y1
    scale = min((src_h - 1) / box_h, (src_w - 1) / box_w, CROP_SCALE)
    new_w, new_h = box_w * scale, box_h * scale
    cx, cy = x1 + box_w / 2, y1 + box_h / 2

    nx1 = max(0, int(cx - new_w / 2))
    ny1 = max(0, int(cy - new_h / 2))
    nx2 = min(src_w - 1, int(cx + new_w / 2))
    ny2 = min(src_h - 1, int(cy + new_h / 2))

    cropped = frame[ny1:ny2 + 1, nx1:nx2 + 1]
    return cv2.resize(cropped, (INPUT_SIZE, INPUT_SIZE))


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def check_liveness(session, frame, box):
    """Returns the softmax probability of the "real" class (index 1) for
    the face at box=(x1,y1,x2,y2) in frame. BGR, raw 0-255 floats — no
    normalization, matching the official MiniFASNet preprocessing."""
    crop = _crop_face(frame, box)
    blob = np.transpose(crop.astype(np.float32), (2, 0, 1))[None, ...]
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: blob})[0][0]
    return float(_softmax(logits)[1])


class LivenessGate:
    """Per-slot rolling liveness vote. A face only reaches recognition
    once a fraction of its recent liveness checks clear the threshold;
    sustained failure (not a single bad frame) is what escalates to a
    "Spoof detected" label."""

    def __init__(self):
        self.history     = collections.defaultdict(lambda: collections.deque(maxlen=WINDOW))
        self.spoof_streak = collections.defaultdict(int)

    def reset(self):
        self.history.clear()
        self.spoof_streak.clear()

    def update(self, slot_id, live_prob):
        """Returns (state, live_prob) where state is "live", "checking",
        or "spoof"."""
        hist = self.history[slot_id]
        hist.append(live_prob >= RAW_LIVE_THRESHOLD)
        ratio = sum(hist) / len(hist)

        if ratio >= PASS_RATIO:
            self.spoof_streak[slot_id] = 0
            return "live", live_prob

        self.spoof_streak[slot_id] += 1
        if self.spoof_streak[slot_id] >= CONFIRM_SPOOF_AFTER:
            return "spoof", live_prob
        return "checking", live_prob
