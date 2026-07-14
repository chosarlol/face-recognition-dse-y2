"""
dl-resnetv2/webcam.py — UPGRADED (DNN SSD detector, no MediaPipe)

Reliability layer (mirrors dl-arcface/webcam.py, the reference pattern
also applied to ml-lbph/webcam.py):
  1. Frame-quality gate — runs before recognition, on the raw crop:
     rejects faces that are too small, cut off / off-center, too dark/
     bright, or too blurry, and tells the user why instead of guessing.
  2. Liveness gate (liveness/liveness.py, shared across all three
     modules) — a MiniFASNet-V2 anti-spoofing classifier must judge the
     face live, averaged over a rolling window, before recognition runs
     at all. Sustained failure shows "Spoof detected"; Unknown is only
     ever shown for a face that passed liveness but wasn't recognized.
  3. Confidence gate — a name is only accepted if BOTH the SVM
     probability AND the cosine similarity to that class's stored
     centroid clear their thresholds. The SVM alone always forces a
     pick between known classes even for a total stranger; the centroid
     check catches that case.
  4. Temporal stability — a name is only displayed once it wins a
     clear majority over the last N recognition attempts (frames that
     fail the quality gate don't count against or reset this window).
     Otherwise shows "Verifying..." rather than flickering or guessing.
"""

import sys
import os
import cv2
import pickle
import collections
import time
import numpy as np
from PIL import Image

import torch
from facenet_pytorch import InceptionResnetV1
from sklearn.preprocessing import normalize

BASE_DIR         = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from liveness.liveness import load_model as load_liveness_model, check_liveness, LivenessGate

CLASSIFIER_PATH  = os.path.join(BASE_DIR, "dl-facenet", "facenet_classifier.pkl")
IMAGE_SIZE       = (160, 160)
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

# ── confidence gate ──────────────────────────────────────────────────────────
# Empirically checked against dl-facenet/embeddings_train.pkl (155 SSD-crop
# embeddings, no alignment — see train.py's header for why MTCNN was
# dropped). Same-person centroid similarity ranged 0.23-0.97, cross-person
# up to 0.74 — a much noisier, overlapping distribution than dl-arcface's
# (ArcFace tolerates unaligned crops far better than FaceNet does; FaceNet
# was designed around MTCNN-aligned input). No single threshold cleanly
# separates same- vs cross-person here, so SIM_THRESHOLD is set low and
# mostly acts as a sanity floor against degenerate embeddings — the SVM
# probability is doing most of the confidence-gating work for this module.
PROB_THRESHOLD  = 0.70
SIM_THRESHOLD   = 0.20

# ── frame-quality gate ───────────────────────────────────────────────────────
MIN_FACE_AREA_RATIO = 0.05   # face bbox area / frame area — tune per camera distance
EDGE_MARGIN         = 4      # px; bbox touching the frame border = cut off
# Calibrated against this project's own captures (split_dataset/val): even
# the sharpest stored 160x160 crops only score 6-17 on Laplacian variance at
# 200x200, so a generic "blurry photo" threshold (60-100) would reject good
# frames outright. This floor only catches genuine motion blur.
BLUR_THRESHOLD      = 4.0
MIN_BRIGHTNESS      = 40
MAX_BRIGHTNESS      = 215

# ── temporal stability ───────────────────────────────────────────────────────
HISTORY_LEN          = 10
MIN_FRAMES_DECISION  = 6
AGREE_RATIO          = 0.7

DNN_PROTO  = os.path.join(BASE_DIR, "source", "deploy.prototxt")
DNN_MODEL  = os.path.join(BASE_DIR, "source", "res10_300x300_ssd_iter_140000.caffemodel")
DNN_CONF   = 0.55


def load_detector():
    if os.path.exists(DNN_PROTO) and os.path.exists(DNN_MODEL):
        print("[INFO] Using DNN SSD detector.")
        return cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL), "dnn"
    print("[WARN] DNN not found, using Haar Cascade.")
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'), "haar"


def detect_faces(detector, mode, frame):
    if mode == "dnn":
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     1.0, (300, 300), (104, 177, 123))
        detector.setInput(blob)
        out   = detector.forward()
        boxes = []
        for i in range(out.shape[2]):
            if out[0, 0, i, 2] > DNN_CONF:
                box = out[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                boxes.append((x1, y1, x2, y2))
        return boxes
    else:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        return [(x, y, x+w, y+h) for (x, y, w, h) in faces] if len(faces) else []


def check_box_quality(frame_shape, box):
    """Geometry checks on the detection box itself. Returns a reason
    string if the face should be rejected before recognition, else None."""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box
    if x1 <= EDGE_MARGIN or y1 <= EDGE_MARGIN or x2 >= w - EDGE_MARGIN or y2 >= h - EDGE_MARGIN:
        return "Center your face"
    area_ratio = ((x2 - x1) * (y2 - y1)) / (w * h)
    if area_ratio < MIN_FACE_AREA_RATIO:
        return "Move closer"
    return None


def check_crop_quality(face_bgr):
    """Pixel-level checks on the cropped face. Returns a reason string
    if the crop should be rejected before recognition, else None."""
    gray   = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.resize(gray, (200, 200))
    brightness = gray_r.mean()
    if brightness < MIN_BRIGHTNESS:
        return "Too dark"
    if brightness > MAX_BRIGHTNESS:
        return "Too bright"
    if cv2.Laplacian(gray_r, cv2.CV_64F).var() < BLUR_THRESHOLD:
        return "Hold still"
    return None


def get_embedding(facenet, face_bgr):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb).resize(IMAGE_SIZE)
    t = torch.tensor(np.array(face_pil)).permute(2, 0, 1).float()
    t = (t - 127.5) / 128.0
    t = t.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = facenet(t).cpu().numpy()[0]
    return emb


def predict(svm, le, centroids, embedding):
    """SVM probability + cosine similarity to the predicted class's
    centroid both have to clear their thresholds, else "Unknown"."""
    emb_n = normalize(embedding.reshape(1, -1))[0]
    probs = svm.predict_proba(emb_n.reshape(1, -1))[0]
    idx   = np.argmax(probs)
    name  = le.inverse_transform([idx])[0]
    prob  = float(probs[idx])
    sim   = float(np.dot(emb_n, centroids[name]))

    if prob < PROB_THRESHOLD or sim < SIM_THRESHOLD:
        return "Unknown", prob
    return name, prob


class Stability:
    """Per-slot rolling vote. A name is only reported once it holds a
    clear majority over the last HISTORY_LEN recognition attempts."""

    def __init__(self):
        self.slots = collections.defaultdict(lambda: collections.deque(maxlen=HISTORY_LEN))

    def reset(self):
        self.slots.clear()

    def update(self, slot_id, name, conf):
        hist = self.slots[slot_id]
        hist.append((name, conf))
        if len(hist) < MIN_FRAMES_DECISION:
            return "Verifying...", 0.0, False

        names     = [n for n, _ in hist]
        mode_name = max(set(names), key=names.count)
        ratio     = names.count(mode_name) / len(names)
        if ratio < AGREE_RATIO:
            return "Verifying...", 0.0, False

        mode_conf = float(np.mean([c for n, c in hist if n == mode_name]))
        return mode_name, mode_conf, True


def draw_box(frame, x1, y1, x2, y2, name, conf):
    if name == "Verifying...":
        color, label = (0, 165, 255), "Verifying..."
    elif name == "Unknown":
        color, label = (30, 30, 220), "Unknown"
    else:
        color, label = (0, 220, 80), f"{name}  {conf*100:.0f}%"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def draw_quality_reason(frame, x1, y1, x2, y2, reason):
    color = (0, 165, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(reason, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, reason, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def draw_liveness_state(frame, x1, y1, x2, y2, state):
    if state == "spoof":
        color, label = (255, 0, 255), "Spoof detected"
    else:
        color, label = (0, 165, 255), "Checking liveness..."
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def main():
    print("Loading FaceNet model...")
    facenet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
    print("Model loaded.")

    print("Loading liveness model...")
    liveness_session = load_liveness_model()
    print("Liveness model loaded.")

    print("Loading classifier...")
    with open(CLASSIFIER_PATH, "rb") as f:
        data = pickle.load(f)
    svm       = data["svm"]
    le        = data["label_encoder"]
    centroids = data["centroids"]
    print("Classifier loaded.")

    detector, mode = load_detector()

    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return
    print("Camera started. Press q to quit.")

    stability = Stability()
    liveness_gate = LivenessGate()
    fps_count, fps, last_fps = 0, 0, time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes = detect_faces(detector, mode, frame)
        if not boxes:
            stability.reset()
            liveness_gate.reset()

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            reason = check_box_quality(frame.shape, (x1, y1, x2, y2)) \
                or check_crop_quality(face_crop)
            if reason:
                draw_quality_reason(frame, x1, y1, x2, y2, reason)
                continue

            live_prob = check_liveness(liveness_session, frame, (x1, y1, x2, y2))
            state, _  = liveness_gate.update(i, live_prob)
            if state != "live":
                draw_liveness_state(frame, x1, y1, x2, y2, state)
                continue

            emb        = get_embedding(facenet, face_crop)
            name, conf = predict(svm, le, centroids, emb)

            s_name, s_conf, _ = stability.update(i, name, conf)
            draw_box(frame, x1, y1, x2, y2, s_name, s_conf)

        fps_count += 1
        if time.time() - last_fps >= 1.0:
            fps       = fps_count
            fps_count = 0
            last_fps  = time.time()
        cv2.putText(frame, f"FPS: {fps}  |  DNN + FaceNet + SVM", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow("Face Recognition — FaceNet  (press Q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()