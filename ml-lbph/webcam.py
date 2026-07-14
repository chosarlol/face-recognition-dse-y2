"""
ml-lbph/webcam.py  —  UPGRADED

Changes vs original:
  - DNN SSD detector instead of Haar (better accuracy, fewer false positives)
  - "Unknown" detection when LBPH distance exceeds CONF_THRESHOLD
  - Rolling 5-frame smoothing for stable labels
  - FPS counter
  - Multi-face support

Usage:
    python ml-lbph/webcam.py
"""

import sys
import os
import cv2
import pickle
import collections
import time
import numpy as np

BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

MODEL_PATH      = os.path.join(BASE_DIR, "ml-lbph", "lbph_model.xml")
LABELS_PATH     = os.path.join(BASE_DIR, "ml-lbph", "labels.pkl")
CONF_THRESHOLD  = 80.0    # LBPH distance threshold — tune between 50–80
IMG_SIZE        = (100, 100)
SMOOTH_FRAMES   = 5

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
    """Returns list of (x1, y1, x2, y2)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
                boxes.append((x1, y1, x2, y2, gray))
        return boxes
    else:
        faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        return [(x, y, x+w, y+h, gray) for (x, y, w, h) in faces] if len(faces) else []


def draw_box(frame, x1, y1, x2, y2, name, dist):
    is_known = name != "Unknown"
    color    = (0, 220, 80) if is_known else (30, 30, 220)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{name}  d:{dist:.0f}" if is_known else "Unknown"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def main():
    print("Loading LBPH model...")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    print("Model loaded.")

    with open(LABELS_PATH, "rb") as f:
        label_map = pickle.load(f)
    id_to_name = {v: k for k, v in label_map.items()}
    print(f"Labels loaded: {list(label_map.keys())}")

    detector, mode = load_detector()

    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return
    print("Camera started. Press q to quit.")

    smooth    = collections.defaultdict(lambda: collections.deque(maxlen=SMOOTH_FRAMES))
    fps_count, fps, last_fps = 0, 0, time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(detector, mode, frame)

        for i, (x1, y1, x2, y2, gray) in enumerate(faces):
            face_crop = gray[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            face_resized      = cv2.resize(face_crop, IMG_SIZE)
            label_id, dist    = recognizer.predict(face_resized)
            pred_name         = id_to_name.get(label_id, "Unknown")

            if dist > CONF_THRESHOLD:
                pred_name = "Unknown"

            # smooth
            smooth[i].append((pred_name, dist))
            names  = [n for n, _ in smooth[i]]
            s_name = max(set(names), key=names.count)
            s_dist = float(np.mean([d for n, d in smooth[i] if n == s_name]))

            draw_box(frame, x1, y1, x2, y2, s_name, s_dist)

        # FPS
        fps_count += 1
        if time.time() - last_fps >= 1.0:
            fps       = fps_count
            fps_count = 0
            last_fps  = time.time()
        cv2.putText(frame, f"FPS: {fps}  |  LBPH", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Face Recognition — LBPH  (press Q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()