"""
dl-arcface/capture.py  —  NEW

Same capture flow as the root capture.py (DNN SSD detector, 80/20 train/val
split into split_dataset/), kept self-contained here so dl-arcface/ stands
on its own like ml-lbph/ and dl-facenet/. Writes to the same shared
split_dataset/ tree, so images captured here are usable by all three
pipelines (and vice versa — you don't need to recapture chosar/phirak).

Usage:
    python dl-arcface/capture.py --name Josar
    python dl-arcface/capture.py --name Borom --camera 1

Run inside the 'frs-arcface' conda env (see dl-arcface/train.py header for why).
"""

import cv2
import os
import sys
import time
import random
import argparse
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

# ── config ───────────────────────────────────────────────────────────────────
DNN_PROTO        = os.path.join(BASE_DIR, "source", "deploy.prototxt")
DNN_MODEL        = os.path.join(BASE_DIR, "source", "res10_300x300_ssd_iter_140000.caffemodel")
DNN_CONF_THRESH  = 0.60
TOTAL_IMAGES     = 100
TRAIN_RATIO      = 0.8
CAPTURE_INTERVAL = 0.5   # seconds between saves
IMG_SIZE         = (160, 160)   # matches existing split_dataset images
# ─────────────────────────────────────────────────────────────────────────────


def load_detector():
    if os.path.exists(DNN_PROTO) and os.path.exists(DNN_MODEL):
        net = cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL)
        print("[INFO] Using DNN SSD detector.")
        return net, "dnn"
    else:
        print("[WARN] DNN model not found in source/, falling back to Haar Cascade.")
        return cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        ), "haar"


def detect(detector, mode, frame):
    if mode == "dnn":
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     1.0, (300, 300), (104, 177, 123))
        detector.setInput(blob)
        out = detector.forward()
        faces = []
        for i in range(out.shape[2]):
            conf = out[0, 0, i, 2]
            if conf > DNN_CONF_THRESH:
                box = out[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                faces.append((max(0, x1), max(0, y1),
                               min(w, x2), min(h, y2)))
        return faces   # list of (x1,y1,x2,y2)
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        return [(x, y, x+w, y+h) for (x, y, w, h) in dets] if len(dets) else []


def setup_dirs(name):
    train_dir = os.path.join(BASE_DIR, "split_dataset", "train", name)
    val_dir   = os.path.join(BASE_DIR, "split_dataset", "val",   name)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir,   exist_ok=True)
    return train_dir, val_dir


def capture_dataset(name, camera_idx):
    detector, mode = load_detector()
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera_idx}.")
        sys.exit(1)

    train_dir, val_dir = setup_dirs(name)
    collected = []
    count     = 0
    last_save = 0

    print(f"\n[INFO] Collecting {TOTAL_IMAGES} images for '{name}' ...")
    print("[INFO] Press 'q' to stop early.\n")

    while count < TOTAL_IMAGES:
        ret, frame = cap.read()
        if not ret:
            break

        now    = time.time()
        disp   = frame.copy()
        faces  = detect(detector, mode, frame)

        if faces:
            # largest face
            x1, y1, x2, y2 = max(faces, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))

            if now - last_save >= CAPTURE_INTERVAL:
                crop    = frame[y1:y2, x1:x2]
                resized = cv2.resize(crop, IMG_SIZE)
                collected.append(resized)
                count   += 1
                last_save = now
                print(f"  Captured {count}/{TOTAL_IMAGES}", end="\r")

            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 220, 80), 2)
        else:
            cv2.putText(disp, "No face detected", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 220), 2)

        # progress bar
        bar = int(count / TOTAL_IMAGES * 200)
        cv2.rectangle(disp, (10, 50), (210, 68), (50, 50, 50), -1)
        cv2.rectangle(disp, (10, 50), (10 + bar, 68), (0, 220, 80), -1)
        cv2.putText(disp, f"{count}/{TOTAL_IMAGES}", (215, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow(f"Capture (ArcFace): {name}", disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # ── split & save ──────────────────────────────────────────────────────
    random.shuffle(collected)
    split      = int(len(collected) * TRAIN_RATIO)
    train_imgs = collected[:split]
    val_imgs   = collected[split:]

    for i, img in enumerate(train_imgs):
        cv2.imwrite(os.path.join(train_dir, f"{name}_{i+1:03d}.jpg"), img)
    for i, img in enumerate(val_imgs):
        cv2.imwrite(os.path.join(val_dir, f"{name}_{i+1:03d}.jpg"), img)

    print(f"\n[DONE] '{name}': {len(train_imgs)} train  +  {len(val_imgs)} val images saved.")
    print("       Now run:  python dl-arcface/train.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Dataset Capture (ArcFace pipeline)")
    parser.add_argument("--name",   required=True,    help="Person's name (used as folder label)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    args = parser.parse_args()
    capture_dataset(args.name, args.camera)
