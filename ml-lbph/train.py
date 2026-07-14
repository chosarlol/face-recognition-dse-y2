"""
ml-lbph/train.py  —  UPGRADED

Changes vs original:
  - Uses DNN SSD detector instead of Haar Cascade (better face alignment)
  - Reads directly from split_dataset/train/ (matches new capture.py output)
  - Saves label map as labels.pkl (replaces class_names.txt for consistency)
  - Prints per-class image count during training

Usage:
    python ml-lbph/train.py
"""

import sys
import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm

BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

DATASET_DIR = os.path.join(BASE_DIR, "split_dataset", "train")
OUT_DIR     = os.path.join(BASE_DIR, "ml-lbph")
MODEL_OUT   = os.path.join(OUT_DIR, "lbph_model.xml")
LABELS_OUT  = os.path.join(OUT_DIR, "labels.pkl")

DNN_PROTO   = os.path.join(BASE_DIR, "source", "deploy.prototxt")
DNN_MODEL   = os.path.join(BASE_DIR, "source", "res10_300x300_ssd_iter_140000.caffemodel")
DNN_CONF    = 0.55
IMG_SIZE    = (100, 100)   # LBPH works best with consistent size


def load_detector():
    if os.path.exists(DNN_PROTO) and os.path.exists(DNN_MODEL):
        print("[INFO] Using DNN SSD detector.")
        return cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL), "dnn"
    print("[WARN] DNN not found, using Haar Cascade fallback.")
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'), "haar"


def detect_face(detector, mode, gray_img, color_img):
    """Returns the largest face crop (grayscale) or None."""
    if mode == "dnn":
        h, w = color_img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(color_img, (300, 300)),
                                     1.0, (300, 300), (104, 177, 123))
        detector.setInput(blob)
        out   = detector.forward()
        best  = None
        best_area = 0
        for i in range(out.shape[2]):
            conf = out[0, 0, i, 2]
            if conf > DNN_CONF:
                box = out[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area = area
                    best = (x1, y1, x2, y2)
        if best:
            x1, y1, x2, y2 = best
            return gray_img[y1:y2, x1:x2]
    else:
        faces = detector.detectMultiScale(gray_img, 1.1, 5, minSize=(50, 50))
        if len(faces):
            x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
            return gray_img[y:y+h, x:x+w]
    return None


def build_dataset(detector, mode):
    classes = sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ])
    if not classes:
        raise ValueError(f"No class folders in '{DATASET_DIR}'. Run capture.py first.")

    print(f"[INFO] Found {len(classes)} people: {classes}\n")

    label_map = {name: idx for idx, name in enumerate(classes)}
    faces, labels = [], []

    for name in classes:
        class_dir = os.path.join(DATASET_DIR, name)
        images    = [f for f in os.listdir(class_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"  '{name}' — {len(images)} images")
        count = 0
        for fname in tqdm(images, desc=f"    processing", leave=False):
            img   = cv2.imread(os.path.join(class_dir, fname))
            gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face  = detect_face(detector, mode, gray, img)
            if face is not None and face.size > 0:
                face_resized = cv2.resize(face, IMG_SIZE)
                faces.append(face_resized)
                labels.append(label_map[name])
                count += 1
        print(f"    → {count} valid faces extracted")

    return faces, labels, label_map


def main():
    detector, mode = load_detector()
    faces, labels, label_map = build_dataset(detector, mode)

    if not faces:
        print("[ERROR] No faces extracted. Check your split_dataset/train/ folder.")
        return

    print(f"\n[INFO] Training LBPH on {len(faces)} samples...")
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8
    )
    recognizer.train(faces, np.array(labels))

    os.makedirs(OUT_DIR, exist_ok=True)
    recognizer.save(MODEL_OUT)
    with open(LABELS_OUT, "wb") as f:
        pickle.dump(label_map, f)

    print(f"[INFO] Model saved   → {MODEL_OUT}")
    print(f"[INFO] Labels saved  → {LABELS_OUT}")
    print("\n LBPH training complete!")


if __name__ == "__main__":
    main()