"""
ml-lbph/val.py  —  UPGRADED

Validates the LBPH model on split_dataset/val/.
Prints per-class accuracy, confusion matrix, overall accuracy.
"""

import sys
import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

VAL_DIR         = os.path.join(BASE_DIR, "split_dataset", "val")
MODEL_PATH      = os.path.join(BASE_DIR, "ml-lbph", "lbph_model.xml")
LABELS_PATH     = os.path.join(BASE_DIR, "ml-lbph", "labels.pkl")
CONF_THRESHOLD  = 60.0    # LBPH distance — lower = more confident; reject above this
IMG_SIZE        = (100, 100)

DNN_PROTO  = os.path.join(BASE_DIR, "source", "deploy.prototxt")
DNN_MODEL  = os.path.join(BASE_DIR, "source", "res10_300x300_ssd_iter_140000.caffemodel")
DNN_CONF   = 0.55


def load_detector():
    if os.path.exists(DNN_PROTO) and os.path.exists(DNN_MODEL):
        return cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL), "dnn"
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'), "haar"


def detect_face(detector, mode, gray, color):
    if mode == "dnn":
        h, w = color.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(color, (300, 300)),
                                     1.0, (300, 300), (104, 177, 123))
        detector.setInput(blob)
        out = detector.forward()
        best, best_area = None, 0
        for i in range(out.shape[2]):
            if out[0, 0, i, 2] > DNN_CONF:
                box = out[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = np.clip(box.astype(int), 0, [w, h, w, h])
                area = (x2-x1)*(y2-y1)
                if area > best_area:
                    best_area, best = area, (x1, y1, x2, y2)
        if best:
            x1, y1, x2, y2 = best
            return gray[y1:y2, x1:x2]
    else:
        faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        if len(faces):
            x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
            return gray[y:y+h, x:x+w]
    return None


def main():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)

    with open(LABELS_PATH, "rb") as f:
        label_map = pickle.load(f)
    id_to_name = {v: k for k, v in label_map.items()}

    detector, mode = load_detector()

    classes = sorted([
        d for d in os.listdir(VAL_DIR)
        if os.path.isdir(os.path.join(VAL_DIR, d))
    ])
    print(f"[INFO] Validating on {len(classes)} classes: {classes}\n")

    y_true, y_pred = [], []
    n_unknown = 0

    for name in classes:
        class_dir = os.path.join(VAL_DIR, name)
        images    = [f for f in os.listdir(class_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for fname in tqdm(images, desc=f"  {name}"):
            img  = cv2.imread(os.path.join(class_dir, fname))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face = detect_face(detector, mode, gray, img)

            if face is None or face.size == 0:
                continue

            face_resized      = cv2.resize(face, IMG_SIZE)
            label_id, dist    = recognizer.predict(face_resized)
            pred_name         = id_to_name.get(label_id, "Unknown")

            if dist > CONF_THRESHOLD:
                pred_name = "Unknown"
                n_unknown += 1

            y_true.append(name)
            y_pred.append(pred_name)

    all_labels = list(classes) + (["Unknown"] if n_unknown else [])

    print("\n" + "=" * 55)
    print("  LBPH VALIDATION RESULTS")
    print("=" * 55)
    print(classification_report(y_true, y_pred, labels=all_labels, zero_division=0))

    correct = sum(t == p for t, p in zip(y_true, y_pred))
    print(f"  Overall Accuracy : {correct / len(y_true) * 100:.1f}%  ({correct}/{len(y_true)})")
    print(f"  Flagged Unknown  : {n_unknown}")

    print("\n  Confusion Matrix (rows=true, cols=pred):")
    cm     = confusion_matrix(y_true, y_pred, labels=all_labels)
    header = "         " + "  ".join(f"{l[:7]:>7}" for l in all_labels)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {all_labels[i][:7]:>7}  " + "  ".join(f"{v:>7}" for v in row))


if __name__ == "__main__":
    main()