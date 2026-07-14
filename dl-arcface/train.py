"""
dl-arcface/train.py  —  NEW (add this folder alongside ml-lbph and dl-facenet)

Trains an ArcFace-based face recognition system:
  1. Detects faces with the same DNN SSD detector used elsewhere in this repo
     (source/deploy.prototxt + res10_300x300_ssd_iter_140000.caffemodel) —
     detection stays fast, only recognition uses the heavier model.
  2. Extracts 512-d ArcFace embeddings for every detected face in
     split_dataset/train/, using insightface's pretrained buffalo_l
     recognition model (w600k_r50.onnx).
  3. Trains a linear SVM on top of the (L2-normalized) embeddings.
  4. Saves the classifier to dl-arcface/models/arcface_classifier.pkl

Why ArcFace over FaceNet:
  - Trained with an additive angular margin loss, which produces embeddings
    with noticeably better class separation than FaceNet's triplet loss —
    translates directly into fewer false matches at the same recall.
  - buffalo_l (ResNet50 backbone, trained on WebFace600K) is one of the
    strongest openly-available recognition models, well above FaceNet's
    VGGFace2-trained InceptionResnetV1 on standard benchmarks (LFW/CFP-FP/
    AgeDB).

IMPORTANT — separate conda env required:
  Installing `insightface` alongside this repo's existing 'frs' env forces
  numpy 2.x and pulls in a conflicting second OpenCV build that breaks
  facenet-pytorch (needs numpy<2) and corrupts ml-lbph's cv2.face module
  (opencv-contrib vs opencv-python both installing into site-packages/cv2).
  So dl-arcface/ runs in its own env, 'frs-arcface', left untouched by the
  other two pipelines:

      conda create -n frs-arcface python=3.10 -y
      conda activate frs-arcface
      pip install opencv-contrib-python==4.11.0.86 numpy insightface onnxruntime scikit-learn tqdm pillow

  (opencv-contrib-python is pinned to 4.11 — OpenCV 5.x dropped
  cv2.dnn.readNetFromCaffe, which the shared SSD detector needs.)

  The buffalo_l model pack (~280MB) auto-downloads to ~/.insightface/ on
  first run, same as facenet-pytorch caching its weights to ~/.cache.

Usage:
    conda activate frs-arcface
    python dl-arcface/train.py
"""

import sys
import os
import pickle
import numpy as np
import cv2
from tqdm import tqdm

from insightface.utils import ensure_available
from insightface.model_zoo import model_zoo
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

DATASET_DIR  = os.path.join(BASE_DIR, "split_dataset", "train")
OUT_DIR      = os.path.join(BASE_DIR, "dl-arcface", "models")
MODEL_OUT    = os.path.join(OUT_DIR, "arcface_classifier.pkl")
EMBED_OUT    = os.path.join(OUT_DIR, "embeddings_train.pkl")

DNN_PROTO    = os.path.join(BASE_DIR, "source", "deploy.prototxt")
DNN_MODEL    = os.path.join(BASE_DIR, "source", "res10_300x300_ssd_iter_140000.caffemodel")
DNN_CONF     = 0.55

ARCFACE_PACK = os.path.join(os.path.expanduser("~"), ".insightface", "models", "buffalo_l")
ARCFACE_ONNX = os.path.join(ARCFACE_PACK, "w600k_r50.onnx")
# ─────────────────────────────────────────────────────────────────────────────


def load_detector():
    if os.path.exists(DNN_PROTO) and os.path.exists(DNN_MODEL):
        print("[INFO] Using DNN SSD detector.")
        return cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL), "dnn"
    print("[WARN] DNN not found, using Haar Cascade fallback.")
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'), "haar"


def detect_face(detector, mode, color_img):
    """Returns the largest face crop (BGR) or None."""
    if mode == "dnn":
        h, w = color_img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(color_img, (300, 300)),
                                     1.0, (300, 300), (104, 177, 123))
        detector.setInput(blob)
        out   = detector.forward()
        best, best_area = None, 0
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
            return color_img[y1:y2, x1:x2]
    else:
        gray  = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        if len(faces):
            x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
            return color_img[y:y+h, x:x+w]
    return None


def load_arcface():
    if not os.path.exists(ARCFACE_ONNX):
        print("[INFO] Downloading ArcFace (buffalo_l) recognition model (~280MB, one-time)...")
        ensure_available("models", "buffalo_l")
    print(f"[INFO] Loading ArcFace recognition model on 'cpu'...")
    rec = model_zoo.get_model(ARCFACE_ONNX, providers=["CPUExecutionProvider"])
    rec.prepare(ctx_id=-1)
    return rec


def get_embedding(rec, face_bgr):
    try:
        face_resized = cv2.resize(face_bgr, rec.input_size)
        return rec.get_feat(face_resized).flatten()
    except Exception as e:
        print(f"  [WARN] Skipping face: {e}")
        return None


def build_embeddings(detector, mode, rec):
    classes = sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ])
    if not classes:
        raise ValueError(f"No class folders found in '{DATASET_DIR}'. Run capture.py first.")

    print(f"[INFO] Found {len(classes)} people: {classes}\n")
    embeddings, labels = [], []

    for label in classes:
        class_dir = os.path.join(DATASET_DIR, label)
        images = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        print(f"  '{label}' — {len(images)} images")
        count = 0
        for fname in tqdm(images, desc=f"    embedding", leave=False):
            img = cv2.imread(os.path.join(class_dir, fname))
            if img is None:
                continue
            face = detect_face(detector, mode, img)
            if face is None or face.size == 0:
                continue
            emb = get_embedding(rec, face)
            if emb is not None:
                embeddings.append(emb)
                labels.append(label)
                count += 1
        print(f"    → {count} valid embeddings extracted")

    print(f"\n[INFO] Total valid embeddings: {len(embeddings)}")
    return np.array(embeddings), np.array(labels)


def train_svm(embeddings, labels):
    X  = normalize(embeddings)
    le = LabelEncoder()
    y  = le.fit_transform(labels)

    n_classes = len(set(y))
    cv_folds  = min(5, n_classes)
    svm       = SVC(kernel='linear', C=1.0, probability=True)

    print("[INFO] Running cross-validation...")
    scores = cross_val_score(svm, X, y, cv=cv_folds, scoring='accuracy')
    print(f"[INFO] CV Accuracy: {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")

    svm.fit(X, y)
    return svm, le


def compute_centroids(embeddings, labels):
    """Per-class mean of L2-normalized embeddings, re-normalized.

    webcam.py uses this as a second gate alongside the SVM probability:
    the SVM always forces a pick between known classes even for a total
    stranger, so we also require the live embedding to be reasonably
    close (cosine sim) to the predicted class's centroid.
    """
    X = normalize(embeddings)
    centroids = {}
    for label in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == label]
        centroids[label] = normalize(X[idx].mean(axis=0, keepdims=True))[0]
    return centroids


def save_artifacts(svm, le, centroids, embeddings, labels):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump({"svm": svm, "label_encoder": le, "centroids": centroids}, f)
    print(f"[INFO] Classifier saved → {MODEL_OUT}")

    with open(EMBED_OUT, "wb") as f:
        pickle.dump({"embeddings": embeddings, "labels": labels}, f)
    print(f"[INFO] Embeddings saved  → {EMBED_OUT}")


def main():
    detector, mode = load_detector()
    rec = load_arcface()

    embeddings, labels = build_embeddings(detector, mode, rec)

    if len(embeddings) == 0:
        print("[ERROR] No valid embeddings. Check split_dataset/train/ has images.")
        return

    svm, le = train_svm(embeddings, labels)
    centroids = compute_centroids(embeddings, labels)
    save_artifacts(svm, le, centroids, embeddings, labels)
    print("\n✅ Done! Now run:  python dl-arcface/webcam.py")


if __name__ == "__main__":
    main()
