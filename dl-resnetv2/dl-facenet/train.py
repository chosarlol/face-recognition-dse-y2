"""
dl-facenet/train.py  —  NEW (add this folder alongside dl-resnetv2)

Trains a FaceNet-based face recognition system:
  1. Detects faces with the same DNN SSD detector used elsewhere in this
     repo (source/deploy.prototxt + res10_300x300_ssd_iter_140000.caffemodel)
  2. Extracts 512-d embeddings for every detected face in
     split_dataset/train/, using pretrained FaceNet (VGGFace2 weights)
  3. Trains a linear SVM on top
  4. Saves classifier to dl-facenet/facenet_classifier.pkl

NOTE: this used to detect+align faces with MTCNN instead of the SSD
detector. That made training embeddings and live webcam embeddings come
from two different cropping pipelines (webcam.py has always used the SSD
crop directly, no MTCNN) — invisible with only a loose SVM-probability
threshold, but it broke the embedding-similarity gate added for the
reliability layer (correct, high-confidence predictions were failing
the centroid-similarity check because their embeddings lived in a
different alignment space than the MTCNN-built centroids). Switching to
the SSD crop here matches webcam.py and keeps train/inference consistent
— same FaceNet embedding network, just a consistently-cropped input,
matching how dl-arcface and ml-lbph are already structured.

Why better than fine-tuning InceptionResNetV2:
  - Pretrained on 3M+ face images — no training from scratch
  - Works with 10–20 images per person (yours needed 100+)
  - ~99% accuracy on standard benchmarks
  - No TF GPU warning — pure PyTorch

Install:
    pip install facenet-pytorch torch torchvision scikit-learn tqdm
"""

import sys
import os
import pickle
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

import torch
from facenet_pytorch import InceptionResnetV1
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_DIR  = os.path.join(BASE_DIR, "split_dataset", "train")
OUT_DIR      = os.path.join(BASE_DIR, "dl-facenet")
MODEL_OUT    = os.path.join(OUT_DIR, "facenet_classifier.pkl")
EMBED_OUT    = os.path.join(OUT_DIR, "embeddings_train.pkl")

DNN_PROTO    = os.path.join(BASE_DIR, "source", "deploy.prototxt")
DNN_MODEL    = os.path.join(BASE_DIR, "source", "res10_300x300_ssd_iter_140000.caffemodel")
DNN_CONF     = 0.55

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE   = (160, 160)
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


def load_facenet():
    print(f"[INFO] Loading FaceNet on '{DEVICE}'...")
    return InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)


def get_embedding(model, face_bgr):
    try:
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb).resize(IMAGE_SIZE)
        t = torch.tensor(np.array(face_pil)).permute(2, 0, 1).float()
        t = (t - 127.5) / 128.0
        t = t.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = model(t).cpu().numpy()[0]
        return emb
    except Exception as e:
        print(f"  [WARN] Skipping face: {e}")
        return None


def build_embeddings(detector, mode, model):
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
            emb = get_embedding(model, face)
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
    model = load_facenet()
    embeddings, labels = build_embeddings(detector, mode, model)

    if len(embeddings) == 0:
        print("[ERROR] No valid embeddings. Check split_dataset/train/ has images.")
        return

    svm, le = train_svm(embeddings, labels)
    centroids = compute_centroids(embeddings, labels)
    save_artifacts(svm, le, centroids, embeddings, labels)
    print("\n✅ Done! Now run:  python dl-resnetv2/webcam.py")


if __name__ == "__main__":
    main()