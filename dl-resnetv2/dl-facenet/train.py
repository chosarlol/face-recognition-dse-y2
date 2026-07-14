"""
dl-facenet/train.py  —  NEW (add this folder alongside dl-resnetv2)

Trains a FaceNet-based face recognition system:
  1. Loads pretrained FaceNet (VGGFace2 weights)
  2. Extracts 512-d embeddings for every image in split_dataset/train/
  3. Trains a linear SVM on top
  4. Saves classifier to dl-facenet/facenet_classifier.pkl

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
from PIL import Image
from tqdm import tqdm

import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR  = os.path.join(BASE_DIR, "split_dataset", "train")
OUT_DIR      = os.path.join(BASE_DIR, "dl-facenet")
MODEL_OUT    = os.path.join(OUT_DIR, "facenet_classifier.pkl")
EMBED_OUT    = os.path.join(OUT_DIR, "embeddings_train.pkl")

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE     = 160
# ─────────────────────────────────────────────────────────────────────────────


def load_facenet():
    print(f"[INFO] Loading FaceNet on '{DEVICE}'...")
    model = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
    mtcnn = MTCNN(image_size=IMG_SIZE, margin=20, device=DEVICE, keep_all=False)
    return model, mtcnn


def get_embedding(model, mtcnn, img_path):
    try:
        img  = Image.open(img_path).convert("RGB")
        face = mtcnn(img)
        if face is None:
            return None
        face = face.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = model(face).cpu().numpy()[0]
        return emb
    except Exception as e:
        print(f"  [WARN] Skipping {os.path.basename(img_path)}: {e}")
        return None


def build_embeddings(model, mtcnn):
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
        for fname in tqdm(images, desc=f"    embedding", leave=False):
            emb = get_embedding(model, mtcnn, os.path.join(class_dir, fname))
            if emb is not None:
                embeddings.append(emb)
                labels.append(label)

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


def save_artifacts(svm, le, embeddings, labels):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump({"svm": svm, "label_encoder": le}, f)
    print(f"[INFO] Classifier saved → {MODEL_OUT}")

    with open(EMBED_OUT, "wb") as f:
        pickle.dump({"embeddings": embeddings, "labels": labels}, f)
    print(f"[INFO] Embeddings saved  → {EMBED_OUT}")


def main():
    model, mtcnn = load_facenet()
    embeddings, labels = build_embeddings(model, mtcnn)

    if len(embeddings) == 0:
        print("[ERROR] No valid embeddings. Check split_dataset/train/ has images.")
        return

    svm, le = train_svm(embeddings, labels)
    save_artifacts(svm, le, embeddings, labels)
    print("\n✅ Done! Now run:  python dl-resnetv2/webcam.py")


if __name__ == "__main__":
    main()