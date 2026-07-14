"""
dl-facenet/val.py  —  NEW

Validates the FaceNet classifier on split_dataset/val/
Prints: per-class accuracy, confusion matrix, overall accuracy.
"""

import sys
import os
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

BASE_DIR          = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VAL_DIR           = os.path.join(BASE_DIR, "split_dataset", "val")
CLASSIFIER_PATH   = os.path.join(BASE_DIR, "dl-facenet", "facenet_classifier.pkl")
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
CONFIDENCE_THRESHOLD = 0.65


def load_models():
    print(f"[INFO] Loading FaceNet on '{DEVICE}'...")
    model = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
    mtcnn = MTCNN(image_size=160, margin=20, device=DEVICE, keep_all=False)
    with open(CLASSIFIER_PATH, "rb") as f:
        data = pickle.load(f)
    return model, mtcnn, data["svm"], data["label_encoder"]


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
    except Exception:
        return None


def main():
    model, mtcnn, svm, le = load_models()

    classes = sorted([
        d for d in os.listdir(VAL_DIR)
        if os.path.isdir(os.path.join(VAL_DIR, d))
    ])
    print(f"[INFO] Validating on {len(classes)} classes: {classes}\n")

    y_true, y_pred = [], []
    n_unknown = 0

    for label in classes:
        class_dir = os.path.join(VAL_DIR, label)
        images    = [f for f in os.listdir(class_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for fname in tqdm(images, desc=f"  {label}"):
            emb = get_embedding(model, mtcnn, os.path.join(class_dir, fname))
            if emb is None:
                continue

            emb_n = normalize(emb.reshape(1, -1))
            probs = svm.predict_proba(emb_n)[0]
            idx   = np.argmax(probs)
            conf  = probs[idx]
            pred  = le.inverse_transform([idx])[0]

            if conf < CONFIDENCE_THRESHOLD:
                pred = "Unknown"
                n_unknown += 1

            y_true.append(label)
            y_pred.append(pred)

    all_labels = list(classes) + (["Unknown"] if n_unknown else [])

    print("\n" + "=" * 55)
    print("  VALIDATION RESULTS")
    print("=" * 55)
    print(classification_report(y_true, y_pred,
                                labels=all_labels, zero_division=0))

    correct = sum(t == p for t, p in zip(y_true, y_pred))
    print(f"  Overall Accuracy : {correct / len(y_true) * 100:.1f}%  "
          f"({correct}/{len(y_true)})")
    print(f"  Flagged Unknown  : {n_unknown}")

    print("\n  Confusion Matrix (rows=true, cols=pred):")
    cm     = confusion_matrix(y_true, y_pred, labels=all_labels)
    header = "         " + "  ".join(f"{l[:7]:>7}" for l in all_labels)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {all_labels[i][:7]:>7}  " + "  ".join(f"{v:>7}" for v in row))


if __name__ == "__main__":
    main()