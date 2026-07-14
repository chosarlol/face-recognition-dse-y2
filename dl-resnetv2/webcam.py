"""
dl-resnetv2/webcam.py — UPGRADED (DNN SSD detector, no MediaPipe)
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

BASE_DIR             = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

CLASSIFIER_PATH      = os.path.join(BASE_DIR, "dl-facenet", "facenet_classifier.pkl")
IMAGE_SIZE           = (160, 160)
CONFIDENCE_THRESHOLD = 0.65
DEVICE               = "cuda" if torch.cuda.is_available() else "cpu"
SMOOTH_FRAMES        = 5

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


def get_embedding(facenet, face_bgr):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb).resize(IMAGE_SIZE)
    t = torch.tensor(np.array(face_pil)).permute(2, 0, 1).float()
    t = (t - 127.5) / 128.0
    t = t.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = facenet(t).cpu().numpy()[0]
    return emb


def predict(svm, le, embedding):
    emb_n = normalize(embedding.reshape(1, -1))
    probs = svm.predict_proba(emb_n)[0]
    idx   = np.argmax(probs)
    return le.inverse_transform([idx])[0], float(probs[idx])


def draw_box(frame, x1, y1, x2, y2, name, conf):
    color = (0, 220, 80) if name != "Unknown" else (30, 30, 220)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{name}  {conf*100:.0f}%" if name != "Unknown" else "Unknown"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def main():
    print("Loading FaceNet model...")
    facenet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
    print("Model loaded.")

    print("Loading classifier...")
    with open(CLASSIFIER_PATH, "rb") as f:
        data = pickle.load(f)
    svm = data["svm"]
    le  = data["label_encoder"]
    print("Classifier loaded.")

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

        boxes = detect_faces(detector, mode, frame)

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            emb        = get_embedding(facenet, face_crop)
            name, conf = predict(svm, le, emb)

            smooth[i].append((name, conf))
            names  = [n for n, _ in smooth[i]]
            s_name = max(set(names), key=names.count)
            s_conf = float(np.mean([c for n, c in smooth[i] if n == s_name]))

            if s_conf < CONFIDENCE_THRESHOLD:
                s_name, s_conf = "Unknown", s_conf

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