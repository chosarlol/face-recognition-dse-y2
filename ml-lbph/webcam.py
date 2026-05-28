import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from helpers.dnn_detector import DNNFaceDetector
from helpers.haar_detector import HaarFaceDetector

# FIXED: Changed from hardcoded Linux paths to project relative paths
MODEL_PATH = "machineface/trainer.yml"
LABELS_PATH = "machineface/labels.txt"

# -------------------------------
# Load label map
# -------------------------------
def load_labels(path):
    if not os.path.exists(path):
        print(f"❌ Error: Code map file missing at '{path}'. Please run train.py successfully first!")
        sys.exit(1)
        
    label_map = {}
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            id_, name = line.strip().split(",", 1)
            label_map[int(id_)] = name
    return label_map

label_map = load_labels(LABELS_PATH)

# -------------------------------
# Create detector objects
# -------------------------------
dnn_detector = DNNFaceDetector()
haar_detector = HaarFaceDetector()

# -------------------------------
# Load recognizer
# -------------------------------
if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: Model file missing at '{MODEL_PATH}'. Please run train.py successfully first!")
    sys.exit(1)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

# -------------------------------
# Start camera
# -------------------------------
cam = cv2.VideoCapture(1)

if not cam.isOpened():
    raise IOError("Cannot open webcam")

print("Camera started. Press q to quit.")

try:
    while True:
        ret, img = cam.read()
        if not ret:
            print("Failed to capture frame")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ============================================
        # Choose one detector only:
        faces = dnn_detector.detect(img)
        # faces = haar_detector.detect(gray)
        # ============================================

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]

            if face.size == 0:
                continue

            # Must match training preprocessing
            face = cv2.resize(face, (200, 200))

            predicted_id, conf = recognizer.predict(face)

            # UPDATED: Lowered threshold from 80 to 60 for strict filtering
            # Lower confidence score values indicate better matching accuracy in LBPH
            if conf < 60.0:
                name = label_map.get(predicted_id, "Unknown")
                text = f"{name} ({conf:.1f})"
                color = (0, 255, 0) # Green box for trusted matches
            else:
                text = "Unknown"
                color = (0, 0, 255) # Red box for strangers or weak matches

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                img,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        cv2.imshow("Face Recognition", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cam.release()
    cv2.destroyAllWindows()
