import os
import cv2
import numpy as np

DATASET_PATH = "split_dataset/train"
MODEL_PATH = "machineface/trainer.yml"
LABELS_PATH = "machineface/labels.txt"

os.makedirs("machineface", exist_ok=True)

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_map = {}
current_id = 0

for person in os.listdir(DATASET_PATH):
    person_folder = os.path.join(DATASET_PATH, person)

    if not os.path.isdir(person_folder):
        continue

    has_images = False  # Track if this person actually has valid training data

    for image in os.listdir(person_folder):
        img_path = os.path.join(person_folder, image)

        if not image.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Optional but highly recommended for LBPH consistency
        img = cv2.resize(img, (200, 200))

        faces.append(img)
        labels.append(current_id)
        has_images = True

    # Only assign label mapping and increment ID if images were successfully processed
    if has_images:
        label_map[current_id] = person
        current_id += 1

# Check if we actually found any training data before running the trainer
if len(faces) == 0:
    print("Error: No valid images found in the dataset folder. Check your images!")
else:
    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_PATH)

    # Save labels mapping file cleanly
    with open(LABELS_PATH, "w") as f:
        for id_, name in label_map.items():
            f.write(f"{id_},{name}\n")

    print("🎉 Training complete!")
    print("Model saved to:", MODEL_PATH)
    print("Labels saved to:", LABELS_PATH)
    print("Mapped Classes:", label_map)

