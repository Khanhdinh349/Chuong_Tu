# train.py
import os
import cv2
import pickle
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet

detector = MTCNN()
embedder = FaceNet()
BASE_DIR = os.path.abspath(".")
dataset_dir = os.path.join(BASE_DIR, "dataset")

face_data = {}

def extract_face_mtcnn(img_path):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)
    if results:
        x, y, w, h = results[0]['box']
        x, y = max(0, x), max(0, y)
        face = img[y:y+h, x:x+w]
        return cv2.resize(face, (160, 160))
    return None

for person in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person)
    if os.path.isdir(person_dir):
        face_data[person] = []
        for img_file in os.listdir(person_dir):
            path = os.path.join(person_dir, img_file)
            face = extract_face_mtcnn(path)
            if face is not None:
                emb = embedder.embeddings([face])[0]
                face_data[person].append(emb)
                print(f"[INFO] Trained: {person}/{img_file}")

output_dir = os.path.join(BASE_DIR, "Code", "embeddings")
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "face_cosine_data.pkl"), "wb") as f:
    pickle.dump(face_data, f)

print("[DONE] Training completed & saved.")
