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
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

face_data = {}

def extract_face(path):
    img = cv2.imread(path)
    results = detector.detect_faces(img)
    if results:
        x, y, w, h = results[0]['box']
        x, y = max(0, x), max(0, y)
        face = img[y:y+h, x:x+w]
        return cv2.resize(face, (160, 160))
    return None

for user in os.listdir(DATASET_DIR):
    user_dir = os.path.join(DATASET_DIR, user)
    if os.path.isdir(user_dir):
        face_data[user] = []
        for img_file in os.listdir(user_dir):
            path = os.path.join(user_dir, img_file)
            face = extract_face(path)
            if face is not None:
                emb = embedder.embeddings([face])[0]
                face_data[user].append(emb)
                print(f"[INFO] Trained: {user}/{img_file}")

output_dir = os.path.join(BASE_DIR, "Code", "embeddings")
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "face_cosine_data.pkl"), "wb") as f:
    pickle.dump(face_data, f)

print("[DONE] Face embeddings saved successfully.")
