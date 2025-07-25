# realtime.py
import cv2
import os
import numpy as np
import pickle
from keras_facenet import FaceNet
from mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_similarity
import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
RELAY_PINS = [5, 6, 13, 19]  # 4 tủ demo
for pin in RELAY_PINS:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

detector = MTCNN()
embedder = FaceNet()
BASE_DIR = os.path.abspath(".")
with open(os.path.join(BASE_DIR, "Code", "embeddings", "face_cosine_data.pkl"), "rb") as f:
    face_data = pickle.load(f)

embeddings = []
labels = []
for name, embs in face_data.items():
    embeddings.extend(embs)
    labels.extend([name]*len(embs))
embeddings = np.array(embeddings)

def match_face(emb, threshold=0.7):
    sims = cosine_similarity([emb], embeddings)
    max_sim = np.max(sims)
    if max_sim >= threshold:
        return labels[np.argmax(sims)], max_sim
    return "unknown", max_sim

def open_locker(index):
    pin = RELAY_PINS[index]
    GPIO.output(pin, GPIO.HIGH)
    print(f"Mở tủ số {index+1}")
    time.sleep(3)
    GPIO.output(pin, GPIO.LOW)

cap = cv2.VideoCapture(0)
print("🔍 Nhận diện khuôn mặt...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    faces = detector.detect_faces(frame)
    for result in faces:
        x, y, w, h = result['box']
        face = frame[y:y+h, x:x+w]
        try:
            face = cv2.resize(face, (160, 160))
            emb = embedder.embeddings([face])[0]
            label, score = match_face(emb)
            color = (0, 255, 0) if label != "unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{label} ({score:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if label != "unknown":
                idx = labels.index(label) % 4
                open_locker(idx)

        except Exception as e:
            print("[ERROR]", e)

    cv2.imshow("Face Locker", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()
