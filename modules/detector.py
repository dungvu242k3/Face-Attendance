import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from ultralytics import YOLO


class FaceDetector:
    def __init__(self, device='cuda'):
        yolo_model_path = "./models/yolov8m.pt"
        self.device = device
        self.detector = YOLO(yolo_model_path)
        self.detector.to(device)

        self.embedder = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    def detect_faces(self, frame, conf=0.5):
        results = self.detector.predict(source=frame, conf=conf, device=self.device, verbose=False)[0]
        boxes = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2, y2))
        return boxes

    def extract_faces(self, frame, boxes, size=160):
        faces = []
        for (x1, y1, x2, y2) in boxes:
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue
            face = cv2.resize(face, (size, size))
            faces.append(face)
        return faces

    def preprocess(self, face_img):
        face = cv2.resize(face_img, (160, 160))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype(np.float32) / 255.0
        face = (face - 0.5) / 0.5
        face = torch.tensor(face).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return face

    def get_embeddings(self, face_img):
        face_tensor = self.preprocess(face_img)
        with torch.no_grad():
            embedding = self.embedder(face_tensor).cpu().numpy()[0]
        return embedding / np.linalg.norm(embedding)
