import os
import numpy as np
import cv2
import onnxruntime as ort
from models.utils import cosine_similarity
import numpy as np



class FaceRecognition:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            print(f"[FaceRecognition] WARNING: model not found: {model_path}. Inference will fail until you provide it.")
        self.model_path = model_path
        self.session = None
        try:
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        except Exception as e:
            print(f"[FaceRecognition] ONNX init: {e}")
        self.input_size = (112, 112)

    def match_face(self, face_emb, face_db, threshold=0.6):
        if face_emb is None or len(face_db) == 0:
            return None

        best_id, best_sim = None, 0.0
        for emp_id, emb in face_db.items():
            sim = cosine_similarity(face_emb, emb)
            if sim > best_sim:
                best_id, best_sim = emp_id, sim
        if best_id and best_sim >= threshold:
            print(f"[FaceRecognition] Match found: {best_id} ({best_sim:.2f})")
            return best_id
        else:
            return None
    def preprocess(self, img):
        img = cv2.resize(img, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5   # normalize to [-1,1]
        img = np.transpose(img, (2,0,1))[None, ...]
        return img

    def get_face_embedding(self, face_img):
        x = self.preprocess(face_img)
        if self.session is None:
            return np.zeros((512,), dtype=np.float32)

        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name  # lấy đúng tensor embedding
        out = self.session.run([output_name], {input_name: x})[0]
        vec = out.reshape(-1).astype(np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-6)
        return vec
