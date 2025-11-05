import os
import numpy as np
import onnxruntime as ort
import torch
import torchreid
import cv2
from PIL import Image

from models.utils import cosine_similarity


class ReIDModel:
    def __init__(self, onnx_path=""):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = "torchreid"
        self.ort_sess = None
        self.model = None

        if onnx_path and os.path.exists(onnx_path):
            try:
                self.ort_sess = ort.InferenceSession(
                    onnx_path, providers=["CPUExecutionProvider"]
                )
                self.mode = "onnx"
                print(f"[ReID] Using ONNX model: {onnx_path}")
            except Exception as e:
                print(f"[ReID] ONNX init failed, fallback to torchreid: {e}")


        if self.mode == "torchreid":
            self.model = torchreid.models.build_model(
                name="osnet_x1_0", num_classes=1000, pretrained=True
            )
            self.model.eval()
            self.model.to(self.device)

            self.transform = torchreid.data.transforms.build_transforms(
                height=256,
                width=128,
                norm_mean=[0.485, 0.456, 0.406],
                norm_std=[0.229, 0.224, 0.225],
            )[0]

    # ----------------------------------------------------------------------
    def preprocess_onnx(self, img):
        img = cv2.resize(img, (128, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = np.transpose(img, (2, 0, 1))[None, ...]
        return img

    # ----------------------------------------------------------------------
    def get_body_embedding(self, body_img):

        if body_img is None:
            print("[ReIDModel] Warning: body_img is None.")
            return np.zeros(512, dtype=np.float32).tolist()

        if not isinstance(body_img, np.ndarray):
            try:
                body_img = np.array(body_img)
            except Exception as e:
                print(f"[ReIDModel] Invalid image type: {type(body_img)}, error: {e}")
                return np.zeros(512, dtype=np.float32).tolist()

        if body_img.size == 0:
            print("[ReIDModel] Warning: body_img is empty.")
            return np.zeros(512, dtype=np.float32).tolist()

        if self.mode == "onnx" and self.ort_sess is not None:
            x = self.preprocess_onnx(body_img)
            out = self.ort_sess.run(
                None, {self.ort_sess.get_inputs()[0].name: x}
            )[0]
            vec = out.reshape(-1).astype(np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-6)
            return vec.tolist()

        with torch.no_grad():
            body_img = cv2.resize(body_img, (128, 256))
            body_img = cv2.cvtColor(body_img, cv2.COLOR_BGR2RGB)
            body_img = Image.fromarray(body_img)

            x = self.transform(body_img).unsqueeze(0).to(self.device)
            feat = self.model(x)
            vec = feat.squeeze().cpu().numpy().astype(np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-6)
            return vec.tolist()

    def match_employee(self, track_emb, body_db, threshold=0.75):
        """
        So khớp embedding hiện tại (track_emb) với database nhân viên.
        Trả về employee_id nếu cosine similarity vượt ngưỡng.
        """
        best_id, best_sim = None, 0.0
        for emp_id, emb in body_db.items():
            sim = cosine_similarity(track_emb, emb)
            if sim > best_sim:
                best_id, best_sim = emp_id, sim

        if best_id is not None and best_sim >= threshold:
            print(f"[ReID] Match found: {best_id} ({best_sim:.2f})")
        return best_id if best_sim >= threshold else None
