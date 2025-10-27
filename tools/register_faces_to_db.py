import os
import cv2
import numpy as np
from models.face_recognition import FaceRecognition
from database.employee_db import EmployeeDB
from models.utils import load_yaml

def register_faces_from_folder(root="data/people_custom", config_path="config/config.yaml"):
    cfg = load_yaml(config_path)
    face_model_path = cfg["models"]["face_recognition"]

    face_model = FaceRecognition(model_path=face_model_path)
    emp_db = EmployeeDB()

    print("[INFO] Bắt đầu đăng ký khuôn mặt từ thư mục:", root)
    for emp_id in sorted(os.listdir(root)):
        face_dir = os.path.join(root, emp_id, "face")
        if not os.path.isdir(face_dir):
            continue

        embeddings = []
        for img_name in os.listdir(face_dir):
            if not img_name.lower().endswith((".jpg", ".png")):
                continue

            img_path = os.path.join(face_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            emb = face_model.get_face_embedding(img)
            embeddings.append(emb)

        if len(embeddings) == 0:
            print(f"[WARN] {emp_id}: không có ảnh hợp lệ.")
            continue
        emb_avg = np.mean(np.stack(embeddings, axis=0), axis=0)
        emb_avg = emb_avg / (np.linalg.norm(emb_avg) + 1e-6)

        emp_db.add_employee(emp_id, emb_avg, None)
        print(f"[OK] Đã lưu {emp_id} với {len(embeddings)} ảnh mặt.")

    print("[DONE] Hoàn tất đăng ký vào database.")

if __name__ == "__main__":
    register_faces_from_folder("data/people_custom", "config/config.yaml")
