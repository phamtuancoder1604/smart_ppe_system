import cv2
import numpy as np
from models.face_recognition import FaceRecognition
from database.employee_db import EmployeeDB
from models.utils import load_yaml, cosine_similarity

def test_face(image_path, config_path="config/config.yaml"):
  
    cfg = load_yaml(config_path)
    face_model_path = cfg["models"]["face_recognition"]
    face_model = FaceRecognition(model_path=face_model_path)
    emp_db = EmployeeDB()
    face_db = emp_db.get_face_db()

    print(f"[INFO] Loaded {len(face_db)} nhân viên trong database.")
    img = cv2.imread(image_path)
    if img is None:
        print("[ERROR] Không đọc được ảnh:", image_path)
        return
    emb = face_model.get_face_embedding(img)
    best_id, best_score = None, 0
    for emp_id, db_emb in face_db.items():
        if isinstance(db_emb, dict):
            if "centroid" in db_emb:
                db_emb = np.array(db_emb["centroid"], dtype=np.float32)
            elif "samples" in db_emb and len(db_emb["samples"]) > 0:
                db_emb = np.mean(np.stack(db_emb["samples"], axis=0), axis=0)
            else:
                continue
        else:
            db_emb = np.array(db_emb, dtype=np.float32)

        score = cosine_similarity(emb, db_emb)
        if score > best_score:
            best_score, best_id = score, emp_id

        if score > best_score:
            best_score, best_id = score, emp_id

    print("\n===== KẾT QUẢ NHẬN DIỆN =====")
    print(f"Ảnh test: {image_path}")
    if best_id and best_score >= 0.8:
        print(f" Nhận diện: {best_id} (độ tương đồng {best_score:.3f})")
    else:
        print(f" Không tìm thấy kết quả tin cậy (max score {best_score:.3f})")

if __name__ == "__main__":
    test_face("data/test_images/messi_1.jpg")
