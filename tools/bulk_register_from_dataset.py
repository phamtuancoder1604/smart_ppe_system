import os
import glob
import argparse
import cv2
import numpy as np

from database.employee_db import EmployeeDB
from models.face_detector import FaceDetector  
from models.face_alignment import FaceAligner  
from models.face_recognition import FaceRecognition 
from models.person_detection import PersonDetector  
from models.reid import ReIDModel  
from models.utils import cosine_similarity


def variance_of_laplacian(img_gray):
    return cv2.Laplacian(img_gray, cv2.CV_64F).var()


def collect_face_embeddings(img_paths, face_detector, face_aligner, face_model,
                            max_samples=8, min_size=80, min_sharp=60.0,
                            dedup_cos=0.985):
    """
    - Tự động phát hiện mặt (nếu chưa crop), align -> 112x112, filter chất lượng,
      khử trùng lặp theo cosine.
    - Trả về list[np.ndarray] (đã L2 normalize).
    """
    embs = []
    for p in img_paths:
        img = cv2.imread(p)
        if img is None:
            continue

        faces = face_detector.detect_faces(img)
        if not faces:
            continue
        fx1, fy1, fx2, fy2, _ = max(faces, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        aligned = face_aligner.align(img, (fx1, fy1, fx2, fy2))
        if aligned is None:
            continue

        if min(aligned.shape[:2]) < min_size:
            continue
        sharp = variance_of_laplacian(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY))
        if sharp < min_sharp:
            continue

        emb = face_model.get_face_embedding(aligned)  # (512,)
        duplicate = False
        for e in embs:
            c = cosine_similarity(emb, e)
            if c >= dedup_cos:
                duplicate = True
                break
        if duplicate:
            continue

        embs.append(emb)
        if len(embs) >= max_samples:
            break
    return embs


def collect_body_embeddings(img_paths, person_detector, reid_model,
                            max_samples=8, dedup_cos=0.985, min_area=80 * 160):
    """
    - Dò người bằng YOLOv8, cắt bbox lớn nhất, lấy embedding body (ReID).
    - Khử trùng lặp để không lưu 10 ảnh giống nhau.
    """
    embs = []
    for p in img_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        persons = person_detector.detect_persons(img, conf=0.5)
        if not persons:
            continue
        x1, y1, x2, y2, _ = max(persons, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        if (x2 - x1) * (y2 - y1) < min_area:
            continue
        crop = img[int(y1):int(y2), int(x1):int(x2)]
        if crop.size == 0:
            continue
        emb = reid_model.get_body_embedding(crop)

        duplicate = False
        for e in embs:
            c = cosine_similarity(emb, e)
            if c >= dedup_cos:
                duplicate = True
                break
        if duplicate:
            continue

        embs.append(emb)
        if len(embs) >= max_samples:
            break
    return embs


def main():
    parser = argparse.ArgumentParser(
        description="Bulk register faces + bodies from a dataset (100 people).")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Thư mục gốc dataset. Cấu trúc mặc định: person_id/face/*.jpg, person_id/body/*.jpg")
    parser.add_argument("--face_glob", type=str, default="face/*.jpg",
                        help="Pattern tìm ảnh mặt tương đối trong từng thư mục người (VD: face/*.jpg)")
    parser.add_argument("--body_glob", type=str, default="body/*.jpg",
                        help="Pattern tìm ảnh body tương đối trong từng thư mục người (VD: body/*.jpg)")
    parser.add_argument("--max_face", type=int, default=8, help="Số ảnh mặt tối đa lưu/nhân viên")
    parser.add_argument("--max_body", type=int, default=8, help="Số ảnh body tối đa lưu/nhân viên")
    parser.add_argument("--threshold", type=float, default=0.80,
                        help="Ngưỡng match tham khảo (không dùng trong đăng ký, chỉ gợi ý)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Bỏ qua emp_id đã có trong DB")
    args = parser.parse_args()

    print("[INFO] Loading models...")
    face_detector = FaceDetector()  
    face_aligner = FaceAligner()
    face_model = FaceRecognition(model_path="models/weights/buffalo_s/w600k_mbf.onnx")
    person_det = PersonDetector(weights="models/weights/yolov8n.pt")
    reid_model = ReIDModel()

    emp_db = EmployeeDB()

    person_dirs = [d for d in sorted(os.listdir(args.dataset_root))
                   if os.path.isdir(os.path.join(args.dataset_root, d))]
    print(f"[INFO] Found {len(person_dirs)} identities in dataset.")

    total_new = 0
    for emp_id in person_dirs:
        pid_dir = os.path.join(args.dataset_root, emp_id)
        if args.skip_existing and emp_id in emp_db.get_face_db():
            print(f"[SKIP] {emp_id} đã có trong DB.")
            continue

        face_paths = sorted(glob.glob(os.path.join(pid_dir, args.face_glob)))
        body_paths = sorted(glob.glob(os.path.join(pid_dir, args.body_glob)))

        if not face_paths and not body_paths:
            print(f"[WARN] {emp_id}: không tìm thấy ảnh theo patterns {args.face_glob} / {args.body_glob}")
            continue

        face_embs = []
        if face_paths:
            face_embs = collect_face_embeddings(
                face_paths, face_detector, face_aligner, face_model,
                max_samples=args.max_face
            )

        body_embs = []
        if body_paths:
            body_embs = collect_body_embeddings(
                body_paths, person_det, reid_model,
                max_samples=args.max_body
            )

        if not face_embs and not body_embs:
            print(f"[WARN] {emp_id}: không có mẫu hợp lệ.")
            continue

        if hasattr(emp_db, "add_face_samples") and face_embs:
            emp_db.add_face_samples(emp_id, face_embs)
        elif face_embs:

            f = np.mean(np.stack(face_embs, axis=0), axis=0)
            f = f / (np.linalg.norm(f) + 1e-6)
            emp_db.add_employee(emp_id, f, None)

        if body_embs:
            b = np.mean(np.stack(body_embs, axis=0), axis=0)
            b = b / (np.linalg.norm(b) + 1e-6)
            if hasattr(emp_db, "bodies"):
                emp_db.bodies[emp_id] = b
                emp_db._save_json_bodies()
            else:
                emp_db.add_employee(emp_id, None, b)

        total_new += 1
        print(f"[OK] {emp_id}: saved {len(face_embs)} face samples, {len(body_embs)} body samples.")

    print(f"[DONE] Đăng ký mới/ cập nhật: {total_new} identities.")
    print("[TIP] Nên đặt threshold nhận diện >= 0.80 cho ArcFace; "
          "và lọc kết quả bằng mode trên 5–10 khung để ổn định tên.")


if __name__ == "__main__":
    main()
