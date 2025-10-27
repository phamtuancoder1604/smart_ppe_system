from database.employee_db import EmployeeDB
from database.log_db import log_ppe_violation
from models.face_recognition import FaceRecognition
from models.person_detection import PersonDetector
from models.tracking import TrackerOCSortLite
from models.reid import ReIDModel
from models.ppe_detection import PPEDetector
from models.utils import IoU, crop_xyxy, load_yaml
from models.face_detector import FaceDetector
from models.face_alignment import FaceAligner
import time
import cv2
import numpy as np

def variance_of_laplacian(img_gray):
    return cv2.Laplacian(img_gray, cv2.CV_64F).var()

def capture_face_samples(frame, face_boxes, face_aligner, face_model, need=5,
                         min_size=80, min_sharp=60.0, dedup=0.985):
    samples = []
    embs = []
    if not face_boxes:
        return embs
    fx1, fy1, fx2, fy2, _ = max(face_boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
    for _ in range(80):
        aligned = face_aligner.align(frame, (fx1, fy1, fx2, fy2))
        if aligned is None:
            continue
        g = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        if aligned.shape[0] < min_size or aligned.shape[1] < min_size:
            continue
        if variance_of_laplacian(g) < min_sharp:
            continue
        emb = face_model.get_face_embedding(aligned)
        is_dup = False
        for e in embs:
            c = float(np.dot(emb, e) / (np.linalg.norm(emb)*np.linalg.norm(e) + 1e-6))
            if c >= dedup:
                is_dup = True
                break
        if not is_dup:
            embs.append(emb)
            samples.append(aligned.copy())
        if len(embs) >= need:
            break
        time.sleep(0.05)
    return embs
def apply_nms(detections, iou_thresh=0.45):

    if len(detections) == 0:
        return detections

    boxes = [d[:4] for d in detections]
    scores = [float(d[4]) for d in detections]

    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes,
        scores=scores,
        score_threshold=0.3,
        nms_threshold=iou_thresh
    )

    if len(indices) == 0:
        return []

    if isinstance(indices, np.ndarray):
        indices = indices.flatten().tolist()
    elif isinstance(indices[0], (list, tuple)):
        indices = [i[0] for i in indices]

    return [detections[i] for i in indices]

def run_face_checkin(face_detector, face_aligner, face_model, det_model, reid_model, emp_db, source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[Check-in] Unable to open source: {source}")
        return

    print("[Check-in] Press 's' to enroll new employee, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = face_detector.detect_faces(frame)
        persons = det_model.detect_persons(frame, conf=0.5)
        persons = apply_nms(persons, iou_thresh=0.45)
        for (fx1, fy1, fx2, fy2, fconf) in faces:
            cv2.rectangle(frame, (int(fx1), int(fy1)), (int(fx2), int(fy2)), (255, 0, 0), 2)

            aligned_face = face_aligner.align(frame, (fx1, fy1, fx2, fy2))
            if aligned_face is not None:
                face_emb = face_model.get_face_embedding(aligned_face)
                emp_id, score = emp_db.match_face(face_emb, threshold=0.6)

                label = f"{emp_id} ({score:.2f})" if emp_id else "Unknown"
                color = (0, 255, 0) if emp_id else (0, 0, 255)

                cv2.putText(frame, label, (int(fx1), int(fy1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        for (x1, y1, x2, y2, conf) in persons:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            body_crop = frame[int(y1):int(y2), int(x1):int(x2)]
            if body_crop.size != 0:
                body_emb = reid_model.get_body_embedding(body_crop)
                emp_id, score = emp_db.match_body(body_emb, threshold=0.6)

                label = f"{emp_id} ({score:.2f})" if emp_id else "Unknown"
                color = (0, 255, 0) if emp_id else (0, 0, 255)

                cv2.putText(frame, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Check-in", frame)
        k = cv2.waitKey(1) & 0xFF

        if k == ord('s'):
            if not persons or not faces:
                print("[Check-in] Không tìm thấy cả người và mặt cùng lúc.")
                continue

            best = None
            best_iou = 0
            for (px1, py1, px2, py2, _) in persons:
                for (fx1, fy1, fx2, fy2, _) in faces:
                    iou = IoU((px1, py1, px2, py2), (fx1, fy1, fx2, fy2))
                    if iou > best_iou:
                        best_iou = iou
                        best = (px1, py1, px2, py2, fx1, fy1, fx2, fy2)

            if best is None:
                print("[Check-in] Không tìm được cặp person–face phù hợp.")
                continue

            px1, py1, px2, py2, fx1, fy1, fx2, fy2 = best
            body_crop = frame[int(py1):int(py2), int(px1):int(px2)]
            if body_crop.size == 0:
                print("[Check-in] Lỗi crop body, thử lại.")
                continue

            print("[Check-in] Bắt đầu chụp 5 ảnh khuôn mặt (mỗi 0.5s)...")
            face_embs = []
            for i in range(5):
                ret, frame_cap = cap.read()
                if not ret:
                    continue
                aligned_face = face_aligner.align(frame_cap, (fx1, fy1, fx2, fy2))
                if aligned_face is None:
                    continue

                # Hiển thị tiến trình
                cv2.imshow("Check-in", aligned_face)
                cv2.waitKey(100)

                face_emb = face_model.get_face_embedding(aligned_face)
                face_embs.append(face_emb)
                print(f"  → Ảnh {i + 1}/5 chụp xong")

                cv2.waitKey(500)

            if len(face_embs) == 0:
                print("[Check-in] Không lấy được ảnh khuôn mặt hợp lệ.")
                continue

            body_emb = reid_model.get_body_embedding(body_crop)
            emp_id = input("Enter employee_id to register: ").strip()
            if not emp_id:
                print("[Check-in] Skipped empty ID.")
                continue

            face_embs = np.array(face_embs, dtype=np.float32)
            face_emb = np.mean(face_embs, axis=0)
            face_emb = face_emb / (np.linalg.norm(face_emb) + 1e-6)
            emp_db.add_employee(emp_id, face_emb, body_emb)
            print(f"[Check-in] Registered {emp_id} với 5 ảnh khuôn mặt trung bình.")

        elif k == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
def run_workplace_monitoring(source, face_detector, face_aligner, face_model, det_model, tracker, reid_model, emp_db, cfg, ppe_model=None):

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[Monitor] Unable to open source: {source}")
        return

    reid_thresh = cfg["thresholds"]["reid_match"]
    face_thresh = cfg["thresholds"]["face_match"]
    id_map = {}

    print("[Monitor] Running hybrid Face + Body recognition...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        persons = det_model.detect_persons(frame, conf=0.5)
        tracks = tracker.update(frame, persons)

        faces = face_detector.detect_faces(frame)
        face_db = emp_db.get_face_db()
        body_db = emp_db.get_body_db()


        for t in tracks:
            x1, y1, x2, y2, tid = map(int, t[:5])
            body_crop = crop_xyxy(frame, (x1, y1, x2, y2))
            if x2 - x1 < 80 or y2 - y1 < 120:
                continue
            emp_name = id_map.get(tid, None)


            if emp_name is None:
                matched_face = None


                for (fx1, fy1, fx2, fy2, fconf) in faces:
                    if IoU((x1, y1, x2, y2), (fx1, fy1, fx2, fy2)) > 0.4:
                        face_crop = crop_xyxy(frame, (fx1, fy1, fx2, fy2))
                        aligned_face = face_aligner.align(frame, (fx1, fy1, fx2, fy2))
                        if aligned_face is not None:
                            face_emb = face_model.get_face_embedding(aligned_face)
                            matched_face = face_model.match_face(face_emb, face_db, threshold=face_thresh)

                        if matched_face:
                            emp_name = matched_face
                            break


                if emp_name is None:
                    body_emb = reid_model.get_body_embedding(body_crop)
                    matched_body = reid_model.match_employee(body_emb, body_db, threshold=reid_thresh)
                    if matched_body:
                        emp_name = matched_body


                if emp_name:
                    id_map[tid] = emp_name

                if ppe_model:
                    ppe_items = ppe_model.detect_ppe(frame, conf=cfg["thresholds"]["ppe_confidence"])
                    if len(ppe_items) > 0:
                        ppe_boxes = [[*pb, conf] for (pb, pclass, conf) in ppe_items]
                        ppe_boxes = apply_nms(ppe_boxes, iou_thresh=0.45)
                        ppe_items = [(ppe_boxes[i][:4], ppe_items[i][1], ppe_boxes[i][4]) for i in
                                     range(len(ppe_boxes))]
                        for (pb, pclass, conf) in ppe_items:
                            cv2.rectangle(frame, (int(pb[0]), int(pb[1])), (int(pb[2]), int(pb[3])), (255, 255, 0), 2)
                            cv2.putText(frame, pclass, (int(pb[0]), int(pb[1]) - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            label = f"ID:{tid} ({emp_name if emp_name else 'Unknown'})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Monitor - Hybrid Face + Body Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["checkin", "monitor"], required=True)
    p.add_argument("--source", default="0")
    p.add_argument("--config", default="config/config.yaml")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    src = int(args.source) if str(args.source).isdigit() else args.source
    face_detector = FaceDetector(cfg["models"]["face_detector"])
    face_aligner = FaceAligner(cfg["models"]["face_alignment"])
    face_model = FaceRecognition(cfg["models"]["face_embedding"])
    det_model = PersonDetector(cfg["models"]["person_detector"])
    tracker = TrackerOCSortLite()
    reid_model = ReIDModel(cfg.get("models",{}).get("reid_onnx",""))
    ppe_model = PPEDetector(cfg["models"]["ppe_detector"])
    emp_db = EmployeeDB()

    if args.mode == "checkin":
        run_face_checkin(face_detector, face_aligner, face_model, det_model, reid_model, emp_db, src)

    else:
        run_workplace_monitoring(src, face_detector, face_aligner, face_model, det_model, tracker, reid_model, emp_db, cfg, ppe_model)
if __name__ == "__main__":
    main()
