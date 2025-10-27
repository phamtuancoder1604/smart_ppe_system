import cv2
import time
import numpy as np
from ultralytics import YOLO
from models.tracking import TrackerOCSortLite
from models.reid import ReIDModel
from database.employee_db import EmployeeDB


def test_tracking(video_source=0):
    print("[INFO] Starting Tracking + ReID + YOLOv8m ...")

    model = YOLO("models/weights/yolov8l.pt")

    tracker = TrackerOCSortLite(max_age=30, iou_thresh=0.5, reid_thresh=0.7)
    reid_model = ReIDModel()
    emp_db = EmployeeDB()

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("[ERROR] Cannot open video source:", video_source)
        return

    id_map = {}
    fps_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False, conf=0.2, iou=0.6, imgsz=960)
        detections = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls == 0 and conf > 0.3:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append([x1, y1, x2, y2, conf])

        if len(detections) == 0:
            cv2.imshow("Tracking + ReID", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        detections = np.array(detections)
        tracks = tracker.update(frame, detections, reid_model)

        for t in tracks:
            x1, y1, x2, y2, tid = map(int, t[:5])
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            if tid not in id_map:
                emb = reid_model.get_body_embedding(crop)
                emp_id = reid_model.match_employee(emb, emp_db.get_body_db(), threshold=0.75)
                id_map[tid] = emp_id if emp_id else None

            emp_name = id_map.get(tid, "Unknown") or "Unknown"
            color = (0, 255, 0) if emp_name != "Unknown" else (0, 0, 255)
            label = f"ID:{tid} ({emp_name})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        fps = 1.0 / (time.time() - fps_time)
        fps_time = time.time()
        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("YOLOv8m + OC-SORT + ReID", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_tracking("data/videos/kids.mp4")
