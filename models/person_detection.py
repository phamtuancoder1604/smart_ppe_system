import os
import numpy as np
from ultralytics import YOLO
import cv2
class PersonDetector:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            print(f"[PersonDetector] WARNING: model not found: {model_path}. Use a valid YOLOv8 *.pt")
        self.model = YOLO(model_path)

    def detect_persons(self, frame, conf=0.5, min_size=100,iou_thresh = 0.4):
        results = self.model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            conf_val = float(box.conf[0])
            if cls == 0 and conf_val >= conf:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                if w * h > min_size * min_size:  # bỏ box nhỏ
                    detections.append([x1, y1, x2, y2, conf_val])
        if len(detections) > 0:
            boxes = np.array([d[:4] for d in detections])
            scores = np.array([d[4] for d in detections])

            indices = cv2.dnn.NMSBoxes(
                bboxes=boxes.tolist(),
                scores=scores.tolist(),
                score_threshold=conf,
                nms_threshold=iou_thresh
            )

            if len(indices) > 0:
                if isinstance(indices, np.ndarray):
                    indices = indices.flatten().tolist()
                elif isinstance(indices[0], (list, tuple)):
                    indices = [i[0] for i in indices]
                detections = [detections[i] for i in indices]

        return detections


