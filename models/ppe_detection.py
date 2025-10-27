import os
from ultralytics import YOLO
import numpy as np
import cv2
class PPEDetector:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            print(f"[PPEDetector] WARNING: PPE model not found: {model_path}. Provide a fine-tuned YOLOv8 *.pt")
        self.model = YOLO(model_path)

    def detect_ppe(self, frame, conf=0.4,iou_thresh=0.4):
        res = self.model(frame, verbose=False)[0]
        out = []
        names = res.names
        for i in range(len(res.boxes)):
            cls = int(res.boxes.cls[i].item())
            c = float(res.boxes.conf[i].item())
            label = names.get(cls, str(cls))
            if c >= conf and label in ['helmet', 'vest', 'mask', 'glove', 'goggles']:
                x1, y1, x2, y2 = res.boxes.xyxy[i].cpu().numpy().tolist()
                out.append(((x1,y1,x2,y2), label, c))
        if len(out) > 0:
            boxes = np.array([o[0] for o in out])
            scores = np.array([o[2] for o in out])

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
                out = [out[i] for i in indices]

        return out
