import numpy as np
import yaml
import cv2

def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
    return float(np.dot(a, b) / denom)

def IoU(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(1.0, (boxA[2] - boxA[0])) * max(1.0, (boxA[3] - boxA[1]))
    boxBArea = max(1.0, (boxB[2] - boxB[0])) * max(1.0, (boxB[3] - boxB[1]))
    return float(inter / (boxAArea + boxBArea - inter + 1e-6))

def crop_xyxy(frame, box):
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    x1 = max(0, min(w-1, x1)); x2 = max(0, min(w, x2))
    y1 = max(0, min(h-1, y1)); y2 = max(0, min(h, y2))
    return frame[y1:y2, x1:x2].copy()

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
