
from insightface.app import FaceAnalysis

class FaceDetector:
    def __init__(self, model_path="buffalo_l", det_size=(640,640)):
        self.app = FaceAnalysis(name=model_path, providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=det_size)

    def detect_faces(self, frame, conf_thresh=0.5):
        faces = self.app.get(frame)
        boxes = []
        for f in faces:
            if f.det_score >= conf_thresh:
                x1, y1, x2, y2 = f.bbox.astype(int)
                boxes.append([x1, y1, x2, y2, float(f.det_score)])
        return boxes

