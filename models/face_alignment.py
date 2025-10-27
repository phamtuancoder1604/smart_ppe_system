import cv2
import numpy as np
import onnxruntime as ort

class FaceAligner:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    def align(self, img, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        face = img[y1:y2, x1:x2]
        if face is None or face.size == 0:
            print("[FaceAligner] Warning: empty face crop, skipping.")
            return None
        inp = cv2.resize(face, (192, 192)).astype(np.float32)
        inp = (inp / 255.0 - 0.5) / 0.5
        inp = np.transpose(inp, (2, 0, 1))[None, ...]
        out = self.session.run(None, {self.session.get_inputs()[0].name: inp})[0][0]
        lm = out.reshape(-1, 2)
        landmarks = lm.copy()
        landmarks[:, 0] = landmarks[:, 0] * (x2 - x1) / 192 + x1
        landmarks[:, 1] = landmarks[:, 1] * (y2 - y1) / 192 + y1
        five = np.float32([
            landmarks[96], landmarks[97],
            landmarks[54], landmarks[76], landmarks[82]
        ])
        ref = np.float32([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ])

        M = cv2.estimateAffinePartial2D(five, ref)[0]
        if M is None:
            print("[FaceAligner] Warning: failed to estimate affine.")
            return None

        aligned = cv2.warpAffine(img, M, (112, 112), borderValue=0)
        return aligned
