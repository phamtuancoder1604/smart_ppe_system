
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from models.utils import cosine_similarity

class Track:
    def __init__(self, box, tid, embedding=None):
        self.box = np.array(box[:4], dtype=float)
        self.tid = int(tid)
        self.miss = 0
        self.prev_center = self._center(self.box)
        self.embedding_last = embedding  # vector ReID cuối

    def _center(self, b):
        return np.array([(b[0]+b[2])/2.0, (b[1]+b[3])/2.0], dtype=float)


class TrackerOCSortLite:
    def __init__(self, max_age=30, iou_thresh=0.5, reid_thresh=0.7):
        self.max_age = max_age
        self.iou_thresh = iou_thresh
        self.reid_thresh = reid_thresh
        self.tracks = []
        self.next_id = 1
        self.prev_gray = None
    def _iou(self, a, b):
        xA, yA = max(a[0], b[0]), max(a[1], b[1])
        xB, yB = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, xB-xA) * max(0, yB-yA)
        areaA = max(1.0,(a[2]-a[0]))*max(1.0,(a[3]-a[1]))
        areaB = max(1.0,(b[2]-b[0]))*max(1.0,(b[3]-b[1]))
        return inter / (areaA + areaB - inter + 1e-6)

    def _predict_with_flow(self, frame_gray):
        if self.prev_gray is None or len(self.tracks) == 0:
            self.prev_gray = frame_gray.copy()
            return
        prev_pts = np.array([t.prev_center for t in self.tracks], dtype=np.float32).reshape(-1,1,2)
        next_pts, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, frame_gray, prev_pts, None)
        for i, t in enumerate(self.tracks):
            if st[i][0] == 1:
                flow = next_pts[i,0,:] - prev_pts[i,0,:]
                t.box[0::2] += flow[0]
                t.box[1::2] += flow[1]
                t.prev_center = t.prev_center + flow
        self.prev_gray = frame_gray.copy()

    def update(self, frame, detections, reid_model=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._predict_with_flow(gray)

        dets = np.array([d[:4] for d in detections], dtype=float) if len(detections)>0 else np.zeros((0,4), dtype=float)
        cost = np.ones((len(self.tracks), len(dets)), dtype=float)

        # B1: match theo IoU
        for i, tr in enumerate(self.tracks):
            for j, db in enumerate(dets):
                iou = self._iou(tr.box, db)
                cost[i,j] = 1.0 - iou

        assigned_tracks, assigned_dets = [], []
        if len(self.tracks) > 0 and len(dets) > 0:
            r, c = linear_sum_assignment(cost)
            for i, j in zip(r, c):
                if 1.0 - cost[i,j] >= self.iou_thresh:
                    assigned_tracks.append((i, j))

        used_dets = set()
        for (i, j) in assigned_tracks:
            emb = None
            if reid_model is not None:
                crop = frame[int(dets[j][1]):int(dets[j][3]), int(dets[j][0]):int(dets[j][2])]
                emb = reid_model.get_body_embedding(crop)
            self.tracks[i].box = dets[j].copy()
            if emb is not None:
                self.tracks[i].embedding_last = emb
            self.tracks[i].miss = 0
            self.tracks[i].prev_center = np.array([(dets[j][0]+dets[j][2])/2.0, (dets[j][1]+dets[j][3])/2.0], dtype=float)
            used_dets.add(j)

        # B2: ReID match cho các detection chưa dùng
        for j in range(len(dets)):
            if j in used_dets:
                continue
            if reid_model is None:
                continue
            crop = frame[int(dets[j][1]):int(dets[j][3]), int(dets[j][0]):int(dets[j][2])]
            emb = reid_model.get_body_embedding(crop)
            best_sim, best_idx = 0.0, -1
            for i, tr in enumerate(self.tracks):
                if tr.miss > 0 and tr.embedding_last is not None:
                    sim = cosine_similarity(emb, tr.embedding_last)
                    if sim > best_sim:
                        best_sim, best_idx = sim, i
            if best_sim >= self.reid_thresh and best_idx >= 0:
                print(f"[ReID] Khôi phục ID {self.tracks[best_idx].tid} (sim={best_sim:.2f})")
                self.tracks[best_idx].box = dets[j].copy()
                self.tracks[best_idx].miss = 0
                self.tracks[best_idx].embedding_last = emb
                used_dets.add(j)

        # B3: Thêm track mới
        for j in range(len(dets)):
            if j not in used_dets:
                emb = None
                if reid_model is not None:
                    crop = frame[int(dets[j][1]):int(dets[j][3]), int(dets[j][0]):int(dets[j][2])]
                    emb = reid_model.get_body_embedding(crop)
                self.tracks.append(Track(dets[j], self.next_id, emb))
                self.next_id += 1

        # B4: Giữ track còn sống
        outputs, alive_tracks = [], []
        for t in self.tracks:
            t.miss += 1
            if t.miss <= self.max_age:
                alive_tracks.append(t)
                outputs.append(np.array([t.box[0], t.box[1], t.box[2], t.box[3], t.tid], dtype=float))
        self.tracks = alive_tracks
        return outputs
