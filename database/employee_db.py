import json, os, numpy as np
from models.utils import cosine_similarity

class EmployeeDB:
    def __init__(self, face_path='data/embeddings/face_embeddings.json',
                 body_path='data/embeddings/body_embeddings.json'):
        self.face_path = face_path
        self.body_path = body_path
        self.faces = self._load_json_faces(face_path)   # multi-sample
        self.bodies = self._load_json_simple(body_path) # 1 emb/body

    # ---------- IO helpers ----------
    def _load_json_faces(self, path):
        if not os.path.exists(path):
            return {}
        try:
            with open(path, 'r') as f:
                raw = json.load(f)
        except json.JSONDecodeError:
            return {}
        faces = {}
        for k, v in raw.items():
            # backward-compat: v có thể là list(float) (1 emb) hoặc dict{"samples":...}
            if isinstance(v, dict) and "samples" in v:
                samples = [np.array(s, dtype=np.float32) for s in v["samples"]]
            else:
                samples = [np.array(v, dtype=np.float32)]
            centroid = np.mean(np.stack(samples, axis=0), axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-6)
            faces[k] = {"samples": samples, "centroid": centroid}
        return faces

    def _load_json_simple(self, path):
        if not os.path.exists(path):
            return {}
        try:
            with open(path, 'r') as f:
                raw = json.load(f)
        except json.JSONDecodeError:
            return {}
        return {k: np.array(v, dtype=np.float32) for k, v in raw.items()}

    def _save_json_faces(self):
        os.makedirs(os.path.dirname(self.face_path), exist_ok=True)
        serial = {k: {"samples": [s.tolist() for s in v["samples"]]} for k, v in self.faces.items()}
        with open(self.face_path, 'w') as f:
            json.dump(serial, f, indent=2)

    def _save_json_bodies(self):
        os.makedirs(os.path.dirname(self.body_path), exist_ok=True)
        serial = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in self.bodies.items()}
        with open(self.body_path, 'w') as f:
            json.dump(serial, f, indent=2)

    # ---------- Add / Update ----------
    def add_employee(self, emp_id, face_emb=None, body_emb=None):
        if face_emb is not None:
            fe = np.array(face_emb, dtype=np.float32)
            fe = fe / (np.linalg.norm(fe) + 1e-6)
            self.faces[emp_id] = {"samples": [fe], "centroid": fe}
            self._save_json_faces()
        if body_emb is not None:
            be = np.array(body_emb, dtype=np.float32)
            be = be / (np.linalg.norm(be) + 1e-6)
            self.bodies[emp_id] = be
            self._save_json_bodies()

    def add_face_samples(self, emp_id, emb_list):
        emb_list = [np.array(e, dtype=np.float32) for e in emb_list]
        emb_list = [e / (np.linalg.norm(e) + 1e-6) for e in emb_list]
        if emp_id not in self.faces:
            self.faces[emp_id] = {"samples": [], "centroid": None}
        self.faces[emp_id]["samples"].extend(emb_list)
        centroid = np.mean(np.stack(self.faces[emp_id]["samples"], axis=0), axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-6)
        self.faces[emp_id]["centroid"] = centroid
        self._save_json_faces()

    def get_face_db(self):
        if not self.faces:
            return {}
        return {eid: data.get("centroid", np.zeros(512, dtype=np.float32)) for eid, data in self.faces.items()}

    def get_body_db(self): return self.bodies

    # ---------- Matching ----------
    def match_face(self, face_emb, threshold=0.92, use_centroid=True):
        q = np.array(face_emb, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-6)

        best_id, best_score = None, -1.0
        for emp_id, pack in self.faces.items():
            if use_centroid and pack.get("centroid") is not None:
                s = cosine_similarity(q, pack["centroid"])
                if s > best_score:
                    best_id, best_score = emp_id, s
            else:
                # max over samples
                scores = [cosine_similarity(q, s) for s in pack.get("samples", [])]
                if scores:
                    smax = float(np.max(scores))
                    if smax > best_score:
                        best_id, best_score = emp_id, smax
        if best_score >= threshold:
            return best_id, float(best_score)
        return None, float(best_score)

    def match_body(self, body_emb, threshold=0.82):
        q = np.array(body_emb, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-6)
        best_id, best_score = None, -1.0
        for emp_id, e in self.bodies.items():
            s = cosine_similarity(q, e)
            if s > best_score:
                best_id, best_score = emp_id, s
        if best_score >= threshold:
            return best_id, float(best_score)
        return None, float(best_score)
