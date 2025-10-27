import cv2
import os
from glob import glob
from models.person_detection import PersonDetector
from models.face_detector import FaceDetector
from models.utils import load_yaml

def extract_faces_bodies(
    input_root="data/raw_people",
    output_root="data/people_custom",
    config_path="config/config.yaml",
    min_face=80,
    min_body_area=8000,
    max_per_person=10
):
    cfg = load_yaml(config_path)
    models_cfg = cfg.get("models", {})

    person_model_path = models_cfg.get("person", "models/weights/yolov8n.pt")
    face_model_path   = models_cfg.get("face_detector", "models/weights/scrfd_person_2.5g.onnx")

    print(f"[CONFIG] Person model: {person_model_path}")
    print(f"[CONFIG] Face detector: {face_model_path}")
    face_det = FaceDetector(model_path=face_model_path)
    person_det = PersonDetector(model_path=person_model_path)
    os.makedirs(output_root, exist_ok=True)
    people_dirs = sorted(os.listdir(input_root))
    for idx, person in enumerate(people_dirs):
        person_dir = os.path.join(input_root, person)
        if not os.path.isdir(person_dir):
            continue
        print(f"[INFO] Processing {person} ...")
        imgs = sorted(glob(os.path.join(person_dir, "*.jpg")))[:max_per_person]
        out_face = os.path.join(output_root, person, "face")
        out_body = os.path.join(output_root, person, "body")
        os.makedirs(out_face, exist_ok=True)
        os.makedirs(out_body, exist_ok=True)
        f_id, b_id = 0, 0
        for img_path in imgs:
            img = cv2.imread(img_path)
            if img is None:
                continue
            persons = person_det.detect_persons(img, conf=0.5)
            for (x1, y1, x2, y2, conf) in persons:
                if (x2 - x1) * (y2 - y1) < min_body_area:
                    continue
                body_crop = img[int(y1):int(y2), int(x1):int(x2)]
                cv2.imwrite(os.path.join(out_body, f"{person}_body_{b_id:03d}.jpg"), body_crop)
                b_id += 1

            faces = face_det.detect_faces(img, conf_thresh=0.25)

            for (fx1, fy1, fx2, fy2, conf) in faces:
                if (fx2 - fx1) < min_face or (fy2 - fy1) < min_face:
                    continue
                face_crop = img[int(fy1):int(fy2), int(fx1):int(fx2)]
                cv2.imwrite(os.path.join(out_face, f"{person}_face_{f_id:03d}.jpg"), face_crop)
                f_id += 1

        print(f"[DONE] {person}: {f_id} faces, {b_id} bodies saved.\n")

if __name__ == "__main__":
    extract_faces_bodies("data/raw_people", "data/people_custom", config_path="config/config.yaml")
