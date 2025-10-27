"""
convert_voc_to_yolo.py
Chuyển đổi annotation PascalVOC (.xml) sang YOLOv8 format (.txt)
---------------------------------------------------------------
Cách dùng:
    python scripts/convert_voc_to_yolo.py \
        --voc_dir data/raw_voc_dataset \
        --yolo_dir data/dataset_ppe \
        --classes person helmet vest
"""

import os
import xml.etree.ElementTree as ET
import argparse
from tqdm import tqdm

def convert_bbox(size, box):
    """Chuyển tọa độ PascalVOC (x_min, y_min, x_max, y_max) → YOLO format (x_center, y_center, w, h)."""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x_center, y_center, w, h = x_center * dw, y_center * dh, w * dw, h * dh
    return (x_center, y_center, w, h)

def convert_annotations(voc_dir, yolo_dir, classes):
    """Chuyển đổi toàn bộ dataset từ PascalVOC sang YOLO format."""
    ann_root = os.path.join(voc_dir, "annotations")
    img_root = os.path.join(voc_dir, "images")

    for subset in ["train", "val"]:
        voc_ann_dir = os.path.join(ann_root, subset)
        voc_img_dir = os.path.join(img_root, subset)
        yolo_img_dir = os.path.join(yolo_dir, "images", subset)
        yolo_lbl_dir = os.path.join(yolo_dir, "labels", subset)
        os.makedirs(yolo_img_dir, exist_ok=True)
        os.makedirs(yolo_lbl_dir, exist_ok=True)

        xml_files = [f for f in os.listdir(voc_ann_dir) if f.endswith(".xml")]
        for xml_file in tqdm(xml_files, desc=f"Converting {subset}"):
            xml_path = os.path.join(voc_ann_dir, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            img_filename = root.find("filename").text
            img_path_src = os.path.join(voc_img_dir, img_filename)
            img_path_dst = os.path.join(yolo_img_dir, img_filename)

            # Copy ảnh sang thư mục YOLO (nếu chưa có)
            if os.path.exists(img_path_src) and not os.path.exists(img_path_dst):
                try:
                    import shutil
                    shutil.copy(img_path_src, img_path_dst)
                except Exception as e:
                    print(f"[WARN] Copy failed: {e}")

            size = root.find("size")
            if size is None:
                continue
            w = int(size.find("width").text)
            h = int(size.find("height").text)

            label_path = os.path.join(yolo_lbl_dir, xml_file.replace(".xml", ".txt"))
            with open(label_path, "w") as out_file:
                for obj in root.findall("object"):
                    cls_name = obj.find("name").text
                    if cls_name not in classes:
                        continue
                    cls_id = classes.index(cls_name)
                    xml_box = obj.find("bndbox")
                    b = (
                        float(xml_box.find("xmin").text),
                        float(xml_box.find("xmax").text),
                        float(xml_box.find("ymin").text),
                        float(xml_box.find("ymax").text),
                    )
                    bb = convert_bbox((w, h), b)
                    out_file.write(f"{cls_id} {' '.join([f'{a:.6f}' for a in bb])}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--voc_dir", type=str, required=True, help="Thư mục chứa dataset PascalVOC (gốc)")
    parser.add_argument("--yolo_dir", type=str, required=True, help="Thư mục đích để lưu YOLO-format dataset")
    parser.add_argument("--classes", nargs="+", required=True, help="Danh sách class theo thứ tự YOLO")
    args = parser.parse_args()

    print(f"[INFO] Chuyển đổi dataset từ {args.voc_dir} → {args.yolo_dir}")
    print(f"[INFO] Classes: {args.classes}")
    convert_annotations(args.voc_dir, args.yolo_dir, args.classes)
    print("[DONE] Chuyển đổi hoàn tất!")
