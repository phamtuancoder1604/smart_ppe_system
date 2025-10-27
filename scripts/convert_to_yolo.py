import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import shutil


SPLIT_DIR = "data/split_dataset"
YOLO_DIR = "data/yolo_dataset"

for subset in ["train", "val"]:
    os.makedirs(os.path.join(YOLO_DIR, subset, "images"), exist_ok=True)
    os.makedirs(os.path.join(YOLO_DIR, subset, "labels"), exist_ok=True)

classes = ["person", "helmet"]  

def convert_bbox(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh

for subset in ["train", "val"]:
    xml_dir = os.path.join(SPLIT_DIR, subset, "annotations")
    img_dir = os.path.join(SPLIT_DIR, subset, "images")
    out_img_dir = os.path.join(YOLO_DIR, subset, "images")
    out_label_dir = os.path.join(YOLO_DIR, subset, "labels")

    for xml_file in tqdm(os.listdir(xml_dir), desc=f"Converting {subset}"):
        if not xml_file.endswith(".xml"):
            continue

        xml_path = os.path.join(xml_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        img_name = root.find("filename").text
        img_src = os.path.join(img_dir, img_name)
        img_dst = os.path.join(out_img_dir, img_name)
        if os.path.exists(img_src) and not os.path.exists(img_dst):
            shutil.copy(img_src, img_dst)

        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)

        txt_name = os.path.splitext(xml_file)[0] + ".txt"
        out_file = open(os.path.join(out_label_dir, txt_name), "w")

        for obj in root.iter("object"):
            cls = obj.find("name").text
            if cls not in classes:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find("bndbox")
            b = (
                float(xmlbox.find("xmin").text),
                float(xmlbox.find("xmax").text),
                float(xmlbox.find("ymin").text),
                float(xmlbox.find("ymax").text),
            )
            bb = convert_bbox((w, h), b)
            out_file.write(f"{cls_id} {' '.join(map(str, bb))}\n")

        out_file.close()

print("Hoan tat chuyen doi XML sang YOLO format.")
print("Dataset YOLO duoc luu tai:", YOLO_DIR)
