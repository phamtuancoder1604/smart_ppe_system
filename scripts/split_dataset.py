import os
import shutil
from sklearn.model_selection import train_test_split

ROOT_DIR = "data/raw_voc_dataset"
IMAGES_DIR = os.path.join(ROOT_DIR, "images")
ANNOTATIONS_DIR = os.path.join(ROOT_DIR, "annotations")
OUTPUT_DIR = "data/split_dataset"

for folder in ["train/images", "train/annotations", "val/images", "val/annotations"]:
    os.makedirs(os.path.join(OUTPUT_DIR, folder), exist_ok=True)

image_files = [
    f for f in os.listdir(IMAGES_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]
xml_files = [
    f for f in os.listdir(ANNOTATIONS_DIR)
    if f.lower().endswith(".xml")
]

print(f"Tổng số ảnh tìm thấy: {len(image_files)}")
print(f"Tổng số annotation tìm thấy: {len(xml_files)}")

valid_pairs = []
for img in image_files:
    base = os.path.splitext(img)[0].lower()
    for xml in xml_files:
        if os.path.splitext(xml)[0].lower() == base:
            valid_pairs.append(img)
            break

print(f"Số cặp ảnh–annotation hợp lệ: {len(valid_pairs)}")

if len(valid_pairs) == 0:
    raise ValueError("Không tìm thấy cặp ảnh–XML nào trùng tên. Hãy kiểm tra lại tên file hoặc định dạng ảnh (.jpg/.png).")

train_files, val_files = train_test_split(valid_pairs, test_size=0.2, random_state=42)

def copy_pairs(file_list, subset):
    for f in file_list:
        base = os.path.splitext(f)[0]
        img_src = os.path.join(IMAGES_DIR, f)
        xml_src = os.path.join(ANNOTATIONS_DIR, f"{base}.xml")
        if not os.path.exists(xml_src):
            xml_src = os.path.join(ANNOTATIONS_DIR, f"{base.lower()}.xml")
        if not os.path.exists(img_src) or not os.path.exists(xml_src):
            print(f"Bỏ qua file không tìm thấy: {f}")
            continue
        shutil.copy(img_src, os.path.join(OUTPUT_DIR, f"{subset}/images", os.path.basename(img_src)))
        shutil.copy(xml_src, os.path.join(OUTPUT_DIR, f"{subset}/annotations", os.path.basename(xml_src)))

print("Đang copy dữ liệu...")
copy_pairs(train_files, "train")
copy_pairs(val_files, "val")

print("\nHoàn tất chia dataset!")
print(f"Train: {len(train_files)} ảnh")
print(f"Val:   {len(val_files)} ảnh")
print(f"Dữ liệu đã lưu tại: {OUTPUT_DIR}")
