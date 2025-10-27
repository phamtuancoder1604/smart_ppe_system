from ultralytics import YOLO
import cv2

model_path = "runs/detect/train7/weights/last.pt"
model = YOLO(model_path)

image_path = "data/yolo_dataset/val/images/hard_hat_workers47.png"#"data/raw_voc_dataset/images/hard_hat_workers334.png"#'data/construction-ppe/images/test/image772.jpg'

results = model.predict(source=image_path, conf=0.3, save=True, show=True)

for result in results:
    annotated_frame = result.plot()
    cv2.imshow("PPE Detection Result", annotated_frame)
    print("\nDetected objects:")
    ignore_classes = ["none"]
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        name = model.names[cls]
        if name in ignore_classes:
            continue
        print(f"- {name}: {conf:.2f}")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q hoặc ESC
            break

cv2.destroyAllWindows()