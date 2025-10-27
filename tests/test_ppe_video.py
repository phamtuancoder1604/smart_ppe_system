import cv2
from ultralytics import YOLO


model = YOLO("runs/detect/train7/weights/best.pt")


video_path = "data/videos/video_test3.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("[ERROR] Cannot open video:", video_path)
    exit()


save_output = True
if save_output:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter("output_ppe.mp4", fourcc, fps, (w, h))


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference YOLO
    results = model(frame, conf=0.4, verbose=False)
    annotated_frame = results[0].plot()
    window_name = "PPE Detection - Press Q to exit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)
    cv2.imshow(window_name, annotated_frame)


    cv2.imshow("PPE Detection - Press Q to exit", annotated_frame)


    if save_output:
        out.write(annotated_frame)


    if cv2.waitKey(1) & 0xFF in [ord("q"), 27]:
        break


cap.release()
if save_output:
    out.release()
cv2.destroyAllWindows()
print("[INFO] Done. Result saved to output_ppe.mp4")
