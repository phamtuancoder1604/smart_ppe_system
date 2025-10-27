# Optional helper: hints to fetch public weights (edit URLs to your mirrors).
# This script prints guidance; it does not auto-download by default due to licensing/availability.
print("""
Place weights under models/weights/ :
- ArcFace MobileFaceNet (ONNX): models/weights/arcface_mobilefaceenet.onnx
- YOLOv8 COCO person: models/weights/yolov8n.pt  (pip: ultralytics -> yolo download)
- YOLOv8 PPE fine-tuned: models/weights/yolov8_ppe_best.pt  (your trained checkpoint)
- (Optional) OSNet ONNX: models/weights/osnet_x1_0.onnx  (else torchreid pretrained is used)

Example to get YOLOv8n:
  from ultralytics import YOLO
  model = YOLO('yolov8n.pt')  # auto-downloads to cache

ArcFace ONNX (InsightFace) and OSNet ONNX: follow their repos to export to ONNX.
""")
