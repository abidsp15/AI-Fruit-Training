from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="dataset/data.yaml",
    epochs=10,
    imgsz=320,
    batch=16,
    device="mps",
    name="fruit_detector"
)