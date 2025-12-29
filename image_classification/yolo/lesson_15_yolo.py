from ultralytics import YOLO

model = YOLO('yolo11n-cls.pt')

results = model.train(data='data', epochs=20, imgsz=64)
