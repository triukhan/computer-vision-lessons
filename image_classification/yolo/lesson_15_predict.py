import numpy as np
from ultralytics import YOLO

model = YOLO('runs/classify/train3/weights/last.pt')

results = model('image_classification/yolo/data/val/rain/rain187.jpg')
names = results[0].names
probs = results[0].probs.data.tolist()

print(names[np.argmax(probs)])
