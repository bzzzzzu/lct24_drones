from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from PIL import Image
import numpy as np

# n/s/m/l/x
#model = YOLO("yolov8s.pt")
model = YOLO("runs/detect/train6/weights/best.pt")
#model = YOLO("yolov10s.pt")
model.cuda()

print(model)

model.info()

#results = model('datasets/drones/images/-_-_-114-_jpg.rf.9ff44cd7d727932af9264515fc35a910.jpg', conf=0.25)
#results = model("https://ultralytics.com/images/bus.jpg", conf=0.25)

#img = Image.open("datasets/drones/images/-_-_-114-_jpg.rf.9ff44cd7d727932af9264515fc35a910.jpg")
img = Image.open("datasets/drones/images/dji_phantom_mountain_cross_2422.png")
results = model(img, conf=0.1)

for r in results:
    annotator = Annotator(img)
    boxes = r.boxes
    for box in boxes:
        b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
        c = box.cls
        annotator.box_label(b, f'{model.names[int(c)]}, conf: {np.round(box.conf.item(), 3)}')
    img = annotator.result()
    Image._show(Image.fromarray(img))
print(results)