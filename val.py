from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from PIL import Image
import numpy as np
import requests
import os

# n/s/m/l/x
#model = YOLO("yolov8s.pt")
#model = YOLO("runs/detect/train6/weights/best.pt")
model = YOLO("runs/detect/train7/weights/best.pt")
#model = YOLO("yolov10s.pt")
model.cuda()

#print(model)

model.info()

#results = model('datasets/drones/images/-_-_-114-_jpg.rf.9ff44cd7d727932af9264515fc35a910.jpg', conf=0.25)
#results = model("https://ultralytics.com/images/bus.jpg", conf=0.25)

#img = Image.open("datasets/drones/images/-_-_-114-_jpg.rf.9ff44cd7d727932af9264515fc35a910.jpg")
#img = Image.open("datasets/drones/images/dji_phantom_mountain_cross_2422.png")
#results = model(img, conf=0.1)

val_image_list = ['https://s3.amazonaws.com/airborne-obj-detection-challenge-training/part1/Images/76903185ad9a44739f36c2ea94f08dcd/156076750541247893376903185ad9a44739f36c2ea94f08dcd.png',
                  'https://s3.amazonaws.com/airborne-obj-detection-challenge-training/part1/Images/348fe7f284274e53bd43e3c8e4c0a574/1555334625674051292348fe7f284274e53bd43e3c8e4c0a574.png',
                  'https://s3.amazonaws.com/airborne-obj-detection-challenge-training/part1/Images/00c28c5653b04bda9c6cfc255ac06494/156276941745694162700c28c5653b04bda9c6cfc255ac06494.png',
                  'https://s3.amazonaws.com/airborne-obj-detection-challenge-training/part1/Images/2a85b429004c4d628fffbcfbe368890d/15676020345424113582a85b429004c4d628fffbcfbe368890d.png',
                  'https://s3.amazonaws.com/airborne-obj-detection-challenge-training/part1/Images/37b867ba11ef4e3694a8178026f9cbc8/157303416099343382937b867ba11ef4e3694a8178026f9cbc8.png']

for val_image in val_image_list:
    raw_name = ''.join(str.split(val_image, '/')[7:])
    if not os.path.exists(raw_name):
        img = Image.open(requests.get(val_image, stream=True).raw).convert('RGB')
        img.save(raw_name)
    else:
        img = Image.open(raw_name)

    results = model(img, conf=0.1, imgsz=1024)
    for r in results:
        annotator = Annotator(img)
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
            c = box.cls
            annotator.box_label(b, f'{model.names[int(c)]}, conf: {np.round(box.conf.item(), 3)}')
        img = annotator.result()
        Image._show(Image.fromarray(img))
    #print(results)