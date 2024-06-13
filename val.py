from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from PIL import Image
import numpy as np
import requests
import os
import json
from PIL import Image, ImageDraw

def save_val_bbox(path, dir):
    names = {
        0: 'copter',
        1: 'plane',
        2: 'heli',
        3: 'bird',
        4: 'winged',
    }
    batched_preds = {}

    if '.json' in path:
        with open(path, 'r') as jsf:
            preds = json.load(jsf)
        last_image = ''
        img = ''
        for p in preds:
            if not p['image_id'] in batched_preds:
                batched_preds[p['image_id']] = []
            batched_preds[p['image_id']].append(p)
        #print(batched_preds)
    else:
        for f in os.listdir(path):
            with open(f'{path}{f}', 'r') as lbf:
                batched_preds[f[:-4]] = []
                preds = lbf.readlines()
                for p in preds:
                    split = str.split(p, ' ')
                    id = int(split[0])
                    box = [float(split[1]), float(split[2]), float(split[3]), float(split[4])]
                    conf = float(split[5])
                    batched_preds[f[:-4]].append(
                        {'bbox': box,
                         'score': conf,
                         'category_id': id,
                        }
                    )

    for p in batched_preds:
        ext = ''
        if os.path.exists(f'datasets/drones_clean/images/{p}.jpg'): ext = '.jpg'
        if os.path.exists(f'datasets/drones_clean/images/{p}.JPEG'): ext = '.JPEG'
        if os.path.exists(f'datasets/drones_clean/images/{p}.png'): ext = '.png'
        img = Image.open(f'datasets/drones_clean/images/{p}{ext}').convert('RGB')
        draw = ImageDraw.Draw(img)
        for detect in batched_preds[p]:
            print(detect)
            if detect['bbox'][0] < 1:
                ybox = [
                    int((detect['bbox'][0] - detect['bbox'][2] / 2) * img.width),
                    int((detect['bbox'][1] - detect['bbox'][3] / 2) * img.height),
                    int((detect['bbox'][0] + detect['bbox'][2] / 2) * img.width),
                    int((detect['bbox'][1] + detect['bbox'][3] / 2) * img.height),
                ]
                draw.rectangle([(ybox[0], ybox[1]), (ybox[2], ybox[3])], outline=(0, 0, 255))
                draw.text((ybox[0], ybox[1] - 26), f'{names[detect["category_id"]]}, {detect["score"]}', font_size=24)
            else:
                draw.rectangle([(detect['bbox'][0], detect['bbox'][1]), (detect['bbox'][0] + detect['bbox'][2], detect['bbox'][1] + detect['bbox'][3])], outline=(0, 0, 255))
                draw.text((detect['bbox'][0], detect['bbox'][1] - 26), f'{names[detect["category_id"]]}, {detect["score"]}', font_size=24)
        img.save(f'datasets/val_results/{dir}/{p}{ext}')
        print(img.size)

if __name__ == '__main__':
    #save_val_bbox('runs/detect/val10/predictions.json', 'relabel_val')
    save_val_bbox('runs/detect/predict/labels/', 'relabel_fixed_birds_val')
    exit()

    # n/s/m/l/x
    #model = YOLO("yolov8s.pt")
    #model = YOLO("runs/detect/train6/weights/best.pt")
    #model = YOLO("runs/detect/train7/weights/best.pt")
    #model = YOLO("runs/detect/train8/weights/best.pt")
    model = YOLO("runs/detect/train19/weights/best.pt")

    #model = YOLO("yolov10s.pt")
    model.cuda()
    model.predict('datasets/drones_clean/filter_val.txt', batch=16, imgsz=1024, conf=0.25, plots=True, save_json=True, save_txt=True, workers=4, agnostic_nms=True, save_conf=True)
    exit()

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