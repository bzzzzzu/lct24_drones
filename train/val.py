from ultralytics import YOLO
import os
import json
from PIL import Image, ImageDraw

# Показывает все что напредсказывалось, по одной картинке в отдельную папку
# Используется для дебага модели
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
        for p in preds:
            if not p['image_id'] in batched_preds:
                batched_preds[p['image_id']] = []
            batched_preds[p['image_id']].append(p)
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
    # n/s/m/l/x
    #model = YOLO("yolov8s.pt")
    #model = YOLO("runs/detect/train6/weights/best.pt")
    #model = YOLO("runs/detect/train7/weights/best.pt")
    #model = YOLO("runs/detect/train8/weights/best.pt")
    #model = YOLO("runs/detect/train19/weights/best.pt")
    model = YOLO("runs/detect/train21/weights/best.pt")

    model.cuda()
    model.predict('datasets/drones_clean/filter_val.txt', batch=16, imgsz=1280, conf=0.25, plots=True, save_json=True, save_txt=True, workers=4, agnostic_nms=True, save_conf=True)

    #save_val_bbox('runs/detect/predict2/labels/', 'relabel_fixed_birds_val_07')