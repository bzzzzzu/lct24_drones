import os
import shutil
from PIL import Image
from ultralytics.utils.plotting import Annotator
import numpy as np
import ultralytics.utils.ops as ops
import json

def get_image_list(list_of_files):
    with open('clean_list.txt', 'w', encoding='utf-8') as clf:
        for l in list_of_files:
            clf.writelines(f'{l}\n')

def get_image_bboxes(list_of_files):
    for f in list_of_files:
        if not os.path.exists(f'datasets/drones_clean/images_with_bbox/{f}'):
            img = Image.open(f'datasets/drones_clean/images/{f}').convert("RGB")
            ann = Annotator(img, font_size=16, line_width=1)
            label_name = f"{'.'.join(str.split(f, '.')[:-1])}.txt"
            with open(f'datasets/drones/labels/{label_name}', 'r') as label_f:
                lines = label_f.readlines()
                for l in lines:
                    box = np.array(str.split(str.strip(l), ' ')[1:]).astype('float64')
                    box = list(ops.xywhn2xyxy(box, img.width, img.height, 0, 0))
                    ann.box_label(box=box, color=(0, 0, 255), txt_color=(255, 255, 255), label=str.split(str.split(str.strip(l), ' ')[0], '.')[0])
                ann.save(f'datasets/drones_clean/images_with_bbox/{f}')

def get_folder_bboxes(folder, save_folder):
    for f in os.listdir(folder):
        img = Image.open(f'{folder}/{f}').convert("RGB")
        ann = Annotator(img, font_size=16, line_width=1)
        label_name = f"{'.'.join(str.split(f, '.')[:-1])}.txt"
        with open(f'{str.replace(folder, "images", "labels")}/{label_name}', 'r') as label_f:
            lines = label_f.readlines()
            for l in lines:
                box = np.array(str.split(str.strip(l), ' ')[1:]).astype('float64')
                box = list(ops.xywhn2xyxy(box, img.width, img.height, 0, 0))
                ann.box_label(box=box, color=(0, 0, 255), txt_color=(255, 255, 255),
                              label=str.split(str.split(str.strip(l), ' ')[0], '.')[0])
            ann.save(f'{save_folder}/{f}')

def copy_image_labels(list_of_files):
    for f in list_of_files:
        label_name = f"{'.'.join(str.split(f, '.')[:-1])}.txt"
        shutil.copy(f'datasets/drones/labels/{label_name}', f'datasets/drones_clean/labels/{label_name}')

def filter_autosplit(list_of_files):
    filter_list = os.listdir('datasets/drones_clean/images_with_bbox/')
    with (open('datasets/drones_clean/autosplit_train.txt', 'r') as as_train,
          open('datasets/drones_clean/filter_train.txt', 'w') as filter_train):
        as_lines = as_train.readlines()
        for l in as_lines:
            part = str.strip(l)[9:]
            if os.path.exists(f'datasets/drones_clean/images_with_bbox/{part}'):
                filter_train.write(l)

    with (open('datasets/drones_clean/autosplit_val.txt', 'r') as as_val,
          open('datasets/drones_clean/filter_val.txt', 'w') as filter_val):
        as_lines = as_val.readlines()
        for l in as_lines:
            part = str.strip(l)[9:]
            if os.path.exists(f'datasets/drones_clean/images_with_bbox/{part}'):
                filter_val.write(l)

def convert_anylabeling_dir_to_yolo(source_json, source_image, dest, names):
    if not os.path.exists(dest):
        os.makedirs(dest)
        os.makedirs(f'{dest}images/')
        os.makedirs(f'{dest}labels/')
    for f in os.listdir(source_json):
        if '.json' in f:
            with open(f'{source_json}{f}', 'r') as jsf:
                label = json.load(jsf)
            image_name = label['imagePath']
            label_name = f"{'.'.join(str.split(label['imagePath'], '.')[:-1])}.txt"
            shutil.copy(f'{source_image}images/{image_name}', f'{dest}images/{image_name}')
            with open(f'{dest}labels/{label_name}', 'w') as lbf:
                for detect in label['shapes']:
                    if detect['label'] in names:
                        id = names[detect["label"]]
                        x = (detect['points'][0][0] + detect['points'][1][0]) / 2 / label['imageWidth']
                        y = (detect['points'][0][1] + detect['points'][1][1]) / 2 / label['imageHeight']
                        w = (detect['points'][1][0] - detect['points'][0][0]) / label['imageWidth']
                        h = (detect['points'][1][1] - detect['points'][0][1]) / label['imageHeight']
                        yolo_line = f'{id} {x} {y} {w} {h}\n'
                        lbf.write(yolo_line)



#list_of_files = os.listdir('datasets/drones_clean/images/')
#filter_autosplit(list_of_files)
#get_folder_bboxes('datasets/airborne/images', 'datasets/airborne/images_bbox')
#convert_anylabeling_dir_to_yolo('datasets/drones_clean/birds_relabel/', 'datasets/drones/', 'datasets/birds_fixed_labels/', {'copter': 0.0, 'plane': 1.0, 'heli': 2.0, 'bird': 3.0, 'winged': 4.0})
get_folder_bboxes('datasets/birds_fixed_labels/images', 'datasets/birds_fixed_labels/images_bbox')