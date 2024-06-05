import os
import shutil
from PIL import Image
from ultralytics.utils.plotting import Annotator
import numpy as np
import ultralytics.utils.ops as ops

list_of_files = os.listdir('datasets/drones_clean/images/')

with open('clean_list.txt', 'w', encoding='utf-8') as clf:
    for l in list_of_files:
        clf.writelines(f'{l}\n')

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

exit()

for f in list_of_files:
    label_name = f"{'.'.join(str.split(f, '.')[:-1])}.txt"
    shutil.copy(f'datasets/drones/labels/{label_name}', f'datasets/drones_clean/labels/{label_name}')