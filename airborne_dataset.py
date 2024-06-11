import os
import time

import numpy
import numpy as np
import requests
import json
import ultralytics.utils.ops as ops
from pprint import pp
from threading import Thread

def download_and_label(examples_list, t_id):
    for example in examples_list:
        #print(example)
        if not os.path.exists(example['local_path']):
            try:
                r = requests.get(example['s3_path'], stream=False)
                if r.ok:
                    with open(example['local_path'], 'wb') as imf:
                        imf.write(r.content)
                    print(f'T{t_id} - Download complete: {example["s3_path"]}')
            except:
                print(f'T{t_id} - Failure to download file {example["s3_path"]}')

        local_label = str.split(example['local_path'], '.')[0]
        local_label = str.replace(local_label, 'images', 'labels')
        local_label = f'{local_label}.txt'
        if not os.path.exists(local_label):
            with open(local_label, 'w') as lbf:
                if 'detect' in example['frame_info']:
                    for d in example['frame_info']['detect']:
                        box = ops.ltwh2xywh(np.array(d['bb']))
                        box[0] = box[0] / example['flight_meta']['resolution']['width']
                        box[1] = box[1] / example['flight_meta']['resolution']['height']
                        box[2] = box[2] / example['flight_meta']['resolution']['width']
                        box[3] = box[3] / example['flight_meta']['resolution']['height']
                        if 'Airplane' in d['id']:
                            box_id = 1.0
                        if 'Helicopter' in d['id']:
                            box_id = 2.0
                        if 'Bird' in d['id']:
                            box_id = 3.0
                        #print(box)
                        write_line = f'{box_id} {box[0]} {box[1]} {box[2]} {box[3]}\n'
                        lbf.write(write_line)

if __name__ == '__main__':
    dataset_dir = 'datasets/airborne/'
    dataset_ratio_airplane = 2
    dataset_ratio_helicopter = 11
    dataset_ratio_bird = 1
    dataset_ratio_empty = 1000
    s3_main = 'https://s3.amazonaws.com/airborne-obj-detection-challenge-training/'
    label_files = [
        'part1/ImageSets/groundtruth.json',
        'part2/ImageSets/groundtruth.json',
        'part3/ImageSets/groundtruth.json',
    ]

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        os.makedirs(f'{dataset_dir}images/')
        os.makedirs(f'{dataset_dir}labels/')

    for l in label_files:
        if not os.path.exists(f'{dataset_dir}groundtruth_{l[4]}.json'):
            r = requests.get(f'{s3_main}{l}', stream=True)
            with open(f'{dataset_dir}groundtruth_{l[4]}.json', 'wb') as lf:
                for chunk in r.iter_content(chunk_size=65536):
                    lf.write(chunk)

    final_examples = []

    label_local = ['groundtruth_1.json',
                   'groundtruth_2.json',
                   'groundtruth_3.json'
                  ]
    global_frame_ids = []
    total_detect = 0
    total_empty = 0

    total_airplane = 0
    total_helicopter = 0
    total_bird = 0

    picked_airplane = 0
    picked_helicopter = 0
    picked_bird = 0
    for l in label_local:
        with open(f'{dataset_dir}{l}', 'r') as jsf:
            labels = json.load(jsf)
        counter_airplane = 0
        counter_helicopter = 0
        counter_bird = 0
        counter_empty = 0
        #print(labels['metadata'])
        for flight_id in labels['samples']:
            h = labels['samples'][flight_id]['metadata']['resolution']['height']
            w = labels['samples'][flight_id]['metadata']['resolution']['width']
            #print(labels['samples'][flight_id]['metadata'])
            #print(f'{h}x{w}')

            # unknown objects, might be drones
            ban_frames = []
            frames_packed = {}
            for frame in labels['samples'][flight_id]['entities']:
                if not frame['blob']['frame'] in frames_packed:
                    frames_packed[frame['blob']['frame']] = {
                        'blob': {'frame': frame['blob']['frame']},
                        'flight_id': frame['flight_id'],
                        'img_name': frame['img_name'],
                    }
                if 'id' in frame:
                    #if frame['id'] not in global_frame_ids:
                    #    global_frame_ids.append(frame['id'])
                    # wrong object
                    if not('Bird' in frame['id'] or 'Airplane' in frame['id'] or 'Helicopter' in frame['id']):
                        ban_frames.append(frame['blob']['frame'])
                    else:
                        # object too small for classification, need around 10x10 after scaling
                        if frame['bb'][2] <= 20.0 or frame['bb'][3] <= 20.0:
                            ban_frames.append(frame['blob']['frame'])
                        else:
                            if not 'detect' in frames_packed[frame['blob']['frame']]:
                                frames_packed[frame['blob']['frame']]['detect'] = []
                            frames_packed[frame['blob']['frame']]['detect'].append(
                                {
                                    'id': frame['id'],
                                    'bb': frame['bb'],
                                }
                            )
                            if 'Airplane' in frame['id']:
                                total_airplane = total_airplane + 1
                            if 'Helicopter' in frame['id']:
                                total_helicopter = total_helicopter + 1
                            if 'Bird' in frame['id']:
                                total_bird = total_bird + 1
            #pp(frames_packed)

            for fr in frames_packed:
                frame = frames_packed[fr]
                if frame['blob']['frame'] not in ban_frames:
                    s3_path = f'{s3_main}part{l[12]}/Images/{frame["flight_id"]}/{frame["img_name"]}'
                    local_path = f'{dataset_dir}images/{frame["img_name"]}'

                    #print(s3_path)
                    #print(local_path)
                    save_example = 0

                    # something detected
                    if 'detect' in frame:
                        for d in frame['detect']:
                            if 'Airplane' in d['id']:
                                if counter_airplane % dataset_ratio_airplane == 0:
                                    save_example = 1
                                    picked_airplane = picked_airplane + 1
                                counter_airplane = counter_airplane + 1
                            if 'Helicopter' in d['id']:
                                if counter_helicopter % dataset_ratio_helicopter == 0:
                                    save_example = 1
                                    picked_helicopter = picked_helicopter + 1
                                counter_helicopter = counter_helicopter + 1
                            if 'Bird' in d['id']:
                                if counter_bird % dataset_ratio_bird == 0:
                                    save_example = 1
                                    picked_bird = picked_bird + 1
                                counter_bird = counter_bird + 1
                        total_detect = total_detect + 1
                    else:
                        if counter_empty % dataset_ratio_empty == 0:
                            save_example = 1
                            total_empty = total_empty + 1
                        counter_empty = counter_empty + 1

                    if save_example == 1:
                        example = {
                            's3_path': s3_path,
                            'local_path': local_path,
                            'flight_meta': labels['samples'][flight_id]['metadata'],
                            'frame_info': frame,
                        }
                        final_examples.append(example)

    # 289190 examples with objects at 1/10
    # 27504 examples without objects at 1/100
    print(f'airplane: {total_airplane}, helicopter: {total_helicopter}, bird: {total_bird}')
    print(global_frame_ids)
    print(f'Airplane at 1/{dataset_ratio_airplane}: {picked_airplane}, Helicopter at 1/{dataset_ratio_helicopter}: {picked_helicopter}, Bird at 1/{dataset_ratio_bird}: {picked_bird}, Empty at 1/{dataset_ratio_empty}: {total_empty}')
    #print(final_examples)
    print(len(final_examples))
    #exit()

    split_examples = numpy.array_split(final_examples, 32)

    threads = []
    for i in range(0, len(split_examples)):
        t = Thread(target=download_and_label, args=(split_examples[i], i))
        threads.append(t)

    for t in threads:
        t.start()
        time.sleep(0.1)

    for t in threads:
        t.join()

    print('done')


    # json content
    # 'metadata'
    # 'samples'
    # -- 'flight_id' x1311
    # ---- 'metadata'
    # ------ 'data_path', 'duration': 119900.0, 'fps': 10.0, 'number_of_frames': 1199, 'resolution': {'height': 2048, 'width': 2448}
    # ---- 'entities'
    # ------ [list of dicts] x1199 or more if several detections per frame
    # -------- 'time', 'flight_id', 'img_name'
    # -------- 'blob': {'frame': 2, 'range_distance_m': nan} # range optional
    # -------- 'labels: {'is_above_horizon': 1} # labels optional
    # -------- 'id': 'Helicopter1' # id optional
    # -------- 'bb': [928.0, 1108.0, 16.0, 16.0] # bb optional, [left, top, width, height]

    # {
    #     'time': 1550844897919368155,
    #     'blob': {
    #         'frame': 480,
    #         'range_distance_m': nan # signifies, it was an unplanned object
    #     },
    #     'id': 'Bird2',
    #     'bb': [1013.4, 515.8, 6.0, 6.0],
    #     'labels': {'is_above_horizon': 1},
    #     'flight_id': '280dc81adbb3420cab502fb88d6abf84',
    #     'img_name': '1550844897919368155280dc81adbb3420cab502fb88d6abf84.png'
    # }