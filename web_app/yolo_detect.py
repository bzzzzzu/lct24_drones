import json
import math
import os
import shutil
import time
from datetime import datetime

import yaml
from pathlib import Path
from typing import Union, Dict, List

from ultralytics import YOLO
import cv2
import torch
from PIL import Image
from werkzeug.utils import secure_filename
import tempfile

WORKDIR = Path(__file__).parent.absolute()
IS_DETECTED, LABELS_IN_PERIOD, SCORES_IN_PERIOD = False, [], []

class YoloModel:
    def __init__(
            self,
            # model_path: str = f"http://triton:8000/yolo/",  # из контейнера
            model_path: str = f"./best_0700_fixed_birds.pt",
            task: str = "detect",
            path_to_config: str = 'drones.yaml',
    ):
        self.model = YOLO(model_path, task=task)
        self.path_to_config = path_to_config
        self.config = self._load_config(path_to_config)
        self.dataset_path: str = self.config["path"]
        self.class_names: Dict = self.config["names"]
        self.imgsz = 1280
        self.conf = 0.25
        self.iou = 0.45
        self.max_det = 1000
        self.save_crop = False
        self.line_width = 3
        self.show_labels = True
        self.show_conf = True

        self.classes = list(self.class_names.keys()) if self.class_names is not None else [0, 1, 2, 3, 4]
        self.class_names_list = list(self.class_names.values()) if self.class_names is not None else ["copter", "plane",
                                                                                                      "heli", "bird",
                                                                                                      "winged"]
        self.device = 0 if torch.cuda.is_available() else 'cpu'

        print("\nINFO: model=", self.model.model_name)
        print(f"INFO: model config: {self.imgsz=}, {self.conf=}, {self.iou=}")
        print('INFO: device', self.device, '\n')

    def detect(self,
               source,
               batch: int = 16,
               stream: bool = False,
               stream_buffer: bool = False,
               save: bool = True,
               project: str = 'uploads',
               name: str = 'results',
               show: bool = False,
               save_txt: bool = True,
               save_conf: bool = True,
               agnostic_nms: bool = True,
               exist_ok: bool = True,
               ):

        results = self.model(
            source=source, stream=stream, stream_buffer=stream_buffer, save=save, project=project, name=name,
            show=show, save_txt=save_txt, save_conf=save_conf, batch=batch, agnostic_nms=agnostic_nms,
            exist_ok=exist_ok, imgsz=self.imgsz, device=self.device, conf=self.conf, data=self.path_to_config,
            max_det=self.max_det, iou=self.iou, save_crop=self.save_crop, line_width=self.line_width,
            show_labels=self.show_labels, show_conf=self.show_conf, classes=self.classes,
        )
        return results

    @staticmethod
    def _load_config(filepath: str):
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return yaml.safe_load(f)
        raise FileNotFoundError

    def process_frame(self, frame, results, timecode_artifacts=None):
        """Метод отрисовки bboxes, labels, conf на каждом frame."""
        global IS_DETECTED, LABELS_IN_PERIOD, SCORES_IN_PERIOD, DETECTION_START_TIME

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            if timecode_artifacts is not None:
                current_time = timecode_artifacts[1] if timecode_artifacts else 0

                if class_ids.size > 0:
                    if not IS_DETECTED:
                        DETECTION_START_TIME = current_time  # Сохранение времени первой детекции
                    IS_DETECTED = True
                    LABELS_IN_PERIOD.extend([self.class_names[i] for i in class_ids])
                    SCORES_IN_PERIOD.extend([float(score) for score in confidences.tolist()])
                else:
                    if IS_DETECTED and (current_time - DETECTION_START_TIME) > 1:
                        # Запись данных, только если 1 секунд прошло без детекции
                        self.save_timecodes(timecode_artifacts, DETECTION_START_TIME, current_time, LABELS_IN_PERIOD,
                                            SCORES_IN_PERIOD)
                        IS_DETECTED, LABELS_IN_PERIOD, SCORES_IN_PERIOD, DETECTION_START_TIME = False, [], [], None

            for box, score, class_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = map(int, box)
                class_name = self.class_names[class_id]
                label = f"{class_name}: {score:.2f}"
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame

    def video_detection(self, path_x, current_datetime=None):
        """Метод детекции в зависимости от типа файла/папки/ссылки."""

        if isinstance(path_x, str):
            filename = os.path.basename(path_x)
            dirname, _ = os.path.splitext(filename)

            # image
            if path_x.lower().endswith(('.png', '.jpg', '.jpeg')):
                results = self.detect(source=path_x, stream=True, stream_buffer=False,
                                      project=f'./uploads/{dirname}/', name='results')
                for frame in results:
                    yield self.process_frame(frame.orig_img, [frame])

            # video
            elif path_x.lower().endswith(('.mp4', '.avi')):
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_output_path = os.path.join(temp_dir, filename)

                    cap = cv2.VideoCapture(path_x)
                    if not cap.isOpened():
                        print(f"Error: Unable to open video file {path_x}")
                        return

                    fps = cap.get(cv2.CAP_PROP_FPS)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fourcc = cv2.VideoWriter_fourcc(*'H264')  # Кодек для MP4

                    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
                    results = self.detect(source=path_x, stream=True, stream_buffer=False,
                                          save=False, save_txt=True, save_conf=True,
                                          project=f'./uploads/{dirname}/',
                                          name='results')

                    frame_cnt = 0
                    for result in results:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame_cnt += 1
                        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                        timecode_artifacts = [frame_cnt, current_time, dirname]

                        processed_frame = self.process_frame(frame, [result], timecode_artifacts)
                        out.write(processed_frame)

                        yield processed_frame

                    # Освобождаем ресурсы
                    cap.release()
                    out.release()
                    cv2.destroyAllWindows()

                    # Перемещаем временное видео в папку results
                    new_dir = os.path.join('./uploads', dirname, 'results')
                    os.makedirs(new_dir, exist_ok=True)
                    final_output_path = os.path.join(new_dir, filename)
                    shutil.move(temp_output_path, final_output_path)

            # link
            elif 'http' in path_x:
                pass

            # folder
            else:
                pass

        # stream
        # 172.21.0.1 - - [14/Jun/2024 23:35:00] "GET /webcam HTTP/1.1" 200 -
        # flask-1   | [ WARN:0@1.416] global cap_v4l.cpp:999 open VIDEOIO(V4L2:/dev/video0): can't open camera by index
        # flask-1   | [ WARN:0@1.416] global cap.cpp:342 open VIDEOIO(V4L2): backend is generally available but can't be used to capture by index

        elif path_x == 0:
            dirname = f'stream_{secure_filename(str(current_datetime))}'
            cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            if not cap.isOpened():
                print(f"Error: Unable to open video file {path_x}")
                return

            while True:
                success, img = cap.read()
                if not success:
                    break

                results = self.detect(source=path_x, stream=True, stream_buffer=True,
                                      project=f'./uploads/{dirname}/', name='results')

                frame_cnt = 0
                for result in results:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_cnt += 1
                    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    timecode_artifacts = [frame_cnt, current_time, dirname]
                    yield self.process_frame(frame, [result], timecode_artifacts)
            cap.release()

        else:
            raise ValueError(f'Source {path_x} is not supported.')

    @staticmethod
    def save_timecodes(timecode_artifacts, start_time, end_time, LABELS_IN_PERIOD, SCORES_IN_PERIOD):
        """Метод для записи временных меток, меток классов в JSON файл."""

        ### timecode_artifacts: [frame_cnt, current_time, dirname]

        timecodes_dir = os.path.join('uploads', timecode_artifacts[2], 'timecodes')
        os.makedirs(timecodes_dir, exist_ok=True)
        timecodes_file = os.path.join(timecodes_dir, f'{timecode_artifacts[2]}.json')

        data = {
            "start_time": start_time,
            "end_time": end_time,
            "labels": LABELS_IN_PERIOD,
            "scores": SCORES_IN_PERIOD,
        }

        if os.path.exists(timecodes_file):
            with open(timecodes_file, 'r+') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    print('FIXME')
                    existing_data = []
                existing_data.append(data)
                f.seek(0)
                json.dump(existing_data, f, indent=4)
        else:
            with open(timecodes_file, 'w') as f:
                json.dump([data], f, indent=4)
