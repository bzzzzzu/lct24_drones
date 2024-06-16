import os
from pathlib import Path
import time
from datetime import datetime
import json
import shutil
from collections import Counter
import yaml
from typing import Union, Dict, List
import re

import torch
from ultralytics import YOLO
import cv2
from PIL import Image
from werkzeug.utils import secure_filename
import tempfile
# import yt_dlp as youtube_dl
# from pytube import YouTube, exceptions

from values import VIDEO_EXTENSIONS, IMG_EXTENSIONS


WORKDIR = Path(__file__).parent.absolute()
TIMEPOINTS = []
CLASS_NAMES = {
    0: "copter",
    1: "plane",
    2: "heli",
    3: "bird",
    4: "winged",
}
BPLA = ["copter", "winged"]
BIRD = ["bird"]
LA = ["plane", "heli"]


class YoloModel:
    """Класс инференса модели."""
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
        self.class_names: Dict = self.config["names"] if self.config is not None else CLASS_NAMES
        self.imgsz = 1280
        self.conf = 0.25
        self.iou = 0.45
        self.max_det = 1000
        self.save_crop = False
        self.line_width = 3
        self.show_labels = True
        self.show_conf = True

        self.classes = list(self.class_names.keys())
        self.class_names_list = list(self.class_names.values())
        self.device = 0 if torch.cuda.is_available() else 'cpu'

        print("\nINFO: model=", self.model.model_name)
        print(f"INFO: model config: {self.imgsz=}, {self.conf=}, {self.iou=}")
        print('INFO: device', self.device, '\n')

    def detect(self,
               source,
               batch: int = 24,
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
        """Метод загрузки конфига."""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return yaml.safe_load(f)
        return None

    @staticmethod
    def process_timecodes(timecodes, min_interval=0.2, merge_gap=0.02):
        """Метод определения самых важных таймкодов."""
        def get_most_common_label(labels):
            if not labels:
                return None
            counter = Counter(labels)
            return counter.most_common(1)[0][0]
        results = []
        start_time = None
        end_time = None
        current_labels = []
        last_detection_time = None

        for entry in timecodes:
            if entry['labels']:
                if start_time is None:
                    start_time = entry['time']
                end_time = entry['time']
                last_detection_time = entry['time']
                current_labels.extend(entry['labels'])
            elif start_time is not None:
                if entry['time'] - last_detection_time <= merge_gap:
                    end_time = entry['time']
                else:
                    if end_time - start_time >= min_interval:
                        most_common_label = get_most_common_label(current_labels)
                        if most_common_label:

                            results.append({
                                'start_time': start_time,
                                'end_time': end_time,
                                "color": "red" if most_common_label in BPLA else "green" if most_common_label in BIRD else "yellow" if most_common_label in LA else "gray",
                            })
                    start_time = None
                    end_time = None
                    current_labels = []
                    last_detection_time = None

        if start_time is not None and (end_time - start_time >= min_interval):
            most_common_label = get_most_common_label(current_labels)
            if most_common_label:
                results.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    "color": "red" if most_common_label in BPLA else "green" if most_common_label in BIRD else "yellow" if most_common_label in LA else "gray",
                })

        return results

    def process_frame(self, frame, results, timecode_artifacts=None):
        """Метод отрисовки bboxes, labels, conf на каждом frame."""
        global TIMEPOINTS

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            if timecode_artifacts is not None:
                current_time = timecode_artifacts[1] if timecode_artifacts else 0
                TIMEPOINTS.append({
                    'time': current_time,
                    'labels': [self.class_names[cls_id] for cls_id in class_ids]
                })

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
        global TIMEPOINTS
        print(f'INFO: {path_x=}')

        if isinstance(path_x, str):
            # IMAGE
            if path_x.lower().endswith(IMG_EXTENSIONS):
                filename = os.path.basename(path_x)
                dirname, _ = os.path.splitext(filename)
                results = self.detect(source=path_x, stream=True, stream_buffer=False,
                                      project=f'./uploads/{dirname}/', name='results')
                for frame in results:
                    yield self.process_frame(frame.orig_img, [frame])

            # VIDEO
            elif path_x.lower().endswith(VIDEO_EXTENSIONS):
                filename = os.path.basename(path_x)
                dirname, _ = os.path.splitext(filename)
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
                    results = self.detect(source=path_x, stream=True, stream_buffer=True,
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

                    cap.release()
                    out.release()
                    cv2.destroyAllWindows()
                    timecodes = self.process_timecodes(TIMEPOINTS)
                    self.save_timecodes(timecodes, dirname)

                    TIMEPOINTS = []
                    # Перемещаем временное видео в папку results
                    new_dir = os.path.join('./uploads', dirname, 'results')
                    os.makedirs(new_dir, exist_ok=True)
                    final_output_path = os.path.join(new_dir, filename)
                    shutil.move(temp_output_path, final_output_path)


            # LINK
            elif self.is_url(path_x):
                pass

                # Критический баг, временно выключено
                '''
                ydl_opts = {
                    "quiet": True,
                    "no_warnings": True,
                    "format": "best",
                    "forceurl": True,
                }

                ydl = youtube_dl.YoutubeDL(ydl_opts)
                info = ydl.extract_info(path_x, download=False)
                url = info["url"]

                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_output_path = os.path.join(temp_dir, f"{current_datetime}.mp4")

                    cap = cv2.VideoCapture(url)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fourcc = cv2.VideoWriter_fourcc(*'H264')  # Кодек для MP4

                    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
                    results = self.detect(source=url, stream=True, stream_buffer=True,
                                          save=False, save_txt=True, save_conf=True,
                                          project=f'./uploads/{current_datetime}/',
                                          name='results')

                    frame_cnt = 0
                    for result in results:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame_cnt += 1
                        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                        timecode_artifacts = [frame_cnt, current_time, current_datetime]

                        processed_frame = self.process_frame(frame, [result], timecode_artifacts)
                        out.write(processed_frame)

                        yield processed_frame

                    cap.release()
                    out.release()
                    cv2.destroyAllWindows()
                    timecodes = self.process_timecodes(TIMEPOINTS)
                    self.save_timecodes(timecodes, current_datetime)

                    TIMEPOINTS = []
                    new_dir = os.path.join('./uploads', current_datetime, 'results')
                    os.makedirs(new_dir, exist_ok=True)
                    final_output_path = os.path.join(new_dir, f"{current_datetime}.mp4")
                    shutil.move(temp_output_path, final_output_path)
                '''

            # FOLDER
            else:
                filename = os.path.basename(path_x)
                dirname, _ = os.path.splitext(filename)
                if os.path.isdir(path_x):
                    for filename in os.listdir(path_x):
                        path_to_file = os.path.join(path_x, filename)
                        with tempfile.TemporaryDirectory() as temp_dir:
                            temp_output_path = os.path.join(temp_dir, filename)


                            cap = cv2.VideoCapture(path_to_file)
                            if not cap.isOpened():
                                print(f"Error: Unable to open video file {path_to_file}")
                                return

                            fps = cap.get(cv2.CAP_PROP_FPS)
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            fourcc = cv2.VideoWriter_fourcc(*'H264')  # Кодек для MP4

                            out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
                            results = self.detect(source=path_to_file, stream=True, stream_buffer=True,
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
                            TIMEPOINTS = []

                            # Перемещаем временное видео в папку results
                            new_dir = os.path.join('./uploads', dirname, 'results')
                            os.makedirs(new_dir, exist_ok=True)
                            final_output_path = os.path.join(new_dir, filename)
                            shutil.move(temp_output_path, final_output_path)


        # WEBCAM
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
    def save_timecodes(timecodes, dirname):
        """Метод для записи временных меток, меток классов в JSON файл."""

        timecodes_dir = os.path.join('uploads', dirname, 'timecodes')
        os.makedirs(timecodes_dir, exist_ok=True)
        timecodes_file = os.path.join(timecodes_dir, f'{dirname}.json')

        with open(timecodes_file, 'w') as f:
            json.dump(timecodes, f, indent=4)
    @staticmethod
    def is_url(path):
        url_regex = re.compile(
            r'^(?:http|ftp)s?://'  # http:// или https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # доменные имена
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # IPv4 адрес
            r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'  # IPv6 адрес
            r'(?::\d+)?'  # опциональный порт
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return re.match(url_regex, path) is not None
