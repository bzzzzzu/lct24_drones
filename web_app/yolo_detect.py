import math
import os
from datetime import datetime

import yaml
from pathlib import Path
from typing import Union, Dict, List

from ultralytics import YOLO
import cv2
import torch
from PIL import Image
from werkzeug.utils import secure_filename

WORKDIR = Path(__file__).parent.absolute()


class YoloModel:
    def __init__(
            self,
            # model_path: str = f"http://triton:8000/yolo/",  # из контейнера
            model_path: str = f"./best_0700_fixed_birds.pt",
            # model_path: str = f"http://localhost:8000/yolo/",  # из контейнера
            task: str = "detect",
            path_to_config: str = 'drones.yaml',
    ):
        self.model = YOLO(model_path, task=task)
        self.path_to_config = path_to_config
        self.config = self._load_config(self.path_to_config)
        self.dataset_path: str = self.config["path"]
        self.class_names: Dict = self.config["names"]
        self.imgsz = self.config.get("imgsz", 1280)
        self.conf = self.config.get("conf", 0.25)
        self.iou = self.config.get("iou", 0.45)
        self.max_det = self.config.get("max_det", 1000)
        self.save_txt = self.config.get("save_txt", True)
        self.save_conf = self.config.get("save_conf", True)
        self.save_crop = self.config.get("save_crop", False)
        self.project = self.config.get("project", './detects')
        self.name = self.config.get("name", 'runs')
        self.line_width = self.config.get("line_width", 3)
        self.show_labels = self.config.get("show_labels", True)
        self.show_conf = self.config.get("show_conf", True)
        self.show = self.config.get("show", False)
        self.stream_buffer = self.config.get("stream_buffer", False)

        self.classes = list(self.class_names.keys()) if self.class_names is not None else [0, 1, 2, 3, 4]
        self.class_names_list = list(self.class_names.values()) if self.class_names is not None else ["copter", "plane", "heli", "bird", "winged"]
        self.device = 0 if torch.cuda.is_available() else 'cpu'

        print("\n\nINFO: model=", self.model.model_name)
        print(f"INFO: model config: {self.imgsz=}, {self.conf=}, {self.iou=}")
        print('INFO: device for inference=', self.device, '\n\n')

    def detect(self,
               source,
               imgsz: int = None,
               device: Union[str, int] = 0,
               stream: bool = False,
               save: bool = True,
               project: str = None,
               name: str = None,
               conf: float = None,
               stream_buffer: bool = None,
               show: bool = None,
               ):

        imgsz = self.imgsz if imgsz is None else imgsz
        project = self.project if project is None else project
        name = self.name if name is None else name
        conf = self.conf if conf is None else conf
        show = self.show if show is None else show
        stream_buffer = self.stream_buffer if stream_buffer is None else stream_buffer

        #if device == 0 and self.device == 'cpu':
        #    device = 'cpu'
        print('INFO: device ->', device)

        results = self.model(
            source=source,
            imgsz=imgsz,
            device=device,
            stream=stream,
            project=project,
            name=name,
            conf=conf,
            data=self.path_to_config,
            max_det=self.max_det,
            iou=self.iou,
            save_txt=self.save_txt,
            save_conf=self.save_conf,
            save_crop=self.save_crop,
            line_width=self.line_width,
            show_labels=self.show_labels,
            show_conf=self.show_conf,
            classes=self.classes,
            stream_buffer=stream_buffer,
            save=save,
            show=show,
            batch=24,
            agnostic_nms=True,
        )
        return results

    @staticmethod
    def _load_config(filepath: str):
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return yaml.safe_load(f)
        raise FileNotFoundError


    @staticmethod
    def get_result_path(full_file_path):
        filename = os.path.basename(full_file_path)
        dirname, _ = os.path.splitext(filename)
        return dirname

    def video_detection(self, path_x):
        print(f'DEBUG: {path_x=}')
        if path_x is None:
            raise ValueError

        if isinstance(path_x, str) and path_x.lower().endswith(('.png', '.jpg', '.jpeg')):
            dirname = self.get_result_path(path_x)

            img = cv2.imread(path_x)
            if img is None:
                print(f"Error: Unable to open image file {path_x}")
                return

            results = self.detect(
                source=path_x,
                stream=True,
                stream_buffer=False,
                project=f'./uploads/{dirname}/',
                name='results'
            )
            for result in results:
                frame = result.orig_img
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)

                for box, score, class_id in zip(boxes, confidences, class_ids):
                    x1, y1, x2, y2 = map(int, box)
                    label = f"{self.class_names[class_id]}: {score:.2f}"
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                yield frame

        # TODO: Добавить распаковку zip
        # TODO: добавить link и folder
        # FIXME: почему то очень большая задержка при live streaming если stream_buffer=True
        elif isinstance(path_x, str):
            dirname = self.get_result_path(path_x)

            cap = cv2.VideoCapture(path_x)
            if not cap.isOpened():
                print(f"Error: Unable to open video file {path_x}")
                return

            # results = self.detect(source=path_x, stream=True, stream_buffer=True)
            results = self.detect(
                source=path_x,
                stream=True,
                stream_buffer=True,
                project=f'./uploads/{dirname}/',
                name='results'
            )

            for result in results:
                frame = result.orig_img
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)

                for box, score, class_id in zip(boxes, confidences, class_ids):
                    x1, y1, x2, y2 = map(int, box)
                    label = f"{self.class_names[class_id]}: {score:.2f}"
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                yield frame
            cap.release()

        elif isinstance(path_x, int) and path_x == 0:
            current_datetime = datetime.now()
            dirname = f'stream_{secure_filename(str(current_datetime))}'

            cap = cv2.VideoCapture(path_x)
            if not cap.isOpened():
                print(f"Error: Unable to open video file {path_x}")
                return

            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))

            while True:
                success, img = cap.read()
                if not success:
                    break

                results = self.detect(
                    source=path_x,
                    stream=True,
                    stream_buffer=True,
                    project=f'./uploads/{dirname}/',
                    name='results'
                )

                for result in results:
                    frame = result.orig_img
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)

                    for box, score, class_id in zip(boxes, confidences, class_ids):
                        x1, y1, x2, y2 = map(int, box)
                        label = f"{self.class_names[class_id]}: {score:.2f}"
                        color = (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    yield frame
            cap.release()