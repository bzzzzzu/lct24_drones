import os
import yaml
from pathlib import Path
from typing import Union, Dict

from ultralytics import YOLO
import cv2
import torch

WORKDIR = Path(__file__).parent.absolute()


class YoloContainerInference:
    def __init__(
            self,
            model_path: str = f"http://localhost:8000/yolo/",  # из контейнера
            task: str = "detect",
            path_to_config: str = '../drones.yaml',
    ):
        self.model = YOLO(model_path, task=task)
        self.path_to_config = path_to_config
        self.config = self._load_config(self.path_to_config)
        self.dataset_path: str = self.config["path"]
        self.class_names: Dict = self.config["names"]
        self.imgsz = self.config.get("imgsz", 1024)
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

        self.classes = list(self.class_names.keys()) if self.class_names is not None else [0, 1, 2, 3, 4]
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        print('\n\nINFO: device for inference=', self.device, '\n\n')

    def detect(self,
               source,
               imgsz: int = None,
               device: Union[str, int] = 0,
               stream: bool = False,
               save: bool = True,
               project: str = None,
               name: str = None,
               conf: float = None
               ):

        imgsz = self.imgsz if imgsz is None else imgsz
        project = self.project if project is None else project
        name = self.name if name is None else name
        conf = self.conf if conf is None else conf

        if device == 0 and self.device == 'cpu':
            device = 'cpu'
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
            save=save,
        )
        return results

    @staticmethod
    def _load_config(filepath: str):
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return yaml.safe_load(f)
        raise FileNotFoundError


# if __name__ == '__main__':
#     def main(source):
#         yolo_inference = YoloContainerInference()
#         if source != 0:
#             results = yolo_inference.detect(source)
#
#         elif source == 0:  # streaming mode
#             results = yolo_inference.detect(source, stream=True)
#             for result in results:
#                 frame = result.orig_img
#                 boxes = result.boxes.xyxy.cpu().numpy()
#                 confidences = result.boxes.conf.cpu().numpy()
#                 class_ids = result.boxes.cls.cpu().numpy().astype(int)
#
#                 for box, score, class_id in zip(boxes, confidences, class_ids):
#                     x1, y1, x2, y2 = map(int, box)
#                     label = f"{yolo_inference.class_names[class_id]}: {score:.2f}"
#                     color = (0, 255, 0)
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                     cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#
#                 cv2.imshow('YOLO Stream', frame)
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break
#             cv2.destroyAllWindows()
#
#     test_folder_imgs = '../utils/test_data/imgs'
#     test_folder_img = '../utils/test_data/imgs/drone_1.jpg'
#     test_folder_videos = '../utils/test_data/videos/'
#     test_streaming = 0
#     test_ling = 'https://www.youtube.com/watch?v=wYR23j4Nfik'
#     test_link_2 = 'https://www.youtube.com/watch?v=VHwEb86s2SI'
#
#     main(test_streaming)
