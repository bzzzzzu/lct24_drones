from pathlib import Path
from typing import List, Union

from ultralytics import YOLO
import yaml
import os
import cv2

WORKDIR = Path(__file__).parent.absolute()


class YoloContainerInference:
    def __init__(
            self,
            path_to_config: str = './drones.yaml',
            **kwargs
    ):
        self.model = YOLO(f"http://localhost:8000/yolo/", task="detect")
        self.path_to_config = path_to_config
        self.config = self._load_config(self.path_to_config)
        self.dataset_path = self.config["path"]
        self.class_names = self.config["names"]

    @staticmethod
    def _load_config(filepath: str):
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return yaml.safe_load(f)
        raise FileNotFoundError

    def get_prediction(
            self,
            source,
            device: Union[str, int] = 0,
            stream: bool = False,
            data='./drones.yaml',
            max_det=1000,
            classes: List = [0, 1, 2, 3, 4],
            save: bool = True,
            save_txt: bool = True,
            save_conf: bool = True,
            save_crop: bool = False,
            project: str = './detects',
            name: str = 'runs',
            imgsz=1024
    ):
        return self.model(
            source=source,
            device=device,
            stream=stream,  # Enable streaming
            data=data,
            max_det=max_det,
            classes=classes,

            save=save,
            save_txt=save_txt,  # save results to *.txt
            save_conf=save_conf,  # save confidences in --save-txt labels
            save_crop=save_crop,  # save cropped prediction boxes
            project=project,
            name=name,
            imgsz=imgsz
        )


def main(source):
    yolo_inference = YoloContainerInference()
    if source != 0:
        results = yolo_inference.get_prediction(source)

    elif source == 0:  # streaming mode
        results = yolo_inference.get_prediction(source, stream=True)
        for result in results:
            frame = result.orig_img
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, score, class_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = map(int, box)
                label = f"{yolo_inference.class_names[class_id]}: {score:.2f}"
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.imshow('YOLO Stream', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()


if __name__ == '__main__':
    test_folder_imgs = './backend/test_data/imgs'
    test_folder_img = './backend/test_data/imgs/drone_1.jpg'
    test_folder_videos = './backend/test_data/videos/'
    test_streaming = 0
    test_ling = 'https://www.youtube.com/watch?v=wYR23j4Nfik'
    test_link_2 = 'https://www.youtube.com/watch?v=VHwEb86s2SI'

    main(test_folder_imgs)
