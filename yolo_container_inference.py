from pathlib import Path
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

    def get_prediction(self, source):
        return self.model(
            source=source,
            device=0,
            stream=True,  # Enable streaming
            data=self.path_to_config,
            max_det=len(self.class_names),
            classes=list(self.class_names.keys()),

            save=True,
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            project='./triton_repo/detects',
            name='runs'
        )


def main(source):
    yolo_inference = YoloContainerInference()
    results = yolo_inference.get_prediction(source)

    if source == 0: # streaming mode
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
    test_folder_imgs = './triton_repo/test_data/imgs'
    test_folder_img = './triton_repo/test_data/imgs/drone_1.jpg'
    test_folder_videos = './triton_repo/test_data/videos/'
    test_streaming = 0

    main(test_streaming)
