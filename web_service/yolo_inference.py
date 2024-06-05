import os
import yaml
from YOLOv8_Efficient.detect import run, ROOT


PATH_TO_YAML = "../drones.yaml"


class YoloInference:
    def __init__(self):
        self.config = self.load_config(PATH_TO_YAML)
        self.dataset_path = self.config["path"]
        self.class_names = self.config["names"]

    def main(
            self,
            weights='../model/yolov8s.pt',  # model path or triton URL
            source='./test_data/airplane_1.jpg',  # ROOT / '0',  # file/dir/URL/glob/screen/0(webcam)
            data='../drones.yaml',  # dataset.yaml path
            imgsz=(640, 640),  # inference size (height, width)
            conf=0.25,  # confidence threshold
            iou=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=False,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            # classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            # update=False,  # update all models
            project='./results/detect',  # save results to project/name
            name='test',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            vid_stride=1,  # video frame-rate stride
            retina_masks=False,  # High resolution masks
            save=False
    ):
        run(
            weights=weights, source=source, data=data, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, device=device,
            view_img=view_img, save_txt=save_txt, save_conf=save_conf, save_crop=save_crop, nosave=nosave,
            agnostic_nms=agnostic_nms, augment=augment, visualize=visualize, project=project, name=name,
            exist_ok=exist_ok, line_thickness=line_thickness, hide_labels=hide_labels, hide_conf=hide_conf, half=half,
            dnn=dnn, vid_stride=vid_stride, retina_masks=retina_masks, save=save
        )
    @staticmethod
    def load_config(filepath: str):
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return yaml.safe_load(f)
        else:
            raise FileNotFoundError()


if __name__ == '__main__':
    yolo = YoloInference()
    yolo.main(
        weights='../model/yolov8s.pt',  # model path or triton URL
        source='./test_data/img.png',  # ROOT / '0',  # file/dir/URL/glob/screen/0(webcam)
        data='../drones.yaml',  # dataset.yaml path
        save=True,
        agnostic_nms=True
    )