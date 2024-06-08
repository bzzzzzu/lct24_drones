from ultralytics import YOLO
from ultralytics.data.utils import autosplit

if __name__ == '__main__':
    model = YOLO("yolov8s.pt")
    model.info()
    model.cuda()

    #autosplit(path="datasets/drones_clean/images/", weights=(0.9, 0.1, 0.0))

    results = model.train(data="drones_clean.yaml", epochs=20, batch=-1, save_period=1, imgsz=1024, workers=8, plots=True)
