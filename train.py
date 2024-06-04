from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8s.pt")
    model.info()
    model.cuda()

    results = model.train(data="drones.yaml", epochs=5, imgsz=640)