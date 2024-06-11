from ultralytics import YOLO
from ultralytics.data.utils import autosplit

if __name__ == '__main__':
    # n/s/m/l/x
    #model = YOLO("yolov8n.pt")
    #autosplit(path="datasets/drones_clean/images/", weights=(0.9, 0.1, 0.0))

    #model = YOLO("yolov8s.pt")
    #results = model.train(data="drones_clean.yaml", device=0, epochs=100, close_mosaic=33, batch=-1, save_period=1, imgsz=1024, workers=8, plots=True)
    #Class     Images  Instances   Box(P          R      mAP50  mAP50-95):
    #all       1630       1803     0.975      0.926       0.95      0.705
    #copter    696        855      0.965        0.9      0.938      0.613
    #plane     182        182      0.991      0.973       0.98      0.863
    #heli      146        146      0.998          1      0.995       0.89
    #bird      67         67       0.96       0.806      0.858      0.473
    #winged    539        553      0.962      0.951       0.98      0.686

    #model = YOLO("yolov8n.pt")
    #model.info()
    #results = model.train(data="drones_clean.yaml", device=0, epochs=100, close_mosaic=33, batch=-1, save_period=1, imgsz=1024, workers=8, plots=True)
    #run aborted - low GPU usage, CPU capped

    #model = YOLO("yolov8m.pt")
    #model.info()
    #results = model.train(data="drones_clean.yaml", device=0, epochs=20, close_mosaic=10, batch=-1, save_period=1, imgsz=1024, workers=8, plots=True)
    #Class     Images  Instances   Box(P          R      mAP50  mAP50-95):
    #all       1630       1803     0.948      0.911      0.934       0.62
    #copter    696        855      0.934      0.846      0.902      0.526
    #plane     182        182      0.971      0.978      0.986        0.8
    #heli      146        146      0.988          1      0.995      0.793
    #bird      67         67       0.895      0.806      0.832      0.391
    #winged    539        553      0.952      0.924      0.956       0.59

    #model = YOLO("yolov8s.pt")
    #model.info()
    #results = model.train(data="drones_clean.yaml", device=0, epochs=20, close_mosaic=10, batch=-1, save_period=1, imgsz=640, workers=8, plots=True)
    #Class     Images  Instances   Box(P          R      mAP50  mAP50-95):
    #all       1630       1803     0.909      0.864      0.899      0.583
    #copter    696        855      0.924      0.786      0.839      0.462
    #plane     182        182      0.957      0.973      0.987      0.778
    #heli      146        146      0.969          1      0.995      0.785
    #bird      67         67       0.767      0.672      0.741      0.332
    #winged    539        553      0.93       0.889      0.934      0.555

    #model = YOLO("yolov8s.pt")
    #model.info()
    #results = model.train(data="drones_clean.yaml", device=0, epochs=20, close_mosaic=10, batch=-1, save_period=1, imgsz=1920, workers=8, plots=True)
    #Class     Images  Instances   Box(P          R      mAP50  mAP50-95):
    #all       1630       1803     0.936      0.921      0.947      0.626
    #copter    696        855      0.94       0.887      0.929      0.553
    #plane     182        182      0.978      0.972       0.99      0.792
    #heli      146        146      0.985          1      0.995       0.77
    #bird      67         67       0.833      0.816      0.853      0.426
    #winged    539        553      0.943      0.931      0.966      0.591