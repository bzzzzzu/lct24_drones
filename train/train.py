# Для тренировки нужно скопировать файлы:
# datasets/drones_clean/images/ и datasets/drones_clean/labels из датасетов задачи, только файлы из filter_train.txt и filter_val.txt
# datasets/birds_fixed_labels/images/ из датасетов задачи, только файлы из train.txt
# datasets/birds_fixed_labels/labels/ из labels.zip

from ultralytics import YOLO

if __name__ == '__main__':
    # n/s/m/l/x

    # sgd with fixed birds, 100 epoch, 1280 size
    model = YOLO("yolov8s.pt")
    model.info()
    results = model.train(data="drones_clean_with_fixed_birds.yaml", device=0, epochs=100, close_mosaic=20, batch=-1, save_period=1, imgsz=1280, workers=10, plots=True, optimizer='SGD')
    #Class     Images  Instances   Box(P          R      mAP50  mAP50-95):
    #all       1630       1803     0.974       0.93       0.96        0.7
    #copter    696        855      0.964      0.911       0.95      0.619
    #plane     182        182      0.985      0.978      0.988      0.869
    #heli      146        146      0.997          1      0.995      0.868
    #bird      67         67       0.948       0.81      0.886      0.458
    #winged    539        553      0.975      0.949      0.982      0.683