Обнаружение дронов нейросетью

Основные компоненты решения:

docker-compose.yml
docker/ - разворачивание решения на сервере

web_app/ - веб-сервер + инференс

triton_repo/ - модели для Triton Inference Server (опционально)

train/ - инструкции по обучению модели (опционально)

Используемые технологии - Yolov8, PyTorch, Flask, Triton, OpenCV, Docker

Развертывание решения:
  залить репозиторий вместе с моделью на сервер
  docker-compose up --build
  localhost:5000 или http://ip_сервера:5000