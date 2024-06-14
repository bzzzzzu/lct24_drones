# FROM python:3.10-slim
# FROM borda/docker_python-opencv-ffmpeg
FROM borda/docker_python-opencv-ffmpeg:gpu-py3.10-cv4.10.0

# RUN apt-get update && apt-get -o Acquire::Retries=5 install ffmpeg libsm6 libxext6 -y

COPY ./docker/flask/requirements.txt requirements.txt
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

# Expose the Triton server port
EXPOSE 5000
WORKDIR web_app

# Start the Triton server with the model repository
CMD ["python", "flaskapp.py"]
# CMD pip list