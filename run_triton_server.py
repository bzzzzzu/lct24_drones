from pathlib import Path
import os
import contextlib
import subprocess
import time
from ultralytics import YOLO
from tritonclient.http import InferenceServerClient

WORKDIR = Path(__file__).parent.absolute()

class YoloWithTritonServer:
    def __init__(self, model_pt: str="yolov8s.pt"):
        self.model_pt = model_pt
        self.model_name = "yolo"
        self.triton_repo_path = Path(WORKDIR) / "triton_repo"

        self.triton_model_path = self.triton_repo_path / self.model_name
        self.tag = "nvcr.io/nvidia/tritonserver:23.09-py3"  # 6.4 GB
        self.container_id = None
        self.triton_client = None

        if not os.path.exists(Path(WORKDIR) / self.model_pt):
            self.load_and_convert_model()

    def load_and_convert_model(self):
        model_pt = YOLO(f"{self.model_pt}")
        onnx_file = model_pt.export(format="onnx", dynamic=True)

        os.makedirs(Path(self.triton_model_path) / "1", exist_ok=True)
        Path(onnx_file).rename(self.triton_model_path / "1" / "model.onnx")
        config_path = os.path.join(self.triton_model_path, "config.pbtxt")
        with open(config_path, 'a'):
            os.utime(config_path, None)

    def start_container(self):
        subprocess.call(f"docker pull {self.tag}", shell=True)

        self.container_id = (
            subprocess.check_output(
                f"docker run -it --ipc=host --gpus all -d --rm -v {self.triton_repo_path}:/models -p 8000:8000 {self.tag} tritonserver --model-repository=/models",
                shell=True,
            )
            .decode("utf-8")
            .strip()
        )

        self.triton_client = InferenceServerClient(url="localhost:8000", verbose=False, ssl=False)
        for _ in range(10):
            with contextlib.suppress(Exception):
                assert self.triton_client.is_model_ready("yolo")
                break
            time.sleep(1)

    def stop_container(self):
        if self.triton_client:
            # Kill and remove the container at the end of the test
            subprocess.call(f"docker kill {self.container_id}", shell=True)


if __name__ == "__main__":
    triton_server = YoloWithTritonServer()
    triton_server.start_container()
    # triton_server.stop_container()
