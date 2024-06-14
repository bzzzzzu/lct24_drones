from pathlib import Path
import os
import shutil
import contextlib
import subprocess
import time
from ultralytics import YOLO
from tritonclient.http import InferenceServerClient

WORKDIR = Path(__file__).parent.absolute()

CONFIG_YOLO = """platform: "onnxruntime_onnx"
max_batch_size: 0
input [
{
  name: "images"
  data_type: TYPE_FP32
  dims: [ 1,3,1024,1024 ]
}
]
output [
{
  name: "output0"
  data_type: TYPE_FP32
  dims: [-1, -1, -1]
}
]
"""

class YoloWithTritonServer:
    def __init__(self, model_pt: str='./best_0700_fixed_birds.pt'):
        self.model_pt = model_pt
        self.model_name = "yolo"
        self.triton_repo_path = Path(WORKDIR) / "triton_repo"

        self.triton_model_path = self.triton_repo_path / self.model_name
        self.tag = "nvcr.io/nvidia/tritonserver:23.09-py3"  # 6.4 GB
        self.container_id = None
        self.triton_client = None

        if Path(self.triton_repo_path, 'yolo').is_dir():
            shutil.rmtree(Path(self.triton_repo_path, 'yolo'))

        self.load_and_convert_model()

    def load_and_convert_model(self):
        if os.path.exists(self.model_pt):
            model_pt = YOLO(f"{self.model_pt}")
            onnx_file = model_pt.export(format="onnx", dynamic=True, batch=16)

            os.makedirs(Path(self.triton_model_path) / "1", exist_ok=True)
            Path(onnx_file).rename(self.triton_model_path / "1" / "model.onnx")
            config_path = os.path.join(self.triton_model_path, "config.pbtxt")
            with open(config_path, 'w') as config_file:
                os.utime(config_path, None)
                # config_file.write(CONFIG_YOLO)
        else:
            raise FileNotFoundError(f"Model path {self.model_pt} does not exist")

    def start_container(self):
        subprocess.call(f"docker pull {self.tag}", shell=True)
        self.container_id = (
            subprocess.check_output(
                f"docker run -it --ipc=host --gpus all -d -v {self.triton_repo_path}:/models -p 8000:8000 {self.tag} tritonserver --model-repository=/models",
                shell=True,
            )
            .decode("utf-8")
            .strip()
        )
        print(f"Started container {self.container_id}")

        # Wait for the Triton server to start
        self.triton_client = InferenceServerClient(url="localhost:8000", verbose=False, ssl=False)

        # Wait until model is ready
        for i in range(10):
            try:
                if self.triton_client.is_model_ready(self.model_name):
                    print("Model is ready")
                    break
            except Exception as e:
                print(f"Waiting for model to be ready ({i + 1}/10): {e}")
            time.sleep(1)
        else:
            print("Model is not ready after waiting")
            self.stop_container()
            raise RuntimeError("Failed to start Triton server with the model")

    def stop_container(self):
        if self.container_id:
            # Kill and remove the container
            subprocess.call(f"docker kill {self.container_id}", shell=True)
            print(f"Stopped and removed container {self.container_id}")


if __name__ == "__main__":
    triton_server = YoloWithTritonServer()
    try:
        triton_server.start_container()
    except Exception as e:
        print(f"Error: {e}")
        triton_server.stop_container()
