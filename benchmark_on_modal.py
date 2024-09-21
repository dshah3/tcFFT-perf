import subprocess
import os
import sys
import datetime

import modal
from modal import Image, Stub

"""
For tcfft_half_speed: 
GPU_MEM=80 modal run benchmark_on_modal.py \
    --compile-command "nvcc speed.cpp tcfft_doit_half.cpp tcfft_half.cu -o tcfft_half_speed -std=c++11 -lcublas -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 -res-usage -lcudart -lfftw3 -lcufft -lineinfo -Xcompiler -fopenmp" \
    --run-command "./tcfft_half_speed -b 1048576 -n 256"

For tcfft_half_accuracy:
GPU_MEM=80 modal run benchmark_on_modal.py \
    --compile-command "nvcc accuracy.cpp tcfft_doit_half.cpp tcfft_half.cu -o tcfft_half_accuracy -std=c++11 -lcublas -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 -res-usage -lcudart -lfftw3 -lcufft -lineinfo -Xcompiler -fopenmp" \
    --run-command "./tcfft_half_accuracy -b 32 -n 256"

For cufft_half_speed:
GPU_MEM=80 modal run benchmark_on_modal.py \
    --compile-command "nvcc speed.cpp cufft_doit_half.cpp -o cufft_half_speed -std=c++11 -lcublas -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 -res-usage -lcudart -lfftw3 -lcufft -lineinfo -Xcompiler -fopenmp" \
    --run-command "./cufft_half_speed -b 1048576 -n 256"

For cufft_half_accuracy:
GPU_MEM=80 modal run benchmark_on_modal.py \
    --compile-command "nvcc accuracy.cpp cufft_doit_half.cpp -o cufft_half_accuracy -std=c++11 -lcublas -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 -res-usage -lcudart -lfftw3 -lcufft -lineinfo -Xcompiler -fopenmp" \
    --run-command "./cufft_half_accuracy -b 32 -n 256"

Extensions to 2D ones are simple
"""


GPU_NAME_TO_MODAL_CLASS_MAP = {
    "H100": modal.gpu.H100,
    "A100": modal.gpu.A100,
    "A10G": modal.gpu.A10G,
}
N_GPUS = int(os.environ.get("N_GPUS", 1))
GPU_MEM = int(os.environ.get("GPU_MEM", 40))
GPU_NAME = os.environ.get("GPU_NAME", "H100")
if GPU_NAME == "H100":
    GPU_CONFIG = GPU_NAME_TO_MODAL_CLASS_MAP[GPU_NAME](count=N_GPUS)
else:
    GPU_CONFIG = GPU_NAME_TO_MODAL_CLASS_MAP[GPU_NAME](count=N_GPUS, size=str(GPU_MEM) + 'GB')

APP_NAME = "tcFFT benchmark run"

image = (
    Image.from_registry("totallyvyom/cuda-env:latest-2")
    .pip_install("huggingface_hub==0.20.3", "hf-transfer==0.1.5")
    .env(
        dict(
            HUGGINGFACE_HUB_CACHE="/pretrained",
            HF_HUB_ENABLE_HF_TRANSFER="1",
            TQDM_DISABLE="true",
        )
    )
    .run_commands(
    "wget -q https://github.com/Kitware/CMake/releases/download/v3.28.1/cmake-3.28.1-Linux-x86_64.sh",
    "bash cmake-3.28.1-Linux-x86_64.sh --skip-license --prefix=/usr/local",
    "rm cmake-3.28.1-Linux-x86_64.sh",
    "ln -s /usr/local/bin/cmake /usr/bin/cmake",)
    .run_commands(
        "apt-get install -y --allow-change-held-packages libcudnn8 libcudnn8-dev",
        "apt-get install -y openmpi-bin openmpi-doc libopenmpi-dev kmod sudo",
        "git clone https://github.com/NVIDIA/cudnn-frontend.git /root/cudnn-frontend",
        "cd /root/cudnn-frontend && mkdir build && cd build && cmake .. && make"
    )
    .run_commands(
        "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin && \
        mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
        apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub && \
        add-apt-repository \"deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /\" && \
        apt-get update"
    ).run_commands(
        "apt-get install -y nsight-systems-2023.3.3"
    )
    .run_commands(
        "apt-get install -y libfftw3-dev"
    )
)

stub = modal.App(APP_NAME)

def execute_command(command: str):
    command_args = command.split(" ")
    print(f"{command_args = }")
    subprocess.run(command_args, stdout=sys.stdout, stderr=subprocess.STDOUT)

@stub.function(
    gpu=GPU_CONFIG,
    image=image,
    allow_concurrent_inputs=4,
    container_idle_timeout=900,
    mounts=[modal.Mount.from_local_dir("./", remote_path="/root/")],
    # volumes={"/cuda-env": modal.Volume.from_name("cuda-env")},
)
def run_benchmark(compile_command: str, run_command: str):
    execute_command("pwd")
    execute_command("ls")
    execute_command(compile_command)
    execute_command(run_command)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    execute_command("mkdir report1_" + timestamp)
    execute_command("mv /root/report1.nsys-rep /root/report1_" + timestamp + "/")
    execute_command("mv /root/report1.qdstrm /root/report1_" + timestamp + "/")
    execute_command("mv /root/report1_" + timestamp + "/" + " /cuda-env/")

    return None

@stub.local_entrypoint()
def inference_main(compile_command: str, run_command: str):
    results = run_benchmark.remote(compile_command, run_command)
    return results