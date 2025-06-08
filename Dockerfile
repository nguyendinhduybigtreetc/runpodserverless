FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Cài công cụ hệ thống, dev tool, CUDA env
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    software-properties-common \
    && add-apt-repository ppa:savoury1/ffmpeg4 -y \
    && apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && apt-get clean

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

WORKDIR /workspace

RUN git clone https://github.com/deepbeepmeep/Wan2GP.git
WORKDIR /workspace/Wan2GP
RUN pip install -r requirements.txt

RUN pip install sageattention==1.0.6

WORKDIR /workspace/Wan2GP

CMD ["python", "wgp.py", "--help"]
