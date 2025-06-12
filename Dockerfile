FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Cài công cụ hệ thống, dev tool, CUDA env
#RUN add-apt-repository ppa:savoury1/ffmpeg4 -y
RUN apt-get update
RUN apt-get install ffmpeg -y

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

WORKDIR /workspace

RUN git clone https://github.com/deepbeepmeep/Wan2GP.git
WORKDIR /workspace/Wan2GP
RUN pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu124
RUN pip install -r requirements.txt


RUN pip install sageattention==1.0.6


WORKDIR /workspace/Wan2GP

CMD ["python", "wgp.py", "--help"]
