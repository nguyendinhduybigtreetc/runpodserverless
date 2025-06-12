FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

# Cài công cụ hệ thống, dev tool, CUDA env
#RUN apt-get add-apt-repository ppa:savoury1/ffmpeg4 -y \
#    && apt-get update && apt-get install -y \
#    ffmpeg \

WORKDIR /workspace

RUN git clone https://github.com/deepbeepmeep/Wan2GP.git
WORKDIR /workspace/Wan2GP
RUN pip install -r requirements.txt

RUN nvidia-smi
RUN git clone https://github.com/thu-ml/SageAttention.git
WORKDIR /workspace/Wan2GP/SageAttention
RUN TORCH_CUDA_ARCH_LIST=Turing python setup.py install --user


WORKDIR /workspace/Wan2GP

CMD ["python", "wgp.py", "--help"]
