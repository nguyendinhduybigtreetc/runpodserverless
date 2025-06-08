FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Cài các gói hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    git \
    software-properties-common \
    && add-apt-repository ppa:savoury1/ffmpeg4 -y \
    && apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && apt-get clean

# Set biến môi trường CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Tạo thư mục làm việc
WORKDIR /workspace

# Clone và cài Wan2GP
RUN git clone https://github.com/deepbeepmeep/Wan2GP.git
WORKDIR /workspace/Wan2GP
RUN pip install -r requirements.txt

# Clone và cài SageAttention
WORKDIR /workspace
RUN git clone https://github.com/thu-ml/SageAttention.git
WORKDIR /workspace/SageAttention
RUN pip install -e .

# Trở lại thư mục chính
WORKDIR /workspace/Wan2GP

# Lệnh mặc định
CMD ["python", "wgp.py", "--help"]
