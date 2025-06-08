FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Cài đặt các công cụ cần thiết và ffmpeg từ PPA savoury1
RUN apt-get update && apt-get install -y \
    git \
    software-properties-common \
    && add-apt-repository ppa:savoury1/ffmpeg4 -y \
    && apt-get update && apt-get install -y ffmpeg \
    && apt-get clean

# Làm việc trong thư mục chính
WORKDIR /workspace

# Clone và cài đặt Wan2GP
RUN git clone https://github.com/deepbeepmeep/Wan2GP.git
WORKDIR /workspace/Wan2GP
RUN pip install -r requirements.txt

# Clone và cài đặt SageAttention
WORKDIR /workspace
RUN git clone https://github.com/thu-ml/SageAttention.git
WORKDIR /workspace/SageAttention
RUN pip install -e .

# Set working directory mặc định khi container khởi chạy
WORKDIR /workspace/Wan2GP

# Lệnh sẽ chạy khi khởi động serverless (có thể override)
COPY . .

CMD ["python", "runpod_serverless.py"]
