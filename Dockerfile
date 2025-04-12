# Sử dụng Ubuntu 20.04 làm base image
FROM ubuntu:20.04

# Tránh các prompt tương tác trong quá trình cài đặt
ENV DEBIAN_FRONTEND=noninteractive

# Cài đặt Python 3.8 và các gói phụ thuộc hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    python3.8-dev \
    build-essential \
    git \
    cmake \
    wget \
    libopenblas-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Cập nhật pip và cài đặt virtualenv
RUN python3.8 -m pip install --upgrade pip
RUN pip install virtualenv

# Thiết lập thư mục làm việc
WORKDIR /app

# Tạo thư mục cho mã nguồn và dữ liệu đã xử lý
RUN mkdir -p /app/code /app/processed_data /app/data /app/preprocess /app/reasoning

# Sao chép tất cả các thư mục và file từ thư mục hiện tại
COPY code/ /app/code/
COPY processed_data/ /app/processed_data/
COPY data/ /app/data/ 
COPY preprocess/ /app/preprocess/ 
COPY reasoning/ /app/reasoning/ 
COPY *.py *.yml *.txt *.md /app/ 

# Tạo và kích hoạt môi trường ảo
RUN virtualenv -p python3.8 /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Cài đặt PyTorch và các phụ thuộc
RUN pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Cài đặt các gói phụ thuộc chính
RUN pip install \
    fastapi==0.95.0 \
    uvicorn==0.21.1 \
    pandas==1.5.3 \
    numpy==1.24.2 \
    scikit-learn==1.2.2 \
    requests==2.28.2 \
    pydantic==1.10.7 \
    tqdm==4.65.0

# Cài đặt các gói phụ thuộc PyTorch Geometric
RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.13.1+cpu.html
RUN pip install torch-geometric==2.2.0

# Cài đặt DGL
RUN pip install dgl==1.0.1 -f https://data.dgl.ai/wheels/repo.html

# Cài đặt các gói phụ thuộc khác
RUN pip install \
    requests \
    bert-score==0.3.13 \
    nltk==3.8.1 \
    transformers==4.27.1

# Tải dữ liệu NLTK cần thiết
RUN python -c "import nltk; nltk.download('punkt')"

# Thiết lập biến môi trường đường dẫn
ENV PYTHONPATH="/app:/app/code"

# Mở cổng 8000 để FastAPI có thể truy cập từ bên ngoài container
EXPOSE 8000

# Chỉ định thư mục làm việc là thư mục code
WORKDIR /app/code

# Khởi chạy ứng dụng FastAPI từ thư mục code
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]