FROM python:3.10-slim

# 安裝系統相依套件
RUN apt-get update && apt-get install -y \
    build-essential libopenblas-dev libssl-dev libffi-dev git \
    && rm -rf /var/lib/apt/lists/*

# 設定工作目錄
WORKDIR /app

# 升級 pip 並安裝 PyTorch（CPU 版本）
RUN pip install --upgrade pip
RUN pip install torch==2.2.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 避免 git 安全性警告
RUN git config --global --add safe.directory /app

# 複製 requirements 並安裝
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --upgrade gradio


# 把整個專案複製進來（包含 main.py、server.py 等）
COPY . .

# 啟動 container 時執行 main.py
CMD ["python", "main.py"]
