# ML Framework Docker镜像
# 支持NVIDIA GPU + CUDA 13.0的GPU加速机器学习框架

# 使用NVIDIA官方CUDA基础镜像，支持CUDA 13.0
FROM nvidia/cuda:13.0-devel-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-venv \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    pkg-config \
    libhdf5-dev \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# 创建Python符号链接
RUN ln -s /usr/bin/python3.11 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# 升级pip和安装基础工具
RUN pip install --upgrade pip setuptools wheel

# 复制项目文件
COPY requirements.txt .
COPY setup.py .
COPY MANIFEST.in .
COPY README.md .
COPY LICENSE .

# 复制源代码和配置
COPY src/ src/
COPY configs/ configs/
COPY examples/ examples/
COPY docs/ docs/

# 创建必要的目录
RUN mkdir -p data models logs plots tests

# 安装Python依赖 - 分阶段安装以优化缓存
# 1. 安装核心依赖
RUN pip install --no-cache-dir \
    numpy>=1.26.0 \
    pandas>=2.2.0 \
    scikit-learn>=1.5.0 \
    scipy>=1.14.0 \
    matplotlib>=3.9.0 \
    seaborn>=0.13.0 \
    pyyaml>=5.4.0 \
    click>=8.0.0 \
    joblib>=1.0.0 \
    tqdm>=4.61.0

# 2. 安装PyTorch (CUDA 13.0支持)
RUN pip install --no-cache-dir \
    torch>=2.8.0 \
    torchvision>=0.18.0 \
    torchaudio>=2.8.0 \
    --index-url https://download.pytorch.org/whl/cu130

# 3. 安装TensorFlow (CUDA 13.0支持)
RUN pip install --no-cache-dir \
    tensorflow>=2.18.0 \
    keras>=3.6.0

# 4. 安装数据处理和可视化库
RUN pip install --no-cache-dir \
    plotly>=5.24.0 \
    bokeh>=3.6.0 \
    openpyxl>=3.0.0 \
    xlrd>=2.0.0 \
    h5py>=3.1.0 \
    pillow>=8.0.0

# 5. 安装ML工具库
RUN pip install --no-cache-dir \
    shap>=0.39.0 \
    lime>=0.2.0 \
    optuna>=4.0.0 \
    mlflow>=2.16.0 \
    wandb>=0.18.0 \
    tensorboard>=2.18.0

# 6. 安装Web框架和部署工具
RUN pip install --no-cache-dir \
    flask>=3.0.0 \
    fastapi>=0.115.0 \
    uvicorn>=0.32.0 \
    streamlit>=1.40.0

# 7. 安装开发工具
RUN pip install --no-cache-dir \
    pytest>=8.3.0 \
    pytest-cov>=6.0.0 \
    black>=24.10.0 \
    jupyter>=1.1.0 \
    ipykernel>=6.29.0 \
    ipywidgets>=8.1.0

# 安装项目本身
RUN pip install -e .

# 设置GPU环境优化
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_VISIBLE_DEVICES=0

# 创建非root用户
RUN useradd -m -u 1000 mluser && \
    chown -R mluser:mluser /app
USER mluser

# 暴露端口
EXPOSE 8000 8080 8501 5000

# 设置健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import torch, tensorflow as tf; print('GPU:', torch.cuda.is_available(), len(tf.config.list_physical_devices('GPU')))" || exit 1

# 创建启动脚本
COPY <<EOF /app/start.sh
#!/bin/bash
set -e

echo "🚀 ML Framework Docker容器启动 (CUDA 13.0)"
echo "============================================"

# 检查GPU环境
echo "🔍 检查GPU环境..."
python -c "
import torch
import tensorflow as tf
print(f'PyTorch版本: {torch.__version__}')
print(f'PyTorch CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU设备: {torch.cuda.get_device_name(0)}')
    print(f'CUDA版本: {torch.version.cuda}')

print(f'TensorFlow版本: {tf.__version__}')
tf_gpus = tf.config.list_physical_devices('GPU')
print(f'TensorFlow GPU可用: {len(tf_gpus) > 0}')
"

# 运行GPU优化
echo "⚙️ 应用GPU优化设置..."
python -c "
from src.ml_framework.gpu_utils import gpu_manager
gpu_manager.optimize_pytorch_gpu()
gpu_manager.optimize_tensorflow_gpu()
print('GPU优化完成')
"

# 根据参数启动不同服务
case "\$1" in
    "jupyter")
        echo "📓 启动Jupyter Notebook..."
        jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''
        ;;
    "api")
        echo "🌐 启动FastAPI服务 (CUDA 13.0支持)..."
        uvicorn src.ml_framework.api:app --host 0.0.0.0 --port 8000
        ;;
    "streamlit")
        echo "📊 启动Streamlit应用 (CUDA 13.0支持)..."
        streamlit run src/ml_framework/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
        ;;
    "test")
        echo "🧪 运行测试 (CUDA 13.0环境)..."
        python test_framework.py
        python test_gpu.py
        ;;
    "train")
        echo "🎯 启动训练模式..."
        python -m ml_framework.cli train "\$@"
        ;;
    "shell")
        echo "💻 启动交互式shell (CUDA 13.0环境)..."
        /bin/bash
        ;;
    *)
        echo "📖 ML Framework使用说明"
        echo "可用命令:"
        echo "  jupyter   - 启动Jupyter Notebook (端口8888)"
        echo "  api       - 启动FastAPI服务 (端口8000)"
        echo "  streamlit - 启动Streamlit应用 (端口8501)"
        echo "  test      - 运行框架测试"
        echo "  train     - 训练模型"
        echo "  shell     - 启动交互式shell"
        echo ""
        echo "示例:"
        echo "  docker run --gpus all -p 8888:8888 ml-framework jupyter"
        echo "  docker run --gpus all -p 8000:8000 ml-framework api"
        echo "  docker run --gpus all -v \$(pwd)/data:/app/data ml-framework train --data data/my_data.csv --target target"
        ;;
esac
EOF

RUN chmod +x /app/start.sh

# 默认启动命令
CMD ["/app/start.sh"]