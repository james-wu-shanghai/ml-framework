# Docker CUDA 13.0 构建配置 - 2025年9月6日

## 构建说明

本Docker镜像已升级支持CUDA 13.0，包含以下主要变更：

### 1. 基础镜像升级
- **FROM**: `nvidia/cuda:12.4-devel-ubuntu22.04` → `nvidia/cuda:13.0-devel-ubuntu22.04`

### 2. 深度学习框架升级
- **PyTorch**: 升级至2.8+版本，支持CUDA 13.0
- **TensorFlow**: 升级至2.18+版本，支持CUDA 13.0

### 3. 依赖包版本升级
- 核心科学计算库：NumPy 1.26+, Pandas 2.2+, Scikit-learn 1.5+
- 可视化工具：Matplotlib 3.9+, Seaborn 0.13+, Plotly 5.24+
- MLOps工具：MLflow 2.16+, Wandb 0.18+, Optuna 4.0+
- 开发工具：Black 24.10+, pytest 8.3+, mypy 1.13+

## 构建命令

### 开发环境构建
```bash
docker build -t ml-framework:cuda13 .
```

### 生产环境构建（如果有Dockerfile.prod）
```bash
docker build -f Dockerfile.prod -t ml-framework:cuda13-prod .
```

### 带版本标签构建
```bash
docker build -t ml-framework:v0.1.0-cuda13 .
```

## 运行命令

### 启动Jupyter Notebook
```bash
docker run --gpus all -p 8888:8888 -v $(pwd)/data:/app/data ml-framework:cuda13 jupyter
```

### 启动FastAPI服务
```bash
docker run --gpus all -p 8000:8000 ml-framework:cuda13 api
```

### 启动Streamlit应用
```bash
docker run --gpus all -p 8501:8501 ml-framework:cuda13 streamlit
```

### 运行测试
```bash
docker run --gpus all ml-framework:cuda13 test
```

### 训练模型
```bash
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models ml-framework:cuda13 train --data data/my_data.csv --target target
```

### 交互式Shell
```bash
docker run --gpus all -it ml-framework:cuda13 shell
```

## 验证GPU支持

在容器内运行以下命令验证CUDA 13.0支持：

```python
import torch
import tensorflow as tf

# 验证PyTorch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")

# 验证TensorFlow
print(f"TensorFlow版本: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU设备数量: {len(gpus)}")
```

## Docker Compose 配置示例

```yaml
version: '3.8'
services:
  ml-framework:
    build: .
    image: ml-framework:cuda13
    ports:
      - "8888:8888"  # Jupyter
      - "8000:8000"  # FastAPI
      - "8501:8501"  # Streamlit
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - CUDA_VISIBLE_DEVICES=0
```

## 注意事项

### 1. NVIDIA驱动要求
- 确保主机安装了支持CUDA 13.0的NVIDIA驱动
- 推荐驱动版本：545.23.06或更高

### 2. Docker版本要求
- Docker Engine: 19.03+
- nvidia-container-toolkit: 最新版本

### 3. 内存要求
- 推荐8GB+系统内存
- GPU显存：4GB+（训练大模型需要更多）

### 4. 兼容性说明
- 某些旧的PyTorch模型可能需要重新训练或转换
- TensorFlow 2.18+的API变化可能影响现有代码

## 故障排除

### GPU不可用
```bash
# 检查NVIDIA驱动
nvidia-smi

# 检查Docker GPU支持
docker run --rm --gpus all nvidia/cuda:13.0-base-ubuntu22.04 nvidia-smi
```

### 依赖冲突
```bash
# 重新构建镜像（不使用缓存）
docker build --no-cache -t ml-framework:cuda13 .
```

### 容器启动失败
```bash
# 查看容器日志
docker logs <container_id>

# 进入容器调试
docker run --gpus all -it ml-framework:cuda13 bash
```