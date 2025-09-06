# ML Framework Docker部署指南

## 🐳 概述

ML Framework提供了完整的Docker化部署方案，支持NVIDIA 4090 + CUDA 12.4的GPU加速环境。

## 📋 前置要求

### 系统要求
- Docker Desktop 20.10+ (Windows/Mac) 或 Docker Engine (Linux)
- NVIDIA GPU驱动 (如需GPU支持)
- NVIDIA Container Toolkit (如需GPU支持)

### GPU支持配置
1. **安装NVIDIA驱动**：确保安装最新的NVIDIA驱动
2. **安装Docker Desktop**：从官网下载并安装
3. **配置NVIDIA Container Toolkit**：
   ```bash
   # Linux
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

## 🚀 快速开始

### 方法1：使用快速启动脚本（推荐）

**Windows**：
```cmd
start_docker.bat
```

**Linux/Mac**：
```bash
chmod +x start_docker.sh
./start_docker.sh
```

### 方法2：使用Python构建脚本

```bash
# 构建完整版镜像
python build_docker.py --type full

# 构建生产版镜像
python build_docker.py --type prod

# 构建并部署服务
python build_docker.py --deploy

# 查看使用说明
python build_docker.py --usage
```

### 方法3：手动构建和运行

```bash
# 1. 构建镜像
docker build -t ml-framework:latest .

# 2. 运行容器（选择一种）
# Jupyter Notebook
docker run -it --gpus all -p 8888:8888 -v $(pwd)/data:/app/data ml-framework:latest jupyter

# FastAPI服务
docker run -it --gpus all -p 8000:8000 -v $(pwd)/data:/app/data ml-framework:latest api

# 交互式Shell
docker run -it --gpus all -v $(pwd)/data:/app/data ml-framework:latest shell
```

## 🛠️ Docker Compose部署

### 启动所有服务
```bash
# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f ml-framework

# 停止服务
docker-compose down
```

### 可用服务
- **Jupyter Notebook**: http://localhost:8888
- **FastAPI API**: http://localhost:8000
- **Streamlit应用**: http://localhost:8501
- **MLflow**: http://localhost:5000
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379

## 🔧 镜像变体

### 完整版镜像 (Dockerfile)
- 包含所有依赖和工具
- 支持Jupyter、开发工具
- 适合开发和实验
- 镜像大小：~8GB

### 生产版镜像 (Dockerfile.prod)
- 只包含运行时依赖
- 优化的轻量级镜像
- 适合生产部署
- 镜像大小：~4GB

## 📊 使用示例

### 1. 训练模型
```bash
# 准备数据文件到data目录
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  ml-framework:latest \
  train --data data/my_dataset.csv --target target_column
```

### 2. 启动API服务
```bash
docker run -d --gpus all \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  --name ml-api \
  ml-framework:latest api
```

### 3. 批量预测
```bash
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  ml-framework:latest \
  predict --model models/my_model.joblib --data data/new_data.csv
```

### 4. 开发环境
```bash
# 启动带有卷挂载的开发环境
docker run -it --gpus all \
  -p 8888:8888 \
  -v $(pwd):/app \
  ml-framework:latest shell

# 在容器内
python examples/basic_classification.py
jupyter notebook --ip=0.0.0.0 --allow-root
```

## 🎯 最佳实践

### 1. 数据管理
```bash
# 创建数据卷
docker volume create ml-data
docker volume create ml-models

# 使用数据卷
docker run --gpus all \
  -v ml-data:/app/data \
  -v ml-models:/app/models \
  ml-framework:latest
```

### 2. 环境变量配置
```bash
# 使用环境文件
echo "CUDA_VISIBLE_DEVICES=0" > .env
echo "BATCH_SIZE=128" >> .env

docker run --gpus all --env-file .env ml-framework:latest
```

### 3. 网络配置
```bash
# 创建自定义网络
docker network create ml-network

# 在网络中运行服务
docker run --gpus all --network ml-network ml-framework:latest
```

### 4. 资源限制
```bash
# 限制GPU内存和CPU使用
docker run --gpus all \
  --memory=16g \
  --cpus=8 \
  --shm-size=8g \
  ml-framework:latest
```

## 🔍 故障排除

### 常见问题

**1. GPU不可用**
```bash
# 检查NVIDIA驱动
nvidia-smi

# 检查Docker GPU支持
docker run --rm --gpus all nvidia/cuda:12.4-base-ubuntu22.04 nvidia-smi

# 检查容器内GPU
docker run --rm --gpus all ml-framework:latest python -c "import torch; print(torch.cuda.is_available())"
```

**2. 内存不足**
```bash
# 增加Docker内存限制
docker run --gpus all --memory=32g --shm-size=16g ml-framework:latest

# 清理Docker缓存
docker system prune -f
docker volume prune -f
```

**3. 端口冲突**
```bash
# 使用不同端口
docker run --gpus all -p 8889:8888 ml-framework:latest jupyter

# 查看端口使用
netstat -tulpn | grep :8888
```

**4. 权限问题**
```bash
# 使用用户ID映射
docker run --gpus all -u $(id -u):$(id -g) ml-framework:latest

# 修改文件权限
sudo chown -R $USER:$USER data/ models/
```

### 日志调试
```bash
# 查看容器日志
docker logs container_name

# 实时查看日志
docker logs -f container_name

# 进入运行中的容器
docker exec -it container_name /bin/bash
```

## 📈 性能优化

### 1. GPU性能
```bash
# 启用所有GPU
docker run --gpus all ml-framework:latest

# 指定特定GPU
docker run --gpus device=0 ml-framework:latest

# 设置GPU内存限制
docker run --gpus all --env CUDA_MEM_FRACTION=0.8 ml-framework:latest
```

### 2. 网络性能
```bash
# 使用host网络模式
docker run --gpus all --network host ml-framework:latest

# 优化共享内存
docker run --gpus all --shm-size=16g ml-framework:latest
```

### 3. 存储性能
```bash
# 使用tmpfs for临时数据
docker run --gpus all --tmpfs /tmp:rw,noexec,nosuid,size=4g ml-framework:latest

# 使用绑定挂载而不是卷
docker run --gpus all -v /host/path:/container/path:cached ml-framework:latest
```

## 🚀 生产部署

### Kubernetes部署
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-framework
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-framework
  template:
    metadata:
      labels:
        app: ml-framework
    spec:
      containers:
      - name: ml-framework
        image: ml-framework:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
          requests:
            memory: "8Gi"
        ports:
        - containerPort: 8000
```

### Docker Swarm部署
```bash
# 初始化Swarm
docker swarm init

# 部署Stack
docker stack deploy -c docker-compose.yml ml-framework
```

这个Docker化方案为你的ML Framework提供了完整的容器化支持，充分利用了NVIDIA 4090的GPU性能！🚀