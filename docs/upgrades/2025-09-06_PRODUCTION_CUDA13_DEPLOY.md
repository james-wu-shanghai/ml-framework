# 生产环境 CUDA 13.0 部署指南 - 2025年9月6日

## 📋 概述

本文档描述如何在生产环境中部署支持CUDA 13.0的ML Framework。生产环境Dockerfile已优化为轻量级镜像，专注于运行时性能和安全性。

## 🏗️ 生产环境特性

### 1. 镜像优化
- **基础镜像**: `nvidia/cuda:13.0-runtime-ubuntu22.04` (精简运行时版本)
- **镜像大小**: 相比开发版本减少约40%
- **安全性**: 移除开发工具，仅保留运行时必需组件
- **启动速度**: 优化的依赖安装和缓存策略

### 2. 依赖精简
- **核心ML库**: PyTorch 2.8+, TensorFlow 2.18+, Scikit-learn 1.5+
- **Web框架**: FastAPI 0.115+, Flask 3.0+, Uvicorn 0.32+
- **数据处理**: NumPy 1.26+, Pandas 2.2+
- **移除组件**: Jupyter, 开发工具, 可视化库

### 3. 性能优化
- **Python字节码**: 禁用`.pyc`文件写入
- **缓存清理**: 自动清理pip缓存
- **非root用户**: 安全的用户权限设置

## 🚀 构建和部署

### 1. 构建生产镜像

```bash
# 基础构建
docker build -f Dockerfile.prod -t ml-framework:prod-cuda13 .

# 带版本标签构建
docker build -f Dockerfile.prod -t ml-framework:v0.1.0-prod-cuda13 .

# 多平台构建（如需要）
docker buildx build --platform linux/amd64 -f Dockerfile.prod -t ml-framework:prod-cuda13 .
```

### 2. 验证镜像

```bash
# 快速验证
docker run --gpus all ml-framework:prod-cuda13 python -c "import torch, tensorflow as tf; print('GPU:', torch.cuda.is_available())"

# 健康检查
docker run --gpus all --rm ml-framework:prod-cuda13 python -c "
import torch
import tensorflow as tf
print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
print(f'TensorFlow: {tf.__version__}, GPU: {len(tf.config.list_physical_devices(\"GPU\"))}')
"
```

### 3. 生产环境运行

#### API服务部署
```bash
# 单容器部署
docker run -d \
  --name ml-framework-api \
  --gpus all \
  -p 8000:8000 \
  --restart unless-stopped \
  -e PYTHONUNBUFFERED=1 \
  -v /path/to/models:/app/models:ro \
  -v /path/to/logs:/app/logs \
  ml-framework:prod-cuda13

# 健康检查
curl http://localhost:8000/health
```

#### 使用Docker Compose
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  ml-api:
    image: ml-framework:prod-cuda13
    container_name: ml-framework-prod
    restart: unless-stopped
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs
      - ./data:/app/data:ro
    environment:
      - PYTHONUNBUFFERED=1
      - ML_FRAMEWORK_LOGGING_LEVEL=INFO
      - ML_FRAMEWORK_MODELS_DIR=/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # 可选：Redis缓存
  redis:
    image: redis:7-alpine
    container_name: ml-redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  # 可选：监控
  prometheus:
    image: prom/prometheus:latest
    container_name: ml-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

volumes:
  redis_data:
```

#### 部署命令
```bash
# 使用Docker Compose部署
docker-compose -f docker-compose.prod.yml up -d

# 查看日志
docker-compose -f docker-compose.prod.yml logs -f ml-api

# 扩容（如需要）
docker-compose -f docker-compose.prod.yml up -d --scale ml-api=3
```

## 🔧 生产环境配置

### 1. 环境变量配置

```bash
# 核心配置
export ML_FRAMEWORK_LOGGING_LEVEL=INFO
export ML_FRAMEWORK_MODELS_DIR=/app/models
export ML_FRAMEWORK_DATA_DIR=/app/data

# GPU配置
export CUDA_VISIBLE_DEVICES=0
export NVIDIA_VISIBLE_DEVICES=all

# 性能调优
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### 2. 卷挂载配置

```bash
# 模型目录（只读）
-v /production/models:/app/models:ro

# 数据目录（只读）
-v /production/data:/app/data:ro

# 日志目录（读写）
-v /production/logs:/app/logs

# 配置文件
-v /production/config.yaml:/app/configs/production.yaml:ro
```

### 3. 资源限制

```yaml
# Docker Compose资源限制
deploy:
  resources:
    limits:
      memory: 8G
      cpus: '4.0'
    reservations:
      memory: 4G
      cpus: '2.0'
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## 🔐 安全配置

### 1. 用户权限
- 容器以非root用户(mluser)运行
- 模型和数据目录使用只读挂载
- 限制网络访问和端口暴露

### 2. 镜像安全扫描

```bash
# 使用Trivy扫描
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy:latest image ml-framework:prod-cuda13

# 使用Clair扫描（如有）
clair-scanner ml-framework:prod-cuda13
```

### 3. 运行时安全

```bash
# 限制容器权限
docker run \
  --security-opt no-new-privileges \
  --read-only \
  --tmpfs /tmp \
  --gpus all \
  ml-framework:prod-cuda13
```

## 📊 监控和日志

### 1. 健康检查

```bash
# API健康检查端点
GET /health
GET /metrics
GET /gpu-status
```

### 2. 日志配置

```python
# 生产环境日志配置
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  handlers:
    - type: "file"
      filename: "/app/logs/ml_framework.log"
      max_bytes: 10485760  # 10MB
      backup_count: 5
    - type: "console"
      stream: "stdout"
```

### 3. 性能监控

```bash
# GPU使用监控
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv --loop=1

# 容器资源监控
docker stats ml-framework-api

# 应用监控（Prometheus格式）
curl http://localhost:8000/metrics
```

## 🚨 故障排除

### 1. 常见问题

#### GPU不可用
```bash
# 检查NVIDIA驱动
nvidia-smi

# 检查容器GPU访问
docker run --gpus all nvidia/cuda:13.0-base nvidia-smi

# 验证PyTorch GPU支持
docker exec ml-framework-api python -c "import torch; print(torch.cuda.is_available())"
```

#### 内存不足
```bash
# 检查GPU内存
nvidia-smi

# 检查系统内存
docker exec ml-framework-api free -h

# 优化内存使用
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
```

#### 性能问题
```bash
# 检查CPU使用
docker exec ml-framework-api top

# 检查IO性能
docker exec ml-framework-api iostat -x 1

# 网络延迟测试
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/health
```

### 2. 日志分析

```bash
# 查看容器日志
docker logs ml-framework-api --tail 100 -f

# 查看应用日志
docker exec ml-framework-api tail -f /app/logs/ml_framework.log

# 日志聚合（如使用ELK）
# 配置Filebeat收集容器日志
```

## 📈 性能优化

### 1. CUDA优化

```bash
# 环境变量优化
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 预分配GPU内存
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

### 2. 并发优化

```bash
# Uvicorn并发配置
uvicorn src.ml_framework.api:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker
```

### 3. 缓存优化

```bash
# Redis缓存配置
export ML_FRAMEWORK_CACHE_BACKEND=redis
export ML_FRAMEWORK_REDIS_URL=redis://redis:6379/0
```

## 🔄 CI/CD集成

### 1. 构建流水线

```yaml
# .github/workflows/production-build.yml
name: Production Build
on:
  push:
    tags: ['v*']

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build production image
        run: |
          docker build -f Dockerfile.prod \
            -t ${{ secrets.REGISTRY }}/ml-framework:${{ github.ref_name }}-prod-cuda13 .
          
      - name: Push to registry
        run: |
          docker push ${{ secrets.REGISTRY }}/ml-framework:${{ github.ref_name }}-prod-cuda13
```

### 2. 部署脚本

```bash
#!/bin/bash
# deploy.sh

set -e

TAG=${1:-latest}
REGISTRY=${REGISTRY:-your-registry.com}

echo "部署ML Framework生产版本: $TAG"

# 拉取最新镜像
docker pull $REGISTRY/ml-framework:$TAG-prod-cuda13

# 停止现有服务
docker-compose -f docker-compose.prod.yml down

# 更新镜像标签
sed -i "s|image: ml-framework:.*|image: $REGISTRY/ml-framework:$TAG-prod-cuda13|" docker-compose.prod.yml

# 启动服务
docker-compose -f docker-compose.prod.yml up -d

# 健康检查
sleep 30
curl -f http://localhost:8000/health || exit 1

echo "部署完成！"
```

现在你的生产环境已经完全支持CUDA 13.0了！🚀