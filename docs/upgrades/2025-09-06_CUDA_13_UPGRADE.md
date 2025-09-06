# CUDA 13.0 升级指南 - 2025年9月6日

## 📋 升级概述

本文档记录了从CUDA 12.4升级到CUDA 13.0的包版本升级情况。

## 🔄 主要升级内容

### 1. 深度学习框架升级

#### PyTorch 升级
- **旧版本**: `torch>=2.1.0,<2.3.0` (CUDA 12.4)
- **新版本**: `torch>=2.8.0,<3.0.0` (CUDA 13.0)
- **说明**: 升级到PyTorch 2.8+以获得CUDA 13.0支持

#### TensorFlow 升级
- **旧版本**: `tensorflow>=2.15.0,<2.16.0` (CUDA 12.4)
- **新版本**: `tensorflow>=2.18.0,<2.19.0` (CUDA 13.0)
- **说明**: TensorFlow 2.18是首个支持CUDA 13.0的版本

### 2. 核心机器学习库升级

#### 科学计算库
- **NumPy**: `1.21.0` → `1.26.0` (兼容性和性能改进)
- **Pandas**: `1.3.0` → `2.2.0` (API改进和性能提升)
- **Scikit-learn**: `1.0.0` → `1.5.0` (新算法和优化)
- **SciPy**: `1.7.0` → `1.14.0` (数值计算优化)

### 3. 可视化库升级

- **Matplotlib**: `3.4.0` → `3.9.0`
- **Seaborn**: `0.11.0` → `0.13.0`
- **Plotly**: `5.0.0` → `5.24.0`
- **Bokeh**: `2.3.0` → `3.6.0`

### 4. 开发工具升级

- **Pytest**: `6.2.0` → `8.3.0`
- **Black**: `21.6.0` → `24.10.0`
- **MyPy**: `0.910` → `1.13.0`
- **Flake8**: `3.9.0` → `7.1.0`

### 5. 模型相关工具升级

- **MLflow**: `1.18.0` → `2.16.0`
- **Wandb**: `0.12.0` → `0.18.0`
- **Optuna**: `2.8.0` → `4.0.0`
- **Ray[tune]**: `1.4.0` → `2.30.0`

### 6. Web框架升级

- **FastAPI**: `0.68.0` → `0.115.0`
- **Flask**: `2.0.0` → `3.0.0`
- **Streamlit**: `0.84.0` → `1.40.0`
- **Uvicorn**: `0.15.0` → `0.32.0`

## 🚀 安装步骤

### 1. 卸载旧版本（可选）
```bash
pip uninstall torch torchvision torchaudio tensorflow
```

### 2. 安装CUDA 13.0版本的PyTorch
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

### 3. 安装所有依赖
```bash
pip install -r requirements.txt
```

### 4. 验证GPU支持
```python
# 验证PyTorch GPU支持
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")

# 验证TensorFlow GPU支持
import tensorflow as tf
print(f"TensorFlow版本: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU设备数量: {len(gpus)}")
```

## ⚠️ 注意事项

### 兼容性警告
1. **PyTorch 2.8+**: 某些旧的模型可能需要适配
2. **TensorFlow 2.18+**: Keras API有所变化
3. **NumPy 1.26+**: 某些废弃的API已移除
4. **Pandas 2.2+**: 数据类型推断更严格

### 建议的升级策略
1. **渐进式升级**: 先在开发环境测试
2. **备份现有环境**: 使用虚拟环境隔离
3. **测试关键功能**: 运行完整的测试套件
4. **更新Docker镜像**: 确保容器环境同步更新

## 🔧 故障排除

### 常见问题

#### 1. CUDA版本不匹配
```bash
# 检查CUDA版本
nvcc --version
nvidia-smi
```

#### 2. PyTorch安装失败
```bash
# 使用特定的CUDA版本索引
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

#### 3. TensorFlow GPU不可用
```bash
# 检查cuDNN安装
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## 📚 参考资源

- [PyTorch CUDA 13.0 支持文档](https://pytorch.org/get-started/locally/)
- [TensorFlow GPU 安装指南](https://www.tensorflow.org/install/gpu)
- [CUDA 13.0 工具包文档](https://developer.nvidia.com/cuda-toolkit)

## 📝 更新记录

- **2025-09-06**: 创建CUDA 13.0升级指南
- **升级原因**: 用户CUDA版本从12.4升级到13.0
- **主要变更**: PyTorch 2.8+, TensorFlow 2.18+支持
- **升级状态**: ✅ 已完成