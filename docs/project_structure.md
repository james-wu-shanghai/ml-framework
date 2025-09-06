# ML Framework 项目结构

```
ml-framework/
├── 📁 src/
│   └── 📁 ml_framework/              # 核心框架代码
│       ├── 📄 __init__.py           # 主入口，导出公共API
│       ├── 📄 core.py               # 框架核心类 MLFramework
│       ├── 📄 config.py             # 配置管理系统
│       ├── 📄 cli.py                # 命令行工具
│       ├── 📁 data/                 # 数据处理模块
│       │   ├── 📄 __init__.py
│       │   ├── 📄 loader.py         # 数据加载器
│       │   ├── 📄 processor.py      # 数据预处理器
│       │   ├── 📄 validator.py      # 数据验证器
│       │   └── 📄 splitter.py       # 数据分割器
│       ├── 📁 models/               # 模型模块
│       │   ├── 📄 __init__.py
│       │   ├── 📄 base.py           # 基础模型类
│       │   └── 📄 registry.py       # 模型注册器
│       ├── 📁 training/             # 训练模块
│       │   ├── 📄 __init__.py
│       │   └── 📄 trainer.py        # 模型训练器
│       ├── 📁 evaluation/           # 评估模块
│       │   ├── 📄 __init__.py
│       │   └── 📄 evaluator.py      # 模型评估器
│       ├── 📁 utils/                # 工具模块
│       │   ├── 📄 __init__.py
│       │   ├── 📄 logger.py         # 日志记录器
│       │   └── 📄 metrics_tracker.py # 指标跟踪器
│       └── 📁 visualization/        # 可视化模块
│           ├── 📄 __init__.py
│           └── 📄 visualizer.py     # 可视化器
├── 📁 examples/                     # 示例代码
│   ├── 📄 basic_classification.py   # 基础分类示例
│   ├── 📄 regression_example.py     # 回归示例
│   └── 📄 quick_start.py           # 快速开始指南
├── 📁 configs/                      # 配置文件
│   └── 📄 default_config.yaml      # 默认配置
├── 📁 tests/                        # 测试代码
├── 📁 docs/                         # 文档
├── 📁 data/                         # 数据目录
├── 📁 models/                       # 模型保存目录
├── 📁 logs/                         # 日志目录
├── 📄 requirements.txt              # 依赖列表
├── 📄 setup.py                      # 安装脚本
├── 📄 MANIFEST.in                   # 打包清单
├── 📄 README.md                     # 项目说明
├── 📄 LICENSE                       # 许可证
├── 📄 test_framework.py             # 框架测试脚本
└── 📄 install_dependencies.py       # 依赖安装脚本
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 方法1: 使用安装脚本
python install_dependencies.py

# 方法2: 直接安装
pip install -r requirements.txt
```

### 2. 测试框架

```bash
python test_framework.py
```

### 3. 运行示例

```bash
# 基础分类示例
python examples/basic_classification.py

# 回归示例
python examples/regression_example.py

# 快速开始
python examples/quick_start.py
```

### 4. 使用命令行工具

```bash
# 初始化项目
python -m ml_framework.cli init

# 训练模型
python -m ml_framework.cli train --data data.csv --target target

# 自动ML
python -m ml_framework.cli auto --data data.csv --target target
```

## 📊 核心功能

### ✅ 已实现功能

- **数据处理**: 加载、清理、预处理、验证
- **模型支持**: scikit-learn算法集成
- **训练系统**: 统一的训练接口
- **评估系统**: 多种评估指标
- **可视化**: 数据和结果可视化
- **配置管理**: 灵活的YAML配置
- **命令行工具**: 完整的CLI接口
- **实验跟踪**: 指标记录和管理

### 🔄 扩展建议

- **深度学习**: PyTorch/TensorFlow模型集成
- **超参数优化**: Optuna/Hyperopt集成
- **模型解释**: SHAP/LIME集成
- **分布式训练**: Ray/Dask支持
- **模型部署**: Flask/FastAPI服务
- **Web界面**: Streamlit/Gradio界面

## 🎯 使用场景

1. **快速原型**: 几行代码完成ML流程
2. **实验管理**: 跟踪和比较不同实验
3. **教学培训**: 学习ML工作流程
4. **生产开发**: 标准化ML项目结构
5. **团队协作**: 统一的开发框架

## 📈 性能特点

- **模块化设计**: 可独立使用各组件
- **配置驱动**: 无需修改代码调整参数
- **链式调用**: 流畅的API设计
- **错误处理**: 完善的异常处理机制
- **日志记录**: 详细的操作日志
- **类型安全**: 类型提示支持

这个ML Framework提供了一个完整的机器学习开发环境，适合从初学者到专业开发者的各种需求！