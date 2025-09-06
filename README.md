# ML Framework

一个功能完整的Python机器学习框架，提供从数据处理到模型部署的完整工作流程。

## ✨ 特性

- 🚀 **快速开始**: 一行代码完成机器学习流程
- 🔧 **模块化设计**: 可灵活组合使用各个组件
- 📊 **丰富的算法**: 支持sklearn, pytorch, tensorflow等主流库
- 📈 **可视化**: 内置丰富的数据和结果可视化功能
- ⚙️ **配置驱动**: 支持YAML/JSON配置文件
- 📝 **实验跟踪**: 集成MLflow等实验管理工具
- 🎯 **多任务支持**: 分类、回归、聚类任务

## 🛠️ 安装

### 从源码安装

```bash
git clone <repository-url>
cd ml-framework
pip install -e .
```

### 安装依赖

```bash
pip install -r requirements.txt
```

## 🚀 快速开始

### 最简单的用法

```python
from ml_framework import quick_start

# 一行代码完成机器学习流程
framework = quick_start(
    data_path='data.csv',
    target_column='target',
    task_type='classification'
)

# 预处理、训练、评估
framework.preprocess_data()
framework.select_model('random_forest')
framework.train()
results = framework.evaluate()

print(f"准确率: {results['accuracy']:.4f}")
```

### 完整的工作流程

```python
from ml_framework import MLFramework

# 1. 初始化框架
framework = MLFramework(config_path='configs/default_config.yaml')

# 2. 加载数据
framework.load_data('data.csv', target_column='target')

# 3. 设置任务类型
framework.set_task_type('classification')

# 4. 数据预处理
framework.preprocess_data()

# 5. 选择和训练模型
framework.select_model('random_forest', n_estimators=100)
framework.train()

# 6. 评估模型
results = framework.evaluate()

# 7. 可视化结果
framework.visualize_results()

# 8. 保存模型
framework.save_model('models/my_model.joblib')
```

## 📋 支持的算法

### 分类算法
- Random Forest
- Logistic Regression
- Support Vector Machine
- Decision Tree
- Neural Networks (PyTorch/TensorFlow)

### 回归算法  
- Random Forest Regressor
- Linear Regression
- Support Vector Regression
- Decision Tree Regressor

### 聚类算法
- K-Means
- DBSCAN
- Hierarchical Clustering

## 📊 数据处理功能

- **数据加载**: 支持CSV, Excel, JSON, Parquet等格式
- **缺失值处理**: 多种填充策略
- **特征工程**: 自动编码、缩放、选择
- **数据验证**: 数据质量检查
- **数据分割**: 训练/测试集分割

## 📈 可视化功能

- 数据分布图
- 特征相关性矩阵
- 模型性能图表
- 学习曲线
- 混淆矩阵
- ROC曲线
- 特征重要性图

## ⚙️ 配置系统

框架支持灵活的配置管理：

```yaml
# configs/default_config.yaml
data:
  batch_size: 32
  test_size: 0.2
  preprocessing:
    scale_features: true
    handle_missing: "mean"

models:
  random_forest:
    n_estimators: 100
    max_depth: null

training:
  validation_split: 0.2
  cross_validation:
    enabled: true
    folds: 5
```

## 📁 项目结构

```
ml-framework/
├── src/ml_framework/           # 核心框架代码
│   ├── __init__.py
│   ├── core.py                 # 主框架类
│   ├── config.py               # 配置管理
│   ├── data/                   # 数据处理模块
│   ├── models/                 # 模型模块
│   ├── training/               # 训练模块
│   ├── evaluation/             # 评估模块
│   ├── utils/                  # 工具模块
│   └── visualization/          # 可视化模块
├── examples/                   # 示例代码
├── configs/                    # 配置文件
├── tests/                      # 测试代码
├── docs/                       # 文档
├── requirements.txt            # 依赖列表
└── setup.py                    # 安装脚本
```

## 📚 示例

查看 `examples/` 目录获取更多示例：

- `basic_classification.py` - 基础分类任务
- `regression_example.py` - 回归任务
- `quick_start.py` - 快速开始指南

## 🔧 命令行工具

```bash
# 训练模型
ml-framework train --data data.csv --target target --model random_forest

# 评估模型
ml-framework evaluate --model models/my_model.joblib --data test.csv

# 预测
ml-framework predict --model models/my_model.joblib --data new_data.csv
```

## 🧪 测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_core.py

# 生成覆盖率报告
pytest --cov=ml_framework tests/
```

## 📝 开发指南

### 贡献代码

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

### 代码风格

使用 black 进行代码格式化：

```bash
black src/
```

### 类型检查

使用 mypy 进行类型检查：

```bash
mypy src/
```

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🤝 支持

- 📧 邮箱: support@mlframework.com
- 💬 讨论区: [GitHub Discussions](https://github.com/yourusername/ml-framework/discussions)
- 🐛 问题报告: [GitHub Issues](https://github.com/yourusername/ml-framework/issues)

## 🏆 致谢

感谢以下开源项目的支持：

- [scikit-learn](https://scikit-learn.org/)
- [PyTorch](https://pytorch.org/)
- [TensorFlow](https://tensorflow.org/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)

## 📈 路线图

- [ ] 更多深度学习模型支持
- [ ] 自动超参数优化
- [ ] 模型解释性工具
- [ ] 分布式训练支持
- [ ] Web界面
- [ ] 模型服务部署

---

**ML Framework** - 让机器学习更简单 🚀