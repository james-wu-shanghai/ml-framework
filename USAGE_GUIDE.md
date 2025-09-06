# ML Framework 使用指南

## 🎯 概述

ML Framework 是一个功能完整的Python机器学习框架，提供从数据处理到模型部署的完整工作流程。它集成了sklearn、pytorch等主流ML库，支持分类、回归、聚类等多种任务。

## 🚀 安装和设置

### 1. 环境要求

- Python 3.8+
- 推荐使用虚拟环境

### 2. 安装步骤

```bash
# 1. 克隆项目（如果从Git）
git clone <repository-url>
cd ml-framework

# 2. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装依赖
python install_dependencies.py
# 或者
pip install -r requirements.txt

# 4. 测试安装
python test_framework.py
```

## 📚 基础使用

### 1. 最简单的使用方式

```python
from ml_framework import quick_start

# 一行代码完成整个ML流程
framework = quick_start(
    data_path='your_data.csv',
    target_column='target',
    task_type='classification'  # 或 'regression', 'clustering'
)

# 训练和评估
framework.preprocess_data()
framework.select_model('random_forest')
framework.train()
results = framework.evaluate()

print(f"准确率: {results['accuracy']:.4f}")
```

### 2. 完整的工作流程

```python
from ml_framework import MLFramework

# 1. 初始化框架
framework = MLFramework(config_path='configs/default_config.yaml')

# 2. 加载数据
framework.load_data('data.csv', target_column='target')

# 3. 设置任务类型
framework.set_task_type('classification')

# 4. 数据预处理
framework.preprocess_data(
    scale_features=True,
    handle_missing='mean',
    encode_categorical=True
)

# 5. 选择和配置模型
framework.select_model('random_forest', 
                      n_estimators=100, 
                      max_depth=10)

# 6. 训练模型
framework.train(validation_split=0.2)

# 7. 评估模型
results = framework.evaluate()

# 8. 可视化结果
framework.visualize_results()

# 9. 保存模型
framework.save_model('models/my_model.joblib')
```

## 🔧 高级功能

### 1. 自定义配置

```python
# 使用配置文件
framework = MLFramework(config_path='my_config.yaml')

# 运行时修改配置
framework.config.set('data.batch_size', 64)
framework.config.set('training.epochs', 200)

# 从环境变量加载配置
framework.config.from_env('ML_FRAMEWORK_')
```

### 2. 模型比较

```python
models = ['random_forest', 'logistic_regression', 'svm']
results = {}

for model_name in models:
    # 创建独立的框架实例
    temp_framework = MLFramework()
    temp_framework.load_data('data.csv', target_column='target')
    temp_framework.set_task_type('classification')
    temp_framework.preprocess_data()
    
    temp_framework.select_model(model_name)
    temp_framework.train()
    
    results[model_name] = temp_framework.evaluate()

# 比较结果
comparison_df = framework.evaluator.compare_models(results)
best_model = framework.evaluator.get_best_model(results)
print(f"最佳模型: {best_model}")
```

### 3. 自动机器学习

```python
# 自动选择最佳模型和参数
results = framework.auto_ml(
    data_path='data.csv',
    target_column='target',
    task_type='classification'
)
```

### 4. 实验跟踪

```python
# 开始实验跟踪
framework.metrics_tracker.start_run('experiment_1')

# 记录参数
framework.metrics_tracker.log_parameters({
    'model': 'random_forest',
    'n_estimators': 100,
    'max_depth': 10
})

# 训练和评估
framework.train()
results = framework.evaluate()

# 记录指标
framework.metrics_tracker.log_metrics(results)

# 结束实验
framework.metrics_tracker.end_run()
```

## 🎨 可视化功能

```python
# 数据可视化
framework.visualizer.plot_data_distribution(data)
framework.visualizer.plot_correlation_matrix(data)

# 模型性能可视化
framework.visualize_results()

# 特征重要性
framework.visualizer.plot_feature_importance(
    model, feature_names, top_k=20
)

# 创建仪表板
framework.visualizer.create_dashboard(data, results)
```

## 💻 命令行工具

### 1. 初始化项目

```bash
python -m ml_framework.cli init
```

### 2. 训练模型

```bash
python -m ml_framework.cli train \
    --data data.csv \
    --target target_column \
    --model random_forest \
    --task-type classification \
    --output models/my_model.joblib
```

### 3. 评估模型

```bash
python -m ml_framework.cli evaluate \
    --model models/my_model.joblib \
    --data test_data.csv \
    --target target_column
```

### 4. 预测

```bash
python -m ml_framework.cli predict \
    --model models/my_model.joblib \
    --data new_data.csv \
    --output predictions.csv
```

### 5. 自动ML

```bash
python -m ml_framework.cli auto \
    --data data.csv \
    --target target_column \
    --task-type classification
```

## 📋 支持的数据格式

- **CSV**: `.csv`
- **Excel**: `.xlsx`, `.xls`
- **JSON**: `.json`
- **Parquet**: `.parquet`
- **Pickle**: `.pkl`, `.pickle`

```python
# 加载不同格式的数据
framework.load_data('data.csv', target_column='target')
framework.load_data('data.xlsx', target_column='target')
framework.load_data('data.json', target_column='target')
```

## 🤖 支持的模型

### 分类模型
- `random_forest`: 随机森林
- `logistic_regression`: 逻辑回归
- `svm`: 支持向量机
- `decision_tree`: 决策树

### 回归模型
- `random_forest`: 随机森林回归
- `linear_regression`: 线性回归
- `svm`: 支持向量回归
- `decision_tree`: 决策树回归

### 聚类模型
- `kmeans`: K均值聚类

```python
# 使用不同模型
framework.select_model('random_forest', n_estimators=100)
framework.select_model('logistic_regression', max_iter=1000)
framework.select_model('svm', kernel='rbf', C=1.0)
```

## ⚠️ 常见问题

### 1. 内存不足

```python
# 减少批次大小
framework.config.set('data.batch_size', 16)

# 使用特征选择
framework.config.set('preprocessing.feature_selection_k', 20)
```

### 2. 训练速度慢

```python
# 减少模型复杂度
framework.select_model('random_forest', n_estimators=50)

# 使用并行处理
framework.select_model('random_forest', n_jobs=-1)
```

### 3. 精度不高

```python
# 尝试不同的预处理
framework.preprocess_data(scaler='robust', handle_missing='knn')

# 尝试不同的模型
framework.select_model('svm', kernel='rbf', C=10)

# 使用交叉验证
framework.config.set('training.cross_validation.enabled', True)
```

## 🔧 扩展开发

### 1. 添加自定义模型

```python
from ml_framework.models.base import BaseModel

class MyModel(BaseModel):
    def fit(self, X, y, **kwargs):
        # 实现训练逻辑
        pass
    
    def predict(self, X):
        # 实现预测逻辑
        pass

# 注册模型
framework.model_registry.models['classification']['my_model'] = MyModel
```

### 2. 添加自定义预处理

```python
def custom_preprocessing(data):
    # 自定义预处理逻辑
    return processed_data

# 应用自定义预处理
framework.data_processor.add_custom_step(custom_preprocessing)
```

### 3. 添加自定义评估指标

```python
def custom_metric(y_true, y_pred):
    # 计算自定义指标
    return score

# 使用自定义指标
results = framework.evaluate(custom_metrics={'my_metric': custom_metric})
```

## 📊 最佳实践

### 1. 数据准备
- 检查数据质量和一致性
- 处理缺失值和异常值
- 进行特征工程
- 确保数据分布合理

### 2. 模型选择
- 从简单模型开始
- 使用交叉验证评估
- 比较多个模型
- 考虑模型的可解释性

### 3. 实验管理
- 记录所有实验参数
- 保存模型和结果
- 使用版本控制
- 建立评估基线

### 4. 性能优化
- 监控资源使用
- 使用并行处理
- 优化数据加载
- 合理选择批次大小

这个使用指南涵盖了ML Framework的主要功能和使用方法。框架设计遵循了模块化、可扩展和易用的原则，适合各种机器学习项目的需求。