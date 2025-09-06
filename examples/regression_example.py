"""
回归任务示例

展示如何使用ML Framework进行回归任务
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from sklearn.datasets import make_regression

from ml_framework import MLFramework


def create_regression_data():
    """创建回归示例数据"""
    print("创建示例回归数据...")
    
    # 生成回归数据
    X, y = make_regression(
        n_samples=1000,
        n_features=15,
        n_informative=10,
        noise=0.1,
        random_state=42
    )
    
    # 转换为DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    data = pd.DataFrame(X, columns=feature_names)
    data['target'] = y
    
    # 保存数据
    os.makedirs('../data', exist_ok=True)
    data.to_csv('../data/sample_regression_data.csv', index=False)
    
    print(f"回归数据已保存，形状: {data.shape}")
    return data


def regression_example():
    """回归示例"""
    print("=" * 50)
    print("回归任务示例")
    print("=" * 50)
    
    # 创建数据
    data = create_regression_data()
    
    # 初始化框架
    framework = MLFramework()
    
    # 完整流程
    framework.load_data('../data/sample_regression_data.csv', target_column='target')
    framework.set_task_type('regression')
    framework.preprocess_data()
    
    # 训练随机森林回归器
    framework.select_model('random_forest')
    framework.train()
    
    # 评估
    results = framework.evaluate()
    
    print("\n回归模型性能:")
    for metric, value in results.items():
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value:.4f}")
    
    # 可视化
    framework.visualize_results()
    
    # 保存模型
    os.makedirs('../models', exist_ok=True)
    framework.save_model('../models/random_forest_regressor.joblib')
    
    print("\n回归示例完成！")


if __name__ == "__main__":
    regression_example()