"""
快速开始指南

演示ML Framework的最简单用法
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ml_framework import quick_start
import pandas as pd
from sklearn.datasets import load_iris


def quick_start_example():
    """快速开始示例"""
    print("ML Framework 快速开始示例")
    print("=" * 40)
    
    # 1. 准备数据
    print("1. 准备数据...")
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['target'] = iris.target
    
    # 保存数据
    os.makedirs('../data', exist_ok=True)
    data.to_csv('../data/iris.csv', index=False)
    
    # 2. 快速开始
    print("2. 快速训练模型...")
    framework = quick_start(
        data_path='../data/iris.csv',
        target_column='target',
        task_type='classification'
    )
    
    # 3. 预处理和训练
    print("3. 数据预处理...")
    framework.preprocess_data()
    
    print("4. 选择和训练模型...")
    framework.select_model('random_forest')
    framework.train()
    
    # 4. 评估
    print("5. 评估模型...")
    results = framework.evaluate()
    
    print(f"准确率: {results['accuracy']:.4f}")
    print(f"F1分数: {results['f1']:.4f}")
    
    print("\n快速开始完成！")


if __name__ == "__main__":
    quick_start_example()