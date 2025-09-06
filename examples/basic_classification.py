"""
基础示例：使用ML Framework进行分类任务

这个示例展示了如何使用ML Framework进行完整的机器学习工作流程：
1. 数据加载
2. 数据预处理
3. 模型训练
4. 模型评估
5. 结果可视化
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

from ml_framework import MLFramework


def create_sample_data():
    """创建示例数据"""
    print("创建示例分类数据...")
    
    # 生成分类数据
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    # 转换为DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    data = pd.DataFrame(X, columns=feature_names)
    data['target'] = y
    
    # 保存数据
    os.makedirs('../data', exist_ok=True)
    data.to_csv('../data/sample_classification_data.csv', index=False)
    
    print(f"数据已保存，形状: {data.shape}")
    return data


def basic_classification_example():
    """基础分类示例"""
    print("=" * 50)
    print("基础分类示例")
    print("=" * 50)
    
    # 创建示例数据
    data = create_sample_data()
    
    # 1. 初始化框架
    print("\n1. 初始化ML Framework...")
    framework = MLFramework(config_path='../configs/default_config.yaml')
    
    # 2. 加载数据
    print("\n2. 加载数据...")
    framework.load_data('../data/sample_classification_data.csv', target_column='target')
    
    # 3. 设置任务类型
    print("\n3. 设置任务类型...")
    framework.set_task_type('classification')
    
    # 4. 数据预处理
    print("\n4. 数据预处理...")
    framework.preprocess_data()
    
    # 5. 选择和训练模型
    print("\n5. 训练随机森林模型...")
    framework.select_model('random_forest', n_estimators=100)
    framework.train()
    
    # 6. 评估模型
    print("\n6. 评估模型...")
    results = framework.evaluate()
    
    print("\n模型性能:")
    for metric, value in results.items():
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value:.4f}")
    
    # 7. 可视化结果
    print("\n7. 可视化结果...")
    framework.visualize_results()
    
    # 8. 获取特征重要性
    print("\n8. 特征重要性:")
    importance = framework.get_feature_importance()
    if importance:
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for feature, score in sorted_features[:10]:
            print(f"  {feature}: {score:.4f}")
    
    # 9. 保存模型
    print("\n9. 保存模型...")
    os.makedirs('../models', exist_ok=True)
    framework.save_model('../models/random_forest_classifier.joblib')
    
    print("\n示例完成！")
    return framework


def auto_ml_example():
    """自动机器学习示例"""
    print("=" * 50)
    print("自动机器学习示例")
    print("=" * 50)
    
    # 使用自动ML功能
    framework = MLFramework()
    
    print("运行自动ML流程...")
    results = framework.auto_ml(
        data_path='../data/sample_classification_data.csv',
        target_column='target',
        task_type='classification'
    )
    
    print("\n自动ML结果:")
    for metric, value in results.items():
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value:.4f}")
    
    return framework


def model_comparison_example():
    """模型比较示例"""
    print("=" * 50)
    print("模型比较示例")
    print("=" * 50)
    
    framework = MLFramework()
    framework.load_data('../data/sample_classification_data.csv', target_column='target')
    framework.set_task_type('classification')
    framework.preprocess_data()
    
    # 测试多个模型
    models = ['random_forest', 'logistic_regression']
    results = {}
    
    for model_name in models:
        print(f"\n训练 {model_name}...")
        
        # 创建新的框架实例避免状态冲突
        temp_framework = MLFramework()
        temp_framework.load_data('../data/sample_classification_data.csv', target_column='target')
        temp_framework.set_task_type('classification')
        temp_framework.preprocess_data()
        
        temp_framework.select_model(model_name)
        temp_framework.train()
        
        model_results = temp_framework.evaluate()
        results[model_name] = model_results
        
        print(f"  准确率: {model_results.get('accuracy', 0):.4f}")
        print(f"  F1分数: {model_results.get('f1', 0):.4f}")
    
    # 比较结果
    print("\n模型比较:")
    comparison_df = framework.evaluator.compare_models(results)
    print(comparison_df)
    
    # 找出最佳模型
    best_model = framework.evaluator.get_best_model(results)
    print(f"\n最佳模型: {best_model}")


if __name__ == "__main__":
    try:
        # 运行基础示例
        framework = basic_classification_example()
        
        # 运行自动ML示例
        # auto_ml_example()
        
        # 运行模型比较示例
        # model_comparison_example()
        
    except Exception as e:
        print(f"示例运行失败: {str(e)}")
        import traceback
        traceback.print_exc()