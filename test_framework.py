"""
框架测试脚本

测试ML Framework的基本功能
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

try:
    from ml_framework import MLFramework, quick_start
    print("✅ ML Framework 导入成功")
except ImportError as e:
    print(f"❌ ML Framework 导入失败: {e}")
    sys.exit(1)


def test_basic_functionality():
    """测试基本功能"""
    print("\n" + "="*50)
    print("测试基本功能")
    print("="*50)
    
    try:
        # 1. 创建测试数据
        print("1. 创建测试数据...")
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            n_classes=2,
            random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        data = pd.DataFrame(X, columns=feature_names)
        data['target'] = y
        
        # 保存测试数据
        os.makedirs('data', exist_ok=True)
        data.to_csv('data/test_data.csv', index=False)
        print(f"   测试数据形状: {data.shape}")
        
        # 2. 测试框架初始化
        print("2. 测试框架初始化...")
        framework = MLFramework()
        print("   ✅ 框架初始化成功")
        
        # 3. 测试数据加载
        print("3. 测试数据加载...")
        framework.load_data('data/test_data.csv', target_column='target')
        print("   ✅ 数据加载成功")
        
        # 4. 测试任务类型设置
        print("4. 测试任务类型设置...")
        framework.set_task_type('classification')
        print("   ✅ 任务类型设置成功")
        
        # 5. 测试数据预处理
        print("5. 测试数据预处理...")
        framework.preprocess_data()
        print("   ✅ 数据预处理成功")
        
        # 6. 测试模型选择
        print("6. 测试模型选择...")
        framework.select_model('random_forest', n_estimators=10)  # 小一点以加快测试
        print("   ✅ 模型选择成功")
        
        # 7. 测试模型训练
        print("7. 测试模型训练...")
        framework.train()
        print("   ✅ 模型训练成功")
        
        # 8. 测试模型评估
        print("8. 测试模型评估...")
        results = framework.evaluate()
        print("   ✅ 模型评估成功")
        print(f"   准确率: {results.get('accuracy', 0):.4f}")
        
        # 9. 测试预测
        print("9. 测试预测...")
        predictions = framework.predict(framework.features.iloc[:5])
        print(f"   ✅ 预测成功，预测结果: {predictions}")
        
        # 10. 测试模型保存
        print("10. 测试模型保存...")
        os.makedirs('models', exist_ok=True)
        framework.save_model('models/test_model.joblib')
        print("    ✅ 模型保存成功")
        
        print("\n🎉 所有基本功能测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_quick_start():
    """测试快速开始功能"""
    print("\n" + "="*50)
    print("测试快速开始功能")
    print("="*50)
    
    try:
        framework = quick_start(
            data_path='data/test_data.csv',
            target_column='target',
            task_type='classification'
        )
        
        framework.preprocess_data()
        framework.select_model('random_forest', n_estimators=10)
        framework.train()
        results = framework.evaluate()
        
        print(f"快速开始测试成功，准确率: {results.get('accuracy', 0):.4f}")
        print("🎉 快速开始功能测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 快速开始测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration():
    """测试配置系统"""
    print("\n" + "="*50)
    print("测试配置系统")
    print("="*50)
    
    try:
        # 使用配置文件
        framework = MLFramework(config_path='configs/default_config.yaml')
        print("✅ 配置文件加载成功")
        
        # 测试配置获取
        batch_size = framework.config.get('data.batch_size')
        print(f"   批次大小: {batch_size}")
        
        # 测试配置设置
        framework.config.set('data.batch_size', 64)
        new_batch_size = framework.config.get('data.batch_size')
        print(f"   新批次大小: {new_batch_size}")
        
        print("🎉 配置系统测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 配置系统测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("ML Framework 功能测试")
    print("=" * 60)
    
    results = []
    
    # 运行测试
    results.append(test_basic_functionality())
    results.append(test_quick_start())
    results.append(test_configuration())
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！框架功能正常。")
        return 0
    else:
        print("❌ 部分测试失败，请检查错误信息。")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)