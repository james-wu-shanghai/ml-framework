"""
GPU功能测试脚本

测试NVIDIA 4090 + CUDA 12.4环境下的GPU加速功能
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import time
import numpy as np


def test_gpu_detection():
    """测试GPU检测功能"""
    print("\n" + "="*50)
    print("🔍 GPU检测测试")
    print("="*50)
    
    try:
        from ml_framework import gpu_manager
        
        # 获取GPU信息
        gpu_info = gpu_manager.get_gpu_info()
        
        print("GPU检测结果:")
        print(f"  PyTorch可用: {gpu_info['pytorch_available']}")
        print(f"  TensorFlow可用: {gpu_info['tensorflow_available']}")
        print(f"  CUDA可用: {gpu_info['cuda_available']}")
        
        if gpu_info['gpu_devices']:
            print("\nGPU设备信息:")
            for device in gpu_info['gpu_devices']:
                print(f"  GPU {device['id']}: {device['name']}")
                print(f"    显存: {device['total_memory'] / 1024**3:.1f} GB")
                print(f"    计算能力: {device['major']}.{device['minor']}")
        
        # 推荐批次大小
        batch_size = gpu_manager.recommend_batch_size('medium')
        print(f"\n推荐批次大小: {batch_size}")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU检测失败: {e}")
        return False


def test_pytorch_gpu():
    """测试PyTorch GPU功能"""
    print("\n" + "="*50)
    print("🔥 PyTorch GPU测试")
    print("="*50)
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("❌ PyTorch CUDA不可用")
            return False
        
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        
        # 创建测试张量
        print("\n测试GPU张量操作...")
        device = torch.device('cuda:0')
        
        # CPU vs GPU性能测试
        size = 5000
        
        # CPU测试
        start_time = time.time()
        a_cpu = torch.randn(size, size)
        b_cpu = torch.randn(size, size)
        c_cpu = torch.mm(a_cpu, b_cpu)
        cpu_time = time.time() - start_time
        
        # GPU测试
        start_time = time.time()
        a_gpu = torch.randn(size, size, device=device)
        b_gpu = torch.randn(size, size, device=device)
        torch.cuda.synchronize()  # 确保GPU操作完成
        
        start_time = time.time()
        c_gpu = torch.mm(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        print(f"CPU矩阵乘法时间: {cpu_time:.3f}秒")
        print(f"GPU矩阵乘法时间: {gpu_time:.3f}秒")
        print(f"GPU加速比: {cpu_time/gpu_time:.1f}x")
        
        # 检查GPU内存使用
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**2
        print(f"\nGPU内存使用:")
        print(f"  已分配: {memory_allocated:.1f} MB")
        print(f"  已保留: {memory_reserved:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch GPU测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tensorflow_gpu():
    """测试TensorFlow GPU功能"""
    print("\n" + "="*50)
    print("🤖 TensorFlow GPU测试")
    print("="*50)
    
    try:
        import tensorflow as tf
        
        print(f"TensorFlow版本: {tf.__version__}")
        
        # 检查GPU设备
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print("❌ TensorFlow GPU不可用")
            return False
        
        print(f"可用GPU数量: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")
        
        # GPU vs CPU性能测试
        print("\n测试GPU张量操作...")
        size = 5000
        
        # CPU测试
        with tf.device('/CPU:0'):
            start_time = time.time()
            a_cpu = tf.random.normal([size, size])
            b_cpu = tf.random.normal([size, size])
            c_cpu = tf.linalg.matmul(a_cpu, b_cpu)
            cpu_time = time.time() - start_time
        
        # GPU测试
        with tf.device('/GPU:0'):
            start_time = time.time()
            a_gpu = tf.random.normal([size, size])
            b_gpu = tf.random.normal([size, size])
            c_gpu = tf.linalg.matmul(a_gpu, b_gpu)
            gpu_time = time.time() - start_time
        
        print(f"CPU矩阵乘法时间: {cpu_time:.3f}秒")
        print(f"GPU矩阵乘法时间: {gpu_time:.3f}秒")
        print(f"GPU加速比: {cpu_time/gpu_time:.1f}x")
        
        # 测试混合精度
        try:
            policy = tf.keras.mixed_precision.global_policy()
            print(f"\n混合精度策略: {policy.name}")
        except Exception as e:
            print(f"混合精度检查失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ TensorFlow GPU测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ml_framework_gpu():
    """测试ML框架的GPU功能"""
    print("\n" + "="*50)
    print("🧠 ML Framework GPU测试")
    print("="*50)
    
    try:
        from ml_framework import MLFramework, gpu_manager
        import pandas as pd
        from sklearn.datasets import make_classification
        
        # 创建测试数据
        X, y = make_classification(
            n_samples=2000,
            n_features=50,
            n_informative=30,
            n_classes=2,
            random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        data = pd.DataFrame(X, columns=feature_names)
        data['target'] = y
        
        # 保存测试数据
        os.makedirs('data', exist_ok=True)
        data.to_csv('data/gpu_test_data.csv', index=False)
        
        # 使用GPU配置初始化框架
        framework = MLFramework(config_path='configs/gpu_config.yaml')
        
        # 优化GPU设置
        gpu_manager.optimize_pytorch_gpu()
        gpu_manager.optimize_tensorflow_gpu()
        
        # 推荐GPU批次大小
        recommended_batch_size = gpu_manager.recommend_batch_size('medium')
        framework.config.set('data.batch_size', recommended_batch_size)
        
        print(f"使用GPU优化批次大小: {recommended_batch_size}")
        
        # 测试完整的ML流程
        framework.load_data('data/gpu_test_data.csv', target_column='target')
        framework.set_task_type('classification')
        framework.preprocess_data()
        
        # 使用更大的随机森林（利用CPU多核）
        framework.select_model('random_forest', n_estimators=200, n_jobs=-1)
        
        start_time = time.time()
        framework.train()
        training_time = time.time() - start_time
        
        results = framework.evaluate()
        
        print(f"训练时间: {training_time:.2f}秒")
        print(f"准确率: {results.get('accuracy', 0):.4f}")
        
        # 检查GPU内存使用
        memory_info = gpu_manager.get_memory_usage()
        if memory_info:
            print("\nGPU内存使用情况:")
            for framework_name, devices in memory_info.items():
                for device, info in devices.items():
                    allocated_mb = info['allocated'] / 1024**2
                    print(f"  {framework_name} {device}: {allocated_mb:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ ML Framework GPU测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("🚀 NVIDIA 4090 + CUDA 12.4 GPU功能测试")
    print("=" * 60)
    
    results = []
    
    # 运行所有测试
    results.append(("GPU检测", test_gpu_detection()))
    results.append(("PyTorch GPU", test_pytorch_gpu()))
    results.append(("TensorFlow GPU", test_tensorflow_gpu()))
    results.append(("ML Framework GPU", test_ml_framework_gpu()))
    
    # 汇总结果
    print("\n" + "="*60)
    print("🏆 GPU测试结果汇总")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{len(results)} 测试通过")
    
    if passed == len(results):
        print("\n🎉 所有GPU功能测试通过！")
        print("你的NVIDIA 4090 + CUDA 12.4环境配置完美！")
        
        print("\n💡 GPU使用建议:")
        print("1. 使用大批次训练以充分利用GPU性能")
        print("2. 定期清理GPU缓存: torch.cuda.empty_cache()")
        print("3. 监控GPU温度和内存使用")
        print("4. 考虑使用混合精度训练以提升性能")
        
        return 0
    else:
        print("\n⚠️ 部分GPU功能存在问题，请检查:")
        print("1. NVIDIA驱动是否最新")
        print("2. CUDA 12.4是否正确安装")
        print("3. PyTorch/TensorFlow是否支持CUDA 12.4")
        
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)