"""
GPU优化依赖安装脚本

专门为NVIDIA 4090 + CUDA 12.4环境优化的安装脚本
"""

import subprocess
import sys
import os


def run_command(command, description):
    """运行命令并显示进度"""
    print(f"\n🔧 {description}...")
    print(f"执行命令: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} 成功")
        if result.stdout:
            print(f"输出: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失败")
        print(f"错误: {e.stderr}")
        return False


def check_gpu_environment():
    """检查GPU环境"""
    print("🔍 检查GPU环境...")
    
    # 检查NVIDIA-SMI
    try:
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA驱动检测成功")
            print("GPU信息:")
            print(result.stdout.split('\n')[8:12])  # 显示GPU信息行
        else:
            print("❌ 未检测到NVIDIA驱动")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi命令未找到，请确保安装了NVIDIA驱动")
        return False
    
    # 检查CUDA版本
    try:
        result = subprocess.run("nvcc --version", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ CUDA工具包检测成功")
            cuda_info = [line for line in result.stdout.split('\n') if 'release' in line.lower()]
            if cuda_info:
                print(f"CUDA版本: {cuda_info[0].strip()}")
        else:
            print("⚠️ CUDA工具包未检测到（这对于PyTorch/TensorFlow可能不是必需的）")
    except FileNotFoundError:
        print("⚠️ nvcc命令未找到，CUDA工具包可能未安装")
    
    return True


def install_pytorch_cuda():
    """安装支持CUDA 12.4的PyTorch"""
    print("\n📦 安装PyTorch (CUDA 12.4支持)...")
    
    # PyTorch官方CUDA 12.4支持的安装命令
    pytorch_command = (
        "pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 "
        "--index-url https://download.pytorch.org/whl/cu124"
    )
    
    return run_command(pytorch_command, "PyTorch CUDA版本安装")


def install_tensorflow():
    """安装TensorFlow"""
    print("\n📦 安装TensorFlow...")
    
    # TensorFlow 2.15+ 原生支持CUDA 12.4
    tensorflow_command = "pip install tensorflow==2.15.0"
    
    return run_command(tensorflow_command, "TensorFlow安装")


def install_other_dependencies():
    """安装其他依赖"""
    print("\n📦 安装其他依赖包...")
    
    # 核心依赖
    core_deps = [
        "numpy>=1.21.0",
        "pandas>=1.3.0", 
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pyyaml>=5.4.0",
        "click>=8.0.0",
        "joblib>=1.0.0",
        "tqdm>=4.61.0"
    ]
    
    for dep in core_deps:
        if not run_command(f"pip install {dep}", f"安装 {dep.split('>=')[0]}"):
            return False
    
    return True


def verify_installation():
    """验证安装"""
    print("\n🔬 验证GPU支持安装...")
    
    # 验证PyTorch CUDA支持
    pytorch_check = """
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"当前GPU: {torch.cuda.get_device_name(0)}")
"""
    
    try:
        result = subprocess.run([sys.executable, "-c", pytorch_check], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ PyTorch验证:")
            print(result.stdout)
        else:
            print("❌ PyTorch验证失败:")
            print(result.stderr)
    except Exception as e:
        print(f"❌ PyTorch验证出错: {e}")
    
    # 验证TensorFlow GPU支持
    tensorflow_check = """
import tensorflow as tf
print(f"TensorFlow版本: {tf.__version__}")
print(f"GPU可用: {len(tf.config.list_physical_devices('GPU')) > 0}")
if len(tf.config.list_physical_devices('GPU')) > 0:
    print(f"GPU设备: {tf.config.list_physical_devices('GPU')}")
"""
    
    try:
        result = subprocess.run([sys.executable, "-c", tensorflow_check], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ TensorFlow验证:")
            print(result.stdout)
        else:
            print("❌ TensorFlow验证失败:")
            print(result.stderr)
    except Exception as e:
        print(f"❌ TensorFlow验证出错: {e}")


def main():
    """主函数"""
    print("=" * 60)
    print("🚀 ML Framework GPU优化安装脚本")
    print("🎯 目标硬件: NVIDIA 4090 + CUDA 12.4")
    print("=" * 60)
    
    # 1. 检查GPU环境
    if not check_gpu_environment():
        print("\n❌ GPU环境检查失败，建议检查NVIDIA驱动安装")
        return 1
    
    # 2. 安装PyTorch CUDA版本
    if not install_pytorch_cuda():
        print("\n❌ PyTorch安装失败")
        return 1
    
    # 3. 安装TensorFlow
    if not install_tensorflow():
        print("\n❌ TensorFlow安装失败")
        return 1
    
    # 4. 安装其他依赖
    if not install_other_dependencies():
        print("\n❌ 其他依赖安装失败")
        return 1
    
    # 5. 验证安装
    verify_installation()
    
    print("\n" + "=" * 60)
    print("🎉 GPU优化安装完成！")
    print("\n📝 接下来的步骤:")
    print("1. 运行 'python test_framework.py' 测试框架")
    print("2. 运行 'python examples/basic_classification.py' 测试示例")
    print("3. 检查GPU使用情况: nvidia-smi")
    print("\n💡 提示:")
    print("- 如果遇到CUDA内存问题，可以在代码中添加:")
    print("  torch.cuda.empty_cache()")
    print("- TensorFlow GPU内存增长控制:")
    print("  tf.config.experimental.set_memory_growth(gpu, True)")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())