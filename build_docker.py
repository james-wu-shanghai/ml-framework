"""
Docker构建和部署脚本

自动化构建ML Framework的Docker镜像
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path


def run_command(command, description, check=True):
    """运行命令并显示进度"""
    print(f"\n🔧 {description}...")
    print(f"执行命令: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=check, 
                              capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print(f"错误: {result.stderr}")
        print(f"✅ {description} 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失败: {e}")
        if e.stderr:
            print(f"错误详情: {e.stderr}")
        return False


def check_docker():
    """检查Docker环境"""
    print("🔍 检查Docker环境...")
    
    # 检查Docker是否安装
    if not run_command("docker --version", "检查Docker版本", check=False):
        print("❌ Docker未安装或不可用")
        return False
    
    # 检查Docker Compose
    if not run_command("docker-compose --version", "检查Docker Compose版本", check=False):
        print("⚠️ Docker Compose未安装，将使用docker命令")
    
    # 检查NVIDIA Docker支持
    if not run_command("docker run --rm --gpus all nvidia/cuda:12.4-base-ubuntu22.04 nvidia-smi", 
                      "检查NVIDIA Docker支持", check=False):
        print("⚠️ NVIDIA Docker支持不可用，将构建CPU版本")
        return False
    
    return True


def build_image(image_type="full", tag="latest"):
    """构建Docker镜像"""
    print(f"\n📦 构建{image_type}版本镜像...")
    
    dockerfile = "Dockerfile" if image_type == "full" else "Dockerfile.prod"
    image_name = f"ml-framework:{tag}"
    
    build_command = f"docker build -f {dockerfile} -t {image_name} ."
    
    if run_command(build_command, f"构建{image_name}镜像"):
        print(f"✅ 镜像 {image_name} 构建成功")
        return True
    else:
        print(f"❌ 镜像 {image_name} 构建失败")
        return False


def test_image(image_name="ml-framework:latest"):
    """测试Docker镜像"""
    print(f"\n🧪 测试镜像 {image_name}...")
    
    # 测试基本功能
    test_commands = [
        f"docker run --rm {image_name} python -c 'import ml_framework; print(\"ML Framework导入成功\")'",
        f"docker run --rm --gpus all {image_name} python -c 'import torch; print(f\"PyTorch CUDA: {{torch.cuda.is_available()}}\")'",
        f"docker run --rm --gpus all {image_name} python -c 'import tensorflow as tf; print(f\"TensorFlow GPU: {{len(tf.config.list_physical_devices(\"GPU\"))}}\")'",
    ]
    
    all_passed = True
    for i, cmd in enumerate(test_commands, 1):
        if not run_command(cmd, f"测试 {i}/{len(test_commands)}", check=False):
            all_passed = False
    
    if all_passed:
        print("✅ 所有测试通过")
        return True
    else:
        print("⚠️ 部分测试失败")
        return False


def deploy_services():
    """部署服务"""
    print("\n🚀 部署服务...")
    
    if not Path("docker-compose.yml").exists():
        print("❌ docker-compose.yml文件不存在")
        return False
    
    # 启动服务
    deploy_commands = [
        "docker-compose down",  # 停止现有服务
        "docker-compose build",  # 构建镜像
        "docker-compose up -d jupyter api streamlit",  # 启动核心服务
    ]
    
    for cmd in deploy_commands:
        if not run_command(cmd, f"执行: {cmd}"):
            return False
    
    print("\n✅ 服务部署完成")
    print("📝 可用服务:")
    print("  - Jupyter Notebook: http://localhost:8888")
    print("  - FastAPI: http://localhost:8000")
    print("  - Streamlit: http://localhost:8501")
    
    return True


def show_usage():
    """显示使用说明"""
    print("\n📖 Docker镜像使用说明")
    print("=" * 50)
    print("基本使用:")
    print("  # 交互式运行")
    print("  docker run -it --gpus all -v $(pwd)/data:/app/data ml-framework:latest shell")
    print()
    print("  # 启动Jupyter")
    print("  docker run -d --gpus all -p 8888:8888 ml-framework:latest jupyter")
    print()
    print("  # 启动API服务")
    print("  docker run -d --gpus all -p 8000:8000 ml-framework:latest api")
    print()
    print("  # 训练模型")
    print("  docker run --gpus all -v $(pwd)/data:/app/data ml-framework:latest train --data data/my_data.csv --target target")
    print()
    print("使用Docker Compose:")
    print("  # 启动所有服务")
    print("  docker-compose up -d")
    print()
    print("  # 查看服务状态")
    print("  docker-compose ps")
    print()
    print("  # 查看日志")
    print("  docker-compose logs -f")
    print()
    print("  # 停止服务")
    print("  docker-compose down")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="ML Framework Docker构建脚本")
    parser.add_argument("--type", choices=["full", "prod"], default="full",
                       help="镜像类型 (full: 完整版, prod: 生产版)")
    parser.add_argument("--tag", default="latest", help="镜像标签")
    parser.add_argument("--no-test", action="store_true", help="跳过测试")
    parser.add_argument("--deploy", action="store_true", help="部署服务")
    parser.add_argument("--usage", action="store_true", help="显示使用说明")
    
    args = parser.parse_args()
    
    if args.usage:
        show_usage()
        return 0
    
    print("🐳 ML Framework Docker构建工具")
    print("=" * 40)
    
    # 检查环境
    if not check_docker():
        print("❌ Docker环境检查失败")
        return 1
    
    # 构建镜像
    image_name = f"ml-framework:{args.tag}"
    if not build_image(args.type, args.tag):
        print("❌ 镜像构建失败")
        return 1
    
    # 测试镜像
    if not args.no_test:
        if not test_image(image_name):
            print("⚠️ 镜像测试有问题，但构建成功")
    
    # 部署服务
    if args.deploy:
        if not deploy_services():
            print("❌ 服务部署失败")
            return 1
    
    print("\n🎉 Docker构建完成!")
    show_usage()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())