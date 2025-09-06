"""
依赖安装脚本

安装ML Framework所需的依赖包
"""

import subprocess
import sys
import os


def install_package(package):
    """安装包"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    """主函数"""
    print("ML Framework 依赖安装脚本")
    print("=" * 40)
    
    # 核心依赖
    core_packages = [
        'numpy>=1.21.0',
        'pandas>=1.3.0', 
        'scikit-learn>=1.0.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'pyyaml>=5.4.0',
        'click>=8.0.0',
        'joblib>=1.0.0'
    ]
    
    print("安装核心依赖包...")
    failed_packages = []
    
    for package in core_packages:
        print(f"安装 {package}...")
        if install_package(package):
            print(f"  ✅ {package} 安装成功")
        else:
            print(f"  ❌ {package} 安装失败")
            failed_packages.append(package)
    
    print("\n" + "=" * 40)
    if failed_packages:
        print(f"❌ {len(failed_packages)} 个包安装失败:")
        for pkg in failed_packages:
            print(f"  - {pkg}")
        print("\n请手动安装失败的包:")
        print(f"pip install {' '.join(failed_packages)}")
        return 1
    else:
        print("🎉 所有核心依赖安装成功！")
        print("\n你现在可以运行:")
        print("  python test_framework.py")
        return 0


if __name__ == "__main__":
    sys.exit(main())