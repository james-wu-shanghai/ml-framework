#!/bin/bash

# ML Framework Docker快速启动脚本

set -e

echo "🚀 ML Framework Docker快速启动"
echo "================================"

# 检查Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker未安装，请先安装Docker"
    exit 1
fi

# 检查NVIDIA Docker支持
if ! docker run --rm --gpus all nvidia/cuda:12.4-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "⚠️ NVIDIA Docker支持不可用，将使用CPU模式"
    GPU_FLAG=""
else
    echo "✅ NVIDIA Docker支持可用"
    GPU_FLAG="--gpus all"
fi

# 构建镜像
echo "📦 构建Docker镜像..."
docker build -t ml-framework:latest .

# 创建数据目录
mkdir -p data models logs plots

# 显示选项菜单
echo ""
echo "选择启动模式:"
echo "1) Jupyter Notebook (端口8888)"
echo "2) FastAPI服务 (端口8000)"
echo "3) Streamlit应用 (端口8501)"
echo "4) 交互式Shell"
echo "5) 运行测试"
echo "6) 使用Docker Compose启动所有服务"

read -p "请选择 (1-6): " choice

case $choice in
    1)
        echo "🔬 启动Jupyter Notebook..."
        docker run -it --rm $GPU_FLAG \
            -p 8888:8888 \
            -v $(pwd)/data:/app/data \
            -v $(pwd)/models:/app/models \
            -v $(pwd)/examples:/app/examples \
            ml-framework:latest jupyter
        ;;
    2)
        echo "🌐 启动FastAPI服务..."
        docker run -it --rm $GPU_FLAG \
            -p 8000:8000 \
            -v $(pwd)/data:/app/data \
            -v $(pwd)/models:/app/models \
            ml-framework:latest api
        ;;
    3)
        echo "📊 启动Streamlit应用..."
        docker run -it --rm $GPU_FLAG \
            -p 8501:8501 \
            -v $(pwd)/data:/app/data \
            -v $(pwd)/models:/app/models \
            ml-framework:latest streamlit
        ;;
    4)
        echo "💻 启动交互式Shell..."
        docker run -it --rm $GPU_FLAG \
            -v $(pwd)/data:/app/data \
            -v $(pwd)/models:/app/models \
            -v $(pwd)/logs:/app/logs \
            -v $(pwd)/plots:/app/plots \
            ml-framework:latest shell
        ;;
    5)
        echo "🧪 运行测试..."
        docker run --rm $GPU_FLAG \
            ml-framework:latest test
        ;;
    6)
        echo "🚀 启动所有服务..."
        if command -v docker-compose &> /dev/null; then
            docker-compose up -d
            echo "✅ 服务已启动"
            echo "📝 可用服务:"
            echo "  - Jupyter: http://localhost:8888"
            echo "  - FastAPI: http://localhost:8000"
            echo "  - Streamlit: http://localhost:8501"
            echo ""
            echo "查看状态: docker-compose ps"
            echo "查看日志: docker-compose logs -f"
            echo "停止服务: docker-compose down"
        else
            echo "❌ Docker Compose未安装"
            exit 1
        fi
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac