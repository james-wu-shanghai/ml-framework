@echo off
REM ML Framework Docker快速启动脚本 (Windows版本)

echo 🚀 ML Framework Docker快速启动
echo ================================

REM 检查Docker
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker未安装，请先安装Docker Desktop
    pause
    exit /b 1
)

REM 检查NVIDIA Docker支持
docker run --rm --gpus all nvidia/cuda:12.4-base-ubuntu22.04 nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo ⚠️ NVIDIA Docker支持不可用，将使用CPU模式
    set GPU_FLAG=
) else (
    echo ✅ NVIDIA Docker支持可用
    set GPU_FLAG=--gpus all
)

REM 构建镜像
echo 📦 构建Docker镜像...
docker build -t ml-framework:latest .

REM 创建数据目录
if not exist data mkdir data
if not exist models mkdir models
if not exist logs mkdir logs
if not exist plots mkdir plots

REM 显示选项菜单
echo.
echo 选择启动模式:
echo 1) Jupyter Notebook (端口8888)
echo 2) FastAPI服务 (端口8000)
echo 3) Streamlit应用 (端口8501)
echo 4) 交互式Shell
echo 5) 运行测试
echo 6) 使用Docker Compose启动所有服务

set /p choice=请选择 (1-6): 

if "%choice%"=="1" (
    echo 🔬 启动Jupyter Notebook...
    docker run -it --rm %GPU_FLAG% ^
        -p 8888:8888 ^
        -v %cd%/data:/app/data ^
        -v %cd%/models:/app/models ^
        -v %cd%/examples:/app/examples ^
        ml-framework:latest jupyter
) else if "%choice%"=="2" (
    echo 🌐 启动FastAPI服务...
    docker run -it --rm %GPU_FLAG% ^
        -p 8000:8000 ^
        -v %cd%/data:/app/data ^
        -v %cd%/models:/app/models ^
        ml-framework:latest api
) else if "%choice%"=="3" (
    echo 📊 启动Streamlit应用...
    docker run -it --rm %GPU_FLAG% ^
        -p 8501:8501 ^
        -v %cd%/data:/app/data ^
        -v %cd%/models:/app/models ^
        ml-framework:latest streamlit
) else if "%choice%"=="4" (
    echo 💻 启动交互式Shell...
    docker run -it --rm %GPU_FLAG% ^
        -v %cd%/data:/app/data ^
        -v %cd%/models:/app/models ^
        -v %cd%/logs:/app/logs ^
        -v %cd%/plots:/app/plots ^
        ml-framework:latest shell
) else if "%choice%"=="5" (
    echo 🧪 运行测试...
    docker run --rm %GPU_FLAG% ^
        ml-framework:latest test
) else if "%choice%"=="6" (
    echo 🚀 启动所有服务...
    docker-compose --version >nul 2>&1
    if errorlevel 1 (
        echo ❌ Docker Compose未安装
        pause
        exit /b 1
    )
    
    docker-compose up -d
    echo ✅ 服务已启动
    echo 📝 可用服务:
    echo   - Jupyter: http://localhost:8888
    echo   - FastAPI: http://localhost:8000
    echo   - Streamlit: http://localhost:8501
    echo.
    echo 查看状态: docker-compose ps
    echo 查看日志: docker-compose logs -f
    echo 停止服务: docker-compose down
) else (
    echo ❌ 无效选择
    pause
    exit /b 1
)

pause