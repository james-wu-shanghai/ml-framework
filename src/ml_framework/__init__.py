"""
ML Framework - 一个功能完整的机器学习框架

这个框架提供了机器学习项目开发所需的所有核心组件，包括：
- 数据预处理和特征工程
- 多种机器学习模型（传统ML和深度学习）
- 模型训练和评估
- 实验跟踪和模型管理
- 可视化工具
- 部署工具

主要模块：
- data: 数据处理模块
- models: 模型定义模块
- training: 训练器模块
- evaluation: 评估模块
- utils: 工具函数模块
- visualization: 可视化模块
"""

__version__ = "0.1.0"
__author__ = "ML Framework Team"
__email__ = "team@mlframework.com"

# 导入主要组件
from .core import MLFramework
from .config import Config
from .data import DataProcessor, DataLoader
from .models import ModelRegistry, BaseModel
from .training import Trainer
from .evaluation import Evaluator
from .utils import Logger, MetricsTracker
from .visualization import Visualizer
from .gpu_utils import gpu_manager

# 定义公共API
__all__ = [
    'MLFramework',
    'Config',
    'DataProcessor',
    'DataLoader', 
    'ModelRegistry',
    'BaseModel',
    'Trainer',
    'Evaluator',
    'Logger',
    'MetricsTracker',
    'Visualizer',
    'gpu_manager',
]

# 框架初始化
def setup_framework(config_path=None):
    """
    初始化ML框架
    
    Args:
        config_path (str, optional): 配置文件路径
        
    Returns:
        MLFramework: 框架实例
    """
    return MLFramework(config_path=config_path)

# 快速开始函数
def quick_start(data_path, target_column, task_type='classification'):
    """
    快速开始机器学习项目
    
    Args:
        data_path (str): 数据文件路径
        target_column (str): 目标列名
        task_type (str): 任务类型 ('classification' 或 'regression')
        
    Returns:
        MLFramework: 配置好的框架实例
    """
    framework = MLFramework()
    framework.load_data(data_path, target_column=target_column)
    framework.set_task_type(task_type)
    return framework