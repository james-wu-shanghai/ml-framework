"""
模型注册器
"""

import pickle
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class ModelRegistry:
    """
    模型注册器
    
    管理所有可用的机器学习模型
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化模型注册器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 注册默认模型
        self._register_default_models()
    
    def _register_default_models(self):
        """注册默认模型"""
        self.models = {
            'classification': {
                'random_forest': RandomForestClassifier,
                'logistic_regression': LogisticRegression,
                'svm': SVC,
                'decision_tree': DecisionTreeClassifier,
            },
            'regression': {
                'random_forest': RandomForestRegressor,
                'linear_regression': LinearRegression,
                'svm': SVR,
                'decision_tree': DecisionTreeRegressor,
            },
            'clustering': {
                'kmeans': KMeans,
            }
        }
    
    def get_model(self, model_name: str, task_type: str = 'classification', **kwargs):
        """
        获取模型实例
        
        Args:
            model_name: 模型名称
            task_type: 任务类型
            **kwargs: 模型参数
            
        Returns:
            模型实例
        """
        if task_type not in self.models:
            raise ValueError(f"不支持的任务类型: {task_type}")
        
        if model_name not in self.models[task_type]:
            raise ValueError(f"不支持的模型: {model_name} for {task_type}")
        
        model_class = self.models[task_type][model_name]
        
        # 合并配置和参数
        model_config = self.config.get(model_name, {})
        model_config.update(kwargs)
        
        try:
            model = model_class(**model_config)
            self.logger.info(f"创建模型: {model_name} ({task_type})")
            return model
        except Exception as e:
            self.logger.error(f"模型创建失败: {str(e)}")
            raise
    
    def list_models(self, task_type: Optional[str] = None) -> Dict[str, Any]:
        """
        列出可用模型
        
        Args:
            task_type: 任务类型筛选
            
        Returns:
            模型字典
        """
        if task_type:
            return self.models.get(task_type, {})
        return self.models
    
    def save_model(self, model, path: Union[str, Path], **kwargs):
        """
        保存模型
        
        Args:
            model: 模型实例
            path: 保存路径
            **kwargs: 保存参数
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # 根据文件扩展名选择保存方法
            if path.suffix == '.joblib':
                joblib.dump(model, path, **kwargs)
            else:
                with open(path, 'wb') as f:
                    pickle.dump(model, f, **kwargs)
            
            self.logger.info(f"模型已保存到: {path}")
            
        except Exception as e:
            self.logger.error(f"模型保存失败: {str(e)}")
            raise
    
    def load_model(self, path: Union[str, Path], **kwargs):
        """
        加载模型
        
        Args:
            path: 模型路径
            **kwargs: 加载参数
            
        Returns:
            模型实例
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在: {path}")
        
        try:
            if path.suffix == '.joblib':
                model = joblib.load(path, **kwargs)
            else:
                with open(path, 'rb') as f:
                    model = pickle.load(f, **kwargs)
            
            self.logger.info(f"模型已加载: {path}")
            return model
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}")
            raise