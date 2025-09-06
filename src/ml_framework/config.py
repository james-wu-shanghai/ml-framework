"""
配置管理模块
"""

import os
import yaml
import json
from typing import Any, Dict, Optional, Union
from pathlib import Path


class Config:
    """
    配置管理类
    
    支持从YAML、JSON文件或字典加载配置，并提供配置访问和更新功能。
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置
        
        Args:
            config_path: 配置文件路径
        """
        self.config = {}
        self._load_default_config()
        
        if config_path:
            self.load_from_file(config_path)
    
    def _load_default_config(self):
        """加载默认配置"""
        self.config = {
            'data': {
                'batch_size': 32,
                'test_size': 0.2,
                'random_state': 42,
                'preprocessing': {
                    'scale_features': True,
                    'handle_missing': 'mean',
                    'encode_categorical': True
                }
            },
            'models': {
                'random_forest': {
                    'n_estimators': 100,
                    'max_depth': None,
                    'random_state': 42
                },
                'logistic_regression': {
                    'max_iter': 1000,
                    'random_state': 42
                },
                'neural_network': {
                    'hidden_layers': [128, 64],
                    'activation': 'relu',
                    'optimizer': 'adam',
                    'learning_rate': 0.001
                }
            },
            'training': {
                'epochs': 100,
                'early_stopping': True,
                'patience': 10,
                'validation_split': 0.2,
                'cross_validation': {
                    'enabled': True,
                    'folds': 5
                }
            },
            'evaluation': {
                'metrics': ['accuracy', 'precision', 'recall', 'f1'],
                'regression_metrics': ['mse', 'mae', 'r2'],
                'save_predictions': True,
                'confusion_matrix': True
            },
            'visualization': {
                'figure_size': [10, 8],
                'dpi': 300,
                'style': 'seaborn',
                'save_plots': True,
                'plot_format': 'png'
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': 'ml_framework.log'
            },
            'tracking': {
                'enabled': True,
                'backend': 'mlflow',
                'experiment_name': 'ml_framework_experiments',
                'auto_log': True
            },
            'paths': {
                'data_dir': 'data',
                'models_dir': 'models',
                'logs_dir': 'logs',
                'plots_dir': 'plots'
            }
        }
    
    def load_from_file(self, config_path: str):
        """
        从文件加载配置
        
        Args:
            config_path: 配置文件路径
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    file_config = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    file_config = json.load(f)
                else:
                    raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
            
            # 合并配置
            self._merge_config(file_config)
            
        except Exception as e:
            raise ValueError(f"配置文件加载失败: {str(e)}")
    
    def load_from_dict(self, config_dict: Dict[str, Any]):
        """
        从字典加载配置
        
        Args:
            config_dict: 配置字典
        """
        self._merge_config(config_dict)
    
    def _merge_config(self, new_config: Dict[str, Any]):
        """
        合并配置
        
        Args:
            new_config: 新配置字典
        """
        def deep_merge(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_merge(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_merge(self.config, new_config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键，支持点分隔的嵌套键（如 'data.batch_size'）
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        设置配置值
        
        Args:
            key: 配置键，支持点分隔的嵌套键
            value: 配置值
        """
        keys = key.split('.')
        config = self.config
        
        # 导航到父级字典
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # 设置值
        config[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]):
        """
        批量更新配置
        
        Args:
            updates: 更新字典
        """
        for key, value in updates.items():
            self.set(key, value)
    
    def save_to_file(self, config_path: str):
        """
        保存配置到文件
        
        Args:
            config_path: 配置文件路径
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(self.config, f, default_flow_style=False, ensure_ascii=False)
                elif config_path.suffix.lower() == '.json':
                    json.dump(self.config, f, indent=2, ensure_ascii=False)
                else:
                    raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
                    
        except Exception as e:
            raise ValueError(f"配置文件保存失败: {str(e)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        获取完整配置字典
        
        Returns:
            配置字典
        """
        return self.config.copy()
    
    def from_env(self, prefix: str = 'ML_FRAMEWORK_'):
        """
        从环境变量加载配置
        
        Args:
            prefix: 环境变量前缀
        """
        env_config = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # 移除前缀并转换为小写
                config_key = key[len(prefix):].lower().replace('_', '.')
                
                # 尝试转换值类型
                try:
                    # 尝试解析为JSON
                    env_config[config_key] = json.loads(value)
                except json.JSONDecodeError:
                    # 如果不是JSON，则保持字符串
                    env_config[config_key] = value
        
        # 更新配置
        for key, value in env_config.items():
            self.set(key, value)
    
    def validate(self) -> bool:
        """
        验证配置的有效性
        
        Returns:
            是否有效
        """
        required_keys = [
            'data.batch_size',
            'training.epochs',
            'logging.level'
        ]
        
        for key in required_keys:
            if self.get(key) is None:
                raise ValueError(f"缺少必需的配置项: {key}")
        
        return True
    
    def __str__(self) -> str:
        """字符串表示"""
        return yaml.dump(self.config, default_flow_style=False)
    
    def __repr__(self) -> str:
        """对象表示"""
        return f"Config({len(self.config)} sections)"