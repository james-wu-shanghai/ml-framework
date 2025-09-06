"""
模型训练器
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.model_selection import train_test_split
import logging


class Trainer:
    """
    模型训练器
    
    提供模型训练的统一接口
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化训练器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 训练历史
        self.training_history = {}
    
    def train(self, 
              model, 
              X: pd.DataFrame, 
              y: pd.Series,
              validation_split: Optional[float] = None,
              **kwargs) -> Any:
        """
        训练模型
        
        Args:
            model: 模型实例
            X: 特征数据
            y: 目标变量
            validation_split: 验证集比例
            **kwargs: 其他训练参数
            
        Returns:
            训练后的模型
        """
        self.logger.info("开始模型训练")
        start_time = time.time()
        
        # 分割训练和验证数据
        if validation_split:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, 
                test_size=validation_split,
                random_state=self.config.get('random_state', 42),
                stratify=y if self._is_classification_task(y) else None
            )
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None
        
        try:
            # 训练模型
            if hasattr(model, 'fit'):
                if X_val is not None and self._supports_validation(model):
                    # 支持验证集的模型
                    model.fit(X_train, y_train, 
                             validation_data=(X_val, y_val),
                             **kwargs)
                else:
                    # 标准训练
                    model.fit(X_train, y_train, **kwargs)
            else:
                raise ValueError("模型不支持训练")
            
            # 记录训练时间
            training_time = time.time() - start_time
            
            # 记录训练历史
            self.training_history = {
                'training_time': training_time,
                'training_samples': len(X_train),
                'validation_samples': len(X_val) if X_val is not None else 0,
                'features': list(X.columns),
                'model_type': type(model).__name__
            }
            
            self.logger.info(f"模型训练完成，耗时: {training_time:.2f}秒")
            return model
            
        except Exception as e:
            self.logger.error(f"模型训练失败: {str(e)}")
            raise
    
    def _is_classification_task(self, y: pd.Series) -> bool:
        """判断是否为分类任务"""
        return y.dtype == 'object' or len(y.unique()) < len(y) * 0.05
    
    def _supports_validation(self, model) -> bool:
        """检查模型是否支持验证集"""
        # 这里可以扩展更多模型的检查
        validation_supported = [
            'XGBClassifier', 'XGBRegressor',
            'LGBMClassifier', 'LGBMRegressor',
            'CatBoostClassifier', 'CatBoostRegressor'
        ]
        return type(model).__name__ in validation_supported
    
    def get_training_history(self) -> Dict[str, Any]:
        """获取训练历史"""
        return self.training_history