"""
基础模型类
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional


class BaseModel(ABC):
    """
    基础模型抽象类
    
    定义所有模型的通用接口
    """
    
    def __init__(self, **kwargs):
        self.params = kwargs
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """训练模型"""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        pass
    
    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """预测概率（可选）"""
        return None
    
    def get_params(self) -> Dict[str, Any]:
        """获取参数"""
        return self.params
    
    def set_params(self, **params):
        """设置参数"""
        self.params.update(params)