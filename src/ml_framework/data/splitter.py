"""
数据分割器
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
from sklearn.model_selection import train_test_split
import logging


class DataSplitter:
    """
    数据分割器
    
    提供各种数据分割方法
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def split(self, 
              X: pd.DataFrame, 
              y: pd.Series,
              test_size: float = 0.2,
              random_state: Optional[int] = None,
              stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        分割数据
        
        Args:
            X: 特征数据
            y: 目标变量
            test_size: 测试集比例
            random_state: 随机种子
            stratify: 是否分层抽样
            
        Returns:
            训练和测试数据
        """
        stratify_param = y if stratify and self._is_classification(y) else None
        
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_param
        )
    
    def _is_classification(self, y: pd.Series) -> bool:
        """判断是否为分类问题"""
        return y.dtype == 'object' or len(y.unique()) < len(y) * 0.05