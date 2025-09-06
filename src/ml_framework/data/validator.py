"""
数据验证器
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging


class DataValidator:
    """
    数据验证器
    
    检查数据质量和一致性
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def validate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        验证数据
        
        Args:
            data: 数据
            
        Returns:
            验证结果
        """
        results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }
        
        # 基本统计
        results['statistics'] = {
            'shape': data.shape,
            'missing_values': data.isnull().sum().to_dict(),
            'duplicates': data.duplicated().sum(),
            'memory_usage': data.memory_usage(deep=True).sum()
        }
        
        # 检查缺失值
        missing_ratio = data.isnull().sum() / len(data)
        high_missing_cols = missing_ratio[missing_ratio > 0.5].index.tolist()
        
        if high_missing_cols:
            results['warnings'].append(f"高缺失值列: {high_missing_cols}")
        
        # 检查重复行
        if data.duplicated().sum() > 0:
            results['warnings'].append(f"发现 {data.duplicated().sum()} 行重复数据")
        
        # 检查空数据框
        if data.empty:
            results['errors'].append("数据框为空")
            results['is_valid'] = False
        
        return results