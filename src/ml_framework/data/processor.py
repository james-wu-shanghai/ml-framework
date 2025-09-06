"""
数据预处理器
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder, OrdinalEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
import logging


class DataProcessor:
    """
    数据预处理器
    
    提供数据清理、特征工程、数据转换等功能
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化数据处理器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 存储预处理器实例
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_selectors = {}
        
        # 记录处理步骤
        self.processing_steps = []
        self.is_fitted = False
    
    def process(self, 
                features: pd.DataFrame, 
                target: Optional[pd.Series] = None,
                task_type: str = 'classification',
                **kwargs) -> Dict[str, Any]:
        """
        完整的数据预处理流程
        
        Args:
            features: 特征数据
            target: 目标变量
            task_type: 任务类型
            **kwargs: 其他参数
            
        Returns:
            处理后的数据字典
        """
        self.logger.info("开始数据预处理")
        
        # 复制数据避免修改原始数据
        X = features.copy()
        y = target.copy() if target is not None else None
        
        # 1. 数据清理
        X = self.clean_data(X)
        
        # 2. 处理缺失值
        X = self.handle_missing_values(X)
        
        # 3. 编码分类变量
        X = self.encode_categorical_features(X)
        
        # 4. 特征缩放
        X = self.scale_features(X)
        
        # 5. 特征选择
        if target is not None:
            X = self.select_features(X, y, task_type)
        
        # 6. 处理目标变量
        if y is not None:
            y = self.process_target(y, task_type)
        
        self.is_fitted = True
        self.logger.info(f"数据预处理完成，最终形状: {X.shape}")
        
        return {
            'features': X,
            'target': y,
            'feature_names': list(X.columns),
            'processing_steps': self.processing_steps
        }
    
    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        使用已拟合的处理器转换新数据
        
        Args:
            features: 特征数据
            
        Returns:
            转换后的数据
        """
        if not self.is_fitted:
            raise ValueError("处理器尚未拟合，请先调用process方法")
        
        X = features.copy()
        
        for step in self.processing_steps:
            if step['type'] == 'clean':
                X = self.clean_data(X)
            elif step['type'] == 'impute':
                X = self._apply_imputation(X, step['imputer'])
            elif step['type'] == 'encode':
                X = self._apply_encoding(X, step['encoder'], step['columns'])
            elif step['type'] == 'scale':
                X = self._apply_scaling(X, step['scaler'])
            elif step['type'] == 'select':
                X = self._apply_feature_selection(X, step['selector'])
        
        return X
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        数据清理
        
        Args:
            data: 输入数据
            
        Returns:
            清理后的数据
        """
        self.logger.info("执行数据清理")
        
        # 移除重复行
        initial_shape = data.shape
        data = data.drop_duplicates()
        removed_duplicates = initial_shape[0] - data.shape[0]
        
        if removed_duplicates > 0:
            self.logger.info(f"移除重复行: {removed_duplicates}")
        
        # 移除全为空的列
        data = data.dropna(axis=1, how='all')
        
        # 记录处理步骤
        self.processing_steps.append({
            'type': 'clean',
            'removed_duplicates': removed_duplicates
        })
        
        return data
    
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        处理缺失值
        
        Args:
            data: 输入数据
            
        Returns:
            处理后的数据
        """
        self.logger.info("处理缺失值")
        
        # 获取配置
        strategy = self.config.get('preprocessing', {}).get('handle_missing', 'mean')
        
        # 分别处理数值和分类特征
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        
        # 处理数值特征
        if len(numeric_columns) > 0:
            if strategy == 'knn':
                imputer = KNNImputer(n_neighbors=5)
            else:
                imputer = SimpleImputer(strategy=strategy)
            
            data[numeric_columns] = imputer.fit_transform(data[numeric_columns])
            self.imputers['numeric'] = imputer
        
        # 处理分类特征
        if len(categorical_columns) > 0:
            imputer = SimpleImputer(strategy='most_frequent')
            data[categorical_columns] = imputer.fit_transform(data[categorical_columns])
            self.imputers['categorical'] = imputer
        
        # 记录处理步骤
        self.processing_steps.append({
            'type': 'impute',
            'imputer': self.imputers,
            'strategy': strategy
        })
        
        return data
    
    def encode_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        编码分类特征
        
        Args:
            data: 输入数据
            
        Returns:
            编码后的数据
        """
        if not self.config.get('preprocessing', {}).get('encode_categorical', True):
            return data
        
        self.logger.info("编码分类特征")
        
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_columns) == 0:
            return data
        
        # 对每个分类列进行编码
        for col in categorical_columns:
            unique_values = data[col].nunique()
            
            if unique_values <= 10:  # 低基数：使用独热编码
                encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                encoded = encoder.fit_transform(data[[col]])
                
                # 创建新列名
                feature_names = [f"{col}_{category}" for category in encoder.categories_[0]]
                encoded_df = pd.DataFrame(encoded, columns=feature_names, index=data.index)
                
                # 替换原列
                data = data.drop(columns=[col])
                data = pd.concat([data, encoded_df], axis=1)
                
                self.encoders[col] = encoder
                
            else:  # 高基数：使用标签编码
                encoder = LabelEncoder()
                data[col] = encoder.fit_transform(data[col].astype(str))
                self.encoders[col] = encoder
        
        # 记录处理步骤
        self.processing_steps.append({
            'type': 'encode',
            'encoder': self.encoders,
            'columns': list(categorical_columns)
        })
        
        return data
    
    def scale_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        特征缩放
        
        Args:
            data: 输入数据
            
        Returns:
            缩放后的数据
        """
        if not self.config.get('preprocessing', {}).get('scale_features', True):
            return data
        
        self.logger.info("特征缩放")
        
        # 只对数值特征进行缩放
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) == 0:
            return data
        
        # 选择缩放器
        scaler_type = self.config.get('preprocessing', {}).get('scaler', 'standard')
        
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        # 应用缩放
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
        self.scalers['features'] = scaler
        
        # 记录处理步骤
        self.processing_steps.append({
            'type': 'scale',
            'scaler': scaler,
            'columns': list(numeric_columns)
        })
        
        return data
    
    def select_features(self, data: pd.DataFrame, target: pd.Series, task_type: str) -> pd.DataFrame:
        """
        特征选择
        
        Args:
            data: 特征数据
            target: 目标变量
            task_type: 任务类型
            
        Returns:
            选择后的特征
        """
        k = self.config.get('preprocessing', {}).get('feature_selection_k', min(50, data.shape[1]))
        
        if k >= data.shape[1]:
            return data
        
        self.logger.info(f"特征选择：从 {data.shape[1]} 个特征中选择 {k} 个")
        
        # 选择评分函数
        if task_type == 'classification':
            score_func = f_classif
        else:
            score_func = f_regression
        
        # 应用特征选择
        selector = SelectKBest(score_func=score_func, k=k)
        selected_features = selector.fit_transform(data, target)
        
        # 获取选中的特征名
        selected_columns = data.columns[selector.get_support()].tolist()
        result_data = pd.DataFrame(selected_features, columns=selected_columns, index=data.index)
        
        self.feature_selectors['features'] = selector
        
        # 记录处理步骤
        self.processing_steps.append({
            'type': 'select',
            'selector': selector,
            'selected_features': selected_columns
        })
        
        return result_data
    
    def process_target(self, target: pd.Series, task_type: str) -> pd.Series:
        """
        处理目标变量
        
        Args:
            target: 目标变量
            task_type: 任务类型
            
        Returns:
            处理后的目标变量
        """
        if task_type == 'classification' and target.dtype == 'object':
            encoder = LabelEncoder()
            encoded_target = encoder.fit_transform(target)
            self.encoders['target'] = encoder
            return pd.Series(encoded_target, index=target.index, name=target.name)
        
        return target
    
    def _apply_imputation(self, data: pd.DataFrame, imputers: Dict) -> pd.DataFrame:
        """应用缺失值填充"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        
        if 'numeric' in imputers and len(numeric_columns) > 0:
            data[numeric_columns] = imputers['numeric'].transform(data[numeric_columns])
        
        if 'categorical' in imputers and len(categorical_columns) > 0:
            data[categorical_columns] = imputers['categorical'].transform(data[categorical_columns])
        
        return data
    
    def _apply_encoding(self, data: pd.DataFrame, encoders: Dict, original_columns: List[str]) -> pd.DataFrame:
        """应用分类编码"""
        for col in original_columns:
            if col in data.columns and col in encoders:
                encoder = encoders[col]
                
                if isinstance(encoder, OneHotEncoder):
                    encoded = encoder.transform(data[[col]])
                    feature_names = [f"{col}_{category}" for category in encoder.categories_[0]]
                    encoded_df = pd.DataFrame(encoded, columns=feature_names, index=data.index)
                    
                    data = data.drop(columns=[col])
                    data = pd.concat([data, encoded_df], axis=1)
                else:
                    data[col] = encoder.transform(data[col].astype(str))
        
        return data
    
    def _apply_scaling(self, data: pd.DataFrame, scaler) -> pd.DataFrame:
        """应用特征缩放"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            data[numeric_columns] = scaler.transform(data[numeric_columns])
        return data
    
    def _apply_feature_selection(self, data: pd.DataFrame, selector) -> pd.DataFrame:
        """应用特征选择"""
        selected_features = selector.transform(data)
        selected_columns = data.columns[selector.get_support()].tolist()
        return pd.DataFrame(selected_features, columns=selected_columns, index=data.index)
    
    def get_feature_importance(self, method: str = 'mutual_info') -> Optional[Dict[str, float]]:
        """
        获取特征重要性
        
        Args:
            method: 计算方法
            
        Returns:
            特征重要性字典
        """
        if 'features' in self.feature_selectors:
            selector = self.feature_selectors['features']
            if hasattr(selector, 'scores_'):
                feature_names = self.processing_steps[-1]['selected_features']
                scores = selector.scores_[selector.get_support()]
                return dict(zip(feature_names, scores))
        
        return None