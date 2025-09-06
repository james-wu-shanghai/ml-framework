"""
数据加载器
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging


class DataLoader:
    """
    数据加载器类
    
    支持从多种格式加载数据：CSV, Excel, JSON, Parquet, SQL等
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化数据加载器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def load(self, data_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        加载数据
        
        Args:
            data_path: 数据文件路径
            **kwargs: 加载参数
            
        Returns:
            DataFrame
        """
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        
        # 根据文件扩展名选择加载方法
        extension = data_path.suffix.lower()
        
        if extension == '.csv':
            return self._load_csv(data_path, **kwargs)
        elif extension in ['.xlsx', '.xls']:
            return self._load_excel(data_path, **kwargs)
        elif extension == '.json':
            return self._load_json(data_path, **kwargs)
        elif extension == '.parquet':
            return self._load_parquet(data_path, **kwargs)
        elif extension in ['.pkl', '.pickle']:
            return self._load_pickle(data_path, **kwargs)
        else:
            raise ValueError(f"不支持的文件格式: {extension}")
    
    def _load_csv(self, data_path: Path, **kwargs) -> pd.DataFrame:
        """加载CSV文件"""
        default_params = {
            'encoding': 'utf-8',
            'index_col': None
        }
        default_params.update(kwargs)
        
        try:
            df = pd.read_csv(data_path, **default_params)
            self.logger.info(f"成功加载CSV文件: {data_path}, 形状: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"CSV文件加载失败: {str(e)}")
            raise
    
    def _load_excel(self, data_path: Path, **kwargs) -> pd.DataFrame:
        """加载Excel文件"""
        default_params = {
            'engine': 'openpyxl'
        }
        default_params.update(kwargs)
        
        try:
            df = pd.read_excel(data_path, **default_params)
            self.logger.info(f"成功加载Excel文件: {data_path}, 形状: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Excel文件加载失败: {str(e)}")
            raise
    
    def _load_json(self, data_path: Path, **kwargs) -> pd.DataFrame:
        """加载JSON文件"""
        default_params = {
            'orient': 'records'
        }
        default_params.update(kwargs)
        
        try:
            df = pd.read_json(data_path, **default_params)
            self.logger.info(f"成功加载JSON文件: {data_path}, 形状: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"JSON文件加载失败: {str(e)}")
            raise
    
    def _load_parquet(self, data_path: Path, **kwargs) -> pd.DataFrame:
        """加载Parquet文件"""
        try:
            df = pd.read_parquet(data_path, **kwargs)
            self.logger.info(f"成功加载Parquet文件: {data_path}, 形状: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Parquet文件加载失败: {str(e)}")
            raise
    
    def _load_pickle(self, data_path: Path, **kwargs) -> pd.DataFrame:
        """加载Pickle文件"""
        try:
            df = pd.read_pickle(data_path, **kwargs)
            self.logger.info(f"成功加载Pickle文件: {data_path}, 形状: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Pickle文件加载失败: {str(e)}")
            raise
    
    def load_from_sql(self, query: str, connection, **kwargs) -> pd.DataFrame:
        """
        从SQL数据库加载数据
        
        Args:
            query: SQL查询语句
            connection: 数据库连接
            **kwargs: 其他参数
            
        Returns:
            DataFrame
        """
        try:
            df = pd.read_sql(query, connection, **kwargs)
            self.logger.info(f"成功从SQL加载数据, 形状: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"SQL数据加载失败: {str(e)}")
            raise
    
    def load_from_url(self, url: str, file_type: str = 'csv', **kwargs) -> pd.DataFrame:
        """
        从URL加载数据
        
        Args:
            url: 数据URL
            file_type: 文件类型
            **kwargs: 其他参数
            
        Returns:
            DataFrame
        """
        try:
            if file_type.lower() == 'csv':
                df = pd.read_csv(url, **kwargs)
            elif file_type.lower() == 'json':
                df = pd.read_json(url, **kwargs)
            else:
                raise ValueError(f"不支持的URL文件类型: {file_type}")
            
            self.logger.info(f"成功从URL加载数据: {url}, 形状: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"URL数据加载失败: {str(e)}")
            raise
    
    def save_data(self, data: pd.DataFrame, output_path: Union[str, Path], **kwargs):
        """
        保存数据
        
        Args:
            data: 要保存的数据
            output_path: 输出路径
            **kwargs: 保存参数
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        extension = output_path.suffix.lower()
        
        try:
            if extension == '.csv':
                data.to_csv(output_path, index=False, **kwargs)
            elif extension in ['.xlsx', '.xls']:
                data.to_excel(output_path, index=False, **kwargs)
            elif extension == '.json':
                data.to_json(output_path, orient='records', **kwargs)
            elif extension == '.parquet':
                data.to_parquet(output_path, **kwargs)
            elif extension in ['.pkl', '.pickle']:
                data.to_pickle(output_path, **kwargs)
            else:
                raise ValueError(f"不支持的保存格式: {extension}")
            
            self.logger.info(f"数据已保存到: {output_path}")
            
        except Exception as e:
            self.logger.error(f"数据保存失败: {str(e)}")
            raise
    
    def get_data_info(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取数据信息
        
        Args:
            data: DataFrame
            
        Returns:
            数据信息字典
        """
        info = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict(),
            'missing_values': data.isnull().sum().to_dict(),
            'memory_usage': data.memory_usage(deep=True).sum(),
            'numeric_columns': list(data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(data.select_dtypes(include=['object', 'category']).columns)
        }
        
        return info