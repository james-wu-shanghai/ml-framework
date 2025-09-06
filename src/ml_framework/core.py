"""
核心框架类，提供统一的机器学习工作流程
"""

import os
import yaml
import logging
from typing import Optional, Dict, Any, Union
import pandas as pd
import numpy as np

from .config import Config
from .data import DataProcessor, DataLoader
from .models import ModelRegistry
from .training import Trainer
from .evaluation import Evaluator
from .utils import Logger, MetricsTracker
from .visualization import Visualizer


class MLFramework:
    """
    机器学习框架核心类
    
    提供完整的机器学习工作流程，包括数据处理、模型训练、评估和部署。
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化ML框架
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config = Config(config_path)
        
        # 初始化组件
        self.logger = Logger(self.config.get('logging', {}))
        self.data_processor = DataProcessor(self.config.get('data', {}))
        self.data_loader = DataLoader(self.config.get('data', {}))
        self.model_registry = ModelRegistry(self.config.get('models', {}))
        self.trainer = Trainer(self.config.get('training', {}))
        self.evaluator = Evaluator(self.config.get('evaluation', {}))
        self.visualizer = Visualizer(self.config.get('visualization', {}))
        self.metrics_tracker = MetricsTracker(self.config.get('tracking', {}))
        
        # 状态变量
        self.data = None
        self.features = None
        self.target = None
        self.model = None
        self.task_type = None
        
        self.logger.info("ML Framework 初始化完成")
    
    def load_data(self, data_path: str, target_column: str, **kwargs) -> 'MLFramework':
        """
        加载数据
        
        Args:
            data_path: 数据文件路径
            target_column: 目标列名
            **kwargs: 其他参数
            
        Returns:
            self: 支持链式调用
        """
        try:
            self.data = self.data_loader.load(data_path, **kwargs)
            self.target = target_column
            
            # 分离特征和目标
            if target_column in self.data.columns:
                self.features = self.data.drop(columns=[target_column])
                self.target_values = self.data[target_column]
            else:
                raise ValueError(f"目标列 '{target_column}' 不存在于数据中")
            
            self.logger.info(f"数据加载成功: {self.data.shape}")
            return self
            
        except Exception as e:
            self.logger.error(f"数据加载失败: {str(e)}")
            raise
    
    def set_task_type(self, task_type: str) -> 'MLFramework':
        """
        设置任务类型
        
        Args:
            task_type: 任务类型 ('classification', 'regression', 'clustering')
            
        Returns:
            self: 支持链式调用
        """
        valid_types = ['classification', 'regression', 'clustering']
        if task_type not in valid_types:
            raise ValueError(f"任务类型必须是: {valid_types}")
        
        self.task_type = task_type
        self.logger.info(f"任务类型设置为: {task_type}")
        return self
    
    def preprocess_data(self, **kwargs) -> 'MLFramework':
        """
        数据预处理
        
        Args:
            **kwargs: 预处理参数
            
        Returns:
            self: 支持链式调用
        """
        try:
            processed_data = self.data_processor.process(
                self.features, 
                self.target_values,
                task_type=self.task_type,
                **kwargs
            )
            
            self.features = processed_data['features']
            self.target_values = processed_data['target']
            
            self.logger.info("数据预处理完成")
            return self
            
        except Exception as e:
            self.logger.error(f"数据预处理失败: {str(e)}")
            raise
    
    def select_model(self, model_name: str, **kwargs) -> 'MLFramework':
        """
        选择模型
        
        Args:
            model_name: 模型名称
            **kwargs: 模型参数
            
        Returns:
            self: 支持链式调用
        """
        try:
            self.model = self.model_registry.get_model(
                model_name, 
                task_type=self.task_type,
                **kwargs
            )
            
            self.logger.info(f"模型选择: {model_name}")
            return self
            
        except Exception as e:
            self.logger.error(f"模型选择失败: {str(e)}")
            raise
    
    def train(self, **kwargs) -> 'MLFramework':
        """
        训练模型
        
        Args:
            **kwargs: 训练参数
            
        Returns:
            self: 支持链式调用
        """
        if self.model is None:
            raise ValueError("请先选择模型")
        if self.features is None or self.target_values is None:
            raise ValueError("请先加载和预处理数据")
        
        try:
            # 开始训练
            self.model = self.trainer.train(
                self.model,
                self.features,
                self.target_values,
                **kwargs
            )
            
            self.logger.info("模型训练完成")
            return self
            
        except Exception as e:
            self.logger.error(f"模型训练失败: {str(e)}")
            raise
    
    def evaluate(self, test_data: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        """
        评估模型
        
        Args:
            test_data: 测试数据
            **kwargs: 评估参数
            
        Returns:
            评估结果字典
        """
        if self.model is None:
            raise ValueError("请先训练模型")
        
        try:
            if test_data is not None:
                test_features = test_data.drop(columns=[self.target])
                test_target = test_data[self.target]
            else:
                # 使用训练数据评估（简化版本）
                test_features = self.features
                test_target = self.target_values
            
            results = self.evaluator.evaluate(
                self.model,
                test_features,
                test_target,
                task_type=self.task_type,
                **kwargs
            )
            
            # 记录指标
            self.metrics_tracker.log_metrics(results)
            
            self.logger.info("模型评估完成")
            return results
            
        except Exception as e:
            self.logger.error(f"模型评估失败: {str(e)}")
            raise
    
    def predict(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        使用模型进行预测
        
        Args:
            data: 预测数据
            
        Returns:
            预测结果
        """
        if self.model is None:
            raise ValueError("请先训练模型")
        
        try:
            # 数据预处理（如果需要）
            if isinstance(data, pd.DataFrame):
                processed_data = self.data_processor.transform(data)
            else:
                processed_data = data
            
            predictions = self.model.predict(processed_data)
            self.logger.info(f"预测完成，样本数: {len(predictions)}")
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"预测失败: {str(e)}")
            raise
    
    def visualize_results(self, **kwargs):
        """
        可视化结果
        
        Args:
            **kwargs: 可视化参数
        """
        if self.model is None:
            raise ValueError("请先训练模型")
        
        try:
            self.visualizer.plot_model_performance(
                self.model,
                self.features,
                self.target_values,
                task_type=self.task_type,
                **kwargs
            )
            
            self.logger.info("结果可视化完成")
            
        except Exception as e:
            self.logger.error(f"可视化失败: {str(e)}")
            raise
    
    def save_model(self, path: str, **kwargs):
        """
        保存模型
        
        Args:
            path: 保存路径
            **kwargs: 保存参数
        """
        if self.model is None:
            raise ValueError("没有可保存的模型")
        
        try:
            self.model_registry.save_model(self.model, path, **kwargs)
            self.logger.info(f"模型已保存到: {path}")
            
        except Exception as e:
            self.logger.error(f"模型保存失败: {str(e)}")
            raise
    
    def load_model(self, path: str, **kwargs) -> 'MLFramework':
        """
        加载模型
        
        Args:
            path: 模型路径
            **kwargs: 加载参数
            
        Returns:
            self: 支持链式调用
        """
        try:
            self.model = self.model_registry.load_model(path, **kwargs)
            self.logger.info(f"模型已从 {path} 加载")
            return self
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}")
            raise
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        获取特征重要性
        
        Returns:
            特征重要性字典
        """
        if self.model is None:
            raise ValueError("请先训练模型")
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                importance = dict(zip(
                    self.features.columns,
                    self.model.feature_importances_
                ))
                return importance
            elif hasattr(self.model, 'coef_'):
                importance = dict(zip(
                    self.features.columns,
                    abs(self.model.coef_).flatten()
                ))
                return importance
            else:
                self.logger.warning("模型不支持特征重要性分析")
                return None
                
        except Exception as e:
            self.logger.error(f"特征重要性分析失败: {str(e)}")
            raise
    
    def auto_ml(self, data_path: str, target_column: str, task_type: str = 'classification', **kwargs):
        """
        自动机器学习流程
        
        Args:
            data_path: 数据路径
            target_column: 目标列名
            task_type: 任务类型
            **kwargs: 其他参数
            
        Returns:
            评估结果
        """
        try:
            # 自动执行完整流程
            self.load_data(data_path, target_column)
            self.set_task_type(task_type)
            self.preprocess_data()
            
            # 自动选择最佳模型
            best_model = self._auto_select_model(task_type)
            self.select_model(best_model)
            
            # 训练和评估
            self.train()
            results = self.evaluate()
            
            self.logger.info("自动ML流程完成")
            return results
            
        except Exception as e:
            self.logger.error(f"自动ML流程失败: {str(e)}")
            raise
    
    def _auto_select_model(self, task_type: str) -> str:
        """
        自动选择最佳模型
        
        Args:
            task_type: 任务类型
            
        Returns:
            模型名称
        """
        if task_type == 'classification':
            return 'random_forest'
        elif task_type == 'regression':
            return 'random_forest'
        else:
            return 'kmeans'