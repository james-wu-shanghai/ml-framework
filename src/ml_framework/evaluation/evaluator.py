"""
模型评估器
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)
import logging


class Evaluator:
    """
    模型评估器
    
    提供各种评估指标的计算
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化评估器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def evaluate(self, 
                 model, 
                 X: pd.DataFrame, 
                 y: pd.Series,
                 task_type: str = 'classification',
                 **kwargs) -> Dict[str, Any]:
        """
        评估模型性能
        
        Args:
            model: 模型实例
            X: 特征数据
            y: 真实标签
            task_type: 任务类型
            **kwargs: 其他参数
            
        Returns:
            评估结果字典
        """
        self.logger.info(f"开始模型评估 ({task_type})")
        
        try:
            # 获取预测结果
            if hasattr(model, 'predict'):
                y_pred = model.predict(X)
            else:
                raise ValueError("模型不支持预测")
            
            # 根据任务类型计算指标
            if task_type == 'classification':
                results = self._evaluate_classification(y, y_pred, model, X)
            elif task_type == 'regression':
                results = self._evaluate_regression(y, y_pred)
            elif task_type == 'clustering':
                results = self._evaluate_clustering(X, y_pred)
            else:
                raise ValueError(f"不支持的任务类型: {task_type}")
            
            # 添加基本信息
            results.update({
                'task_type': task_type,
                'model_type': type(model).__name__,
                'num_samples': len(y),
                'num_features': X.shape[1] if hasattr(X, 'shape') else len(X[0])
            })
            
            self.logger.info("模型评估完成")
            return results
            
        except Exception as e:
            self.logger.error(f"模型评估失败: {str(e)}")
            raise
    
    def _evaluate_classification(self, y_true, y_pred, model, X) -> Dict[str, Any]:
        """评估分类模型"""
        results = {}
        
        # 基本指标
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        results['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        results['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        results['confusion_matrix'] = cm.tolist()
        
        # 分类报告
        results['classification_report'] = classification_report(y_true, y_pred, output_dict=True)
        
        # 如果支持概率预测
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X)
                results['has_probabilities'] = True
                
                # 计算ROC AUC等指标
                from sklearn.metrics import roc_auc_score
                if len(np.unique(y_true)) == 2:  # 二分类
                    results['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                else:  # 多分类
                    results['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
                    
            except Exception:
                results['has_probabilities'] = False
        else:
            results['has_probabilities'] = False
        
        return results
    
    def _evaluate_regression(self, y_true, y_pred) -> Dict[str, Any]:
        """评估回归模型"""
        results = {}
        
        # 基本指标
        results['mse'] = mean_squared_error(y_true, y_pred)
        results['rmse'] = np.sqrt(results['mse'])
        results['mae'] = mean_absolute_error(y_true, y_pred)
        results['r2'] = r2_score(y_true, y_pred)
        
        # 平均绝对百分比误差
        results['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # 残差统计
        residuals = y_true - y_pred
        results['residuals_mean'] = np.mean(residuals)
        results['residuals_std'] = np.std(residuals)
        
        return results
    
    def _evaluate_clustering(self, X, cluster_labels) -> Dict[str, Any]:
        """评估聚类模型"""
        results = {}
        
        try:
            from sklearn.metrics import silhouette_score, calinski_harabasz_score
            
            # 轮廓系数
            results['silhouette_score'] = silhouette_score(X, cluster_labels)
            
            # Calinski-Harabasz指数
            results['calinski_harabasz_score'] = calinski_harabasz_score(X, cluster_labels)
            
            # 聚类数量
            results['num_clusters'] = len(np.unique(cluster_labels))
            
            # 每个聚类的样本数
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            results['cluster_sizes'] = dict(zip(unique_labels.tolist(), counts.tolist()))
            
        except Exception as e:
            self.logger.warning(f"聚类评估部分指标计算失败: {str(e)}")
        
        return results
    
    def compare_models(self, model_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        比较多个模型的性能
        
        Args:
            model_results: 模型结果字典
            
        Returns:
            比较结果DataFrame
        """
        comparison_data = []
        
        for model_name, results in model_results.items():
            row = {'model': model_name}
            
            # 提取主要指标
            if results['task_type'] == 'classification':
                row.update({
                    'accuracy': results.get('accuracy'),
                    'precision': results.get('precision'),
                    'recall': results.get('recall'),
                    'f1': results.get('f1'),
                    'roc_auc': results.get('roc_auc')
                })
            elif results['task_type'] == 'regression':
                row.update({
                    'mse': results.get('mse'),
                    'rmse': results.get('rmse'),
                    'mae': results.get('mae'),
                    'r2': results.get('r2'),
                    'mape': results.get('mape')
                })
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def get_best_model(self, model_results: Dict[str, Dict[str, Any]], metric: str = 'auto') -> str:
        """
        获取最佳模型
        
        Args:
            model_results: 模型结果字典
            metric: 评估指标
            
        Returns:
            最佳模型名称
        """
        if not model_results:
            raise ValueError("没有模型结果可比较")
        
        # 自动选择指标
        if metric == 'auto':
            first_result = next(iter(model_results.values()))
            task_type = first_result['task_type']
            
            if task_type == 'classification':
                metric = 'f1'
            elif task_type == 'regression':
                metric = 'r2'
            else:
                metric = 'silhouette_score'
        
        # 找到最佳模型
        best_model = None
        best_score = None
        
        # 确定是否越大越好
        higher_is_better = metric not in ['mse', 'rmse', 'mae', 'mape']
        
        for model_name, results in model_results.items():
            score = results.get(metric)
            if score is not None:
                if best_score is None:
                    best_model = model_name
                    best_score = score
                elif (higher_is_better and score > best_score) or (not higher_is_better and score < best_score):
                    best_model = model_name
                    best_score = score
        
        return best_model