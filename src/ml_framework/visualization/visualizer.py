"""
可视化器
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging


class Visualizer:
    """
    可视化器
    
    提供各种数据和模型可视化功能
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化可视化器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 设置样式
        self.setup_style()
    
    def setup_style(self):
        """设置绘图样式"""
        style = self.config.get('style', 'seaborn')
        plt.style.use(style if style in plt.style.available else 'default')
        
        # 设置默认图像大小
        figsize = self.config.get('figure_size', [10, 8])
        plt.rcParams['figure.figsize'] = figsize
        
        # 设置DPI
        dpi = self.config.get('dpi', 100)
        plt.rcParams['figure.dpi'] = dpi
    
    def plot_data_distribution(self, data: pd.DataFrame, target_column: Optional[str] = None):
        """
        绘制数据分布图
        
        Args:
            data: 数据
            target_column: 目标列名
        """
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        
        # 数值特征分布
        if len(numeric_columns) > 0:
            n_cols = min(3, len(numeric_columns))
            n_rows = (len(numeric_columns) - 1) // n_cols + 1
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes]
            
            for i, col in enumerate(numeric_columns):
                if i < len(axes):
                    axes[i].hist(data[col].dropna(), bins=30, alpha=0.7)
                    axes[i].set_title(f'{col} 分布')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('频次')
            
            # 隐藏多余的子图
            for i in range(len(numeric_columns), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            self._save_plot('data_distribution_numeric.png')
            plt.show()
        
        # 分类特征分布
        if len(categorical_columns) > 0:
            n_cols = min(2, len(categorical_columns))
            n_rows = (len(categorical_columns) - 1) // n_cols + 1
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5*n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1 or n_cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(categorical_columns):
                if i < len(axes):
                    value_counts = data[col].value_counts()
                    axes[i].bar(range(len(value_counts)), value_counts.values)
                    axes[i].set_title(f'{col} 分布')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('频次')
                    axes[i].set_xticks(range(len(value_counts)))
                    axes[i].set_xticklabels(value_counts.index, rotation=45)
            
            # 隐藏多余的子图
            for i in range(len(categorical_columns), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            self._save_plot('data_distribution_categorical.png')
            plt.show()
    
    def plot_correlation_matrix(self, data: pd.DataFrame):
        """
        绘制相关系数矩阵
        
        Args:
            data: 数据
        """
        # 只选择数值特征
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.shape[1] < 2:
            self.logger.warning("数值特征少于2个，无法绘制相关系数矩阵")
            return
        
        correlation_matrix = numeric_data.corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   fmt='.2f')
        plt.title('特征相关系数矩阵')
        plt.tight_layout()
        self._save_plot('correlation_matrix.png')
        plt.show()
    
    def plot_model_performance(self, 
                             model, 
                             X: pd.DataFrame, 
                             y: pd.Series,
                             task_type: str = 'classification',
                             **kwargs):
        """
        绘制模型性能图
        
        Args:
            model: 模型实例
            X: 特征数据
            y: 真实标签
            task_type: 任务类型
            **kwargs: 其他参数
        """
        predictions = model.predict(X)
        
        if task_type == 'classification':
            self._plot_classification_results(y, predictions, model, X)
        elif task_type == 'regression':
            self._plot_regression_results(y, predictions)
        elif task_type == 'clustering':
            self._plot_clustering_results(X, predictions)
    
    def _plot_classification_results(self, y_true, y_pred, model, X):
        """绘制分类结果"""
        from sklearn.metrics import confusion_matrix, classification_report
        
        # 混淆矩阵
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        self._save_plot('confusion_matrix.png')
        plt.show()
        
        # ROC曲线（二分类）
        if hasattr(model, 'predict_proba') and len(np.unique(y_true)) == 2:
            from sklearn.metrics import roc_curve, roc_auc_score
            
            y_proba = model.predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc = roc_auc_score(y_true, y_proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC曲线 (AUC = {auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--', label='随机分类器')
            plt.xlabel('假正率')
            plt.ylabel('真正率')
            plt.title('ROC曲线')
            plt.legend()
            plt.grid()
            self._save_plot('roc_curve.png')
            plt.show()
    
    def _plot_regression_results(self, y_true, y_pred):
        """绘制回归结果"""
        # 真实值 vs 预测值
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title('真实值 vs 预测值')
        
        # 残差图
        residuals = y_true - y_pred
        plt.subplot(1, 2, 2)
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('预测值')
        plt.ylabel('残差')
        plt.title('残差图')
        
        plt.tight_layout()
        self._save_plot('regression_results.png')
        plt.show()
    
    def _plot_clustering_results(self, X, cluster_labels):
        """绘制聚类结果"""
        # 如果特征数大于2，使用PCA降维
        if X.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_reduced = pca.fit_transform(X)
        else:
            X_reduced = X.iloc[:, :2].values if hasattr(X, 'iloc') else X[:, :2]
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter)
        plt.title('聚类结果')
        plt.xlabel('第一主成分' if X.shape[1] > 2 else '特征1')
        plt.ylabel('第二主成分' if X.shape[1] > 2 else '特征2')
        self._save_plot('clustering_results.png')
        plt.show()
    
    def plot_feature_importance(self, model, feature_names: List[str], top_k: int = 20):
        """
        绘制特征重要性
        
        Args:
            model: 模型实例
            feature_names: 特征名称列表
            top_k: 显示前k个重要特征
        """
        importance = None
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_).flatten()
        
        if importance is not None:
            # 获取前k个重要特征
            indices = np.argsort(importance)[::-1][:top_k]
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(indices)), importance[indices])
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('重要性')
            plt.title(f'前{top_k}个重要特征')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            self._save_plot('feature_importance.png')
            plt.show()
        else:
            self.logger.warning("模型不支持特征重要性分析")
    
    def plot_learning_curve(self, train_scores: List[float], val_scores: List[float]):
        """
        绘制学习曲线
        
        Args:
            train_scores: 训练分数
            val_scores: 验证分数
        """
        plt.figure(figsize=(10, 6))
        plt.plot(train_scores, label='训练分数', marker='o')
        plt.plot(val_scores, label='验证分数', marker='s')
        plt.xlabel('迭代次数')
        plt.ylabel('分数')
        plt.title('学习曲线')
        plt.legend()
        plt.grid()
        self._save_plot('learning_curve.png')
        plt.show()
    
    def _save_plot(self, filename: str):
        """保存图像"""
        if self.config.get('save_plots', False):
            plots_dir = Path(self.config.get('plots_dir', 'plots'))
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            plot_format = self.config.get('plot_format', 'png')
            filepath = plots_dir / filename
            
            plt.savefig(filepath, format=plot_format, dpi=self.config.get('dpi', 100), bbox_inches='tight')
            self.logger.info(f"图像已保存: {filepath}")
    
    def create_dashboard(self, data: pd.DataFrame, model_results: Dict[str, Any]):
        """
        创建仪表板
        
        Args:
            data: 数据
            model_results: 模型结果
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 数据概览
        axes[0, 0].text(0.1, 0.5, f"数据形状: {data.shape}\n缺失值: {data.isnull().sum().sum()}", 
                       transform=axes[0, 0].transAxes, fontsize=12)
        axes[0, 0].set_title('数据概览')
        axes[0, 0].axis('off')
        
        # 目标变量分布
        if 'target' in data.columns:
            data['target'].value_counts().plot(kind='bar', ax=axes[0, 1])
            axes[0, 1].set_title('目标变量分布')
        
        # 模型性能
        if 'accuracy' in model_results:
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            values = [model_results.get(m, 0) for m in metrics]
            axes[1, 0].bar(metrics, values)
            axes[1, 0].set_title('模型性能指标')
            axes[1, 0].set_ylim(0, 1)
        
        # 特征分布
        numeric_features = data.select_dtypes(include=[np.number]).columns[:5]
        if len(numeric_features) > 0:
            data[numeric_features].hist(ax=axes[1, 1], bins=20)
            axes[1, 1].set_title('主要特征分布')
        
        plt.tight_layout()
        self._save_plot('dashboard.png')
        plt.show()