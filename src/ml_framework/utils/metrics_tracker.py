"""
指标跟踪器
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime


class MetricsTracker:
    """
    指标跟踪器
    
    用于跟踪和记录实验指标
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化指标跟踪器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        self.experiment_name = self.config.get('experiment_name', 'default')
        
        # 存储指标
        self.metrics = []
        self.current_run = None
    
    def start_run(self, run_name: Optional[str] = None):
        """
        开始新的实验运行
        
        Args:
            run_name: 运行名称
        """
        if not self.enabled:
            return
        
        self.current_run = {
            'run_name': run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'start_time': datetime.now().isoformat(),
            'metrics': {},
            'parameters': {},
            'artifacts': []
        }
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        记录指标
        
        Args:
            metrics: 指标字典
            step: 步数
        """
        if not self.enabled or not self.current_run:
            return
        
        timestamp = datetime.now().isoformat()
        
        for key, value in metrics.items():
            if key not in self.current_run['metrics']:
                self.current_run['metrics'][key] = []
            
            self.current_run['metrics'][key].append({
                'value': value,
                'timestamp': timestamp,
                'step': step
            })
    
    def log_parameters(self, parameters: Dict[str, Any]):
        """
        记录参数
        
        Args:
            parameters: 参数字典
        """
        if not self.enabled or not self.current_run:
            return
        
        self.current_run['parameters'].update(parameters)
    
    def log_artifact(self, artifact_path: str, artifact_type: str = 'file'):
        """
        记录工件
        
        Args:
            artifact_path: 工件路径
            artifact_type: 工件类型
        """
        if not self.enabled or not self.current_run:
            return
        
        self.current_run['artifacts'].append({
            'path': artifact_path,
            'type': artifact_type,
            'timestamp': datetime.now().isoformat()
        })
    
    def end_run(self):
        """结束当前运行"""
        if not self.enabled or not self.current_run:
            return
        
        self.current_run['end_time'] = datetime.now().isoformat()
        self.metrics.append(self.current_run.copy())
        self.current_run = None
    
    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """获取指标历史"""
        return self.metrics
    
    def save_metrics(self, output_path: str):
        """
        保存指标到文件
        
        Args:
            output_path: 输出路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
    
    def load_metrics(self, input_path: str):
        """
        从文件加载指标
        
        Args:
            input_path: 输入路径
        """
        input_path = Path(input_path)
        
        if input_path.exists():
            with open(input_path, 'r', encoding='utf-8') as f:
                self.metrics = json.load(f)
    
    def get_best_run(self, metric_name: str, mode: str = 'max') -> Optional[Dict[str, Any]]:
        """
        获取最佳运行
        
        Args:
            metric_name: 指标名称
            mode: 模式 ('max' 或 'min')
            
        Returns:
            最佳运行信息
        """
        if not self.metrics:
            return None
        
        best_run = None
        best_value = None
        
        for run in self.metrics:
            if metric_name in run['metrics']:
                # 获取最后一个值
                values = run['metrics'][metric_name]
                if values:
                    current_value = values[-1]['value']
                    
                    if best_value is None:
                        best_run = run
                        best_value = current_value
                    elif (mode == 'max' and current_value > best_value) or \
                         (mode == 'min' and current_value < best_value):
                        best_run = run
                        best_value = current_value
        
        return best_run
    
    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame格式"""
        if not self.metrics:
            return pd.DataFrame()
        
        rows = []
        for run in self.metrics:
            row = {
                'run_name': run['run_name'],
                'start_time': run['start_time'],
                'end_time': run.get('end_time')
            }
            
            # 添加参数
            row.update({f"param_{k}": v for k, v in run.get('parameters', {}).items()})
            
            # 添加最终指标值
            for metric_name, metric_values in run.get('metrics', {}).items():
                if metric_values:
                    row[f"metric_{metric_name}"] = metric_values[-1]['value']
            
            rows.append(row)
        
        return pd.DataFrame(rows)