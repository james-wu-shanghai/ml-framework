"""
GPU工具模块

提供GPU检测、配置和优化功能
"""

import logging
from typing import Dict, Any, Optional
import warnings


class GPUManager:
    """
    GPU管理器
    
    检测和配置GPU环境
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.pytorch_available = False
        self.tensorflow_available = False
        self.cuda_available = False
        
        self._check_gpu_support()
    
    def _check_gpu_support(self):
        """检查GPU支持情况"""
        # 检查PyTorch
        try:
            import torch
            self.pytorch_available = True
            self.cuda_available = torch.cuda.is_available()
            
            if self.cuda_available:
                self.logger.info(f"PyTorch CUDA可用 - 版本: {torch.version.cuda}")
                self.logger.info(f"GPU数量: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    gpu_name = torch.cuda.get_device_name(i)
                    self.logger.info(f"GPU {i}: {gpu_name}")
            else:
                self.logger.warning("PyTorch CUDA不可用")
                
        except ImportError:
            self.logger.warning("PyTorch未安装")
        
        # 检查TensorFlow
        try:
            import tensorflow as tf
            self.tensorflow_available = True
            
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                self.logger.info(f"TensorFlow GPU可用 - 设备数量: {len(gpus)}")
                for gpu in gpus:
                    self.logger.info(f"TensorFlow GPU: {gpu}")
                    
                # 配置GPU内存增长
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    self.logger.info("TensorFlow GPU内存增长已启用")
                except Exception as e:
                    self.logger.warning(f"TensorFlow GPU内存配置失败: {e}")
            else:
                self.logger.warning("TensorFlow GPU不可用")
                
        except ImportError:
            self.logger.warning("TensorFlow未安装")
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """获取GPU信息"""
        info = {
            'pytorch_available': self.pytorch_available,
            'tensorflow_available': self.tensorflow_available,
            'cuda_available': self.cuda_available,
            'gpu_devices': []
        }
        
        # PyTorch GPU信息
        if self.pytorch_available and self.cuda_available:
            try:
                import torch
                info['pytorch_cuda_version'] = torch.version.cuda
                info['pytorch_version'] = torch.__version__
                
                for i in range(torch.cuda.device_count()):
                    device_info = {
                        'id': i,
                        'name': torch.cuda.get_device_name(i),
                        'total_memory': torch.cuda.get_device_properties(i).total_memory,
                        'major': torch.cuda.get_device_properties(i).major,
                        'minor': torch.cuda.get_device_properties(i).minor
                    }
                    info['gpu_devices'].append(device_info)
            except Exception as e:
                self.logger.error(f"获取PyTorch GPU信息失败: {e}")
        
        # TensorFlow GPU信息
        if self.tensorflow_available:
            try:
                import tensorflow as tf
                info['tensorflow_version'] = tf.__version__
                tf_gpus = tf.config.list_physical_devices('GPU')
                info['tensorflow_gpu_count'] = len(tf_gpus)
            except Exception as e:
                self.logger.error(f"获取TensorFlow GPU信息失败: {e}")
        
        return info
    
    def optimize_pytorch_gpu(self):
        """优化PyTorch GPU设置"""
        if not self.pytorch_available or not self.cuda_available:
            return
        
        try:
            import torch
            
            # 启用CuDNN基准测试
            torch.backends.cudnn.benchmark = True
            self.logger.info("PyTorch CuDNN基准测试已启用")
            
            # 设置CUDA设备
            if torch.cuda.device_count() > 0:
                torch.cuda.set_device(0)
                self.logger.info("PyTorch默认CUDA设备设置为0")
                
        except Exception as e:
            self.logger.error(f"PyTorch GPU优化失败: {e}")
    
    def optimize_tensorflow_gpu(self):
        """优化TensorFlow GPU设置"""
        if not self.tensorflow_available:
            return
        
        try:
            import tensorflow as tf
            
            # 配置GPU内存增长
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                # 启用混合精度
                try:
                    tf.keras.mixed_precision.set_global_policy('mixed_float16')
                    self.logger.info("TensorFlow混合精度已启用")
                except Exception as e:
                    self.logger.warning(f"TensorFlow混合精度启用失败: {e}")
                    
        except Exception as e:
            self.logger.error(f"TensorFlow GPU优化失败: {e}")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """获取GPU内存使用情况"""
        memory_info = {}
        
        # PyTorch内存信息
        if self.pytorch_available and self.cuda_available:
            try:
                import torch
                memory_info['pytorch'] = {}
                
                for i in range(torch.cuda.device_count()):
                    device_memory = {
                        'allocated': torch.cuda.memory_allocated(i),
                        'reserved': torch.cuda.memory_reserved(i),
                        'max_allocated': torch.cuda.max_memory_allocated(i),
                        'max_reserved': torch.cuda.max_memory_reserved(i)
                    }
                    memory_info['pytorch'][f'device_{i}'] = device_memory
                    
            except Exception as e:
                self.logger.error(f"获取PyTorch内存信息失败: {e}")
        
        return memory_info
    
    def clear_cache(self):
        """清理GPU缓存"""
        if self.pytorch_available and self.cuda_available:
            try:
                import torch
                torch.cuda.empty_cache()
                self.logger.info("PyTorch GPU缓存已清理")
            except Exception as e:
                self.logger.error(f"清理PyTorch GPU缓存失败: {e}")
    
    def recommend_batch_size(self, model_size: str = 'medium') -> int:
        """根据GPU推荐批次大小"""
        if not self.cuda_available:
            return 32  # CPU默认批次大小
        
        try:
            import torch
            gpu_name = torch.cuda.get_device_name(0).lower()
            
            # 根据GPU型号推荐批次大小
            if '4090' in gpu_name:
                batch_sizes = {'small': 256, 'medium': 128, 'large': 64}
            elif '4080' in gpu_name:
                batch_sizes = {'small': 128, 'medium': 64, 'large': 32}
            elif '3090' in gpu_name or '3080' in gpu_name:
                batch_sizes = {'small': 128, 'medium': 64, 'large': 32}
            else:
                batch_sizes = {'small': 64, 'medium': 32, 'large': 16}
            
            recommended = batch_sizes.get(model_size, 32)
            self.logger.info(f"推荐批次大小: {recommended} (模型大小: {model_size})")
            return recommended
            
        except Exception as e:
            self.logger.error(f"推荐批次大小失败: {e}")
            return 32


# 全局GPU管理器实例
gpu_manager = GPUManager()