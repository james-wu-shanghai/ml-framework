"""
日志记录器
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional


class Logger:
    """
    日志记录器
    
    提供统一的日志记录功能
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化日志记录器
        
        Args:
            config: 日志配置
        """
        self.config = config or {}
        self.setup_logger()
    
    def setup_logger(self):
        """设置日志记录器"""
        # 获取配置
        level = self.config.get('level', 'INFO')
        format_str = self.config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_file = self.config.get('file', None)
        
        # 创建根记录器
        logger = logging.getLogger('ml_framework')
        logger.setLevel(getattr(logging, level.upper()))
        
        # 清除已有的处理器
        logger.handlers.clear()
        
        # 创建格式化器
        formatter = logging.Formatter(format_str)
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 文件处理器
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path, encoding='utf-8')
            file_handler.setLevel(getattr(logging, level.upper()))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        self.logger = logger
    
    def info(self, message: str):
        """记录信息日志"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """记录警告日志"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """记录错误日志"""
        self.logger.error(message)
    
    def debug(self, message: str):
        """记录调试日志"""
        self.logger.debug(message)