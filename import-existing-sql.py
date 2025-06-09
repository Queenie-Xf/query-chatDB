import os
import sys
import logging
import sqlite3
import re
import traceback
from typing import List, Optional
import concurrent.futures
import time
from datetime import datetime

class ErrorLogger:
    def __init__(self, base_dir=None, log_filename=None):
        """
        初始化错误日志记录器
        """
        # 确定日志目录
        if base_dir is None:
            base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        
        # 确保日志目录存在
        os.makedirs(base_dir, exist_ok=True)
        
        # 生成日志文件名
        if log_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f'conversion_errors_{timestamp}.log'
        
        error_log_path = os.path.join(base_dir, log_filename)
        
        # 配置日志记录器
        self.logger = logging.getLogger(f'conversion_logger_{log_filename}')
        self.logger.setLevel(logging.DEBUG)
        
        # 清除已存在的处理器
        self.logger.handlers.clear()
        
        try:
            # 创建文件处理器
            file_handler = logging.FileHandler(error_log_path, encoding='utf-8', mode='w')
            file_handler.setLevel(logging.DEBUG)
            
            # 创建控制台处理器
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            
            # 定义日志格式
            formatter = logging.Formatter(
                '%(asctime)s - [%(levelname)s] - %(message)s\n'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            # 添加处理器
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
            
            print(f"日志将写入: {error_log_path}")
            print(f"当前工作目录: {os.getcwd()}")
            print(f"日志文件绝对路径: {os.path.abspath(error_log_path)}")
            
            # 测试写入权限
            with open(error_log_path, 'a') as f:
                f.write("日志系统初始化成功\n")
        except Exception as e:
            print(f"日志系统初始化失败: {e}")
            print(traceback.format_exc())
            raise
        
        self.log_path = error_log_path

    def log_conversion_info(self, message: str):
        """
        记录普通转换信息
        """
        print(f"记录信息: {message}")
        self.logger.info(message)

    def log_error(self, message: str, exception: Optional[Exception] = None):
        """
        记录错误信息
        """
        error_details = message
        if exception:
            error_details += f"\n异常详情: {traceback.format_exc()}"
        
        print(f"错误: {error_details}")
        self.logger.error(error_details)

class SQLiteConverter:
    def convert_mysql_to_sqlite(self, input_file: str, output_db: str, error_logger: ErrorLogger):
        """
        转换MySQL到SQLite
        """
        error_logger.log_conversion_info(f"开始转换 {input_file}")
        
        try:
            # 转换逻辑...
            error_logger.log_conversion_info(f"成功转换 {input_file}")
        except Exception as e:
            error_logger.log_error(f"转换 {input_file} 失败", e)
            raise

def main():
    """
    主程序入口
    """
    # 为每个文件创建单独的日志
    conversions = [
        ('./data/hotel_location.sql', './data/hotel_location.db'),
        ('./data/hotel_rate.sql', './data/hotel_rate.db')
    ]
    
    for sql_file, db_file in conversions:
        # 为每个文件创建单独的日志记录器
        file_basename = os.path.splitext(os.path.basename(sql_file))[0]
        error_logger = ErrorLogger(log_filename=f'conversion_errors_{file_basename}.log')
        
        converter = SQLiteConverter()
        converter.convert_mysql_to_sqlite(sql_file, db_file, error_logger)

if __name__ == '__main__':
    main()