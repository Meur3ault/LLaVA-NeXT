#!/usr/bin/env python3
"""
监控训练过程中的内存使用情况
"""

import time
import psutil
import torch
import os
import threading
from datetime import datetime

class MemoryMonitor:
    def __init__(self, log_file="memory_log.txt", interval=5):
        self.log_file = log_file
        self.interval = interval
        self.monitoring = False
        self.thread = None
        
    def get_memory_info(self):
        """获取内存信息"""
        # 系统内存
        system_memory = psutil.virtual_memory()
        
        # GPU内存
        gpu_info = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                cached = torch.cuda.memory_reserved(i) / (1024**3)
                max_allocated = torch.cuda.max_memory_allocated(i) / (1024**3)
                gpu_info.append({
                    'device': i,
                    'allocated': allocated,
                    'cached': cached,
                    'max_allocated': max_allocated
                })
        
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'system_memory': {
                'total': system_memory.total / (1024**3),
                'available': system_memory.available / (1024**3),
                'used': system_memory.used / (1024**3),
                'percent': system_memory.percent
            },
            'gpu_memory': gpu_info
        }
    
    def log_memory(self, info):
        """记录内存信息"""
        with open(self.log_file, 'a') as f:
            f.write(f"\n=== {info['timestamp']} ===\n")
            f.write(f"系统内存: {info['system_memory']['used']:.2f}GB / {info['system_memory']['total']:.2f}GB ({info['system_memory']['percent']:.1f}%)\n")
            
            for gpu in info['gpu_memory']:
                f.write(f"GPU {gpu['device']}: 已分配 {gpu['allocated']:.2f}GB, 已缓存 {gpu['cached']:.2f}GB, 最大分配 {gpu['max_allocated']:.2f}GB\n")
    
    def monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                info = self.get_memory_info()
                self.log_memory(info)
                
                # 检查内存使用率
                if info['system_memory']['percent'] > 90:
                    print(f"⚠️ 系统内存使用率过高: {info['system_memory']['percent']:.1f}%")
                
                for gpu in info['gpu_memory']:
                    if gpu['allocated'] > 20:  # 假设GPU有24GB内存
                        print(f"⚠️ GPU {gpu['device']} 内存使用过高: {gpu['allocated']:.2f}GB")
                
                time.sleep(self.interval)
            except Exception as e:
                print(f"监控出错: {e}")
                time.sleep(self.interval)
    
    def start(self):
        """开始监控"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.thread = threading.Thread(target=self.monitor_loop)
        self.thread.daemon = True
        self.thread.start()
        print(f"开始监控内存使用，日志文件: {self.log_file}")
    
    def stop(self):
        """停止监控"""
        self.monitoring = False
        if self.thread:
            self.thread.join()
        print("停止内存监控")

def main():
    monitor = MemoryMonitor()
    
    try:
        monitor.start()
        
        # 模拟训练过程
        print("开始模拟训练...")
        time.sleep(60)  # 监控1分钟
        
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        monitor.stop()

if __name__ == "__main__":
    main()