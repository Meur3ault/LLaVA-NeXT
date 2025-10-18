#!/usr/bin/env python3
"""
调试OOM问题的脚本
该脚本会在训练过程中记录每个批次的详细信息，帮助定位导致OOM的具体数据样本
"""

import os
import sys
import json
import torch
import gc
from datetime import datetime

# 设置环境变量
os.environ['CUDA_HOME'] = os.environ.get('CUDA_HOME', '/usr/local/cuda')
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu:' + os.environ.get('CUDA_HOME', '/usr/local/cuda') + '/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu:' + os.environ.get('CUDA_HOME', '/usr/local/cuda') + '/lib64:' + os.environ.get('LIBRARY_PATH', '')
os.environ['LDFLAGS'] = '-L/usr/lib/x86_64-linux-gnu -L/lib/x86_64-linux-gnu ' + os.environ.get('LDFLAGS', '')

# 导入训练相关模块
from llava.train.train import train, LLaVATrainer, TrainingArguments
from transformers import Trainer
import transformers

# 创建日志文件
debug_log_path = "./debug_oom_log.jsonl"
memory_log_path = "./memory_usage.jsonl"

def log_gpu_memory(step, phase="", extra_info=None):
    """记录GPU内存使用情况"""
    if torch.cuda.is_available():
        memory_info = {
            "step": step,
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "max_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
        }
        
        if extra_info:
            memory_info.update(extra_info)
        
        # 写入日志
        with open(memory_log_path, 'a') as f:
            f.write(json.dumps(memory_info, ensure_ascii=False) + '\n')
        
        return memory_info
    return None

def log_batch_info(step, batch, extra_info=None):
    """记录批次详细信息"""
    batch_info = {
        "step": step,
        "timestamp": datetime.now().isoformat(),
    }
    
    # 记录批次中的张量形状
    if isinstance(batch, dict):
        for key, value in batch.items():
            if torch.is_tensor(value):
                batch_info[f"{key}_shape"] = list(value.shape)
                batch_info[f"{key}_dtype"] = str(value.dtype)
            elif isinstance(value, (list, tuple)) and len(value) > 0 and torch.is_tensor(value[0]):
                batch_info[f"{key}_len"] = len(value)
                batch_info[f"{key}_first_shape"] = list(value[0].shape)
    
    if extra_info:
        batch_info.update(extra_info)
    
    # 写入日志
    with open(debug_log_path, 'a') as f:
        f.write(json.dumps(batch_info, ensure_ascii=False) + '\n')
    
    return batch_info

# 猴子补丁：修改Trainer的training_step方法
original_training_step = Trainer.training_step

def debug_training_step(self, model, inputs):
    """带调试信息的training_step"""
    step = self.state.global_step
    
    # 在前向传播前记录内存
    pre_forward_mem = log_gpu_memory(step, "pre_forward")
    
    # 记录输入批次信息
    batch_info = log_batch_info(step, inputs)
    
    # 打印到控制台
    if step % 10 == 0:  # 每10步打印一次
        print(f"\n{'='*80}")
        print(f"Step {step} - Memory Info:")
        if pre_forward_mem:
            print(f"  Allocated: {pre_forward_mem['allocated_gb']:.2f} GB")
            print(f"  Reserved: {pre_forward_mem['reserved_gb']:.2f} GB")
            print(f"  Max Allocated: {pre_forward_mem['max_allocated_gb']:.2f} GB")
        print(f"Batch Info:")
        for key, value in batch_info.items():
            if key not in ['step', 'timestamp']:
                print(f"  {key}: {value}")
        print(f"{'='*80}\n")
    
    try:
        # 执行原始的training_step
        loss = original_training_step(self, model, inputs)
        
        # 在backward后记录内存
        post_backward_mem = log_gpu_memory(step, "post_backward")
        
        # 手动清理缓存（可选）
        if step % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            after_cleanup_mem = log_gpu_memory(step, "after_cleanup")
        
        return loss
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            # OOM发生时，记录详细信息
            print(f"\n{'!'*80}")
            print(f"OOM occurred at step {step}!")
            print(f"Batch info that caused OOM:")
            for key, value in batch_info.items():
                print(f"  {key}: {value}")
            
            # 记录OOM事件
            oom_info = {
                "step": step,
                "event": "OOM",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "batch_info": batch_info
            }
            with open("oom_event.json", 'w') as f:
                json.dump(oom_info, f, indent=2, ensure_ascii=False)
            
            print(f"\nOOM details saved to oom_event.json")
            print(f"{'!'*80}\n")
        
        # 重新抛出异常
        raise e

# 应用猴子补丁
Trainer.training_step = debug_training_step
LLaVATrainer.training_step = debug_training_step

if __name__ == "__main__":
    print("="*80)
    print("Starting training with OOM debugging enabled")
    print(f"Debug log will be saved to: {debug_log_path}")
    print(f"Memory log will be saved to: {memory_log_path}")
    print("="*80)
    
    # 清空之前的日志
    for log_file in [debug_log_path, memory_log_path]:
        if os.path.exists(log_file):
            os.remove(log_file)
    
    # 运行训练
    train()
