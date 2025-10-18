#!/usr/bin/env python3
"""
检查GPU内存使用情况
"""

import torch
import psutil
import os

def check_system_memory():
    """检查系统内存"""
    memory = psutil.virtual_memory()
    print(f"系统内存:")
    print(f"  总内存: {memory.total / (1024**3):.2f} GB")
    print(f"  可用内存: {memory.available / (1024**3):.2f} GB")
    print(f"  使用率: {memory.percent:.1f}%")

def check_gpu_memory():
    """检查GPU内存"""
    if torch.cuda.is_available():
        print(f"\nGPU内存:")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / (1024**3)
            
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            cached = torch.cuda.memory_reserved(i) / (1024**3)
            free = total_memory - cached
            
            print(f"  GPU {i} ({props.name}):")
            print(f"    总内存: {total_memory:.2f} GB")
            print(f"    已分配: {allocated:.2f} GB")
            print(f"    已缓存: {cached:.2f} GB")
            print(f"    可用: {free:.2f} GB")
    else:
        print("\n未检测到CUDA设备")

def check_environment():
    """检查环境变量"""
    print(f"\n环境变量:")
    env_vars = [
        'CUDA_VISIBLE_DEVICES',
        'PYTORCH_CUDA_ALLOC_CONF',
        'PYTORCH_ALLOC_CONF',
        'NCCL_DEBUG',
        'NCCL_IB_DISABLE',
        'NCCL_P2P_DISABLE'
    ]
    
    for var in env_vars:
        value = os.environ.get(var, '未设置')
        print(f"  {var}: {value}")

def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        print(f"\n清理GPU内存...")
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("✅ GPU内存已清理")

def main():
    print("=== 内存诊断报告 ===")
    check_system_memory()
    check_gpu_memory()
    check_environment()
    
    print(f"\n=== 建议 ===")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / (1024**3)
            
            if total_memory < 16:
                print(f"⚠️  GPU {i} 内存较小 ({total_memory:.1f}GB)，建议:")
                print(f"   - 使用CPU offload")
                print(f"   - 减少batch size")
                print(f"   - 使用gradient checkpointing")
            elif total_memory < 24:
                print(f"⚠️  GPU {i} 内存中等 ({total_memory:.1f}GB)，建议:")
                print(f"   - 使用ZeRO-3 with CPU offload")
                print(f"   - 减少model_max_length")
            else:
                print(f"✅ GPU {i} 内存充足 ({total_memory:.1f}GB)")

if __name__ == "__main__":
    main()