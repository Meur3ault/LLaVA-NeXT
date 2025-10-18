#!/usr/bin/env python3
"""
分析OOM日志的脚本
"""

import json
import os
import sys
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # 无GUI后端
import matplotlib.pyplot as plt

def analyze_memory_log(log_path="./memory_usage.jsonl"):
    """分析内存使用日志"""
    if not os.path.exists(log_path):
        print(f"Memory log not found: {log_path}")
        return
    
    data = []
    with open(log_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    if not data:
        print("No data in memory log")
        return
    
    print(f"\n{'='*80}")
    print("Memory Usage Analysis")
    print(f"{'='*80}")
    print(f"Total records: {len(data)}")
    
    # 按步骤分组
    by_step = defaultdict(list)
    for record in data:
        by_step[record['step']].append(record)
    
    # 找出内存使用最高的步骤
    max_mem_by_step = {}
    for step, records in by_step.items():
        max_allocated = max(r['allocated_gb'] for r in records)
        max_mem_by_step[step] = max_allocated
    
    # 排序找出前10个内存使用最高的步骤
    top_steps = sorted(max_mem_by_step.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print("\nTop 10 steps with highest memory usage:")
    for step, mem in top_steps:
        print(f"  Step {step}: {mem:.2f} GB")
    
    # 分析内存增长趋势
    print("\nMemory growth trend:")
    steps = sorted(by_step.keys())
    if len(steps) > 1:
        first_step_mem = by_step[steps[0]][0]['allocated_gb']
        last_step_mem = by_step[steps[-1]][-1]['allocated_gb']
        growth = last_step_mem - first_step_mem
        print(f"  First step ({steps[0]}): {first_step_mem:.2f} GB")
        print(f"  Last step ({steps[-1]}): {last_step_mem:.2f} GB")
        print(f"  Growth: {growth:+.2f} GB")
    
    # 绘制内存使用曲线
    try:
        steps_list = []
        allocated_list = []
        reserved_list = []
        
        for step in sorted(by_step.keys()):
            for record in by_step[step]:
                if record['phase'] == 'post_backward' or record['phase'] == 'pre_forward':
                    steps_list.append(step)
                    allocated_list.append(record['allocated_gb'])
                    reserved_list.append(record['reserved_gb'])
                    break
        
        plt.figure(figsize=(12, 6))
        plt.plot(steps_list, allocated_list, label='Allocated', marker='o', markersize=2)
        plt.plot(steps_list, reserved_list, label='Reserved', marker='s', markersize=2)
        plt.xlabel('Training Step')
        plt.ylabel('Memory (GB)')
        plt.title('GPU Memory Usage Over Training Steps')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('memory_usage_plot.png', dpi=150, bbox_inches='tight')
        print(f"\nMemory usage plot saved to: memory_usage_plot.png")
    except Exception as e:
        print(f"Failed to create plot: {e}")

def analyze_batch_log(log_path="./debug_oom_log.jsonl"):
    """分析批次信息日志"""
    if not os.path.exists(log_path):
        print(f"Batch log not found: {log_path}")
        return
    
    data = []
    with open(log_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    if not data:
        print("No data in batch log")
        return
    
    print(f"\n{'='*80}")
    print("Batch Information Analysis")
    print(f"{'='*80}")
    print(f"Total batches: {len(data)}")
    
    # 分析输入序列长度分布
    if 'input_ids_shape' in data[0]:
        seq_lengths = [record['input_ids_shape'][1] for record in data if 'input_ids_shape' in record]
        if seq_lengths:
            print("\nSequence length statistics:")
            print(f"  Min: {min(seq_lengths)}")
            print(f"  Max: {max(seq_lengths)}")
            print(f"  Avg: {sum(seq_lengths)/len(seq_lengths):.1f}")
            print(f"  Median: {sorted(seq_lengths)[len(seq_lengths)//2]}")
            
            # 找出序列长度最长的步骤
            max_len_idx = seq_lengths.index(max(seq_lengths))
            max_len_step = data[max_len_idx]['step']
            print(f"\n  Longest sequence at step {max_len_step}: {max(seq_lengths)} tokens")
    
    # 检查图像相关的输入
    image_keys = [key for key in data[0].keys() if 'image' in key.lower() or 'pixel' in key.lower()]
    if image_keys:
        print(f"\nImage-related inputs found: {image_keys}")
        for key in image_keys:
            if key + '_shape' in data[0] or key + '_len' in data[0]:
                shapes_key = key + '_shape' if key + '_shape' in data[0] else key + '_len'
                values = [record.get(shapes_key) for record in data if record.get(shapes_key)]
                if values:
                    print(f"  {key}: {set(values) if len(set(values)) < 10 else 'varied'}")

def analyze_oom_event(event_path="oom_event.json"):
    """分析OOM事件"""
    if not os.path.exists(event_path):
        print(f"\nNo OOM event file found at: {event_path}")
        print("(This is good - means no OOM occurred yet!)")
        return
    
    print(f"\n{'='*80}")
    print("OOM Event Analysis")
    print(f"{'='*80}")
    
    with open(event_path, 'r') as f:
        oom_data = json.load(f)
    
    print(f"OOM occurred at step: {oom_data['step']}")
    print(f"Timestamp: {oom_data['timestamp']}")
    print(f"Error: {oom_data['error']}")
    
    print("\nBatch that caused OOM:")
    for key, value in oom_data['batch_info'].items():
        if key not in ['step', 'timestamp']:
            print(f"  {key}: {value}")

if __name__ == "__main__":
    print("Analyzing OOM debugging logs...\n")
    
    # 分析内存日志
    analyze_memory_log()
    
    # 分析批次日志
    analyze_batch_log()
    
    # 分析OOM事件
    analyze_oom_event()
    
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}\n")
