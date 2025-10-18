#!/usr/bin/env python3
"""
检查数据集分布，找出可能导致OOM的异常样本
"""

import json
import os
import yaml
from PIL import Image
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm

def load_yaml_config(yaml_path):
    """加载YAML配置文件"""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def analyze_dataset(json_path, image_folder, max_samples=1000):
    """分析单个数据集"""
    print(f"\nAnalyzing: {json_path}")
    
    if not os.path.exists(json_path):
        print(f"  ⚠ File not found!")
        return None
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 限制分析的样本数量
    if len(data) > max_samples:
        print(f"  Sampling {max_samples} from {len(data)} total samples")
        import random
        data = random.sample(data, max_samples)
    
    analysis = {
        'total_samples': len(data),
        'text_lengths': [],
        'num_images': [],
        'image_sizes': [],
        'conversation_turns': [],
    }
    
    for item in tqdm(data, desc="  Processing"):
        # 分析对话长度
        if 'conversations' in item:
            total_text_len = sum(len(conv.get('value', '')) for conv in item['conversations'])
            analysis['text_lengths'].append(total_text_len)
            analysis['conversation_turns'].append(len(item['conversations']))
        
        # 分析图像
        if 'image' in item:
            if isinstance(item['image'], list):
                analysis['num_images'].append(len(item['image']))
                # 尝试加载图像获取尺寸
                for img_name in item['image'][:3]:  # 只检查前3张
                    img_path = os.path.join(image_folder, img_name)
                    if os.path.exists(img_path):
                        try:
                            with Image.open(img_path) as img:
                                analysis['image_sizes'].append(img.size)
                        except:
                            pass
            else:
                analysis['num_images'].append(1)
                img_path = os.path.join(image_folder, item['image'])
                if os.path.exists(img_path):
                    try:
                        with Image.open(img_path) as img:
                            analysis['image_sizes'].append(img.size)
                    except:
                        pass
    
    # 计算统计信息
    stats = {}
    
    if analysis['text_lengths']:
        stats['text_length'] = {
            'min': min(analysis['text_lengths']),
            'max': max(analysis['text_lengths']),
            'mean': np.mean(analysis['text_lengths']),
            'median': np.median(analysis['text_lengths']),
            'p95': np.percentile(analysis['text_lengths'], 95),
            'p99': np.percentile(analysis['text_lengths'], 99),
        }
    
    if analysis['num_images']:
        stats['num_images'] = {
            'min': min(analysis['num_images']),
            'max': max(analysis['num_images']),
            'mean': np.mean(analysis['num_images']),
        }
    
    if analysis['image_sizes']:
        widths = [s[0] for s in analysis['image_sizes']]
        heights = [s[1] for s in analysis['image_sizes']]
        pixels = [w*h for w, h in analysis['image_sizes']]
        stats['image_size'] = {
            'max_width': max(widths),
            'max_height': max(heights),
            'max_pixels': max(pixels),
            'mean_pixels': np.mean(pixels),
            'p95_pixels': np.percentile(pixels, 95),
            'p99_pixels': np.percentile(pixels, 99),
        }
    
    if analysis['conversation_turns']:
        stats['conversation_turns'] = {
            'max': max(analysis['conversation_turns']),
            'mean': np.mean(analysis['conversation_turns']),
        }
    
    return stats

def main():
    # 读取配置
    config_path = './scripts/train/nano_single_image.yaml'
    image_folder = './data/images/single'
    
    print(f"{'='*80}")
    print("Dataset Distribution Analysis")
    print(f"{'='*80}")
    
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return
    
    config = load_yaml_config(config_path)
    
    all_stats = []
    
    for dataset in config['datasets']:
        json_path = dataset['json_path']
        sampling_strategy = dataset.get('sampling_strategy', 'all')
        
        stats = analyze_dataset(json_path, image_folder, max_samples=500)
        
        if stats:
            stats['json_path'] = json_path
            stats['sampling_strategy'] = sampling_strategy
            all_stats.append(stats)
            
            print(f"\n  Statistics:")
            if 'text_length' in stats:
                print(f"    Text length:")
                print(f"      Max: {stats['text_length']['max']:,} chars")
                print(f"      Mean: {stats['text_length']['mean']:.0f} chars")
                print(f"      95th percentile: {stats['text_length']['p95']:.0f} chars")
                print(f"      99th percentile: {stats['text_length']['p99']:.0f} chars")
            
            if 'num_images' in stats:
                print(f"    Number of images:")
                print(f"      Max: {stats['num_images']['max']}")
                print(f"      Mean: {stats['num_images']['mean']:.1f}")
            
            if 'image_size' in stats:
                print(f"    Image size:")
                print(f"      Max pixels: {stats['image_size']['max_pixels']:,} ({stats['image_size']['max_width']}x{stats['image_size']['max_height']})")
                print(f"      Mean pixels: {stats['image_size']['mean_pixels']:,.0f}")
                print(f"      95th percentile: {stats['image_size']['p95_pixels']:,.0f}")
                print(f"      99th percentile: {stats['image_size']['p99_pixels']:,.0f}")
            
            if 'conversation_turns' in stats:
                print(f"    Conversation turns:")
                print(f"      Max: {stats['conversation_turns']['max']}")
                print(f"      Mean: {stats['conversation_turns']['mean']:.1f}")
    
    # 保存完整分析结果
    output_path = './dataset_analysis.json'
    with open(output_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n{'='*80}")
    print(f"Full analysis saved to: {output_path}")
    print(f"{'='*80}")
    
    # 给出建议
    print("\n💡 Recommendations based on analysis:")
    
    max_text_len = max((s.get('text_length', {}).get('p99', 0) for s in all_stats), default=0)
    if max_text_len > 10000:
        print(f"  ⚠ Very long text sequences detected (up to {max_text_len:,} chars)")
        print(f"    Consider reducing model_max_length or filtering long samples")
    
    max_pixels = max((s.get('image_size', {}).get('p99_pixels', 0) for s in all_stats), default=0)
    if max_pixels > 1000000:  # > 1 megapixel
        print(f"  ⚠ Very large images detected (up to {max_pixels:,} pixels)")
        print(f"    Consider preprocessing images to resize them")
    
    max_num_images = max((s.get('num_images', {}).get('max', 0) for s in all_stats), default=0)
    if max_num_images > 4:
        print(f"  ⚠ Samples with many images detected (up to {max_num_images} images)")
        print(f"    Multi-image samples consume more memory")

if __name__ == "__main__":
    main()
