#!/usr/bin/env python3
"""
分析NPP API头文件中声明但未实现的函数
"""

import os
import re
import glob
import json
from collections import defaultdict

def extract_function_declarations(header_file):
    """从头文件中提取函数声明"""
    functions = []
    with open(header_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # 查找函数声明（排除注释）
    # 匹配格式: NppStatus function_name(...);
    pattern = r'^\s*(NppStatus)\s+(nppi\w+|npps\w+)\s*\([^{}]*\)\s*;'
    
    lines = content.split('\n')
    for i, line in enumerate(lines):
        # 跳过注释行
        if line.strip().startswith('//') or line.strip().startswith('/*') or line.strip().startswith('*'):
            continue
            
        match = re.search(pattern, line)
        if match:
            return_type = match.group(1).strip()
            function_name = match.group(2).strip()
            
            # 获取完整的函数签名（可能跨行）
            full_line = line
            j = i + 1
            while j < len(lines) and ';' not in full_line:
                full_line += ' ' + lines[j].strip()
                j += 1
            
            functions.append({
                'name': function_name,
                'return_type': return_type,
                'signature': full_line.strip(),
                'file': os.path.basename(header_file)
            })
    
    return functions

def extract_implemented_functions(src_dir):
    """从源代码中提取已实现的函数"""
    implemented = set()
    
    # 扫描所有.cpp和.cu文件
    for pattern in ['**/*.cpp', '**/*.cu']:
        for file_path in glob.glob(os.path.join(src_dir, pattern), recursive=True):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # 查找函数定义
            # 匹配格式: NppStatus function_name(...) {
            pattern_impl = r'(?:__global__\s+|__device__\s+|__host__\s+)*\s*NppStatus\s+(nppi\w+|npps\w+)\s*\([^{}]*\)\s*\{'
            
            matches = re.finditer(pattern_impl, content, re.MULTILINE)
            for match in matches:
                function_name = match.group(1)
                implemented.add(function_name)
    
    return implemented

def categorize_functions(functions):
    """将函数按类别分类"""
    categories = {
        'arithmetic': [],
        'geometry_transforms': [],
        'filtering': [],
        'statistics': [],
        'color_conversion': [],
        'logical': [],
        'comparison': [],
        'morphological': [],
        'threshold': [],
        'data_exchange': [],
        'linear_transforms': [],
        'other': []
    }
    
    for func in functions:
        name = func['name'].lower()
        file_name = func['file'].lower()
        
        if 'arithmetic' in file_name or any(op in name for op in ['add', 'sub', 'mul', 'div', 'abs', 'sqr', 'sqrt', 'exp', 'ln']):
            categories['arithmetic'].append(func)
        elif 'geometry' in file_name or any(op in name for op in ['resize', 'rotate', 'warp', 'affine', 'perspective', 'remap', 'mirror']):
            categories['geometry_transforms'].append(func)
        elif 'filtering' in file_name or any(op in name for op in ['filter', 'gaussian', 'box', 'sobel', 'laplace', 'canny', 'gradient']):
            categories['filtering'].append(func)
        elif 'statistics' in file_name or any(op in name for op in ['mean', 'std', 'min', 'max', 'sum', 'histogram', 'integral']):
            categories['statistics'].append(func)
        elif 'color' in file_name or any(op in name for op in ['rgb', 'yuv', 'hsv', 'gray', 'bgr', 'color']):
            categories['color_conversion'].append(func)
        elif any(op in name for op in ['and', 'or', 'xor', 'not', 'shift']):
            categories['logical'].append(func)
        elif any(op in name for op in ['compare', 'threshold', 'greater', 'less', 'equal']):
            if 'threshold' in name:
                categories['threshold'].append(func)
            else:
                categories['comparison'].append(func)
        elif 'morphological' in file_name or any(op in name for op in ['dilate', 'erode', 'open', 'close', 'morphology']):
            categories['morphological'].append(func)
        elif 'exchange' in file_name or any(op in name for op in ['copy', 'set', 'convert', 'transpose', 'lut']):
            categories['data_exchange'].append(func)
        elif 'linear' in file_name or any(op in name for op in ['magnitude', 'phase', 'polartocomplex', 'complextopolar']):
            categories['linear_transforms'].append(func)
        else:
            categories['other'].append(func)
    
    return categories

def analyze_data_types_and_channels(functions):
    """分析数据类型和通道数的覆盖情况"""
    type_channel_map = defaultdict(set)
    
    for func in functions:
        name = func['name']
        
        # 提取数据类型信息
        type_match = re.search(r'_(\d+[usfcicp]+)_', name)
        if type_match:
            data_type = type_match.group(1)
            
            # 提取通道信息
            channel_match = re.search(r'_C(\d+)([RPCSA]*)_', name)
            if channel_match:
                channels = int(channel_match.group(1))
                mode = channel_match.group(2) if channel_match.group(2) else ''
                type_channel_map[data_type].add(f"C{channels}{mode}")
            else:
                # 单通道或特殊格式
                type_channel_map[data_type].add("C1")
    
    return dict(type_channel_map)

def estimate_implementation_difficulty(func):
    """估算实现难度"""
    name = func['name'].lower()
    
    # 基础操作 - 容易
    if any(op in name for op in ['add', 'sub', 'mul', 'div', 'abs', 'copy', 'set']):
        return 'easy'
    
    # 中等复杂度操作
    elif any(op in name for op in ['filter', 'resize', 'rotate', 'threshold', 'convert']):
        return 'medium'
    
    # 高复杂度操作
    elif any(op in name for op in ['canny', 'watershed', 'morphology', 'warp', 'perspective', 'harris']):
        return 'hard'
    
    # 超高复杂度操作
    elif any(op in name for op in ['fft', 'dct', 'hough', 'optical']):
        return 'very_hard'
    
    return 'medium'

def main():
    api_dir = '/home/cjxu/npp/API-12.8'
    src_dir = '/home/cjxu/npp/src'
    
    print("正在分析API头文件...")
    
    # 提取所有函数声明
    all_functions = []
    header_files = glob.glob(os.path.join(api_dir, 'nppi_*.h')) + glob.glob(os.path.join(api_dir, 'npps_*.h'))
    
    for header_file in header_files:
        functions = extract_function_declarations(header_file)
        all_functions.extend(functions)
        print(f"  {os.path.basename(header_file)}: {len(functions)} 个函数")
    
    print(f"\n总计声明的函数: {len(all_functions)}")
    
    # 提取已实现的函数
    print("\n正在分析已实现的函数...")
    implemented_functions = extract_implemented_functions(src_dir)
    print(f"已实现的函数: {len(implemented_functions)}")
    
    # 找出缺失的函数
    missing_functions = []
    for func in all_functions:
        if func['name'] not in implemented_functions:
            missing_functions.append(func)
    
    print(f"缺失的函数: {len(missing_functions)}")
    
    # 分类分析
    print("\n=== 按类别分析缺失操作 ===")
    categorized_missing = categorize_functions(missing_functions)
    
    report = {
        'summary': {
            'total_declared': len(all_functions),
            'total_implemented': len(implemented_functions),
            'total_missing': len(missing_functions),
            'implementation_rate': f"{len(implemented_functions)/len(all_functions)*100:.1f}%"
        },
        'categories': {}
    }
    
    for category, funcs in categorized_missing.items():
        if not funcs:
            continue
            
        print(f"\n{category.upper()}: {len(funcs)} 个缺失")
        
        # 分析数据类型覆盖
        type_coverage = analyze_data_types_and_channels(funcs)
        
        # 估算实现难度
        difficulty_count = defaultdict(int)
        for func in funcs:
            difficulty = estimate_implementation_difficulty(func)
            difficulty_count[difficulty] += 1
        
        # 列出前10个缺失的函数
        print("  主要缺失函数:")
        for i, func in enumerate(funcs[:10]):
            difficulty = estimate_implementation_difficulty(func)
            print(f"    {i+1}. {func['name']} [{difficulty}]")
        
        if len(funcs) > 10:
            print(f"    ... 还有 {len(funcs)-10} 个函数")
        
        print(f"  数据类型覆盖: {len(type_coverage)} 种类型")
        print(f"  实现难度分布: {dict(difficulty_count)}")
        
        # Convert sets to lists for JSON serialization
        type_coverage_json = {k: list(v) for k, v in type_coverage.items()}
        
        report['categories'][category] = {
            'missing_count': len(funcs),
            'functions': [f['name'] for f in funcs],
            'type_coverage': type_coverage_json,
            'difficulty_distribution': dict(difficulty_count)
        }
    
    # 生成优先级建议
    print("\n=== 实现优先级建议 ===")
    priority_list = []
    
    for category, data in report['categories'].items():
        if data['missing_count'] == 0:
            continue
            
        # 计算优先级分数 (缺失数量 + 易实现程度)
        easy_count = data['difficulty_distribution'].get('easy', 0)
        medium_count = data['difficulty_distribution'].get('medium', 0)
        total_missing = data['missing_count']
        
        priority_score = easy_count * 3 + medium_count * 2 + total_missing * 0.1
        
        priority_list.append({
            'category': category,
            'score': priority_score,
            'missing_count': total_missing,
            'easy_count': easy_count
        })
    
    priority_list.sort(key=lambda x: x['score'], reverse=True)
    
    print("建议实现顺序:")
    for i, item in enumerate(priority_list, 1):
        print(f"  {i}. {item['category']}: {item['missing_count']}个缺失 "
              f"(其中{item['easy_count']}个容易实现) [分数: {item['score']:.1f}]")
    
    # 保存详细报告
    with open('/home/cjxu/npp/missing_apis_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细报告已保存到: /home/cjxu/npp/missing_apis_report.json")

if __name__ == '__main__':
    main()