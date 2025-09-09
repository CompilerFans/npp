#!/usr/bin/env python3
"""
NPP头文件分析工具
分析所有NPP头文件，提取函数声明并分类统计
"""
import os
import re
import sys
from pathlib import Path
from collections import defaultdict
import yaml

def extract_functions_from_header(file_path):
    """
    从单个头文件中提取函数声明
    """
    functions = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        print(f"警告: 无法读取文件 {file_path}")
        return functions
    
    # 移除注释
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
    
    # 正则表达式匹配函数声明
    # 匹配以NppStatus开头或以npp开头的函数
    function_patterns = [
        # 标准函数声明: NppStatus functionName(...)
        r'^\s*NppStatus\s+([a-zA-Z_]\w*)\s*\(',
        # 其他返回类型的npp函数: returnType nppFunctionName(...)
        r'^\s*(?:const\s+)?(?:unsigned\s+)?(?:int|void|float|double|size_t|cudaStream_t|char\s*\*|NppLibraryVersion\s*\*|\w+)\s+(npp[a-zA-Z_]\w*)\s*\(',
        # 处理多行函数声明
        r'^\s*(npp[a-zA-Z_]\w*)\s*\(',
    ]
    
    for pattern in function_patterns:
        matches = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                func_name = match[-1]  # 取最后一个匹配组
            else:
                func_name = match
            
            if func_name and func_name.lower().startswith('npp'):
                functions.append(func_name)
    
    # 去重并排序
    return sorted(list(set(functions)))

def categorize_function(func_name):
    """
    根据函数名对函数进行分类
    """
    name = func_name.lower()
    
    # NPPI (图像处理) 分类
    if name.startswith('nppi'):
        if any(op in name for op in ['add', 'sub', 'mul', 'div', 'abs', 'and', 'or', 'xor', 'not']):
            return 'nppi_arithmetic_and_logical_operations'
        elif any(conv in name for conv in ['rgb', 'yuv', 'bgr', 'hsv', 'xyz', 'lab', 'nv12', 'nv21']):
            return 'nppi_color_conversion'
        elif any(filter_op in name for filter_op in ['filter', 'box', 'gaussian', 'median', 'sobel', 'laplace', 'sharpen']):
            return 'nppi_filtering_functions'
        elif any(geom in name for geom in ['resize', 'rotate', 'mirror', 'flip', 'warp', 'remap', 'affine', 'perspective']):
            return 'nppi_geometry_transforms'
        elif any(morph in name for morph in ['erode', 'dilate', 'open', 'close', 'tophat', 'blackhat', 'gradient']):
            return 'nppi_morphological_operations'
        elif any(thresh in name for thresh in ['threshold', 'compare']):
            return 'nppi_threshold_and_compare_operations'
        elif any(stat in name for stat in ['sum', 'mean', 'min', 'max', 'histogram', 'moment', 'norm']):
            return 'nppi_statistics_functions'
        elif any(data in name for data in ['copy', 'set', 'convert', 'transpose', 'swap']):
            return 'nppi_data_exchange_and_initialization'
        elif any(supp in name for supp in ['malloc', 'free', 'getbuffersize']):
            return 'nppi_support_functions'
        elif any(linear in name for linear in ['magnitude', 'phase', 'powx', 'exp', 'ln', 'sqrt']):
            return 'nppi_linear_transforms'
        else:
            return 'nppi_other'
    
    # NPPS (信号处理) 分类
    elif name.startswith('npps'):
        if any(arith in name for arith in ['add', 'sub', 'mul', 'div', 'abs']):
            return 'npps_arithmetic_and_logical_operations'
        elif any(stat in name for stat in ['sum', 'mean', 'min', 'max', 'stddev', 'norm', 'dotprod']):
            return 'npps_statistics_functions'
        elif any(init in name for init in ['set', 'zero', 'ramp', 'copy']):
            return 'npps_initialization'
        elif any(conv in name for conv in ['convert']):
            return 'npps_conversion_functions'
        elif any(filter_op in name for filter_op in ['fir', 'iir', 'filter', 'median']):
            return 'npps_filtering_functions'
        elif any(transform in name for transform in ['fft', 'dct', 'dft', 'hilbert']):
            return 'npps_transform_functions'
        elif any(logical in name for logical in ['and', 'or', 'xor', 'not', 'shift']):
            return 'npps_logical_operations'
        elif any(win in name for win in ['win', 'bartlett', 'hann', 'hamming', 'blackman', 'kaiser']):
            return 'npps_windowing_functions'
        elif any(thresh in name for thresh in ['threshold']):
            return 'npps_threshold_operations'
        elif any(supp in name for supp in ['malloc', 'free', 'getbuffersize']):
            return 'npps_support_functions'
        else:
            return 'npps_other'
    
    # NPP核心函数
    elif name.startswith('npp') and not name.startswith(('nppi', 'npps')):
        return 'npp_core'
    
    return 'unknown'

def analyze_npp_headers():
    """
    分析所有NPP头文件
    """
    headers_dir = Path('/home/cjxu/npp/build/install/include/opennpp')
    
    if not headers_dir.exists():
        print(f"错误: 头文件目录不存在: {headers_dir}")
        return
    
    # 获取所有.h文件
    header_files = list(headers_dir.glob('*.h'))
    if not header_files:
        print(f"错误: 在 {headers_dir} 中未找到头文件")
        return
    
    print(f"发现 {len(header_files)} 个头文件")
    
    all_functions = {}
    categorized_functions = defaultdict(list)
    
    for header_file in header_files:
        print(f"\n分析文件: {header_file.name}")
        functions = extract_functions_from_header(header_file)
        
        if functions:
            all_functions[header_file.name] = functions
            print(f"  找到 {len(functions)} 个函数")
            
            # 对函数进行分类
            for func in functions:
                category = categorize_function(func)
                categorized_functions[category].append(func)
        else:
            print(f"  未找到函数")
    
    # 统计结果
    print("\n" + "="*80)
    print("分析结果汇总")
    print("="*80)
    
    total_functions = sum(len(funcs) for funcs in all_functions.values())
    print(f"\n总共发现 {total_functions} 个函数声明")
    
    print(f"\n各头文件函数数量:")
    for header, functions in all_functions.items():
        print(f"  {header:<40} : {len(functions):>3} 个函数")
    
    print(f"\n按模块分类统计:")
    for category, functions in sorted(categorized_functions.items()):
        unique_functions = list(set(functions))
        print(f"  {category:<45} : {len(unique_functions):>3} 个函数")
    
    # 详细列表
    print("\n" + "="*80)
    print("详细函数列表")
    print("="*80)
    
    for header, functions in all_functions.items():
        print(f"\n{header}:")
        for func in functions:
            print(f"  - {func}")
    
    print("\n" + "="*80)
    print("按模块分类的函数列表")
    print("="*80)
    
    for category, functions in sorted(categorized_functions.items()):
        unique_functions = sorted(list(set(functions)))
        print(f"\n{category} ({len(unique_functions)} 个函数):")
        for func in unique_functions:
            print(f"  - {func}")
    
    # 与现有配置对比
    print("\n" + "="*80)
    print("与现有api_features.yaml对比")
    print("="*80)
    
    try:
        with open('/home/cjxu/npp/status/api_features.yaml', 'r', encoding='utf-8') as f:
            existing_config = yaml.safe_load(f)
        
        existing_functions = set()
        for module, data in existing_config.items():
            if 'functions' in data:
                for func_info in data['functions']:
                    existing_functions.add(func_info['name'])
        
        discovered_functions = set()
        for functions in all_functions.values():
            discovered_functions.update(functions)
        
        print(f"现有配置中的函数数量: {len(existing_functions)}")
        print(f"头文件中发现的函数数量: {len(discovered_functions)}")
        
        missing_in_config = discovered_functions - existing_functions
        extra_in_config = existing_functions - discovered_functions
        
        print(f"\n配置中缺失的函数 ({len(missing_in_config)} 个):")
        for func in sorted(missing_in_config):
            print(f"  - {func}")
        
        print(f"\n配置中多余的函数 ({len(extra_in_config)} 个):")
        for func in sorted(extra_in_config):
            print(f"  - {func}")
        
    except Exception as e:
        print(f"无法读取现有配置文件: {e}")

if __name__ == "__main__":
    analyze_npp_headers()