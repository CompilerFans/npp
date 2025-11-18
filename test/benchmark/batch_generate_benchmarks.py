#!/usr/bin/env python3
"""
批量生成 NPP Benchmark 文件

基于已有的 unit test 文件，自动生成对应的 benchmark 文件

使用方法:
    # 生成所有算术运算的 benchmarks
    python3 batch_generate_benchmarks.py --module arithmetic
    
    # 生成所有模块的 benchmarks
    python3 batch_generate_benchmarks.py --all
    
    # 只显示将要生成的文件，不实际生成
    python3 batch_generate_benchmarks.py --module arithmetic --dry-run
"""

import argparse
import re
from pathlib import Path
from typing import List, Dict

# Unit test 到 benchmark 的模块映射
MODULE_MAPPING = {
    'nppi_arithmetic_operations': 'arithmetic',
    'nppi_filtering_functions': 'filtering',
    'nppi_geometry_transforms': 'geometry',
    'nppi_color_conversion_operations': 'color',
    'nppi_statistics_functions': 'statistics',
    'nppi_threshold_and_compare_operations': 'threshold',
    'nppi_morphological_operations': 'morphological',
}

# NPP 函数名模式
FUNCTION_PATTERN = re.compile(r'nppi(\w+)_(\d+\w+)_([AC]\d+)(R|I|IR)(Sfs)?')

def parse_function_name(func_name: str) -> Dict[str, str]:
    """解析 NPP 函数名"""
    match = FUNCTION_PATTERN.match(func_name)
    if not match:
        return None
    
    return {
        'base_name': 'nppi' + match.group(1),
        'data_type': match.group(2),
        'channels': match.group(3),
        'op_type': match.group(4),
        'extra': match.group(5) or '',
    }

def find_unit_tests(unit_test_dir: Path, module: str) -> List[str]:
    """查找 unit test 中的所有函数"""
    functions = []
    test_files = list(unit_test_dir.glob('test_*.cpp'))
    
    for test_file in test_files:
        with open(test_file, 'r') as f:
            content = f.read()
            
        # 查找所有 NPP 函数调用
        matches = FUNCTION_PATTERN.findall(content)
        for match in matches:
            func_name = f"nppi{match[0]}_{match[1]}_{match[2]}{match[3]}{match[4]}"
            if func_name not in functions:
                functions.append(func_name)
    
    return sorted(functions)

def generate_benchmark_file(func_info: Dict[str, str], module: str, output_dir: Path, dry_run: bool = False):
    """生成单个 benchmark 文件"""
    
    full_name = f"{func_info['base_name']}_{func_info['data_type']}_{func_info['channels']}{func_info['op_type']}{func_info['extra']}"
    output_file = output_dir / f"benchmark_{func_info['base_name'].lower()}.cpp"
    
    if dry_run:
        print(f"  Would generate: {output_file.name} ({full_name})")
        return False
    
    # 检查是否已存在
    if output_file.exists():
        print(f"  Skip (exists): {output_file.name}")
        return False
    
    # 生成文件（这里简化，实际应该使用完整的模板）
    print(f"  Generate: {output_file.name} ({full_name})")
    
    # TODO: 调用 generate_benchmark.py 或直接生成
    # 这里只是示意，实际可以导入 generate_benchmark 模块
    
    return True

def scan_and_generate(unit_test_base: Path, benchmark_base: Path, module: str = None, dry_run: bool = False):
    """扫描 unit tests 并生成 benchmarks"""
    
    if module:
        modules_to_process = {module: MODULE_MAPPING.get(f'nppi_{module}_operations', module)}
    else:
        modules_to_process = MODULE_MAPPING
    
    total_found = 0
    total_generated = 0
    
    for unit_module, bench_module in modules_to_process.items():
        unit_dir = unit_test_base / unit_module
        if not unit_dir.exists():
            print(f"Warning: Unit test directory not found: {unit_dir}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Module: {bench_module} ({unit_module})")
        print(f"{'='*60}")
        
        # 查找所有测试的函数
        functions = find_unit_tests(unit_dir, unit_module)
        total_found += len(functions)
        
        print(f"Found {len(functions)} unique functions in unit tests")
        
        # 输出目录
        output_dir = benchmark_base / "nppi" / bench_module
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 按基础函数名分组
        function_groups = {}
        for func_name in functions:
            func_info = parse_function_name(func_name)
            if func_info:
                base = func_info['base_name']
                if base not in function_groups:
                    function_groups[base] = []
                function_groups[base].append(func_info)
        
        print(f"Grouped into {len(function_groups)} base functions:")
        for base_name, variants in function_groups.items():
            print(f"  {base_name}: {len(variants)} variants")
            if generate_benchmark_file(variants[0], bench_module, output_dir, dry_run):
                total_generated += 1
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total functions found: {total_found}")
    print(f"  Total benchmarks {'would be' if dry_run else ''} generated: {total_generated}")
    print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(
        description='Batch generate NPP benchmarks from unit tests',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--module', 
                       choices=['arithmetic', 'filtering', 'geometry', 'color', 
                               'statistics', 'threshold', 'morphological'],
                       help='Generate benchmarks for specific module')
    parser.add_argument('--all', action='store_true',
                       help='Generate benchmarks for all modules')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be generated without creating files')
    
    args = parser.parse_args()
    
    # 确定路径
    script_dir = Path(__file__).parent
    unit_test_base = script_dir.parent / "unit" / "nppi"
    benchmark_base = script_dir
    
    if not unit_test_base.exists():
        print(f"Error: Unit test directory not found: {unit_test_base}")
        return 1
    
    # 执行生成
    if args.all:
        scan_and_generate(unit_test_base, benchmark_base, dry_run=args.dry_run)
    elif args.module:
        scan_and_generate(unit_test_base, benchmark_base, args.module, args.dry_run)
    else:
        parser.print_help()
        print("\nPlease specify --module or --all")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
