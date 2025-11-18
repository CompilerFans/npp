#!/usr/bin/env python3
"""
快速生成 NPP Benchmark 文件的脚本

使用方法:
    python3 generate_benchmark.py nppiSub 8u C1 RSfs --module arithmetic
    python3 generate_benchmark.py nppiMul 32f C1 R --module arithmetic
    python3 generate_benchmark.py nppiResize 8u C1 R --module geometry
"""

import argparse
import os
from pathlib import Path

TEMPLATE = """#include "benchmark_base.h"
#include <nppi_{include_header}.h>

using namespace npp_benchmark;

// ============================================================
// Benchmark: {full_function_name}
// ============================================================

static void BM_{full_function_name}_Fixed(benchmark::State& state) {{
    const int width = 1920;
    const int height = 1080;
    {extra_params_decl}
    
    ImageBenchmarkBase<Npp{data_type}> base;
    base.SetupImageMemory(width, height);
    
    NppiSize roi = {{width, height}};
    
    for (auto _ : state) {{
        NppStatus status = {full_function_name}(
            base.d_src1_, base.step_,
            base.d_src2_, base.step_,
            base.d_dst_, base.step_,
            roi{extra_params_call}
        );
        
        base.SyncAndCheckError();
        
        if (status != NPP_NO_ERROR) {{
            state.SkipWithError("{full_function_name} failed");
            break;
        }}
    }}
    
    size_t bytesProcessed = base.ComputeImageBytes(2, 1);
    REPORT_THROUGHPUT(state, bytesProcessed);
    REPORT_CUSTOM_METRIC(state, "Megapixels", (width * height) / 1e6);
    
    base.TeardownImageMemory();
}}

BENCHMARK(BM_{full_function_name}_Fixed)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

static void BM_{full_function_name}_Sizes(benchmark::State& state) {{
    int width = state.range(0);
    int height = state.range(1);
    {extra_params_decl}
    
    ImageBenchmarkBase<Npp{data_type}> base;
    base.SetupImageMemory(width, height);
    
    NppiSize roi = {{width, height}};
    
    for (auto _ : state) {{
        NppStatus status = {full_function_name}(
            base.d_src1_, base.step_,
            base.d_src2_, base.step_,
            base.d_dst_, base.step_,
            roi{extra_params_call}
        );
        
        base.SyncAndCheckError();
        
        if (status != NPP_NO_ERROR) {{
            state.SkipWithError("{full_function_name} failed");
            break;
        }}
    }}
    
    size_t bytesProcessed = base.ComputeImageBytes(2, 1);
    REPORT_THROUGHPUT(state, bytesProcessed);
    REPORT_CUSTOM_METRIC(state, "Megapixels", (width * height) / 1e6);
    
    base.TeardownImageMemory();
}}

BENCHMARK(BM_{full_function_name}_Sizes)
    ->Args({{256, 256}})
    ->Args({{512, 512}})
    ->Args({{1024, 1024}})
    ->Args({{1920, 1080}})
    ->Args({{3840, 2160}})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_MAIN();
"""

MODULE_HEADERS = {
    'arithmetic': 'arithmetic_and_logical_operations',
    'filtering': 'filtering_functions',
    'geometry': 'geometry_transforms',
    'color': 'color_conversion',
    'statistics': 'statistics_functions',
    'threshold': 'threshold_and_compare_operations',
    'morphological': 'morphological_operations',
}

def generate_benchmark(func_name, data_type, channels, op_type, extra_suffix, module):
    """生成 benchmark 文件"""
    
    # 构建完整函数名
    full_name = f"{func_name}_{data_type}_{channels}{op_type}{extra_suffix}"
    
    # 确定头文件
    include_header = MODULE_HEADERS.get(module, 'arithmetic_and_logical_operations')
    
    # 额外参数处理
    extra_params_decl = ""
    extra_params_call = ""
    
    if "Sfs" in extra_suffix:
        extra_params_decl = "const int scaleFactor = 0;"
        extra_params_call = ", scaleFactor"
    
    # 填充模板
    content = TEMPLATE.format(
        full_function_name=full_name,
        data_type=data_type,
        include_header=include_header,
        extra_params_decl=extra_params_decl,
        extra_params_call=extra_params_call
    )
    
    # 确定输出路径
    module_dir = Path(__file__).parent / "nppi" / module
    module_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = module_dir / f"benchmark_{func_name.lower()}.cpp"
    
    # 写入文件
    with open(output_file, 'w') as f:
        f.write(content)
    
    print(f"Generated: {output_file}")
    print(f"Function: {full_name}")
    
    return output_file

def update_cmake(module, benchmark_file):
    """更新 CMakeLists.txt 添加新的 benchmark"""
    cmake_file = Path(__file__).parent / "CMakeLists.txt"
    
    # 读取现有内容
    with open(cmake_file, 'r') as f:
        content = f.read()
    
    # 构建新的 source 变量名
    module_upper = module.upper()
    var_name = f"NPPI_{module_upper}_BENCHMARK_SOURCES"
    
    # 检查是否已存在
    if var_name in content:
        print(f"\nNote: {var_name} already exists in CMakeLists.txt")
        print(f"Please manually add: nppi/{module}/{benchmark_file.name}")
    else:
        print(f"\nAdd these lines to CMakeLists.txt:")
        print(f"""
# {module.capitalize()} benchmarks
set({var_name}
    nppi/{module}/{benchmark_file.name}
)

npp_create_benchmark_target(
    nppi_{module}_benchmark
    "${{{var_name}}}"
    npp
)

target_include_directories(nppi_{module}_benchmark
    PRIVATE
    ${{BENCHMARK_INCLUDE_DIRS}}
)
""")

def main():
    parser = argparse.ArgumentParser(
        description='Generate NPP benchmark files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s nppiSub 8u C1 RSfs --module arithmetic
  %(prog)s nppiMul 32f C1 R --module arithmetic
  %(prog)s nppiResize 8u C3 R --module geometry
  %(prog)s nppiFilter 16u C1 R --module filtering

Modules: arithmetic, filtering, geometry, color, statistics, threshold, morphological
        """
    )
    
    parser.add_argument('function', help='NPP function name (e.g., nppiSub)')
    parser.add_argument('data_type', help='Data type (e.g., 8u, 16u, 32f)')
    parser.add_argument('channels', help='Channels (e.g., C1, C3, C4)')
    parser.add_argument('op_type', help='Operation type (e.g., R, I, IR)')
    parser.add_argument('--extra', default='', help='Extra suffix (e.g., Sfs)')
    parser.add_argument('--module', default='arithmetic', 
                       choices=list(MODULE_HEADERS.keys()),
                       help='NPP module category')
    
    args = parser.parse_args()
    
    output_file = generate_benchmark(
        args.function,
        args.data_type,
        args.channels,
        args.op_type,
        args.extra,
        args.module
    )
    
    update_cmake(args.module, output_file)

if __name__ == '__main__':
    main()
