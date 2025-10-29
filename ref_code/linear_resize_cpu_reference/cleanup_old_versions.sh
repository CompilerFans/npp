#!/bin/bash

echo "========================================="
echo "清理历史临时版本"
echo "========================================="
echo ""

# 删除旧的实现头文件
echo "删除旧的实现头文件..."
rm -f linear_interpolation_cpu.h
rm -f linear_interpolation_cpu_original_backup.h
rm -f linear_interpolation_v1_improved.h
rm -f linear_interpolation_v1_ultimate.h
rm -f linear_interpolation_v2.h
rm -f linear_resize_cpu_final.h

# 删除旧的测试程序和可执行文件
echo "删除旧的测试程序..."
rm -f analyze_2x2_to_3x3.cpp analyze_2x2_to_3x3
rm -f analyze_fractional_downscale.cpp analyze_fractional_downscale
rm -f analyze_integer_upscales.cpp analyze_integer_upscales
rm -f compare_v1_v2_npp.cpp compare_v1_v2_npp
rm -f comprehensive_test.cpp comprehensive_test
rm -f debug_075x.cpp debug_075x
rm -f debug_1d_fractional.cpp debug_1d_fractional
rm -f debug_problem_cases.cpp debug_problem_cases
rm -f deep_pixel_analysis.cpp deep_pixel_analysis
rm -f extract_upscale_weights.cpp extract_upscale_weights
rm -f find_exact_weights.cpp
rm -f reverse_engineer_params.cpp reverse_engineer_params
rm -f test_coord_offset.cpp
rm -f test_cpu_vs_npp.cpp test_cpu_vs_npp
rm -f test_interpolation_bias.cpp
rm -f test_linear_analysis.cpp test_linear_analysis
rm -f test_linear_debug_simple_patterns.cpp test_linear_debug_simple_patterns
rm -f test_nearest_for_upscale.cpp
rm -f test_refactored.cpp test_refactored
rm -f test_v1_improvements.cpp test_v1_improvements
rm -f test_v1_ultimate.cpp test_v1_ultimate
rm -f validate_linear_cpu.cpp validate_linear_cpu

# 删除旧的编译脚本
echo "删除旧的编译脚本..."
rm -f compile_compare_v1_v2.sh
rm -f compile_comprehensive.sh
rm -f compile_cpu_vs_npp.sh
rm -f compile_debug.sh
rm -f compile_debug_problems.sh
rm -f compile_deep_analysis.sh
rm -f compile_linear_analysis.sh
rm -f compile_test_improvements.sh
rm -f compile_validate_linear.sh

# 删除中间分析文档
echo "删除中间分析文档..."
rm -f linear_interpolation_analysis.md
rm -f NON_INTEGER_SCALE_ANALYSIS.md
rm -f MPP_VS_NPP_COMPARISON.md
rm -f V1_VS_V2_ANALYSIS.md
rm -f V1_NPP_PRECISION_ANALYSIS.md
rm -f PRECISION_SUMMARY.md
rm -f V1_OPTIMIZATION_SUMMARY.md
rm -f WORK_SUMMARY.md
rm -f COMPREHENSIVE_ANALYSIS.md
rm -f FRACTIONAL_DOWNSCALE_PATTERN.md
rm -f NPP_FRACTIONAL_ANALYSIS_REFINED.md
rm -f INTEGER_UPSCALE_WEIGHTS_ANALYSIS.md
rm -f CPU_IMPLEMENTATION_SUMMARY.md
rm -f COMPLETE_PROJECT_SUMMARY.md
rm -f FINAL_RESULTS_SUMMARY.md

echo ""
echo "✓ 清理完成"
echo ""
echo "保留的文件:"
echo "  • linear_resize_refactored.h - 最终实现"
echo "  • test_refactored_vs_npp_complete.cpp - 完整测试"
echo "  • test_refactored_vs_npp_complete - 测试可执行文件"
echo "  • REFACTORED_PRECISION_ANALYSIS.md - 精度分析"
echo "  • REFACTORING_SUMMARY.md - 重构总结"
echo "  • NPP_ALGORITHM_DISCOVERED.md - 算法发现"
echo "  • PROJECT_INDEX.md - 项目索引"
echo "  • QUICK_START.md - 快速开始"
echo "  • README.md - 项目说明"
