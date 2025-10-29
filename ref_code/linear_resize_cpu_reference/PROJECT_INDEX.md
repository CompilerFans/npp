# Linear Resize CPU Reference - 项目文件索引

## 核心实现文件

### 生产代码
- `linear_interpolation_cpu.h` - **V1 Improved** (推荐生产使用)
  - 修复了0.25x下采样和非均匀缩放
  - 100%匹配NPP在最常用场景

- `linear_interpolation_v1_ultimate.h` - **V1 Ultimate** (实验版本)
  - 集成分数下采样算法
  - 支持更多复杂场景

- `linear_interpolation_v2.h` - **V2 mpp风格**
  - 基于kunzmi/mpp的可分离实现
  - 参考对比用

### 备份文件
- `linear_interpolation_cpu_original_backup.h` - V1原始版本备份
- `linear_interpolation_v1_improved.h` - V1 Improved独立副本

## 文档（按重要性排序）

### 必读文档

1. **`FINAL_RESULTS_SUMMARY.md`** ⭐⭐⭐
   - 项目最终总结
   - 成果概览和技术发现
   - 实用建议

2. **`README.md`**
   - 项目简介和快速开始
   - 编译和使用说明

3. **`COMPREHENSIVE_ANALYSIS.md`** ⭐⭐
   - 38个测试场景的完整结果
   - 分类统计和问题分析

### 深入分析文档

4. **`NPP_ALGORITHM_DISCOVERED.md`** ⭐⭐⭐
   - 反向工程的核心成果
   - NPP算法框架和参数公式

5. **`V1_OPTIMIZATION_SUMMARY.md`** ⭐⭐
   - V1的优化历程
   - 问题诊断和解决方案

6. **`V1_NPP_PRECISION_ANALYSIS.md`** ⭐
   - 14个测试场景的详细分析
   - 像素级对比数据

7. **`NPP_FRACTIONAL_ANALYSIS_REFINED.md`**
   - 分数下采样的精化分析
   - 未解决问题的讨论

8. **`FRACTIONAL_DOWNSCALE_PATTERN.md`**
   - 分数下采样模式的初步发现

### 对比和参考文档

9. **`V1_VS_V2_ANALYSIS.md`**
   - V1和V2的详细对比
   - 算法差异分析

10. **`MPP_VS_NPP_COMPARISON.md`**
    - mpp和NPP的架构对比
    - kunzmi/mpp分析

11. **`PRECISION_SUMMARY.md`**
    - 精度对比快速参考表

12. **`WORK_SUMMARY.md`**
    - 早期工作总结

### 专题分析

13. **`linear_interpolation_analysis.md`**
    - 早期线性插值分析

14. **`NON_INTEGER_SCALE_ANALYSIS.md`**
    - 非整数缩放的专题研究

## 测试和工具程序

### 综合测试

- `comprehensive_test.cpp` + `compile_comprehensive.sh`
  - 38个场景的全面测试套件
  - **主要验证工具**

- `test_v1_ultimate.cpp` + 编译脚本
  - V1 Ultimate版本的验证
  - 分数下采样重点测试

- `test_v1_improvements.cpp` + `compile_test_improvements.sh`
  - V1 Improved的改进验证

### 对比测试

- `compare_v1_v2_npp.cpp` + `compile_compare_v1_v2.sh`
  - V1 vs V2 vs NPP 三方对比
  - 8个代表性场景

- `validate_linear_cpu.cpp` + `compile_validate_linear.sh`
  - 基础验证测试

### 问题诊断工具

- `debug_problem_cases.cpp` + `compile_debug_problems.sh`
  - **关键调试工具**
  - 深入分析0.25x和非均匀缩放问题

- `debug_1d_fractional.cpp`
  - 1D分数下采样调试
  - 参数验证

### 反向工程工具

- `reverse_engineer_params.cpp` ⭐⭐⭐
  - **核心分析工具**
  - 计算NPP的threshold和attenuation参数

- `analyze_fractional_downscale.cpp`
  - 分数下采样深度分析
  - 逐像素模式探测

- `deep_pixel_analysis.cpp` + `compile_deep_analysis.sh`
  - 像素级深度分析
  - 多种插值方法测试

### 早期探索工具

- `test_linear_analysis.cpp` + `compile_linear_analysis.sh`
  - 早期线性插值分析

- `test_linear_debug_simple_patterns.cpp` + `compile_debug.sh`
  - 简单模式调试

- `analyze_2x2_to_3x3.cpp`
  - 2x2到3x3的专项分析

- `find_exact_weights.cpp`
  - 权重反向计算

- `test_coord_offset.cpp`
  - 坐标偏移测试

- `test_interpolation_bias.cpp`
  - 插值偏差测试

- `test_nearest_for_upscale.cpp`
  - 上采样最近邻测试

## 文件使用建议

### 快速开始
1. 阅读 `README.md` 和 `FINAL_RESULTS_SUMMARY.md`
2. 使用 `linear_interpolation_cpu.h` (V1 Improved)
3. 运行 `comprehensive_test` 验证

### 深入研究
1. 研读 `NPP_ALGORITHM_DISCOVERED.md`
2. 运行 `reverse_engineer_params` 和 `debug_problem_cases`
3. 阅读 `V1_OPTIMIZATION_SUMMARY.md` 了解优化过程

### 继续开发
1. 基于 `linear_interpolation_v1_ultimate.h` 进行改进
2. 使用 `debug_1d_fractional` 调试特定问题
3. 参考 `NPP_FRACTIONAL_ANALYSIS_REFINED.md` 的未解决问题

## 编译脚本汇总

所有编译脚本格式：
```bash
g++ -o <output> <source>.cpp \
    -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 \
    -lnppig -lnppicc -lnppidei -lcudart \
    -std=c++11 -O2
```

主要脚本：
- `compile_comprehensive.sh` - 综合测试
- `compile_compare_v1_v2.sh` - 三方对比
- `compile_debug_problems.sh` - 问题诊断
- 其他编译脚本命名规则: `compile_<程序名>.sh`

## 推荐阅读路径

### 路径1: 实用导向
```
README.md
→ FINAL_RESULTS_SUMMARY.md
→ linear_interpolation_cpu.h
→ comprehensive_test.cpp
```

### 路径2: 研究导向
```
README.md
→ NPP_ALGORITHM_DISCOVERED.md
→ reverse_engineer_params.cpp
→ NPP_FRACTIONAL_ANALYSIS_REFINED.md
→ linear_interpolation_v1_ultimate.h
```

### 路径3: 完整理解
```
README.md
→ WORK_SUMMARY.md (历史)
→ V1_VS_V2_ANALYSIS.md (对比)
→ V1_OPTIMIZATION_SUMMARY.md (优化)
→ NPP_ALGORITHM_DISCOVERED.md (发现)
→ COMPREHENSIVE_ANALYSIS.md (验证)
→ FINAL_RESULTS_SUMMARY.md (总结)
```

## 统计信息

- **总文件数**: 44个
- **实现文件**: 5个 (.h)
- **测试程序**: 18个 (.cpp)
- **编译脚本**: 11个 (.sh)
- **文档**: 15个 (.md)
- **代码行数**: ~2000行实现 + ~4000行测试
- **文档字数**: ~25000字

## 版本历史

1. **V1 Original** - 基础hybrid实现
2. **V1 Improved** - 修复关键问题，44%完美匹配
3. **V1 Ultimate** - 集成分数算法（实验中）
4. **V2 mpp** - 参考实现

当前稳定版本: **V1 Improved**
