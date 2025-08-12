# OpenNPP 测试框架指南

## 概述

这个测试框架提供了一个全面的解决方案，用于对比我们的开源NPP实现与NVIDIA闭源库的结果、性能和功能一致性。

## 架构设计

### 核心组件

1. **测试框架 (`test/framework/`)**
   - `npp_test_framework.h`: 可扩展的测试基类
   - `test_nppi_addc_8u_c1rsfs.h`: 具体函数测试类
   - `test_report.h`: HTML/JSON报告生成器

2. **测试运行器**
   - `test_runner.cpp`: 主测试执行器
   - 支持命令行参数和多种输出格式

3. **实现版本**
   - CPU版本: `nppiAddC_8u_C1RSfs_Ctx_reference`
   - CUDA版本: `nppiAddC_8u_C1RSfs_Ctx_cuda`
   - NVIDIA版本: 动态链接库调用

## 使用方法

### 1. 构建项目

```bash
# 创建构建目录
mkdir build && cd build

# 配置CMake
cmake ..

# 编译所有内容
make -j$(nproc)
```

### 2. 运行测试

#### 运行所有测试
```bash
./test_runner
```

#### 运行特定函数测试
```bash
./test_runner -f nppiAddC_8u_C1RSfs_Ctx
```

#### 生成自定义报告
```bash
./test_runner -r my_test_report -o my_results -v
```

#### 命令行选项
```bash
Options:
  -f, --function FUNCTION   测试特定函数 (默认: all)
  -r, --report NAME         报告名称 (默认: npp_test_report)
  -o, --output DIR          输出目录 (默认: test_reports)
  -v, --verbose             详细输出
  -h, --help                显示帮助
```

### 3. 查看结果

测试完成后，报告文件将生成在指定目录中：

- **HTML报告**: `test_reports/npp_test_report.html`
- **JSON报告**: `test_reports/npp_test_report.json`

## 测试内容

### 功能测试
- 不同图像尺寸 (1x1 到 2048x2048)
- 不同常量值 (0-255)
- 不同缩放因子 (0-16)
- 不同图像模式 (渐变、棋盘格、纯色)
- 边界情况测试

### 性能测试
- CPU vs GPU 性能对比
- GPU vs NVIDIA 性能对比
- 不同数据规模的性能分析

### 精度验证
- 与NVIDIA NPP的像素级精度对比
- 最大误差和平均误差计算
- 容差范围验证

## 扩展新函数

要为新的NPP函数添加测试，只需：

1. **创建测试参数类**
```cpp
class NewFunctionTestParameters : public NPPTest::TestParameters {
    // 定义参数和验证
};
```

2. **创建测试类**
```cpp
class TestNewFunction : public NPPTest::NPPTestBase {
    // 实现run()和getTestCases()
};
```

3. **自动注册**
```cpp
static bool register_new_test = []() {
    NPPTest::TestRegistry::registerTest(std::make_unique<TestNewFunction>());
    return true;
}();
```

## 示例输出

```
=== OpenNPP Test Runner ===
CUDA Version: 12020
Device: NVIDIA RTX 3080 (Compute 8.6)

--- nppiAddC_8u_C1RSfs_Ctx ---
NVIDIA NPP Available: Yes
  ✓ PASS: 32x32, const=50, scale=1, pattern=0
  ✓ PASS: 256x256, const=100, scale=2, pattern=1
  ...

=== Final Results ===
Total tests: 147
Passed: 147
Failed: 0
Success rate: 100.00%

🎉 All tests passed!

=== Test Report Generated ===
HTML Report: test_reports/npp_test_report.html
JSON Report: test_reports/npp_test_report.json
```

## 性能基准

测试框架自动计算以下性能指标：
- CPU执行时间
- GPU执行时间
- NVIDIA NPP执行时间
- GPU相对于CPU的加速比
- 我们实现相对于NVIDIA的加速比

## 故障排除

1. **NVIDIA NPP不可用**
   - 系统会跳过NVIDIA对比，只测试CPU vs GPU
   - 确保安装了CUDA Toolkit

2. **CUDA设备不可用**
   - 检查系统是否有NVIDIA GPU
   - 检查CUDA驱动是否正确安装

3. **编译错误**
   - 确保安装了jsoncpp开发包
   - 检查CUDA版本兼容性

## 文件结构

```
test/
├── framework/           # 测试框架核心
│   ├── npp_test_framework.h
│   ├── test_nppi_addc_8u_c1rsfs.h
│   └── test_report.h
├── test_runner.cpp      # 主测试程序
├── test_nppi_addc_8u_c1rsfs.cpp  # 基础测试
└── test_nppi_addc_8u_c1rsfs_validation.cpp  # 验证测试
```
