# 测试文件重组计划

## 当前状态分析

### arithmetic目录
- **保留原样**：`test_nppi_add.cpp` 和 `test_nppi_arithmetic.cpp` 不重复
  - `test_nppi_add.cpp`: 测试双源操作 (Add, Sub, Mul, Div)
  - `test_nppi_arithmetic.cpp`: 测试常数操作 (AddC, SubC, MulC, DivC)

### support目录问题
1. 功能重复：多个文件都在测试内存分配
2. 命名不清晰：文件名过长且含义模糊
3. 测试分散：相似的测试分布在不同文件中

## 建议的新结构

### arithmetic目录（不变）
```
test/unit/nppi/arithmetic/
├── test_nppi_arithmetic_constant.cpp  # 重命名自test_nppi_arithmetic.cpp
├── test_nppi_arithmetic_dual.cpp      # 重命名自test_nppi_add.cpp
└── test_nppi_arithmetic_validation.cpp # 新增：专门的验证测试
```

### support目录（重构）
```
test/unit/nppi/support/
├── test_memory_allocation.cpp     # 合并所有内存分配测试
├── test_memory_alignment.cpp      # 专门测试对齐和step值
└── test_memory_performance.cpp    # 性能基准测试
```

### validation目录（独立）
```
test/validation/
├── test_nvidia_comparison.cpp     # 合并所有NVIDIA对比测试
└── dlopen_helper.h                # dlopen辅助函数
```

## 重构步骤

### 第1步：合并support目录的内存测试
将以下文件合并为 `test_memory_allocation.cpp`:
- test_nppi_support.cpp 的基础测试
- test_nppi_support_complete_coverage.cpp 的全类型覆盖
- 删除重复的测试用例

### 第2步：创建专门的对齐测试
将step验证相关测试移到 `test_memory_alignment.cpp`:
- test_nppi_support_step_validation.cpp 的所有内容
- 添加更多边界情况测试

### 第3步：统一NVIDIA对比测试
将以下内容移到 validation目录:
- test_nppi_support_nvidia_comparison.cpp
- test_nvidia_comparison_dlopen.cpp
- 创建共享的dlopen加载器

### 第4步：重命名arithmetic测试
- test_nppi_arithmetic.cpp → test_nppi_arithmetic_constant.cpp
- test_nppi_add.cpp → test_nppi_arithmetic_dual.cpp

## 优势
1. **更清晰的组织**：每个文件有明确的职责
2. **减少重复**：合并相似的测试
3. **更好的可维护性**：相关测试在一起
4. **更快的编译**：减少文件数量