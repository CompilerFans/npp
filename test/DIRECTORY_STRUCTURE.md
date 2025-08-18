# Test目录结构设计

## 当前问题
- 存在重复的测试目录（gtest/nppcore 和 nppcore/）
- 目录层级不清晰
- 旧测试文件和新测试文件混在一起

## 建议的新结构

```
test/
├── unit/                      # 单元测试（使用Google Test）
│   ├── nppcore/              # NPP核心功能测试
│   │   ├── test_version.cpp
│   │   ├── test_device.cpp
│   │   ├── test_stream.cpp
│   │   └── test_error.cpp
│   │
│   ├── nppi/                 # NPPI图像处理测试
│   │   ├── support/          # 支持函数测试
│   │   │   ├── test_memory.cpp
│   │   │   └── test_allocation.cpp
│   │   └── arithmetic/       # 算术运算测试
│   │       ├── test_add.cpp
│   │       ├── test_addc.cpp
│   │       ├── test_sub.cpp
│   │       └── test_mul.cpp
│   │
│   └── npps/                 # NPPS信号处理测试（将来）
│       └── ...
│
├── integration/              # 集成测试
│   ├── test_cuda_compatibility.cpp
│   └── test_multi_module.cpp
│
├── validation/               # 与NVIDIA NPP对比验证
│   ├── framework/           # 验证框架
│   │   ├── comparison_base.h
│   │   └── dlopen_loader.cpp
│   ├── test_consistency.cpp
│   └── test_performance.cpp
│
├── benchmark/                # 性能基准测试
│   ├── memory_benchmark.cpp
│   └── arithmetic_benchmark.cpp
│
├── tools/                    # 测试工具和脚本
│   ├── verify_dual_lib.cpp
│   └── test_summary.py
│
├── CMakeLists.txt
└── README.md
```

## 重组步骤

1. 将gtest/下的所有测试移到unit/对应目录
2. 删除旧的nppcore/和nppi/目录
3. 将validation从gtest/移到独立的validation/目录
4. 创建tools/目录存放测试工具
5. 更新CMakeLists.txt