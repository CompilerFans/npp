# Reference Code 参考代码

本目录包含高质量的参考实现和分析文档。

## 目录

### Super Sampling CPU Reference

**位置**：`super_sampling_cpu_reference/`

**内容**：NVIDIA NPP `nppiResize` SUPER模式的高质量CPU参考实现

**质量指标**：
- ✅ 88.5%完美匹配（逐像素bit-exact）
- ✅ 100%在±1范围内（52个形状组合测试）
- ✅ 最大差异仅1像素

**快速开始**：
```bash
cd super_sampling_cpu_reference
./scripts/build_tests.sh
./validate_reference
./test_shapes
```

**文档导航**：
- `README.md` - 快速入门
- `README_SUPER_SAMPLING_ANALYSIS.md` - 完整分析总览
- `ALGORITHM_SUMMARY.md` - 算法总结
- `docs/` - 详细技术文档

**核心文件**：
- `src/super_sampling_cpu.h` - 头文件实现（可直接include使用）
- `tests/reference_super_sampling.cpp` - 验证测试（11个场景）
- `tests/test_extensive_shapes.cpp` - 广泛测试（52个场景）

---

## 添加新的参考实现

当添加新的参考代码时，请遵循以下结构：

```
ref_code/
├── README.md                    # 本文件
└── <feature_name>_reference/
    ├── README.md                # 快速入门
    ├── ALGORITHM_SUMMARY.md     # 算法总结
    ├── src/                     # 源代码
    ├── tests/                   # 测试代码
    ├── scripts/                 # 编译脚本
    └── docs/                    # 详细文档
```

### 文档要求

1. **README.md**：快速入门，5分钟内能运行
2. **ALGORITHM_SUMMARY.md**：算法核心概念和数学推导
3. **测试覆盖**：至少包含验证测试和边缘情况测试
4. **编译脚本**：一键编译所有测试

### 质量标准

- ✅ 与NVIDIA NPP高度匹配（>95%或明确说明差异原因）
- ✅ 详细的算法文档和注释
- ✅ 全面的测试覆盖
- ✅ 清晰的使用示例

---

## 许可证

遵循项目整体许可证。
