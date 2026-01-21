# CUDA NPP Samples - MPP 适配版本

本项目将 NVIDIA CUDA Samples 中的 NPP 相关示例适配到 MPP 开源实现，并提供完整的验证测试框架。

## 项目概述

从原始 cuda-samples 项目中提取了 7 个 NPP 相关示例，全部成功适配到 MPP 库：

### 核心图像处理示例 (6个)

1. **boxFilterNPP** - 盒式滤波器
2. **cannyEdgeDetectorNPP** - Canny 边缘检测
3. **histEqualizationNPP** - 直方图均衡化
4. **FilterBorderControlNPP** - Prewitt 梯度滤波器边界控制
5. **watershedSegmentationNPP** - 分水岭图像分割
6. **batchedLabelMarkersAndLabelCompressionNPP** - 批量连通区域标记

### 辅助示例 (1个)

7. **freeImageInteropNPP** - FreeImage 库集成演示（含图像滤波处理）

## 构建和运行

### 环境要求
- CUDA Toolkit 12.0+
- CMake 3.18+
- 支持 CUDA 的 GPU
- FreeImage 库（可选，用于图像 I/O）

### 编译
```bash
./build.sh
```

### 快速测试
```bash
# 一键运行所有测试和验证
./run_tests.sh
```

### 单独运行示例
```bash
cd build/bin

# 盒式滤波
./boxFilterNPP

# Canny边缘检测
./cannyEdgeDetectorNPP

# Prewitt梯度滤波
./FilterBorderControlNPP

# 直方图均衡化
./histEqualizationNPP

# 分水岭分割
./watershedSegmentationNPP

# 批量连通区域标记
./batchedLabelMarkersAndLabelCompressionNPP

# FreeImage集成
./freeImageInteropNPP
```

## 验证测试框架

本项目提供了完整的验证测试框架，用于对比 MPP 实现与 NVIDIA NPP 官方库的输出结果。

### 测试架构

```
run_tests.sh                    # 主测试脚本
├── 第一阶段：运行 MPP 示例程序
├── 第二阶段：收集输出结果
├── 第三阶段：功能验证（文件存在+大小匹配）
└── 第四阶段：算法正确性验证
    └── verify_results          # C++ 验证程序
        ├── 像素级精确匹配（确定性算法）
        └── 语义等价性验证（非确定性算法）
```

### 验证方法

#### 1. 像素级精确匹配（确定性算法）

适用于：boxFilter、histEqualization、Canny、Prewitt 等确定性算法。

**原理**：相同输入必须产生完全相同的输出。

**指标**：
| 指标 | 计算公式 | 说明 |
|------|----------|------|
| 差异像素数 | `count(MPP[i] != REF[i])` | 值不相等的像素总数 |
| 差异比例 | `差异像素数 / 总像素数 × 100%` | 不匹配像素的百分比 |
| PSNR | `20 × log10(255 / √MSE)` | 峰值信噪比，越高越好 |

**状态判定**：
| 状态 | 条件 | 含义 |
|------|------|------|
| PASS | 差异比例 = 0% | 完全匹配 |
| OK | PSNR >= 40dB | 高质量匹配 |
| WARN | 30dB <= PSNR < 40dB | 可接受的差异 |
| FAIL | PSNR < 30dB | 需要修复 |

#### 2. 语义等价性验证（非确定性算法）

适用于：LabelMarkers、Watershed 等基于 Union-Find 的标记算法。

**为什么需要语义等价性验证？**

Union-Find 算法的标签分配取决于 GPU 线程执行顺序，这是非确定性的。不同实现、甚至同一实现的不同运行，都可能产生不同的标签值。因此，我们不能简单地比较像素值是否相同。

**原理**：验证连通区域划分是否相同，而非标签值是否相同。

如果两个标签图表示相同的连通区域划分，那么：
1. 在 REF 中属于同一标签的任意两个像素，在 MPP 中也必须属于同一标签
2. 在 REF 中属于不同标签的任意两个像素，在 MPP 中也必须属于不同标签

**实现方法**：
```
建立 ref_label -> mpp_label 的双向映射：
- 如果同一个 ref_label 映射到多个不同的 mpp_label → 连通区域被分割
- 如果不同的 ref_label 映射到同一个 mpp_label → 连通区域被错误合并
```

**状态判定**：
| 状态 | 条件 | 含义 |
|------|------|------|
| PASS Perfect | 所有像素值完全相同 | 完美匹配 |
| PASS Semantic Eq | 像素值不同但语义等价 | 连通区域划分相同 |
| WARN Count Match | 标签数相同但连通性有误 | 需要关注 |
| FAIL Connect Err | 连通性错误 | 需要修复 |

### 参考数据

验证需要 NVIDIA NPP 官方库的输出作为参考：

```
reference_nvidia_npp/
├── teapot512_boxFilter.pgm
├── teapot512_cannyEdgeDetection.pgm
├── teapot512_histEqualization.pgm
├── teapot512_boxFilterFII.pgm
├── FilterBorderControl/
│   └── *.pgm
├── watershed/
│   └── *.raw
└── batchedLabelMarkers/
    └── *.raw
```

生成参考数据：
```bash
# 需要安装 NVIDIA NPP 库
./gen_nvidia_ref_batch
```

### 验证报告

运行测试后会生成详细的验证报告：

```bash
./run_tests.sh
# 报告输出: VERIFICATION_REPORT.md
```

报告包含：
- 每个测试用例的详细结果
- 差异比例、PSNR、标签数对比
- 语义等价性验证结果
- 总体统计和通过率

## 实现的 NPP 功能

### 图像滤波
- **盒式滤波** - 均值滤波，支持任意核大小
- **Prewitt 梯度** - X/Y 分量独立输出，支持 L1/L2/Inf 范数

### 边缘检测
- **Canny 边缘检测** - 多阶段算法：高斯滤波 → Sobel 梯度 → 非最大值抑制 → 双阈值检测

### 直方图处理
- **直方图均衡化** - 均匀分布直方图，动态缓冲区管理

### 图像分割
- **分水岭分割** - 基于标记的分水岭算法，梯度计算和优先队列处理
- **连通区域标记** - Union-Find 算法，支持 4-way 和 8-way 连通性

## 技术特性

- **API 兼容** - 与 NVIDIA NPP 完全一致的 API 签名
- **高性能** - CUDA 并行计算优化
- **异步支持** - 支持 CUDA 流和上下文
- **内存高效** - 优化的 GPU 内存管理
- **完整验证** - 像素级和语义级双重验证

## 文件结构

```
cuda-npp-samples/
├── CMakeLists.txt                 # 统一构建配置
├── build.sh                       # 构建脚本
├── run_tests.sh                   # 测试脚本
├── verify_results.cpp             # 验证程序源码
├── Common/                        # 公共工具和数据
├── Samples/4_CUDA_Libraries/      # NPP 示例代码
│   ├── boxFilterNPP/
│   ├── cannyEdgeDetectorNPP/
│   ├── FilterBorderControlNPP/
│   ├── histEqualizationNPP/
│   ├── watershedSegmentationNPP/
│   └── batchedLabelMarkersAndLabelCompressionNPP/
├── reference_nvidia_npp/          # NVIDIA NPP 参考输出
├── mpp_results/                   # MPP 测试输出
└── VERIFICATION_REPORT.md         # 验证报告
```

## 内部 mcMPP 仓库适配说明

在内部 mcMPP 仓库（MACA 平台）中使用时，需要进行以下修改：

### 1. 构建脚本

使用 `build_maca.sh` 替代 `build.sh`：
```bash
./build_maca.sh
```

### 2. CMakeLists.txt 修改

修改第 74 行，将链接库名从 `mpp` 改为 `mppc`：
```cmake
# 原始（NVIDIA 平台）
target_link_libraries(${SAMPLE_NAME} mpp CUDA::cudart CUDA::cudart_static)

# 修改为（MACA 平台）
target_link_libraries(${SAMPLE_NAME} mppc CUDA::cudart CUDA::cudart_static)
```

## CI 集成

测试脚本支持 CI/CD 集成（如 Jenkins），通过退出码反映测试结果：

```bash
./run_tests.sh
echo $?  # 0=全部通过, 非0=失败数量
```

**退出码说明**：
| 退出码 | 含义 |
|--------|------|
| 0 | 所有测试通过 |
| N (N>0) | N 个测试失败 |

**Jenkins 集成示例**：
```groovy
stage('Samples Tests') {
    steps {
        dir('examples/cuda-npp-samples') {
            sh './build.sh'
            sh './run_tests.sh'
        }
    }
    post {
        always {
            archiveArtifacts artifacts: 'examples/cuda-npp-samples/VERIFICATION_REPORT.md'
        }
    }
}
```

**当前测试状态**：
- 非确定性算法测试（LabelMarkers、Watershed）已暂时屏蔽
- 仅运行确定性算法的像素级验证

## 开发说明

本项目展示了 MPP 作为 NVIDIA NPP 开源替代方案的可行性，为 GPU 加速的图像和信号处理应用提供了完整的解决方案。

验证框架的设计考虑了：
- **确定性算法**：要求像素级精确匹配
- **非确定性算法**：使用语义等价性验证，允许标签值不同但连通区域划分必须相同
