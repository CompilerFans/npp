# 反推 NPP 13.4 语义的方法论（实现/测试通用）

语义真值优先级：

1. **5090 差分 golden**：最终裁决。状态按名称比较，值按容差比较。
2. **NPP 库 PTX/SASS 反汇编**：用于从机器码确认算法/边界/校验顺序
   （尤其是文档没写、探针难以穷举的情形，例如 cubic 越界 tap、求和窗口
   的裁剪规则）。
3. **官方头文件注释与文档**：给契约方向，但已知多处与实机不符
   （MaximumError 布局、相对误差分母、16f 可用性等），不能单独作准。

## 1. 定位内核符号

NPP 按模块分库（`/usr/local/cuda/lib64/`）：

| 库 | 模块 |
|---|---|
| libnppif.so | 滤波（Filter/SumWindow/导数/形态学的一部分） |
| libnppig.so | 几何（WarpAffine/Resize/Rotate/Mirror） |
| libnppist.so | 图像统计（CrossCorr/积分/误差） |
| libnppial.so | 算术/逻辑 |
| libnppicc.so | 颜色转换 |
| libnpps.so / libnppc.so | 信号处理 / 核心 |

```bash
cuobjdump -symbols /usr/local/cuda/lib64/libnppif.so | grep -i sumwindow | head
cuobjdump -symbols /usr/local/cuda/lib64/libnppig.so  | grep -i 'warpaffine.*cubic' | head
```

符号名本身经常直接给出算法结构，例如：

- `FilterSumWindowColumnReplicateBorderSharedFunctor`
- `SampleCubicImage` / `SampleCubicSplineImage`
- `...KernelTransform`

## 2. 反汇编单个内核

```bash
# 取 mangled 名后：
cuobjdump -sass -fun '<mangled-name>' /usr/local/cuda/lib64/libnppif.so > /tmp/kernel.sass
# 或先导出该库的 cubin/PTX 再 nvdisasm
```

阅读重点：立即数（窗口半径、权重系数）、边界比较（`<` vs `<=`）、
坐标钳制/回绕、舍入指令（`FRND`/`F2I`）、以及分支结构对应的校验顺序。

## 3. 与探针/差分配合

1. 先用 5090 探针（NPP 直调）扫参数空间，缩小差异范围。
2. 对差异点用 SASS 确认常量/比较方向。
3. 在 mcMPP 实现后用 `NPP_RECORD_DIR` 录 golden、本地 `NPP_VERIFY_DIR`
   回放验证；必要时补 crafted golden 把规则钉死。

## 4. 注意事项

- SASS 依赖架构（RTX 5090 = sm_120），以 5090 上安装的库为准。
- NPP 的 symbol 是 `STB_LOCAL`，只能反汇编、不能链接；不要尝试调用。
- 反汇编输出很大，重定向到文件后 grep/tail，保持工具输出小。
