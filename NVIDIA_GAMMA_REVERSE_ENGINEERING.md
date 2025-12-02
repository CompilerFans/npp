# NVIDIA NPP Gamma完全逆向工程指南

## 🎯 目标

提取NVIDIA NPP的完整Gamma LUT（查找表），找到精确的实现公式。

## 📋 问题背景

当前使用gamma 1.87实现，但仍有±2-3的误差：

```
Input 19: NVIDIA=61, Gamma1.87=64 (error=-3, 需要gamma≈1.815)
Input 22: NVIDIA=67, Gamma1.87=69 (error=-2, 需要gamma≈1.833)
Input 39: NVIDIA=95, Gamma1.87=93 (error=+2, 需要gamma≈1.902)
Input 43: NVIDIA=100, Gamma1.87=98 (error=+2, 需要gamma≈1.902)
```

这说明NVIDIA可能使用：
1. **分段gamma函数**（不同输入范围使用不同gamma值）
2. **修改过的sRGB公式**（不同的阈值或常数）
3. **特殊的查找表生成算法**

## 🚀 运行步骤

### 1. 上传诊断工具到服务器

```bash
# 在本地Mac上
cd ~/Desktop/MPP/npp
scp extract_nvidia_gamma_lut.cpp 你的用户名@服务器地址:~/npp/
scp compile_extract_lut.sh 你的用户名@服务器地址:~/npp/
```

### 2. 在服务器上编译并运行

```bash
# SSH登录服务器
ssh 你的用户名@服务器地址

# 进入项目目录
cd ~/npp

# 编译诊断工具
./compile_extract_lut.sh

# 运行诊断工具
./extract_nvidia_gamma_lut

# 将结果保存到文件
./extract_nvidia_gamma_lut > nvidia_gamma_analysis.txt
```

### 3. 下载结果到本地分析

```bash
# 在本地Mac上
cd ~/Desktop/MPP/npp
scp 你的用户名@服务器地址:~/npp/nvidia_gamma_lut.csv ./
scp 你的用户名@服务器地址:~/npp/nvidia_gamma_analysis.txt ./
```

## 📊 预期输出

程序会输出：

1. **完整的256值Forward Gamma LUT**
   ```
   Input -> Output
     0 ->   0
     1 ->   5
     2 ->   8
     ...
   255 -> 255
   ```

2. **完整的256值Inverse Gamma LUT**

3. **最佳拟合单一gamma值**及其最大误差

4. **误差分布分析**（每16个值采样一次）

5. **CSV文件**：`nvidia_gamma_lut.csv`包含所有数据

## 🔍 后续分析

拿到完整LUT后，可以：

### 方法1: 分段gamma分析

检查是否存在阈值点，在不同区间使用不同gamma值：

```python
import pandas as pd
import numpy as np

df = pd.read_csv('nvidia_gamma_lut.csv')

# 为每个输入计算implied gamma
for i in range(1, 256):
    input_norm = i / 255.0
    output_norm = df.loc[i, 'Forward_Output'] / 255.0
    if output_norm > 0:
        implied_gamma = np.log(input_norm) / np.log(output_norm)
        print(f"Input {i}: implied gamma = {implied_gamma:.3f}")
```

### 方法2: 检查是否为修改的sRGB

尝试不同的sRGB阈值和参数：

```python
# 测试不同阈值
for threshold in np.arange(0.002, 0.005, 0.0001):
    # 测试不同gamma
    for gamma in np.arange(2.0, 2.8, 0.1):
        # 计算误差
        ...
```

### 方法3: 直接使用NVIDIA的LUT

如果找不到公式，直接复制NVIDIA的256值LUT：

```cpp
// 在 nppi_gamma.cu 中
__constant__ Npp8u d_gamma_fwd_lut[256] = {
    0, 5, 8, 11, 13, 15, 17, 19, 21, 23, ...,  // 从CSV复制
};
```

## 🎯 成功标准

找到让所有256个输入值的误差都≤1的实现方式。

## 📝 需要记录的信息

1. 完整的256值LUT
2. 最佳单一gamma拟合结果
3. 误差分布模式
4. 是否存在明显的分段点
5. GPU型号和CUDA版本

---

**准备好后运行诊断工具，我们将基于实际数据找到NVIDIA的精确实现！** 🔬
