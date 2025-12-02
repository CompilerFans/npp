# Gamma逆向工程 - 快速参考

## 📦 创建的文件

1. **extract_nvidia_gamma_lut.cpp** - C++诊断工具，提取NVIDIA NPP的完整256值LUT
2. **compile_extract_lut.sh** - 编译脚本
3. **analyze_gamma_lut.py** - Python分析脚本，自动分析LUT并寻找公式
4. **NVIDIA_GAMMA_REVERSE_ENGINEERING.md** - 完整使用指南

## ⚡ 快速执行流程

### 在服务器上执行（需要GPU）:

```bash
# 1. 上传文件（在本地Mac）
cd ~/Desktop/MPP/npp
scp extract_nvidia_gamma_lut.cpp compile_extract_lut.sh 用户名@服务器:~/npp/

# 2. 编译并运行（在服务器）
ssh 用户名@服务器
cd ~/npp
./compile_extract_lut.sh
./extract_nvidia_gamma_lut > nvidia_gamma_analysis.txt

# 3. 下载结果（在本地Mac）
scp 用户名@服务器:~/npp/nvidia_gamma_lut.csv ./
scp 用户名@服务器:~/npp/nvidia_gamma_analysis.txt ./
```

### 在本地Mac分析:

```bash
cd ~/Desktop/MPP/npp
python3 analyze_gamma_lut.py nvidia_gamma_lut.csv
```

## 🎯 预期结果

分析脚本会告诉你：

1. ✅ **最佳单一gamma值**及其精度
2. 📊 **隐含gamma值的变化趋势**（检查是否为分段函数）
3. 🔍 **修改的sRGB参数**（如果适用）
4. 💾 **现成的C++ LUT代码**（可直接复制使用）

## 🚀 三种可能的解决方案

### 方案1: 单一Gamma值（如果误差≤1）

```cpp
static constexpr float NPP_GAMMA = X.XXXf;  // 从分析结果获取
```

### 方案2: 分段Gamma函数（如果发现不同范围有不同gamma）

```cpp
if (normalized < THRESHOLD) {
    result = powf(normalized, 1.0f / GAMMA_LOW);
} else {
    result = powf(normalized, 1.0f / GAMMA_HIGH);
}
```

### 方案3: 直接使用NVIDIA的LUT（100%精确）

```cpp
__constant__ Npp8u d_gamma_fwd_lut[256] = {
    // 从analyze_gamma_lut.py的输出直接复制
};
```

## 📝 下一步

运行诊断工具后，根据分析结果：

1. 如果找到误差≤1的gamma值 → 更新代码中的NPP_GAMMA常量
2. 如果是分段函数 → 实现分段逻辑
3. 如果无法用公式拟合 → 直接使用NVIDIA的256值LUT

---

**现在就去服务器上运行诊断工具！** 🚀
