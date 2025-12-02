#!/usr/bin/env python3
"""
分析NVIDIA NPP Gamma LUT，寻找精确的实现公式
"""

import csv
import math
import sys

def load_lut(filename):
    """从CSV加载LUT"""
    data = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'input': int(row['Input']),
                'forward': int(row['Forward_Output']),
                'inverse': int(row['Inverse_Output'])
            })
    return data

def analyze_single_gamma(data, gamma_min=1.70, gamma_max=2.00, step=0.001):
    """分析单一gamma值的拟合效果"""
    print(f"\n{'='*70}")
    print(f"分析单一Gamma拟合 (范围: {gamma_min:.2f} - {gamma_max:.2f})")
    print(f"{'='*70}\n")

    best_gamma = gamma_min
    best_max_error = 255
    best_errors = []

    # 测试每个gamma值
    gamma = gamma_min
    while gamma <= gamma_max:
        max_error = 0
        errors = []

        for d in data:
            if d['input'] == 0:
                continue

            # 计算Forward Gamma
            normalized = d['input'] / 255.0
            predicted = round(math.pow(normalized, 1.0 / gamma) * 255.0)
            actual = d['forward']
            error = abs(predicted - actual)
            errors.append(error)

            if error > max_error:
                max_error = error

        # 更新最佳结果
        if max_error < best_max_error:
            best_max_error = max_error
            best_gamma = gamma
            best_errors = errors[:]

        gamma += step

    print(f"最佳Gamma值: {best_gamma:.4f}")
    print(f"最大误差: {best_max_error}")
    print(f"平均误差: {sum(best_errors) / len(best_errors):.2f}")

    # 显示误差分布
    error_counts = {}
    for e in best_errors:
        error_counts[e] = error_counts.get(e, 0) + 1

    print(f"\n误差分布:")
    for error in sorted(error_counts.keys()):
        count = error_counts[error]
        percent = count * 100.0 / len(best_errors)
        print(f"  ±{error}: {count} 像素 ({percent:.1f}%)")

    # 显示有误差的具体像素
    if best_max_error > 1:
        print(f"\n误差>1的像素点:")
        for i, d in enumerate(data):
            if d['input'] == 0:
                continue
            normalized = d['input'] / 255.0
            predicted = round(math.pow(normalized, 1.0 / best_gamma) * 255.0)
            actual = d['forward']
            error = predicted - actual
            if abs(error) > 1:
                print(f"  Input {d['input']:3d}: predicted={predicted:3d}, actual={actual:3d}, error={error:+3d}")

    return best_gamma, best_max_error

def analyze_piecewise_gamma(data):
    """分析是否为分段gamma函数"""
    print(f"\n{'='*70}")
    print(f"分析分段Gamma (检测不同输入范围的隐含gamma值)")
    print(f"{'='*70}\n")

    print(f"{'Input':>5s} {'Output':>6s} {'Implied γ':>10s}")
    print(f"{'-'*5} {'-'*6} {'-'*10}")

    implied_gammas = []
    for d in data:
        if d['input'] == 0 or d['forward'] == 0:
            continue

        input_norm = d['input'] / 255.0
        output_norm = d['forward'] / 255.0

        # output = input^(1/gamma)
        # gamma = log(input) / log(output)
        try:
            implied_gamma = math.log(input_norm) / math.log(output_norm)
            implied_gammas.append((d['input'], implied_gamma))

            # 每8个值打印一次
            if d['input'] % 8 == 0 or d['input'] < 20:
                print(f"{d['input']:5d} {d['forward']:6d} {implied_gamma:10.4f}")
        except:
            pass

    # 分析gamma变化趋势
    print(f"\nGamma变化趋势:")
    ranges = [
        (1, 20, "极低值"),
        (21, 50, "低值"),
        (51, 100, "中低值"),
        (101, 150, "中值"),
        (151, 200, "中高值"),
        (201, 254, "高值")
    ]

    for start, end, label in ranges:
        range_gammas = [g for i, g in implied_gammas if start <= i <= end]
        if range_gammas:
            avg_gamma = sum(range_gammas) / len(range_gammas)
            min_gamma = min(range_gammas)
            max_gamma = max(range_gammas)
            print(f"  {label:8s} (input {start:3d}-{end:3d}): avg={avg_gamma:.4f}, min={min_gamma:.4f}, max={max_gamma:.4f}")

def analyze_modified_srgb(data):
    """分析是否为修改的sRGB公式"""
    print(f"\n{'='*70}")
    print(f"分析修改的sRGB公式")
    print(f"{'='*70}\n")

    # 标准sRGB: threshold=0.0031308, gamma=2.4
    # 测试不同参数组合

    print("测试不同的sRGB参数组合...")

    best_threshold = 0.0031308
    best_srgb_gamma = 2.4
    best_max_error = 255

    for threshold in [i * 0.001 for i in range(1, 10)]:  # 0.001 - 0.009
        for srgb_gamma in [i * 0.1 for i in range(20, 30)]:  # 2.0 - 2.9
            max_error = 0

            for d in data:
                if d['input'] == 0:
                    continue

                normalized = d['input'] / 255.0

                # 修改的sRGB公式
                if normalized <= threshold:
                    result = 12.92 * normalized
                else:
                    result = 1.055 * math.pow(normalized, 1.0 / srgb_gamma) - 0.055

                predicted = round(result * 255.0)
                actual = d['forward']
                error = abs(predicted - actual)

                if error > max_error:
                    max_error = error

            if max_error < best_max_error:
                best_max_error = max_error
                best_threshold = threshold
                best_srgb_gamma = srgb_gamma

    print(f"\n最佳sRGB参数:")
    print(f"  阈值: {best_threshold:.6f}")
    print(f"  Gamma: {best_srgb_gamma:.2f}")
    print(f"  最大误差: {best_max_error}")

    if best_max_error <= 1:
        print(f"\n✅ 找到精确匹配的sRGB公式！")
    elif best_max_error <= 2:
        print(f"\n⚠️  接近匹配 (误差≤2)")
    else:
        print(f"\n❌ sRGB公式不匹配 (误差>{best_max_error})")

def generate_lut_code(data):
    """生成可以直接用于代码的LUT"""
    print(f"\n{'='*70}")
    print(f"生成C++代码 (如果无法找到公式，可直接使用NVIDIA的LUT)")
    print(f"{'='*70}\n")

    print("// Forward Gamma LUT (直接从NVIDIA NPP提取)")
    print("__constant__ Npp8u d_gamma_fwd_lut[256] = {")

    for i in range(0, 256, 16):
        values = [str(data[j]['forward']) for j in range(i, min(i+16, 256))]
        print(f"    {', '.join(values)},")

    print("};")

    print("\n// Inverse Gamma LUT (直接从NVIDIA NPP提取)")
    print("__constant__ Npp8u d_gamma_inv_lut[256] = {")

    for i in range(0, 256, 16):
        values = [str(data[j]['inverse']) for j in range(i, min(i+16, 256))]
        print(f"    {', '.join(values)},")

    print("};")

def main():
    if len(sys.argv) < 2:
        print("用法: python3 analyze_gamma_lut.py nvidia_gamma_lut.csv")
        sys.exit(1)

    filename = sys.argv[1]

    print(f"加载NVIDIA NPP Gamma LUT: {filename}")
    data = load_lut(filename)
    print(f"成功加载 {len(data)} 个数据点\n")

    # 1. 单一gamma分析
    best_gamma, max_error = analyze_single_gamma(data)

    # 2. 分段gamma分析
    analyze_piecewise_gamma(data)

    # 3. 修改的sRGB分析
    analyze_modified_srgb(data)

    # 4. 生成LUT代码
    generate_lut_code(data)

    # 最终建议
    print(f"\n{'='*70}")
    print(f"最终建议")
    print(f"{'='*70}\n")

    if max_error <= 1:
        print(f"✅ 使用单一Gamma值 {best_gamma:.4f} 即可达到±1精度")
    elif max_error <= 2:
        print(f"⚠️  单一Gamma值 {best_gamma:.4f} 可达到±2精度")
        print(f"   考虑:")
        print(f"   1. 直接使用NVIDIA的256值LUT (100%精确)")
        print(f"   2. 分析是否为分段函数")
    else:
        print(f"❌ 无法用简单公式拟合 (最大误差={max_error})")
        print(f"   建议: 直接使用NVIDIA的256值LUT")

if __name__ == '__main__':
    main()
