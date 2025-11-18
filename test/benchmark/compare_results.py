#!/usr/bin/env python3
"""
NPP 性能对比工具

读取两个 Google Benchmark JSON 结果，生成对比报告。

用法:
    python3 compare_results.py mpp.json nvidia.json [output.html]
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

def load_benchmark_results(json_file: str) -> List[Dict]:
    """加载 Google Benchmark JSON 结果"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data.get('benchmarks', [])

def extract_test_results(benchmarks: List[Dict]) -> Dict[str, Dict]:
    """提取测试结果，按测试名称分组"""
    results = {}
    for bench in benchmarks:
        name = bench.get('name', '')
        # 只保留 mean 结果（忽略 median, stddev, cv）
        if '/real_time_mean' in name or (
            '/real_time' not in name and 
            'mean' not in name.lower() and
            'median' not in name.lower()
        ):
            # 清理名称
            clean_name = name.replace('/real_time_mean', '')
            results[clean_name] = {
                'time': bench.get('real_time', 0),
                'cpu_time': bench.get('cpu_time', 0),
                'iterations': bench.get('iterations', 0),
                'bytes_per_second': bench.get('bytes_per_second', 0),
            }
    return results

def compare_results(mpp_results: Dict, nvidia_results: Dict) -> List[Tuple]:
    """对比两个结果，返回对比列表"""
    comparisons = []
    
    for name in sorted(mpp_results.keys()):
        if name not in nvidia_results:
            continue
            
        mpp = mpp_results[name]
        nvidia = nvidia_results[name]
        
        mpp_time = mpp['time']
        nvidia_time = nvidia['time']
        
        if nvidia_time > 0:
            speedup = nvidia_time / mpp_time
            percentage = (speedup) * 100
        else:
            speedup = 0
            percentage = 0
        
        # 评级
        if percentage >= 95:
            rating = '🟢 优秀'
        elif percentage >= 80:
            rating = '🟡 良好'
        elif percentage >= 60:
            rating = '🟠 可接受'
        else:
            rating = '🔴 需优化'
        
        comparisons.append({
            'name': name,
            'mpp_time': mpp_time,
            'nvidia_time': nvidia_time,
            'speedup': speedup,
            'percentage': percentage,
            'rating': rating,
            'mpp_throughput': mpp['bytes_per_second'],
            'nvidia_throughput': nvidia['bytes_per_second'],
        })
    
    return comparisons

def generate_text_report(comparisons: List[Dict]) -> str:
    """生成文本格式的对比报告"""
    report = []
    report.append("=" * 100)
    report.append("NPP 性能对比报告")
    report.append("=" * 100)
    report.append("")
    
    # 表头
    report.append(f"{'测试名称':<50} {'MPP (ms)':>12} {'NVIDIA (ms)':>12} {'性能比':>10} {'评级':>10}")
    report.append("-" * 100)
    
    # 数据行
    for comp in comparisons:
        name = comp['name']
        if len(name) > 48:
            name = name[:45] + "..."
        
        report.append(
            f"{name:<50} "
            f"{comp['mpp_time']:>12.3f} "
            f"{comp['nvidia_time']:>12.3f} "
            f"{comp['percentage']:>9.1f}% "
            f"{comp['rating']:>10}"
        )
    
    report.append("-" * 100)
    
    # 统计信息
    avg_percentage = sum(c['percentage'] for c in comparisons) / len(comparisons) if comparisons else 0
    excellent = sum(1 for c in comparisons if '优秀' in c['rating'])
    good = sum(1 for c in comparisons if '良好' in c['rating'])
    acceptable = sum(1 for c in comparisons if '可接受' in c['rating'])
    needs_opt = sum(1 for c in comparisons if '需优化' in c['rating'])
    
    report.append("")
    report.append("总体统计:")
    report.append(f"  平均性能比: {avg_percentage:.1f}%")
    report.append(f"  优秀 (≥95%):   {excellent} 个")
    report.append(f"  良好 (80-95%): {good} 个")
    report.append(f"  可接受 (60-80%): {acceptable} 个")
    report.append(f"  需优化 (<60%): {needs_opt} 个")
    report.append("")
    report.append("=" * 100)
    
    return "\n".join(report)

def generate_html_report(comparisons: List[Dict], output_file: str):
    """生成 HTML 格式的对比报告"""
    html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>NPP 性能对比报告</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
        h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #4CAF50; color: white; font-weight: bold; }
        tr:hover { background: #f1f1f1; }
        .excellent { color: #4CAF50; font-weight: bold; }
        .good { color: #8BC34A; }
        .acceptable { color: #FF9800; }
        .needs-opt { color: #f44336; font-weight: bold; }
        .summary { background: #e8f5e9; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .summary h2 { margin-top: 0; color: #2E7D32; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 NPP 性能对比报告</h1>
"""
    
    # 添加统计信息
    avg_percentage = sum(c['percentage'] for c in comparisons) / len(comparisons) if comparisons else 0
    excellent = sum(1 for c in comparisons if '优秀' in c['rating'])
    good = sum(1 for c in comparisons if '良好' in c['rating'])
    acceptable = sum(1 for c in comparisons if '可接受' in c['rating'])
    needs_opt = sum(1 for c in comparisons if '需优化' in c['rating'])
    
    html += f"""
        <div class="summary">
            <h2>总体统计</h2>
            <p><strong>平均性能比:</strong> {avg_percentage:.1f}%</p>
            <p>
                <strong>分布:</strong> 
                优秀 {excellent} 个 | 
                良好 {good} 个 | 
                可接受 {acceptable} 个 | 
                需优化 {needs_opt} 个
            </p>
        </div>
        
        <table>
            <thead>
                <tr>
                    <th>测试名称</th>
                    <th>MPP 时间 (ms)</th>
                    <th>NVIDIA 时间 (ms)</th>
                    <th>性能比</th>
                    <th>评级</th>
                </tr>
            </thead>
            <tbody>
"""
    
    # 添加数据行
    for comp in comparisons:
        rating_class = comp['rating'].split()[0].replace('🟢', 'excellent').replace('🟡', 'good').replace('🟠', 'acceptable').replace('🔴', 'needs-opt')
        html += f"""
                <tr>
                    <td>{comp['name']}</td>
                    <td>{comp['mpp_time']:.3f}</td>
                    <td>{comp['nvidia_time']:.3f}</td>
                    <td>{comp['percentage']:.1f}%</td>
                    <td class="{rating_class}">{comp['rating']}</td>
                </tr>
"""
    
    html += """
            </tbody>
        </table>
    </div>
</body>
</html>
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

def main():
    if len(sys.argv) < 3:
        print("用法: python3 compare_results.py <mpp.json> <nvidia.json> [output.html]")
        sys.exit(1)
    
    mpp_file = sys.argv[1]
    nvidia_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    # 检查文件是否存在
    if not Path(mpp_file).exists():
        print(f"错误: 找不到文件 {mpp_file}")
        sys.exit(1)
    if not Path(nvidia_file).exists():
        print(f"错误: 找不到文件 {nvidia_file}")
        sys.exit(1)
    
    # 加载结果
    print("正在加载结果...")
    mpp_benchmarks = load_benchmark_results(mpp_file)
    nvidia_benchmarks = load_benchmark_results(nvidia_file)
    
    # 提取结果
    mpp_results = extract_test_results(mpp_benchmarks)
    nvidia_results = extract_test_results(nvidia_benchmarks)
    
    print(f"MPP 测试数量: {len(mpp_results)}")
    print(f"NVIDIA 测试数量: {len(nvidia_results)}")
    
    # 对比
    comparisons = compare_results(mpp_results, nvidia_results)
    
    if not comparisons:
        print("警告: 没有找到可对比的测试结果")
        sys.exit(0)
    
    # 生成文本报告
    text_report = generate_text_report(comparisons)
    print("\n" + text_report)
    
    # 生成 HTML 报告
    if output_file:
        generate_html_report(comparisons, output_file)
        print(f"\n✅ HTML 报告已生成: {output_file}")
        print(f"   在浏览器中打开查看: file://{Path(output_file).absolute()}")

if __name__ == '__main__':
    main()
