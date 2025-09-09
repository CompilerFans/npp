#!/bin/bash
"""
NPP状态报告生成脚本

使用方法:
1. 先运行测试并生成XML: ./unit_tests --gtest_output=xml:test_results.xml
2. 运行此脚本生成状态报告: ./status/run_status_report.sh
"""

# 设置脚本目录和项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 进入项目根目录
cd "$PROJECT_ROOT"

echo "开始生成NPP API状态报告..."
echo "项目根目录: $PROJECT_ROOT"

# 检查依赖
if ! command -v python3 &> /dev/null; then
    echo "错误: 需要安装Python3"
    exit 1
fi

# 检查PyYAML
if ! python3 -c "import yaml" &> /dev/null; then
    echo "警告: 未找到PyYAML，正在尝试安装..."
    pip3 install pyyaml
fi

# 检查测试结果文件
if [ ! -f "test_results.xml" ]; then
    echo "警告: 未找到test_results.xml，将生成新的测试结果..."
    
    # 检查是否有unit_tests可执行文件
    if [ -f "unit_tests" ]; then
        echo "运行单元测试并生成XML结果..."
        ./unit_tests --gtest_output=xml:test_results.xml --gtest_brief
        echo "测试完成"
    elif [ -f "build/unit_tests" ]; then
        echo "运行单元测试并生成XML结果..."
        ./build/unit_tests --gtest_output=xml:test_results.xml --gtest_brief
        echo "测试完成"
    else
        echo "警告: 未找到unit_tests可执行文件，将在没有测试结果的情况下生成报告"
        # 创建空的XML文件以避免错误
        cat > test_results.xml << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<testsuites tests="0" failures="0" disabled="0" errors="0" time="0">
</testsuites>
EOF
    fi
fi

# 运行状态报告生成器
echo "生成API状态报告..."
python3 "$SCRIPT_DIR/generate_status_report.py" \
    --project-root "$PROJECT_ROOT" \
    --features-file "status/api_features.yaml" \
    --xml-results "test_results.xml" \
    --output "status/API_STATUS.yaml"

# 检查结果
if [ $? -eq 0 ]; then
    echo ""
    echo "状态报告生成成功"
    echo "报告文件: $PROJECT_ROOT/status/API_STATUS.yaml"
    echo ""
    
    # 显示报告摘要
    if [ -f "status/API_STATUS.yaml" ] && command -v python3 &> /dev/null; then
        echo "报告摘要:"
        python3 -c "
import yaml
try:
    with open('status/API_STATUS.yaml', 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    summary = data.get('summary', {})
    print(f'总API数量: {summary.get(\"total_apis\", 0)}')
    print(f'已实现API: {summary.get(\"implemented_apis\", 0)} ({summary.get(\"implementation_rate\", 0)}%)')
    print(f'已测试API: {summary.get(\"tested_apis\", 0)} ({summary.get(\"test_coverage_rate\", 0)}%)')
    print(f'测试通过: {summary.get(\"passed_tests\", 0)} / {summary.get(\"total_tests\", 0)}')
except Exception as e:
    print('无法解析报告摘要')
"
    fi
else
    echo "状态报告生成失败"
    exit 1
fi