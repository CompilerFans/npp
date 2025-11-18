#!/bin/bash
# NPP vs NVIDIA NPP 性能对比脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
BENCHMARK_DIR="$BUILD_DIR/benchmark"
RESULTS_DIR="$PROJECT_ROOT/benchmark_results"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== NPP Performance Benchmark Comparison ===${NC}"
echo ""

# 创建结果目录
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ============================================================
# 步骤 1：编译 MPP 版本并运行性能测试
# ============================================================

echo -e "${YELLOW}Step 1: Building and benchmarking MPP implementation...${NC}"

cd "$PROJECT_ROOT"

# 清理并重新构建 MPP 版本
rm -rf build
mkdir -p build
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_BENCHMARKS=ON \
    -DUSE_NVIDIA_NPP=OFF

# 先编译 MPP 库，再编译 benchmark
echo "Building MPP libraries..."
make npp -j$(nproc)

echo "Building benchmark..."
make nppi_arithmetic_benchmark -j$(nproc)

if [ ! -f "$BENCHMARK_DIR/nppi_arithmetic_benchmark" ]; then
    echo -e "${RED}Error: MPP benchmark executable not found${NC}"
    exit 1
fi

echo "Running MPP benchmarks..."
"$BENCHMARK_DIR/nppi_arithmetic_benchmark" \
    --benchmark_out="$RESULTS_DIR/mpp_${TIMESTAMP}.json" \
    --benchmark_out_format=json \
    --benchmark_repetitions=5 \
    --benchmark_report_aggregates_only=true

# 修复权限问题
chmod 644 "$RESULTS_DIR/mpp_${TIMESTAMP}.json"

echo -e "${GREEN}✓ MPP benchmarks completed${NC}"
echo ""

# ============================================================
# 步骤 2：编译 NVIDIA NPP 版本并运行性能测试
# ============================================================

echo -e "${YELLOW}Step 2: Building and benchmarking NVIDIA NPP implementation...${NC}"

cd "$PROJECT_ROOT"

# 清理并重新构建 NVIDIA NPP 版本
rm -rf build
mkdir -p build
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_BENCHMARKS=ON \
    -DUSE_NVIDIA_NPP=ON

make nppi_arithmetic_benchmark -j$(nproc)

if [ ! -f "$BENCHMARK_DIR/nppi_arithmetic_benchmark" ]; then
    echo -e "${RED}Error: NVIDIA NPP benchmark executable not found${NC}"
    exit 1
fi

echo "Running NVIDIA NPP benchmarks..."
"$BENCHMARK_DIR/nppi_arithmetic_benchmark" \
    --benchmark_out="$RESULTS_DIR/nvidia_npp_${TIMESTAMP}.json" \
    --benchmark_out_format=json \
    --benchmark_repetitions=5 \
    --benchmark_report_aggregates_only=true

# 修复权限问题
chmod 644 "$RESULTS_DIR/nvidia_npp_${TIMESTAMP}.json"

echo -e "${GREEN}✓ NVIDIA NPP benchmarks completed${NC}"
echo ""

# ============================================================
# 步骤 3：生成详细对比报告
# ============================================================

echo ""
echo -e "${YELLOW}Step 3: Generating detailed comparison report...${NC}"
echo ""

# 检查 JSON 文件是否存在
if [ ! -f "$RESULTS_DIR/mpp_${TIMESTAMP}.json" ]; then
    echo -e "${RED}Error: MPP results file not found${NC}"
    exit 1
fi

if [ ! -f "$RESULTS_DIR/nvidia_npp_${TIMESTAMP}.json" ]; then
    echo -e "${RED}Error: NVIDIA NPP results file not found${NC}"
    exit 1
fi

# 使用 Python 脚本生成详细对比
if command -v python3 &> /dev/null; then
    HTML_REPORT="$RESULTS_DIR/comparison_${TIMESTAMP}.html"
    
    python3 "$SCRIPT_DIR/compare_results.py" \
        "$RESULTS_DIR/mpp_${TIMESTAMP}.json" \
        "$RESULTS_DIR/nvidia_npp_${TIMESTAMP}.json" \
        "$HTML_REPORT"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✓ Detailed comparison report generated${NC}"
        echo -e "   HTML Report: ${HTML_REPORT}"
        echo -e "   View in browser: file://${HTML_REPORT}"
    else
        echo -e "${RED}✗ Failed to generate comparison report${NC}"
    fi
else
    echo -e "${RED}Error: Python3 not found${NC}"
    echo "Please install Python3 to generate comparison reports"
    echo ""
    echo "Manual comparison:"
    echo "  MPP results:        $RESULTS_DIR/mpp_${TIMESTAMP}.json"
    echo "  NVIDIA NPP results: $RESULTS_DIR/nvidia_npp_${TIMESTAMP}.json"
fi

echo ""
echo -e "${GREEN}=== Benchmark Comparison Complete ===${NC}"
