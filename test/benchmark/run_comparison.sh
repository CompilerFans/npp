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

echo -e "${GREEN}✓ NVIDIA NPP benchmarks completed${NC}"
echo ""

# ============================================================
# 步骤 3：生成对比报告
# ============================================================

echo -e "${YELLOW}Step 3: Generating comparison report...${NC}"

if command -v python3 &> /dev/null; then
    # 如果有 Python，生成详细的对比报告
    python3 "$SCRIPT_DIR/compare_results.py" \
        "$RESULTS_DIR/mpp_${TIMESTAMP}.json" \
        "$RESULTS_DIR/nvidia_npp_${TIMESTAMP}.json" \
        "$RESULTS_DIR/comparison_${TIMESTAMP}.html"
    
    echo -e "${GREEN}✓ Comparison report generated: $RESULTS_DIR/comparison_${TIMESTAMP}.html${NC}"
else
    echo -e "${YELLOW}Python3 not found, skipping HTML report generation${NC}"
fi

# ============================================================
# 步骤 4：显示简要对比
# ============================================================

echo ""
echo -e "${GREEN}=== Quick Summary ===${NC}"
echo "MPP results:        $RESULTS_DIR/mpp_${TIMESTAMP}.json"
echo "NVIDIA NPP results: $RESULTS_DIR/nvidia_npp_${TIMESTAMP}.json"
echo ""

if command -v jq &> /dev/null; then
    echo "Performance comparison (sample):"
    echo ""
    
    # 提取并对比 Add_8u_C1RSfs 的性能
    mpp_time=$(jq -r '.benchmarks[] | select(.name | contains("BM_nppiAdd_8u_C1RSfs_Fixed")) | .real_time' \
        "$RESULTS_DIR/mpp_${TIMESTAMP}.json" | head -1)
    nvidia_time=$(jq -r '.benchmarks[] | select(.name | contains("BM_nppiAdd_8u_C1RSfs_Fixed")) | .real_time' \
        "$RESULTS_DIR/nvidia_npp_${TIMESTAMP}.json" | head -1)
    
    if [ -n "$mpp_time" ] && [ -n "$nvidia_time" ]; then
        speedup=$(echo "scale=2; $nvidia_time / $mpp_time" | bc)
        echo "  nppiAdd_8u_C1RSfs (1920x1080):"
        echo "    MPP:        ${mpp_time} ms"
        echo "    NVIDIA NPP: ${nvidia_time} ms"
        echo "    Speedup:    ${speedup}x"
    fi
else
    echo -e "${YELLOW}Install 'jq' for quick performance comparison${NC}"
fi

echo ""
echo -e "${GREEN}=== Benchmark Comparison Complete ===${NC}"
