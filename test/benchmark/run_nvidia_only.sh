#!/bin/bash
# 只测试 NVIDIA NPP 性能

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
RESULTS_DIR="$PROJECT_ROOT/benchmark_results"

echo "=== NVIDIA NPP Performance Test ==="
echo ""

# 创建结果目录
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 清理并编译
echo "Building NVIDIA NPP benchmark..."
cd "$PROJECT_ROOT"
rm -rf build
mkdir -p build
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_BENCHMARKS=ON \
    -DUSE_NVIDIA_NPP=ON

echo "Compiling..."
make nppi_arithmetic_benchmark -j$(nproc)

if [ ! -f "benchmark/nppi_arithmetic_benchmark" ]; then
    echo "Error: Benchmark executable not found"
    exit 1
fi

# 运行测试
echo ""
echo "Running NVIDIA NPP benchmarks..."
cd benchmark
./nppi_arithmetic_benchmark \
    --benchmark_out="$RESULTS_DIR/nvidia_npp_${TIMESTAMP}.json" \
    --benchmark_out_format=json \
    --benchmark_repetitions=5 \
    --benchmark_report_aggregates_only=true

echo ""
echo "✓ Test completed!"
echo "Results saved to: $RESULTS_DIR/nvidia_npp_${TIMESTAMP}.json"
