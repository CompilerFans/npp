#!/bin/bash
# =============================================================================
# 一键测试脚本 - NVIDIA NPP Performance Benchmark
# =============================================================================
# 用法：./quick_benchmark.sh
# 功能：自动环境检查、编译、运行性能测试
# =============================================================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印函数
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

# 获取项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
BUILD_DIR="$PROJECT_ROOT/build"
RESULTS_DIR="$PROJECT_ROOT/benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# =============================================================================
# 1. 环境检查
# =============================================================================
print_header "Step 1: Environment Check"

# 检查 CMake
print_info "Checking CMake version..."
if ! command -v cmake &> /dev/null; then
    print_error "CMake not found! Please install CMake >= 3.10"
    exit 1
fi
CMAKE_VERSION=$(cmake --version | head -n1 | awk '{print $3}')
print_success "CMake version: $CMAKE_VERSION"

# 检查 CUDA
print_info "Checking CUDA..."
if ! command -v nvcc &> /dev/null; then
    print_error "CUDA not found! Please install CUDA toolkit"
    exit 1
fi
CUDA_VERSION=$(nvcc -V | grep "release" | awk '{print $6}' | cut -d',' -f1)
print_success "CUDA version: $CUDA_VERSION"

# 检查 GPU
print_info "Checking GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    print_error "nvidia-smi not found! No NVIDIA GPU detected"
    exit 1
fi
GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
print_success "GPU: $GPU_INFO"

# 检查 Git
print_info "Checking Git..."
if ! command -v git &> /dev/null; then
    print_error "Git not found!"
    exit 1
fi
print_success "Git available"

# =============================================================================
# 2. 更新代码
# =============================================================================
print_header "Step 2: Update Source Code"

cd "$PROJECT_ROOT"

# 检查是否是 Git 仓库
if [ -d ".git" ]; then
    print_info "Pulling latest changes..."
    git pull
    
    print_info "Updating submodules..."
    git submodule update --init --recursive
    
    print_success "Source code updated"
else
    print_error "Not a git repository. Please clone from GitHub first."
    exit 1
fi

# 验证 GoogleTest
if [ ! -f "third_party/googletest/CMakeLists.txt" ]; then
    print_error "GoogleTest submodule not found!"
    exit 1
fi
print_success "GoogleTest submodule verified"

# 验证 Google Benchmark
if [ ! -f "third_party/benchmark/CMakeLists.txt" ]; then
    print_error "Google Benchmark submodule not found!"
    exit 1
fi
print_success "Google Benchmark submodule verified"

# =============================================================================
# 3. 清理和配置
# =============================================================================
print_header "Step 3: CMake Configuration"

# 清理旧的构建文件
if [ -d "$BUILD_DIR" ]; then
    print_info "Removing old build directory..."
    rm -rf "$BUILD_DIR"
fi

# 创建构建目录
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# CMake 配置
print_info "Running CMake configuration..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_BENCHMARKS=ON \
    -DUSE_NVIDIA_NPP=ON

if [ $? -eq 0 ]; then
    print_success "CMake configuration successful"
else
    print_error "CMake configuration failed!"
    exit 1
fi

# =============================================================================
# 4. 编译
# =============================================================================
print_header "Step 4: Compilation"

# 获取 CPU 核心数
if command -v nproc &> /dev/null; then
    CORES=$(nproc)
else
    CORES=4
fi

print_info "Compiling with $CORES cores..."
make nppi_arithmetic_benchmark -j$CORES

if [ $? -eq 0 ]; then
    print_success "Compilation successful"
else
    print_error "Compilation failed!"
    exit 1
fi

# 验证可执行文件
if [ ! -f "$BUILD_DIR/benchmark/nppi_arithmetic_benchmark" ]; then
    print_error "Benchmark executable not found!"
    exit 1
fi
print_success "Benchmark executable created"

# =============================================================================
# 5. 运行测试
# =============================================================================
print_header "Step 5: Running Benchmarks"

# 创建结果目录
mkdir -p "$RESULTS_DIR"

cd "$BUILD_DIR/benchmark"

print_info "Running performance tests..."
echo ""

# 运行测试并保存结果
./nppi_arithmetic_benchmark \
    --benchmark_out="$RESULTS_DIR/nvidia_npp_${TIMESTAMP}.json" \
    --benchmark_out_format=json \
    --benchmark_repetitions=5 \
    --benchmark_report_aggregates_only=true \
    --benchmark_counters_tabular=true

if [ $? -eq 0 ]; then
    print_success "Benchmarks completed successfully"
else
    print_error "Benchmarks failed!"
    exit 1
fi

# =============================================================================
# 6. 总结
# =============================================================================
print_header "Summary"

echo -e "${GREEN}✓ All tests completed successfully!${NC}\n"

print_info "System Information:"
echo "  - CMake: $CMAKE_VERSION"
echo "  - CUDA: $CUDA_VERSION"
echo "  - GPU: $GPU_INFO"
echo ""

print_info "Results saved to:"
echo "  - JSON: $RESULTS_DIR/nvidia_npp_${TIMESTAMP}.json"
echo ""

print_info "Executable location:"
echo "  - $BUILD_DIR/benchmark/nppi_arithmetic_benchmark"
echo ""

print_info "To run again:"
echo "  cd $BUILD_DIR/benchmark"
echo "  ./nppi_arithmetic_benchmark"
echo ""

print_info "To view results:"
echo "  cat $RESULTS_DIR/nvidia_npp_${TIMESTAMP}.json"
echo ""

# =============================================================================
# 完成
# =============================================================================
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}     Quick Benchmark Completed! 🎉${NC}"
echo -e "${BLUE}========================================${NC}\n"

exit 0
