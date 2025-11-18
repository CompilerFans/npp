#!/bin/bash
# 一键编译和运行 NPP Benchmark
# 兼容所有 CUDA 版本和服务器环境

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   NPP Benchmark 一键编译运行脚本${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# ============================================
# 1. 检查环境
# ============================================
echo -e "${YELLOW}[1/6] 检查环境...${NC}"

# 检查 CMake
if ! command -v cmake &> /dev/null; then
    echo -e "${RED}错误: 未找到 cmake${NC}"
    exit 1
fi
CMAKE_VERSION=$(cmake --version | head -n1)
echo -e "${GREEN}✓ CMake: $CMAKE_VERSION${NC}"

# 检查 CUDA
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}错误: 未找到 nvcc (CUDA)${NC}"
    exit 1
fi
CUDA_VERSION=$(nvcc -V | grep "release" | sed 's/.*release //' | sed 's/,.*//')
echo -e "${GREEN}✓ CUDA: $CUDA_VERSION${NC}"

# 检查 C++ 编译器
if ! command -v g++ &> /dev/null && ! command -v c++ &> /dev/null; then
    echo -e "${RED}错误: 未找到 C++ 编译器${NC}"
    exit 1
fi
CXX_VERSION=$(g++ --version 2>/dev/null | head -n1 || c++ --version | head -n1)
echo -e "${GREEN}✓ C++: $CXX_VERSION${NC}"

echo ""

# ============================================
# 2. 更新代码
# ============================================
echo -e "${YELLOW}[2/6] 更新代码...${NC}"

# 检查是否在 git 仓库中
if [ -d ".git" ]; then
    echo "正在拉取最新代码..."
    git pull || echo -e "${YELLOW}警告: git pull 失败，继续使用本地代码${NC}"
    
    # 更新 submodules
    if [ -f ".gitmodules" ]; then
        echo "正在更新 submodules..."
        git submodule update --init --recursive || echo -e "${YELLOW}警告: submodule 更新失败${NC}"
    fi
else
    echo -e "${YELLOW}不在 git 仓库中，跳过更新${NC}"
fi

echo ""

# ============================================
# 3. 清理旧的编译结果
# ============================================
echo -e "${YELLOW}[3/6] 清理旧的编译结果...${NC}"

if [ -d "build" ]; then
    echo "删除旧的 build 目录..."
    rm -rf build
fi

mkdir -p build
echo -e "${GREEN}✓ 创建新的 build 目录${NC}"

echo ""

# ============================================
# 4. CMake 配置
# ============================================
echo -e "${YELLOW}[4/6] CMake 配置...${NC}"

cd build

# 选择构建类型
BUILD_TYPE="Release"
USE_NPP="ON"

echo "配置选项:"
echo "  - 构建类型: $BUILD_TYPE"
echo "  - 使用 NVIDIA NPP: $USE_NPP"
echo ""

cmake .. \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DBUILD_BENCHMARKS=ON \
    -DUSE_NVIDIA_NPP=$USE_NPP \
    2>&1 | tee cmake_output.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}错误: CMake 配置失败！${NC}"
    echo "查看详细日志: build/cmake_output.log"
    exit 1
fi

echo -e "${GREEN}✓ CMake 配置成功${NC}"
echo ""

# ============================================
# 5. 编译
# ============================================
echo -e "${YELLOW}[5/6] 编译 benchmark...${NC}"

# 获取 CPU 核心数
if command -v nproc &> /dev/null; then
    CORES=$(nproc)
elif command -v sysctl &> /dev/null; then
    CORES=$(sysctl -n hw.ncpu)
else
    CORES=4
fi

echo "使用 $CORES 个核心并行编译..."

make nppi_arithmetic_benchmark -j$CORES 2>&1 | tee build.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}错误: 编译失败！${NC}"
    echo "查看详细日志: build/build.log"
    
    # 提供诊断信息
    echo ""
    echo -e "${YELLOW}诊断信息:${NC}"
    echo "1. 检查 CUDA 和 NPP 库是否安装："
    echo "   ls -la /usr/local/cuda*/lib64/libnpp*.so"
    echo ""
    echo "2. 检查链接命令："
    echo "   cat build/test/benchmark/CMakeFiles/nppi_arithmetic_benchmark.dir/link.txt"
    echo ""
    echo "3. 查看详细编译输出："
    echo "   cd build && make nppi_arithmetic_benchmark VERBOSE=1"
    
    exit 1
fi

echo -e "${GREEN}✓ 编译成功${NC}"
echo ""

# ============================================
# 6. 运行 benchmark
# ============================================
echo -e "${YELLOW}[6/6] 运行 benchmark...${NC}"

BENCHMARK_BIN="benchmark/nppi_arithmetic_benchmark"

if [ ! -f "$BENCHMARK_BIN" ]; then
    echo -e "${RED}错误: 找不到 benchmark 可执行文件${NC}"
    exit 1
fi

echo "运行: ./$BENCHMARK_BIN"
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   Benchmark 结果${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

./$BENCHMARK_BIN

BENCHMARK_EXIT_CODE=$?

echo ""
echo -e "${BLUE}========================================${NC}"

if [ $BENCHMARK_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Benchmark 运行成功！${NC}"
else
    echo -e "${RED}✗ Benchmark 运行失败（退出码: $BENCHMARK_EXIT_CODE）${NC}"
fi

echo -e "${BLUE}========================================${NC}"
echo ""

# ============================================
# 总结
# ============================================
echo -e "${YELLOW}总结:${NC}"
echo "  - CUDA 版本: $CUDA_VERSION"
echo "  - 构建目录: $(pwd)"
echo "  - 可执行文件: $BENCHMARK_BIN"
echo ""
echo "其他运行选项:"
echo "  - 过滤测试: ./$BENCHMARK_BIN --benchmark_filter=<pattern>"
echo "  - 输出 JSON: ./$BENCHMARK_BIN --benchmark_out=result.json --benchmark_out_format=json"
echo "  - 列出测试: ./$BENCHMARK_BIN --benchmark_list_tests"
echo ""

exit $BENCHMARK_EXIT_CODE
