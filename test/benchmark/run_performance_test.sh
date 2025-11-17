#!/bin/bash

###############################################################################
# NPP Performance Testing Script
# 
# 用途：自动化测试 MPP vs NVIDIA NPP 的性能对比
# 使用：./run_performance_test.sh [项目路径]
#
# 示例：
#   ./run_performance_test.sh                  # 使用当前目录
#   ./run_performance_test.sh /home/jzh/npp    # 指定项目路径
###############################################################################

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_section() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  $1${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
}

# 获取项目路径
if [ -z "$1" ]; then
    PROJECT_DIR=$(pwd)
    print_info "使用当前目录: $PROJECT_DIR"
else
    PROJECT_DIR="$1"
    print_info "使用指定目录: $PROJECT_DIR"
fi

# 检查目录是否存在
if [ ! -d "$PROJECT_DIR" ]; then
    print_error "目录不存在: $PROJECT_DIR"
    exit 1
fi

cd "$PROJECT_DIR"

# 检查是否是 git 仓库
if [ ! -d ".git" ]; then
    print_error "这不是一个 Git 仓库"
    exit 1
fi

# 结果目录
RESULTS_DIR="$PROJECT_DIR/benchmark_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p "$RESULTS_DIR"

# 测试参数
REPETITIONS=5
MIN_TIME=1.0

print_section "Step 0: 环境检查"

# 检查必要的工具
print_info "检查必要的工具..."

if ! command -v git &> /dev/null; then
    print_error "Git 未安装"
    exit 1
fi

if ! command -v cmake &> /dev/null; then
    print_error "CMake 未安装"
    exit 1
fi

if ! command -v nvcc &> /dev/null; then
    print_error "CUDA 未安装或未设置环境变量"
    exit 1
else
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9.]*\).*/\1/p')
    print_success "检测到 CUDA 版本: $CUDA_VERSION"
fi

# 检查 GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    print_success "检测到 GPU: $GPU_INFO"
else
    print_warning "nvidia-smi 不可用，无法检测 GPU"
fi

print_section "Step 1: 更新代码"

# 保存当前分支
CURRENT_BRANCH=$(git branch --show-current)
print_info "当前分支: $CURRENT_BRANCH"

# 检查是否有未提交的修改
if ! git diff-index --quiet HEAD --; then
    print_warning "检测到未提交的修改，将暂存这些修改"
    git stash push -m "Auto stash before performance test - $TIMESTAMP"
    STASHED=true
else
    STASHED=false
fi

# 拉取最新代码
print_info "拉取最新代码..."
git pull origin "$CURRENT_BRANCH"

print_section "Step 2: 测试 MPP 版本"

print_info "清理旧的 build 目录..."
rm -rf build
mkdir -p build
cd build

print_info "配置 MPP 版本 (USE_NVIDIA_NPP=OFF)..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_BENCHMARKS=ON \
    -DUSE_NVIDIA_NPP=OFF

print_info "编译 MPP 版本..."
CPU_CORES=$(nproc)
make nppi_arithmetic_benchmark -j"$CPU_CORES"

if [ ! -f "benchmark/nppi_arithmetic_benchmark" ]; then
    print_error "MPP 版本编译失败"
    exit 1
fi

print_success "MPP 版本编译成功"

print_info "运行 MPP 性能测试..."
cd benchmark
./nppi_arithmetic_benchmark \
    --benchmark_out="$RESULTS_DIR/mpp_results_${TIMESTAMP}.json" \
    --benchmark_out_format=json \
    --benchmark_repetitions="$REPETITIONS" \
    --benchmark_min_time="$MIN_TIME" \
    --benchmark_report_aggregates_only=true \
    | tee "$RESULTS_DIR/mpp_output_${TIMESTAMP}.txt"

print_success "MPP 测试完成"

print_section "Step 3: 测试 NVIDIA NPP 版本"

cd "$PROJECT_DIR"
print_info "清理 build 目录..."
rm -rf build
mkdir -p build
cd build

print_info "配置 NVIDIA NPP 版本 (USE_NVIDIA_NPP=ON)..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_BENCHMARKS=ON \
    -DUSE_NVIDIA_NPP=ON

print_info "编译 NVIDIA NPP 版本..."
make nppi_arithmetic_benchmark -j"$CPU_CORES"

if [ ! -f "benchmark/nppi_arithmetic_benchmark" ]; then
    print_error "NVIDIA NPP 版本编译失败"
    exit 1
fi

print_success "NVIDIA NPP 版本编译成功"

print_info "运行 NVIDIA NPP 性能测试..."
cd benchmark
./nppi_arithmetic_benchmark \
    --benchmark_out="$RESULTS_DIR/nvidia_results_${TIMESTAMP}.json" \
    --benchmark_out_format=json \
    --benchmark_repetitions="$REPETITIONS" \
    --benchmark_min_time="$MIN_TIME" \
    --benchmark_report_aggregates_only=true \
    | tee "$RESULTS_DIR/nvidia_output_${TIMESTAMP}.txt"

print_success "NVIDIA NPP 测试完成"

print_section "Step 4: 生成对比报告"

cd "$PROJECT_DIR"

# 创建对比报告
REPORT_FILE="$RESULTS_DIR/comparison_report_${TIMESTAMP}.txt"

cat > "$REPORT_FILE" << EOF
================================================================================
NPP Performance Comparison Report
================================================================================

测试时间: $(date)
CUDA 版本: $CUDA_VERSION
GPU: $GPU_INFO
测试重复次数: $REPETITIONS
最小测试时间: $MIN_TIME 秒

================================================================================
测试结果文件
================================================================================
MPP JSON:        mpp_results_${TIMESTAMP}.json
NVIDIA NPP JSON: nvidia_results_${TIMESTAMP}.json
MPP 输出:        mpp_output_${TIMESTAMP}.txt
NVIDIA NPP 输出: nvidia_output_${TIMESTAMP}.txt

================================================================================
快速对比 (1920x1080)
================================================================================

EOF

# 如果安装了 jq，添加详细对比
if command -v jq &> /dev/null; then
    echo "MPP Results (1920x1080):" >> "$REPORT_FILE"
    jq -r '.benchmarks[] | select(.name | contains("1920/1080")) | "  \(.name): \(.real_time) ms"' \
        "$RESULTS_DIR/mpp_results_${TIMESTAMP}.json" >> "$REPORT_FILE" || echo "  (解析失败)" >> "$REPORT_FILE"
    
    echo "" >> "$REPORT_FILE"
    echo "NVIDIA NPP Results (1920x1080):" >> "$REPORT_FILE"
    jq -r '.benchmarks[] | select(.name | contains("1920/1080")) | "  \(.name): \(.real_time) ms"' \
        "$RESULTS_DIR/nvidia_results_${TIMESTAMP}.json" >> "$REPORT_FILE" || echo "  (解析失败)" >> "$REPORT_FILE"
    
    echo "" >> "$REPORT_FILE"
    echo "提示: 安装 jq 可获得更详细的对比分析" >> "$REPORT_FILE"
else
    echo "未安装 jq 工具，无法自动生成详细对比" >> "$REPORT_FILE"
    echo "安装方法: sudo apt-get install jq" >> "$REPORT_FILE"
fi

cat >> "$REPORT_FILE" << EOF

================================================================================
结果目录
================================================================================
$RESULTS_DIR

查看详细结果:
  cat $RESULTS_DIR/mpp_output_${TIMESTAMP}.txt
  cat $RESULTS_DIR/nvidia_output_${TIMESTAMP}.txt

================================================================================
EOF

print_success "对比报告已生成"

# 显示报告
cat "$REPORT_FILE"

print_section "完成"

print_success "所有测试完成！"
print_info "结果保存在: $RESULTS_DIR"
echo ""
print_info "文件列表:"
ls -lh "$RESULTS_DIR"/*"${TIMESTAMP}"* | awk '{print "  " $9 " (" $5 ")"}'

# 恢复 stash（如果有）
if [ "$STASHED" = true ]; then
    echo ""
    print_warning "之前暂存的修改可以使用以下命令恢复:"
    echo "  git stash pop"
fi

echo ""
print_info "提示: 可以使用以下命令对比 JSON 结果:"
echo "  # 安装 jq (如果未安装)"
echo "  sudo apt-get install jq"
echo ""
echo "  # 查看 MPP 结果"
echo "  jq '.benchmarks[] | select(.name | contains(\"1920/1080\"))' $RESULTS_DIR/mpp_results_${TIMESTAMP}.json"
echo ""
echo "  # 查看 NVIDIA NPP 结果"
echo "  jq '.benchmarks[] | select(.name | contains(\"1920/1080\"))' $RESULTS_DIR/nvidia_results_${TIMESTAMP}.json"
echo ""

exit 0
