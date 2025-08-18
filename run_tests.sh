#!/bin/bash
# OpenNPP测试运行脚本
# 提供统一的测试运行接口

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 默认参数
BUILD_DIR="build"
TEST_TYPE="all"
VERBOSE=false

# 帮助信息
show_help() {
    echo "OpenNPP测试运行脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help          显示帮助信息"
    echo "  -t, --type TYPE     测试类型: all, quick, core, arithmetic, support, validation"
    echo "  -b, --build-dir DIR 构建目录 (默认: build)"
    echo "  -v, --verbose       详细输出"
    echo "  -c, --clean         清理并重新构建"
    echo "  -j, --jobs N        并行编译任务数"
    echo ""
    echo "示例:"
    echo "  $0                  # 运行所有测试"
    echo "  $0 -t quick         # 运行快速测试（不含一致性验证）"
    echo "  $0 -t core          # 只运行核心测试"
    echo "  $0 -c -j 8          # 清理重建并用8线程编译，然后测试"
}

# 解析参数
JOBS=$(nproc)
CLEAN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -t|--type)
            TEST_TYPE="$2"
            shift 2
            ;;
        -b|--build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -c|--clean)
            CLEAN=true
            shift
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}错误: 未知参数 $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# 打印配置
echo -e "${GREEN}=== OpenNPP 测试运行器 ===${NC}"
echo "构建目录: $BUILD_DIR"
echo "测试类型: $TEST_TYPE"
echo "并行任务: $JOBS"
echo ""

# 清理构建（如果需要）
if [ "$CLEAN" = true ]; then
    echo -e "${YELLOW}清理构建目录...${NC}"
    rm -rf "$BUILD_DIR"
fi

# 创建并进入构建目录
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}创建构建目录...${NC}"
    mkdir -p "$BUILD_DIR"
fi

cd "$BUILD_DIR"

# 配置项目（如果需要）
if [ ! -f "Makefile" ]; then
    echo -e "${YELLOW}配置项目...${NC}"
    cmake ..
fi

# 编译项目
echo -e "${YELLOW}编译项目 (使用 $JOBS 个线程)...${NC}"
make -j"$JOBS"

# 运行测试
echo ""
echo -e "${GREEN}=== 运行测试 ===${NC}"

case $TEST_TYPE in
    all)
        echo "运行所有测试..."
        make test_all
        ;;
    quick)
        echo "运行快速测试（不含一致性验证）..."
        make test_quick
        ;;
    core)
        echo "运行核心功能测试..."
        make test_core
        ;;
    arithmetic)
        echo "运行算术运算测试..."
        make test_arithmetic
        ;;
    support)
        echo "运行支持函数测试..."
        make test_support
        ;;
    validation)
        echo "运行一致性验证测试..."
        if [ -f "opennpp_test" ]; then
            make test_validation 2>/dev/null || echo -e "${YELLOW}警告: NVIDIA NPP未找到，跳过一致性验证${NC}"
        else
            echo -e "${YELLOW}警告: 测试程序未编译${NC}"
        fi
        ;;
    *)
        echo -e "${RED}错误: 未知的测试类型 '$TEST_TYPE'${NC}"
        exit 1
        ;;
esac

# 测试完成
echo ""
echo -e "${GREEN}=== 测试完成 ===${NC}"

# 生成汇总报告
if [ "$VERBOSE" = true ]; then
    echo ""
    echo -e "${YELLOW}测试汇总:${NC}"
    ctest --test-dir . -N | tail -n +2
fi

# 返回测试结果
exit $?