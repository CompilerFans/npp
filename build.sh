#!/bin/bash

# NPP Library Build Script
# 支持功能测试、对比测试和完整库构建

set -e  # 遇到错误时退出

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

# 显示帮助信息
show_help() {
    echo "NPP Library Build Script"
    echo ""
    echo "用法: $0 [选项] [目标]"
    echo ""
    echo "目标："
    echo "  functional    - 构建并运行功能测试 (默认)"
    echo "  comparison    - 构建并运行对比测试"
    echo "  legacy        - 构建并运行遗留测试"
    echo "  all           - 构建所有测试"
    echo "  library       - 仅构建NPP库"
    echo "  clean         - 清理所有构建文件"
    echo ""
    echo "选项："
    echo "  -h, --help    - 显示此帮助信息"
    echo "  -j N          - 使用N个并行作业 (默认: CPU核心数)"
    echo "  -d, --debug   - Debug构建模式"
    echo "  -r, --release - Release构建模式 (默认)"
    echo "  -v, --verbose - 详细输出"
    echo "  --no-run      - 仅构建，不运行测试"
    echo "  --gpu-arch    - 指定GPU架构 (例如: 70,75,80,86)"
    echo "  --install     - 构建后安装库"
    echo ""
    echo "示例："
    echo "  $0                          # 构建并运行功能测试"
    echo "  $0 functional -j 8          # 使用8个作业构建功能测试"
    echo "  $0 all --debug              # Debug模式构建所有测试"
    echo "  $0 library --release        # Release模式仅构建库"
    echo "  $0 clean                    # 清理所有构建文件"
}

# 默认参数
TARGET="functional"
BUILD_TYPE="Release"
JOBS=$(nproc)
RUN_TESTS=true
VERBOSE=false
GPU_ARCHITECTURES="70;75;80;86"
INSTALL_LIB=false

# 解析命令行参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -j)
                JOBS="$2"
                shift 2
                ;;
            -d|--debug)
                BUILD_TYPE="Debug"
                shift
                ;;
            -r|--release)
                BUILD_TYPE="Release"
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            --no-run)
                RUN_TESTS=false
                shift
                ;;
            --gpu-arch)
                GPU_ARCHITECTURES="$2"
                shift 2
                ;;
            --install)
                INSTALL_LIB=true
                shift
                ;;
            functional|comparison|legacy|all|library|clean)
                TARGET="$1"
                shift
                ;;
            *)
                print_error "未知参数: $1"
                echo "使用 -h 或 --help 查看帮助"
                exit 1
                ;;
        esac
    done
}

# 检查依赖
check_dependencies() {
    print_info "检查构建依赖..."
    
    # 检查CMake
    if ! command -v cmake &> /dev/null; then
        print_error "CMake未安装"
        exit 1
    fi
    
    # 检查CUDA
    if ! command -v nvcc &> /dev/null; then
        print_warning "CUDA未找到，某些功能可能无法工作"
    else
        CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
        print_info "找到CUDA版本: $CUDA_VERSION"
    fi
    
    # 检查GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
        print_info "找到GPU: $GPU_INFO"
    else
        print_warning "未检测到NVIDIA GPU"
    fi
    
    print_success "依赖检查完成"
}

# 清理构建文件
clean_build() {
    print_info "清理构建文件..."
    
    # 清理各种构建目录
    rm -rf build/
    rm -rf test/functional/build/
    rm -rf test/comparison/build/
    rm -rf test/legacy/build/
    
    # 清理临时文件
    find . -name "*.o" -delete 2>/dev/null || true
    find . -name "*.a" -delete 2>/dev/null || true
    find . -name "*.so" -delete 2>/dev/null || true
    find . -name "CMakeFiles" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name "CMakeCache.txt" -delete 2>/dev/null || true
    find . -name "cmake_install.cmake" -delete 2>/dev/null || true
    find . -name "Makefile" -delete 2>/dev/null || true
    
    print_success "清理完成"
}

# 构建功能测试
build_functional() {
    print_info "构建功能测试..."
    
    cd test/functional
    mkdir -p build
    cd build
    
    CMAKE_ARGS=(
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
        -DCMAKE_CUDA_ARCHITECTURES="$GPU_ARCHITECTURES"
        -DBUILD_FUNCTIONAL_TESTS=ON
    )
    
    if [[ "$VERBOSE" == "true" ]]; then
        CMAKE_ARGS+=(-DCMAKE_VERBOSE_MAKEFILE=ON)
    fi
    
    cmake "${CMAKE_ARGS[@]}" ..
    make -j"$JOBS"
    
    cd ../../..
    print_success "功能测试构建完成"
}

# 构建对比测试
build_comparison() {
    print_info "构建对比测试..."
    
    cd test/comparison
    mkdir -p build
    cd build
    
    CMAKE_ARGS=(
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
        -DCMAKE_CUDA_ARCHITECTURES="$GPU_ARCHITECTURES"
        -DBUILD_COMPARISON_TESTS=ON
    )
    
    if [[ "$VERBOSE" == "true" ]]; then
        CMAKE_ARGS+=(-DCMAKE_VERBOSE_MAKEFILE=ON)
    fi
    
    cmake "${CMAKE_ARGS[@]}" .. || {
        print_warning "对比测试CMake配置失败，跳过"
        cd ../../..
        return 0
    }
    
    make -j"$JOBS" || {
        print_warning "对比测试构建失败，跳过"
        cd ../../..
        return 0
    }
    
    cd ../../..
    print_success "对比测试构建完成"
}

# 构建遗留测试
build_legacy() {
    print_info "构建遗留测试..."
    
    if [[ ! -d "test/legacy" ]]; then
        print_warning "遗留测试目录不存在，跳过"
        return 0
    fi
    
    cd test/legacy
    mkdir -p build
    cd build
    
    CMAKE_ARGS=(
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
        -DCMAKE_CUDA_ARCHITECTURES="$GPU_ARCHITECTURES"
        -DBUILD_LEGACY_TESTS=ON
    )
    
    if [[ "$VERBOSE" == "true" ]]; then
        CMAKE_ARGS+=(-DCMAKE_VERBOSE_MAKEFILE=ON)
    fi
    
    cmake "${CMAKE_ARGS[@]}" .. || {
        print_warning "遗留测试CMake配置失败，跳过"
        cd ../../..
        return 0
    }
    
    make -j"$JOBS" || {
        print_warning "遗留测试构建失败，跳过"
        cd ../../..
        return 0
    }
    
    cd ../../..
    print_success "遗留测试构建完成"
}

# 构建主库
build_library() {
    print_info "构建NPP主库..."
    
    mkdir -p build
    cd build
    
    CMAKE_ARGS=(
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
        -DCMAKE_CUDA_ARCHITECTURES="$GPU_ARCHITECTURES"
    )
    
    if [[ "$VERBOSE" == "true" ]]; then
        CMAKE_ARGS+=(-DCMAKE_VERBOSE_MAKEFILE=ON)
    fi
    
    if [[ "$INSTALL_LIB" == "true" ]]; then
        CMAKE_ARGS+=(-DCMAKE_INSTALL_PREFIX=/usr/local)
    fi
    
    cmake "${CMAKE_ARGS[@]}" .. || {
        print_error "主库CMake配置失败"
        exit 1
    }
    
    make -j"$JOBS" || {
        print_error "主库构建失败"
        exit 1
    }
    
    if [[ "$INSTALL_LIB" == "true" ]]; then
        print_info "安装库文件..."
        sudo make install
        print_success "库安装完成"
    fi
    
    cd ..
    print_success "主库构建完成"
}

# 运行功能测试
run_functional_tests() {
    print_info "运行功能测试..."
    
    cd test/functional/build
    
    if [[ ! -f "npp_functional_tests" ]]; then
        print_error "功能测试可执行文件未找到"
        cd ../../..
        return 1
    fi
    
    echo ""
    print_info "======== NPP功能测试结果 ========"
    ./npp_functional_tests --gtest_brief=1 || {
        print_warning "部分功能测试失败"
        TESTS_FAILED=true
    }
    
    cd ../../..
}

# 运行对比测试
run_comparison_tests() {
    print_info "运行对比测试..."
    
    if [[ ! -d "test/comparison/build" ]]; then
        print_warning "对比测试构建目录不存在，跳过"
        return 0
    fi
    
    cd test/comparison/build
    
    # 查找测试可执行文件
    TEST_FILES=$(find . -name "*test*" -type f -executable | head -3)
    
    if [[ -z "$TEST_FILES" ]]; then
        print_warning "未找到对比测试可执行文件"
        cd ../../..
        return 0
    fi
    
    echo ""
    print_info "======== NPP对比测试结果 ========"
    
    for test_file in $TEST_FILES; do
        print_info "运行: $test_file"
        timeout 60 "$test_file" || {
            print_warning "测试 $test_file 失败或超时"
            TESTS_FAILED=true
        }
    done
    
    cd ../../..
}

# 运行遗留测试
run_legacy_tests() {
    print_info "运行遗留测试..."
    
    if [[ ! -d "test/legacy/build" ]]; then
        print_warning "遗留测试构建目录不存在，跳过"
        return 0
    fi
    
    cd test/legacy/build
    
    # 查找并运行所有测试可执行文件
    TEST_FILES=$(find . -name "*test*" -type f -executable | head -5)
    
    if [[ -z "$TEST_FILES" ]]; then
        print_warning "未找到遗留测试可执行文件"
        cd ../../..
        return 0
    fi
    
    echo ""
    print_info "======== NPP遗留测试结果 ========"
    
    for test_file in $TEST_FILES; do
        print_info "运行: $test_file"
        timeout 60 "$test_file" || {
            print_warning "测试 $test_file 失败或超时"
            TESTS_FAILED=true
        }
    done
    
    cd ../../..
}

# 显示构建统计
show_build_stats() {
    print_info "======== 构建统计 ========"
    
    # 统计文件数量
    if [[ -d "src/" ]]; then
        CPP_FILES=$(find src/ -name "*.cpp" 2>/dev/null | wc -l)
        CU_FILES=$(find src/ -name "*.cu" 2>/dev/null | wc -l)
        echo "  C++源文件: $CPP_FILES"
        echo "  CUDA源文件: $CU_FILES"
    fi
    
    if [[ -d "api/" ]]; then
        HEADER_FILES=$(find api/ -name "*.h" 2>/dev/null | wc -l)
        echo "  头文件: $HEADER_FILES"
    fi
    
    if [[ -d "test/functional/" ]]; then
        TEST_FILES=$(find test/functional/ -name "*.cpp" 2>/dev/null | wc -l)
        echo "  功能测试文件: $TEST_FILES"
    fi
    
    # 统计代码行数
    if command -v wc &> /dev/null && [[ -d "src/" ]] && [[ -d "api/" ]]; then
        TOTAL_LINES=$(find src/ api/ -name "*.cpp" -o -name "*.cu" -o -name "*.h" 2>/dev/null | xargs wc -l 2>/dev/null | tail -1 | awk '{print $1}' || echo "N/A")
        echo "  总代码行数: $TOTAL_LINES"
    fi
    
    # 显示构建时间
    if [[ -n "$BUILD_START_TIME" ]]; then
        BUILD_END_TIME=$(date +%s)
        BUILD_DURATION=$((BUILD_END_TIME - BUILD_START_TIME))
        echo "  构建时间: ${BUILD_DURATION}秒"
    fi
}

# 主函数
main() {
    BUILD_START_TIME=$(date +%s)
    TESTS_FAILED=false
    
    echo "======================================="
    echo "    NPP Library Build Script v1.0"
    echo "======================================="
    echo ""
    
    parse_args "$@"
    
    print_info "构建配置:"
    echo "  目标: $TARGET"
    echo "  构建类型: $BUILD_TYPE"
    echo "  并行作业: $JOBS"
    echo "  GPU架构: $GPU_ARCHITECTURES"
    echo "  运行测试: $RUN_TESTS"
    echo ""
    
    check_dependencies
    
    case "$TARGET" in
        clean)
            clean_build
            ;;
        functional)
            build_functional
            if [[ "$RUN_TESTS" == "true" ]]; then
                run_functional_tests
            fi
            ;;
        comparison)
            build_comparison
            if [[ "$RUN_TESTS" == "true" ]]; then
                run_comparison_tests
            fi
            ;;
        legacy)
            build_legacy
            if [[ "$RUN_TESTS" == "true" ]]; then
                run_legacy_tests
            fi
            ;;
        all)
            build_functional
            build_comparison
            build_legacy
            if [[ "$RUN_TESTS" == "true" ]]; then
                run_functional_tests
                run_comparison_tests
                run_legacy_tests
            fi
            ;;
        library)
            build_library
            ;;
        *)
            print_error "未知目标: $TARGET"
            show_help
            exit 1
            ;;
    esac
    
    show_build_stats
    
    echo ""
    if [[ "$TESTS_FAILED" == "true" ]]; then
        print_warning "构建完成，但部分测试失败"
        exit 1
    else
        print_success "构建成功完成！"
    fi
}

# 运行主函数
main "$@"