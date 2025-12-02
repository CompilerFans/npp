#!/bin/bash

# Simple build script for NPP Library

echo "=== NPP Library Build Script ==="

# Default values
BUILD_TYPE="Release"
JOBS=$(nproc)
BUILD_TESTS="ON"
BUILD_EXAMPLES="ON"
BUILD_SHARED_LIBS="OFF"
WARNINGS_AS_ERRORS="ON"
# Check environment variable first, then use default
USE_NVIDIA_NPP="OFF"
BUILD_DIR="build"  # Default build directory
CUDA_ARCH=""  # Empty means auto-detect

# Auto-detect GPU architecture
detect_cuda_arch() {
    if command -v nvidia-smi &> /dev/null; then
        # Get compute capability from nvidia-smi (e.g., "8.6" -> "86")
        local compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
        if [ -n "$compute_cap" ]; then
            # Remove the dot: "8.6" -> "86"
            echo "${compute_cap//./}"
            return
        fi
    fi
    # Fallback: try nvcc with __CUDA_ARCH_LIST__
    if command -v nvcc &> /dev/null; then
        # Common fallback architectures
        echo "75"
        return
    fi
    # Default fallback
    echo "75"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        -j)
            JOBS="$2"
            shift 2
            ;;
        --no-tests)
            BUILD_TESTS="OFF"
            shift
            ;;
        --no-examples)
            BUILD_EXAMPLES="OFF"
            shift
            ;;
        --lib-only)
            BUILD_TESTS="OFF"
            BUILD_EXAMPLES="OFF"
            shift
            ;;
        --no-werror)
            WARNINGS_AS_ERRORS="OFF"
            shift
            ;;
        --use-nvidia-npp)
            USE_NVIDIA_NPP="ON"
            shift
            ;;
        --shared)
            BUILD_SHARED_LIBS="ON"
            shift
            ;;
        --arch)
            CUDA_ARCH="$2"
            shift 2
            ;;
        clean)
            echo "Cleaning build directories..."
            rm -rf build/ build-nvidia/
            echo "Clean completed"
            exit 0
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -d, --debug     Debug build"
            echo "  -j N            Use N parallel jobs"
            echo "  --arch N        Set CUDA architecture (e.g., 75, 86, 89). Auto-detected if not specified."
            echo "  --no-tests      Skip building tests"
            echo "  --no-examples   Skip building examples"
            echo "  --lib-only      Build library only (no tests, no examples)"
            echo "  --shared        Build shared libraries (.so/.dll) instead of static (.a/.lib)"
            echo "  --no-werror     Disable warnings as errors"
            echo "  --use-nvidia-npp Use NVIDIA NPP library for tests instead of MPP"
            echo "  clean           Clean build directory"
            echo "  -h, --help      Show help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set build directory based on NPP library choice
if [ "$USE_NVIDIA_NPP" = "ON" ]; then
    BUILD_DIR="build-nvidia"
    NPP_LIB_NAME="NVIDIA NPP"
else
    BUILD_DIR="build"
    NPP_LIB_NAME="MPP"
fi

echo "Using $NPP_LIB_NAME library, build directory: $BUILD_DIR"

# Auto-detect CUDA architecture if not specified
if [ -z "$CUDA_ARCH" ]; then
    CUDA_ARCH=$(detect_cuda_arch)
    echo "Auto-detected CUDA architecture: $CUDA_ARCH"
else
    echo "Using specified CUDA architecture: $CUDA_ARCH"
fi

# Configure CMake
echo "Configuring CMake ($BUILD_TYPE mode)..."
echo "BUILD_TESTS=$BUILD_TESTS, BUILD_EXAMPLES=$BUILD_EXAMPLES, BUILD_SHARED_LIBS=$BUILD_SHARED_LIBS, WARNINGS_AS_ERRORS=$WARNINGS_AS_ERRORS, USE_NVIDIA_NPP=$USE_NVIDIA_NPP"
cmake -S .\
    -B $BUILD_DIR\
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DBUILD_TESTS="$BUILD_TESTS" \
    -DBUILD_EXAMPLES="$BUILD_EXAMPLES" \
    -DBUILD_SHARED_LIBS="$BUILD_SHARED_LIBS" \
    -DNPP_WARNINGS_AS_ERRORS="$WARNINGS_AS_ERRORS" \
    -DUSE_NVIDIA_NPP="$USE_NVIDIA_NPP" \
    -G Ninja

# Build
echo "Building ..."
cmake --build $BUILD_DIR

echo "Build completed successfully!"
