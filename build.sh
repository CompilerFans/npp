#!/bin/bash

# Simple build script for NPP Library

echo "=== NPP Library Build Script ==="

# Default values
JOBS=$(nproc)

BUILD_TYPE="Release"
BUILD_DIR="build"  # Default build directory

BUILD_TESTS="ON"
WARNINGS_AS_ERRORS="ON"

# CUDA/MACA SDK Relative
# Check environment variable first, then use default
USE_NVIDIA_NPP="${USE_NVIDIA_NPP:-OFF}"
GPU_TYPE="METAX_GPU"
WITHOUT_CU_BRIDGE="OFF"
CMAKE_NAME="cmake"

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
        --lib-only)
            BUILD_TESTS="OFF"
            shift
            ;;
        --no-werror)
            WARNINGS_AS_ERRORS="OFF"
            shift
            ;;
        --cuda)
            GPU_TYPE="NVIDIA_GPU"
            shift
            ;;
        --use-nvidia-npp)
            USE_NVIDIA_NPP="ON"
            shift
            ;;
        --maca)
            GPU_TYPE="METAX_GPU"
            CMAKE_NAME="cmake_maca"
            shift
            ;;
        --without-cu-bridge)
            WITHOUT_CU_BRIDGE="ON"
            CMAKE_NAME="cmake"
            shift
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
            echo "  --lib-only      Build library only (no tests, no examples)"
            echo "  --no-werror     Disable warnings as errors"
            echo "  --use-nvidia-npp Use NVIDIA NPP library for tests instead of MPP"
            echo "  --cuda          Build with CUDA SDK"
            echo "  --maca          Build with MACA SDK"
            echo "  --without-cu-bridge          Build with MACA SDK and no CU-BRIDGE dependency"
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

# TODO check combine logic, such as --cuda & --use-nvidia-npp, --maca & --without-cu-bridge
echo "Using $NPP_LIB_NAME library, build directory: $BUILD_DIR"
echo "Configuring CMake ($BUILD_TYPE mode)..."
echo "BUILD_TESTS=$BUILD_TESTS"
echo "WARNINGS_AS_ERRORS=$WARNINGS_AS_ERRORS"
echo "USE_NVIDIA_NPP=$USE_NVIDIA_NPP,CMAKE_NAME=$CMAKE_NAME"
echo "GPU_TYPE=$WARNINGPU_TYPEGS_AS_ERRORS"
echo "WITHOUT_CU_BRIDGE=$WITHOUT_CU_BRIDGE"
echo "CMAKE_NAME=$CMAKE_NAME"

$CMAKE_NAME -S .\
    -B $BUILD_DIR \
    -G Ninja      \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DBUILD_TESTS="$BUILD_TESTS" \
    -DNPP_WARNINGS_AS_ERRORS="$WARNINGS_AS_ERRORS" \
    -DUSE_NVIDIA_NPP="$USE_NVIDIA_NPP" \
    -DGPU_TYPE="$GPU_TYPE" \
    -DWITHOUT_CU_BRIDGE="$WITHOUT_CU_BRIDGE"

# Build
echo "Building ..."
$CMAKE_NAME --build $BUILD_DIR

echo "Build completed successfully!"
