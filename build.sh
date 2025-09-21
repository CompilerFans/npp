#!/bin/bash

# Simple build script for NPP Library

set -e

echo "=== NPP Library Build Script ==="

# Default values
BUILD_TYPE="Release"
JOBS=$(nproc)
BUILD_TESTS="ON"
BUILD_EXAMPLES="ON"
WARNINGS_AS_ERRORS="ON"
# Check environment variable first, then use default
USE_NVIDIA_NPP="${USE_NVIDIA_NPP:-OFF}"
BUILD_DIR="build"  # Default build directory

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
            echo "  --no-tests      Skip building tests"
            echo "  --no-examples   Skip building examples"
            echo "  --lib-only      Build library only (no tests, no examples)"
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

# Configure CMake
echo "Configuring CMake ($BUILD_TYPE mode)..."
echo "BUILD_TESTS=$BUILD_TESTS, BUILD_EXAMPLES=$BUILD_EXAMPLES, WARNINGS_AS_ERRORS=$WARNINGS_AS_ERRORS, USE_NVIDIA_NPP=$USE_NVIDIA_NPP"
cmake -S .\
    -B $BUILD_DIR\
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_CUDA_ARCHITECTURES="89" \
    -DBUILD_TESTS="$BUILD_TESTS" \
    -DBUILD_EXAMPLES="$BUILD_EXAMPLES" \
    -DNPP_WARNINGS_AS_ERRORS="$WARNINGS_AS_ERRORS" \
    -DUSE_NVIDIA_NPP="$USE_NVIDIA_NPP" \
    -G Ninja

# Build
echo "Building ..."
cmake --build $BUILD_DIR

echo "Build completed successfully!"
