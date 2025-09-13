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
USE_NVIDIA_NPP="OFF"

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
            echo "Cleaning build directory..."
            rm -rf build/
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
            echo "  --use-nvidia-npp Use NVIDIA NPP library for tests instead of OpenNPP"
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

# Create build directory
mkdir -p build
cd build

# Configure CMake
echo "Configuring CMake ($BUILD_TYPE mode)..."
echo "BUILD_TESTS=$BUILD_TESTS, BUILD_EXAMPLES=$BUILD_EXAMPLES, WARNINGS_AS_ERRORS=$WARNINGS_AS_ERRORS, USE_NVIDIA_NPP=$USE_NVIDIA_NPP"
cmake .. \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_CUDA_ARCHITECTURES="89" \
    -DBUILD_TESTS="$BUILD_TESTS" \
    -DBUILD_EXAMPLES="$BUILD_EXAMPLES" \
    -DNPP_WARNINGS_AS_ERRORS="$WARNINGS_AS_ERRORS" \
    -DUSE_NVIDIA_NPP="$USE_NVIDIA_NPP"

# Build
echo "Building with $JOBS jobs..."
make -j"$JOBS"

echo "Build completed successfully!"