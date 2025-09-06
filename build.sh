#!/bin/bash

# Simple build script for NPP Library

set -e

echo "=== NPP Library Build Script ==="

# Default values
BUILD_TYPE="Release"
JOBS=$(nproc)

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
cmake .. \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86"

# Build
echo "Building with $JOBS jobs..."
make -j"$JOBS"

echo "Build completed successfully!"