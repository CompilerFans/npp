#!/bin/bash

# CUDA NPP Samples - Standalone Build Script

echo "=== NPP Samples Build Script ==="

# Default values
BUILD_TYPE="Release"
JOBS=$(nproc)
BUILD_SHARED_LIBS="OFF"
USE_NVIDIA_NPP="${USE_NVIDIA_NPP:-OFF}"
WARNINGS_AS_ERRORS="OFF"
BUILD_DIR="build"
CUDA_ARCH="89"
MPP_ROOT_DIR=""
MPP_INCLUDE_DIR=""
MPP_LIBRARY_DIR=""

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
        --shared)
            BUILD_SHARED_LIBS="ON"
            shift
            ;;
        --use-nvidia-npp)
            USE_NVIDIA_NPP="ON"
            BUILD_DIR="build-nvidia"
            shift
            ;;
        --no-werror)
            WARNINGS_AS_ERRORS="OFF"
            shift
            ;;
        --mpp-root)
            MPP_ROOT_DIR="$2"
            shift 2
            ;;
        --mpp-include)
            MPP_INCLUDE_DIR="$2"
            shift 2
            ;;
        --mpp-lib)
            MPP_LIBRARY_DIR="$2"
            shift 2
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
            echo "  -d, --debug         Debug build"
            echo "  -j N                Use N parallel jobs"
            echo "  --shared            Link shared libraries (.so)"
            echo "  --use-nvidia-npp    Use NVIDIA NPP instead of MPP"
            echo "  --no-werror         Disable warnings as errors"
            echo "  --mpp-root DIR      Set MPP root directory"
            echo "  --mpp-include DIR   Set MPP include directory"
            echo "  --mpp-lib DIR       Set MPP library directory"
            echo "  --arch ARCH         Set CUDA architecture (default: 89)"
            echo "  clean               Clean build directories"
            echo "  -h, --help          Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set library name for display
if [ "$USE_NVIDIA_NPP" = "ON" ]; then
    NPP_LIB_NAME="NVIDIA NPP"
else
    NPP_LIB_NAME="MPP"
fi

echo "Using $NPP_LIB_NAME library, build directory: $BUILD_DIR"

# Build CMake arguments
CMAKE_ARGS=(
    -B "$BUILD_DIR"
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
    -DBUILD_SHARED_LIBS="$BUILD_SHARED_LIBS"
    -DNPP_WARNINGS_AS_ERRORS="$WARNINGS_AS_ERRORS"
    -DUSE_NVIDIA_NPP="$USE_NVIDIA_NPP"
    -G Ninja
)

# Add optional MPP paths
if [ -n "$MPP_ROOT_DIR" ]; then
    CMAKE_ARGS+=(-DMPP_ROOT_DIR="$MPP_ROOT_DIR")
fi
if [ -n "$MPP_INCLUDE_DIR" ]; then
    CMAKE_ARGS+=(-DMPP_INCLUDE_DIR="$MPP_INCLUDE_DIR")
fi
if [ -n "$MPP_LIBRARY_DIR" ]; then
    CMAKE_ARGS+=(-DMPP_LIBRARY_DIR="$MPP_LIBRARY_DIR")
fi

# Configure CMake
echo "Configuring CMake ($BUILD_TYPE mode)..."
echo "CUDA_ARCH=$CUDA_ARCH, BUILD_SHARED_LIBS=$BUILD_SHARED_LIBS, USE_NVIDIA_NPP=$USE_NVIDIA_NPP"

cmake -S . "${CMAKE_ARGS[@]}"

if [ $? -ne 0 ]; then
    echo "CMake configuration failed!"
    exit 1
fi

# Build
echo "Building with $JOBS jobs..."
cmake --build "$BUILD_DIR" -j "$JOBS"

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo "Build completed successfully!"
echo "Executables are in: $BUILD_DIR/bin/"
