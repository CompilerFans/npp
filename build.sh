#!/bin/bash

# Simple build script for OpenNPP

echo "=== OpenNPP Build Script ==="

# Create build directory
mkdir -p build
cd build

# Configure CMake
echo "Configuring CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="60;70;75;80"

if [ $? -ne 0 ]; then
    echo "CMake configuration failed"
    exit 1
fi

# Build
echo "Building..."
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo "Build failed"
    exit 1
fi

echo "Build completed successfully!"
echo ""
echo "Available executables:"
echo "  ./test_runner          - Main test runner"
echo "  ./test_nppi_addc       - Basic function test"
echo "  ./test_nppi_addc_validation - Validation against NVIDIA NPP"
echo ""
echo "To run tests:"
echo "  ./test_runner"
echo "  ./test_runner -f nppiAddC_8u_C1RSfs_Ctx -v"