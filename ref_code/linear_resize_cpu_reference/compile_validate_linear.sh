#!/bin/bash

set -e

CUDA_PATH=${CUDA_PATH:-/usr/local/cuda}

echo "Compiling linear interpolation CPU validation test..."
echo "CUDA Path: $CUDA_PATH"

g++ -std=c++11 -O2 \
    -I$CUDA_PATH/include \
    -I. \
    -L$CUDA_PATH/lib64 \
    -o validate_linear_cpu \
    validate_linear_cpu.cpp \
    -lcudart -lnppc -lnppial -lnppig

echo "Compilation successful!"
echo "Run with: ./validate_linear_cpu"
