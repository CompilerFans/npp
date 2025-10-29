#!/bin/bash

set -e

CUDA_PATH=${CUDA_PATH:-/usr/local/cuda}

echo "Compiling linear interpolation analysis test..."
echo "CUDA Path: $CUDA_PATH"

g++ -std=c++11 -O2 \
    -I$CUDA_PATH/include \
    -L$CUDA_PATH/lib64 \
    -o test_linear_analysis \
    test_linear_analysis.cpp \
    -lcudart -lnppc -lnppial -lnppig

echo "Compilation successful!"
echo "Run with: ./test_linear_analysis"
