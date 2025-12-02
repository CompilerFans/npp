#!/bin/bash
# 编译NVIDIA NPP LUT提取工具

# 查找CUDA路径
if [ -d "/usr/local/cuda" ]; then
    CUDA_PATH="/usr/local/cuda"
elif [ -d "/usr/local/cuda-12.6" ]; then
    CUDA_PATH="/usr/local/cuda-12.6"
else
    echo "Error: CUDA not found"
    exit 1
fi

echo "Using CUDA from: $CUDA_PATH"

# 编译
g++ extract_nvidia_gamma_lut.cpp \
    -I${CUDA_PATH}/include \
    -L${CUDA_PATH}/lib64 \
    -L${CUDA_PATH}/targets/x86_64-linux/lib \
    -lcudart \
    -lnppc \
    -lnppial \
    -lnppicc \
    -lnppidei \
    -lnppif \
    -lnppig \
    -lnppim \
    -lnppist \
    -lnppisu \
    -lnppitc \
    -o extract_nvidia_gamma_lut

if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    echo "Run with: ./extract_nvidia_gamma_lut"
else
    echo "Compilation failed!"
    exit 1
fi
