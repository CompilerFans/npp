#!/bin/bash
# 编译NVIDIA NPP AC4 Alpha通道行为测试工具

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
g++ test_nvidia_ac4_alpha.cpp \
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
    -o test_nvidia_ac4_alpha

if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    echo "Run with: ./test_nvidia_ac4_alpha"
else
    echo "Compilation failed!"
    exit 1
fi
