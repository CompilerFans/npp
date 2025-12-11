#!/bin/bash

echo "开始构建MPP和样本程序..."

# 首先构建MPP库（使用主项目的build.sh）
echo "1. 构建MPP库..."
cd ../../
./build.sh --lib-only

# 构建样本程序
echo "2. 构建NPP样本..."
cd examples/cuda-npp-samples
mkdir -p build
cd build
cmake ..
make -j$(nproc)

echo " 构建完成！"
