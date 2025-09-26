#!/bin/bash

echo "开始构建MPP和样本程序..."

# 首先构建MPP库（不编译测试）
echo "1. 构建MPP库..."
cd ../../
mkdir -p build
cd build
cmake -DBUILD_TESTING=OFF ..
make npp -j$(nproc)

# 构建样本程序
echo "2. 构建NPP样本..."
cd ../examples/cuda-npp-samples
mkdir -p build
cd build
cmake ..
make -j$(nproc)

echo " 构建完成！"
