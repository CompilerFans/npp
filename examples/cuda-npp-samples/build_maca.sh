#!/bin/bash

echo "开始构建MPP和样本程序 (MACA)..."

# 保存当前目录
SAMPLES_DIR=$(pwd)

# 首先构建MPP库（使用主项目的build_maca.sh）
echo "1. 构建MPP库..."
cd ../../
source build_maca.sh --lib-only

# 构建样本程序
echo "2. 构建NPP样本..."
cd "$SAMPLES_DIR"
mkdir -p build
cd build

# 使用 cmake_maca 配置
cmake_maca -S .. \
    -B . \
    -DCMAKE_BUILD_TYPE="Release" \
    -DCMAKE_CUDA_ARCHITECTURES="89" \
    -G Ninja

# 使用 cmake_maca 构建
cmake_maca --build .

echo "构建完成！"