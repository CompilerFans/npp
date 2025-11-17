#!/bin/bash
# 部署到服务器的脚本

SERVER_USER="jzh"
SERVER_HOST="your-server-address"  # 修改为你的服务器地址
SERVER_PATH="~/npp"

echo "正在同步项目到服务器..."

rsync -avz --progress \
    --exclude 'build/' \
    --exclude '.git/' \
    --exclude 'benchmark_results/' \
    --exclude '*.o' \
    --exclude '*.so' \
    --exclude '*.a' \
    /Users/jiaozihan/Desktop/MPP/npp-1/ \
    ${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}/

echo ""
echo "同步完成！"
echo ""
echo "在服务器上运行以下命令："
echo "  cd ~/npp"
echo "  chmod +x test/benchmark/*.sh"
echo "  ./test/benchmark/run_comparison.sh"
