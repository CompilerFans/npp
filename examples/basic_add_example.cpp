/**
 * @file basic_add_example.cpp
 * @brief OpenNPP基础加法运算示例
 * 
 * 演示如何使用OpenNPP库进行基本的图像加法运算
 */

#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// 包含OpenNPP头文件
#include "npp.h"

void printCudaError(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error in " << operation << ": " << cudaGetErrorString(error) << std::endl;
        exit(1);
    }
}

void printNppStatus(NppStatus status, const char* operation) {
    if (status != NPP_NO_ERROR) {
        std::cerr << "NPP Error in " << operation << ": " << status << std::endl;
        exit(1);
    }
}

int main() {
    std::cout << "=== OpenNPP基础加法运算示例 ===" << std::endl;
    
    // 获取并显示库版本信息
    const NppLibraryVersion* version = nppGetLibVersion();
    std::cout << "OpenNPP版本: " << version->major << "." 
              << version->minor << "." << version->build << std::endl;
    
    // 获取GPU信息
    int major, minor;
    NppStatus nppStatus = nppGetGpuComputeCapability(&major, &minor);
    printNppStatus(nppStatus, "获取GPU计算能力");
    std::cout << "GPU计算能力: " << major << "." << minor << std::endl;
    
    const char* gpuName = nppGetGpuName();
    if (gpuName) {
        std::cout << "GPU名称: " << gpuName << std::endl;
    } else {
        std::cout << "无法获取GPU名称" << std::endl;
    }
    
    // 设置图像尺寸
    const int width = 64;
    const int height = 64;
    const int imageSize = width * height;
    const size_t imageBytes = imageSize * sizeof(Npp32f);
    
    std::cout << "\n正在处理 " << width << "x" << height << " 的32位浮点图像..." << std::endl;
    
    // 创建主机数据
    std::vector<Npp32f> hostSrc1(imageSize, 1.5f);
    std::vector<Npp32f> hostSrc2(imageSize, 2.5f);
    std::vector<Npp32f> hostDst(imageSize);
    
    std::cout << "源图像1: 所有像素值为 " << hostSrc1[0] << std::endl;
    std::cout << "源图像2: 所有像素值为 " << hostSrc2[0] << std::endl;
    
    // 使用NPP内存分配函数
    Npp32f *devSrc1, *devSrc2, *devDst;
    int pitch;
    
    // 分配源图像1内存
    devSrc1 = nppiMalloc_32f_C1(width, height, &pitch);
    if (devSrc1 == nullptr) {
        std::cerr << "NPP Error: 无法分配源图像1内存" << std::endl;
        exit(1);
    }
    
    // 分配源图像2内存
    devSrc2 = nppiMalloc_32f_C1(width, height, &pitch);
    if (devSrc2 == nullptr) {
        std::cerr << "NPP Error: 无法分配源图像2内存" << std::endl;
        nppiFree(devSrc1);
        exit(1);
    }
    
    // 分配目标图像内存
    devDst = nppiMalloc_32f_C1(width, height, &pitch);
    if (devDst == nullptr) {
        std::cerr << "NPP Error: 无法分配目标图像内存" << std::endl;
        nppiFree(devSrc1);
        nppiFree(devSrc2);
        exit(1);
    }
    
    // 复制数据到GPU
    cudaError_t cudaStatus = cudaMemcpy2D(devSrc1, pitch, hostSrc1.data(), width * sizeof(Npp32f),
                              width * sizeof(Npp32f), height, cudaMemcpyHostToDevice);
    printCudaError(cudaStatus, "复制源图像1到GPU");
    
    cudaStatus = cudaMemcpy2D(devSrc2, pitch, hostSrc2.data(), width * sizeof(Npp32f),
                              width * sizeof(Npp32f), height, cudaMemcpyHostToDevice);
    printCudaError(cudaStatus, "复制源图像2到GPU");
    
    // 执行NPP加法运算
    NppiSize roi = {width, height};
    nppStatus = nppiAdd_32f_C1R(devSrc1, pitch, devSrc2, pitch, devDst, pitch, roi);
    printNppStatus(nppStatus, "NPP加法运算");
    
    // 复制结果回主机
    cudaStatus = cudaMemcpy2D(hostDst.data(), width * sizeof(Npp32f), devDst, pitch,
                              width * sizeof(Npp32f), height, cudaMemcpyDeviceToHost);
    printCudaError(cudaStatus, "复制结果回主机");
    
    // 验证结果
    float expectedValue = hostSrc1[0] + hostSrc2[0];
    bool success = true;
    
    for (int i = 0; i < imageSize; i++) {
        if (std::abs(hostDst[i] - expectedValue) > 1e-5f) {
            success = false;
            std::cerr << "验证失败: 位置 " << i << ", 期望 " << expectedValue 
                      << ", 实际 " << hostDst[i] << std::endl;
            break;
        }
    }
    
    if (success) {
        std::cout << "✓ 加法运算成功! 结果: " << hostDst[0] << " (期望: " << expectedValue << ")" << std::endl;
    } else {
        std::cout << "✗ 加法运算失败!" << std::endl;
    }
    
    // 显示部分结果
    std::cout << "\n结果图像前10个像素值:" << std::endl;
    for (int i = 0; i < std::min(10, imageSize); i++) {
        std::cout << "  [" << i << "] = " << hostDst[i] << std::endl;
    }
    
    // 清理NPP分配的GPU内存
    nppiFree(devSrc1);
    nppiFree(devSrc2);
    nppiFree(devDst);
    
    std::cout << "\n=== 示例完成 ===" << std::endl;
    
    return success ? 0 : 1;
}