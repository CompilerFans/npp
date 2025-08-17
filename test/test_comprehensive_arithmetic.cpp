#include "npp.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>

#ifdef HAVE_NVIDIA_NPP
#include <npp.h>
#endif

/**
 * 全面的算术运算验证测试
 * 测试所有数据类型(8u, 16u, 16s, 32f)的所有算术运算(AddC, SubC, MulC, DivC)
 */

struct TestStats {
    int totalTests = 0;
    int passedTests = 0;
    
    void addResult(bool passed) {
        totalTests++;
        if (passed) passedTests++;
    }
    
    void printSummary() const {
        std::cout << "\n=== 全面算术运算测试结果 ===" << std::endl;
        std::cout << "总测试数: " << totalTests << std::endl;
        std::cout << "通过测试: " << passedTests << std::endl;
        std::cout << "失败测试: " << (totalTests - passedTests) << std::endl;
        std::cout << "成功率: " << std::fixed << std::setprecision(1) 
                  << (100.0 * passedTests / totalTests) << "%" << std::endl;
    }
    
    bool allPassed() const {
        return passedTests == totalTests;
    }
};

template<typename T>
bool testArithmeticOperation(const std::string& opName, 
                           NppStatus (*openFunc)(const T*, int, T, T*, int, NppiSize, int),
                           T sourceValue, T constantValue, int scaleFactor, T expectedResult) {
    std::cout << "测试 " << opName << "..." << std::endl;
    
    const int width = 64;
    const int height = 64;
    const int dataSize = width * height;
    
    // 准备主机数据
    std::vector<T> srcHost(dataSize, sourceValue);
    std::vector<T> dstHost(dataSize, T(0));
    
    // 分配设备内存
    T *srcDev = nullptr, *dstDev = nullptr;
    size_t srcPitch, dstPitch;
    
    cudaError_t cudaResult = cudaMallocPitch((void**)&srcDev, &srcPitch, width * sizeof(T), height);
    if (cudaResult != cudaSuccess) {
        std::cout << "源内存分配失败" << std::endl;
        return false;
    }
    
    cudaResult = cudaMallocPitch((void**)&dstDev, &dstPitch, width * sizeof(T), height);
    if (cudaResult != cudaSuccess) {
        std::cout << "目标内存分配失败" << std::endl;
        cudaFree(srcDev);
        return false;
    }
    
    // 复制数据到设备
    cudaResult = cudaMemcpy2D(srcDev, srcPitch, srcHost.data(), width * sizeof(T),
                             width * sizeof(T), height, cudaMemcpyHostToDevice);
    if (cudaResult != cudaSuccess) {
        std::cout << "主机到设备拷贝失败" << std::endl;
        cudaFree(srcDev);
        cudaFree(dstDev);
        return false;
    }
    
    // 执行操作
    NppiSize roiSize = {width, height};
    NppStatus status = openFunc(srcDev, (int)srcPitch, constantValue, dstDev, (int)dstPitch, roiSize, scaleFactor);
    
    bool testPassed = false;
    if (status == NPP_NO_ERROR) {
        // 复制结果回主机
        cudaResult = cudaMemcpy2D(dstHost.data(), width * sizeof(T), dstDev, dstPitch,
                                 width * sizeof(T), height, cudaMemcpyDeviceToHost);
        
        if (cudaResult == cudaSuccess) {
            // 验证结果
            testPassed = true;
            for (int i = 0; i < dataSize && testPassed; ++i) {
                if (dstHost[i] != expectedResult) {
                    std::cout << "结果不匹配在索引 " << i << ": 得到 " << dstHost[i] 
                              << ", 期望 " << expectedResult << std::endl;
                    testPassed = false;
                }
            }
            
            if (testPassed) {
                std::cout << opName << ": 通过 ✓" << std::endl;
            }
        } else {
            std::cout << "设备到主机拷贝失败" << std::endl;
        }
    } else {
        std::cout << opName << " 执行失败，状态码: " << status << std::endl;
    }
    
    // 清理
    cudaFree(srcDev);
    cudaFree(dstDev);
    
    return testPassed;
}

// 32位浮点数版本(无缩放因子)
bool testArithmeticOperation32f(const std::string& opName, 
                               NppStatus (*openFunc)(const Npp32f*, int, Npp32f, Npp32f*, int, NppiSize),
                               Npp32f sourceValue, Npp32f constantValue, Npp32f expectedResult) {
    std::cout << "测试 " << opName << "..." << std::endl;
    
    const int width = 64;
    const int height = 64;
    const int dataSize = width * height;
    
    std::vector<Npp32f> srcHost(dataSize, sourceValue);
    std::vector<Npp32f> dstHost(dataSize, 0.0f);
    
    Npp32f *srcDev = nullptr, *dstDev = nullptr;
    size_t srcPitch, dstPitch;
    
    cudaError_t cudaResult = cudaMallocPitch((void**)&srcDev, &srcPitch, width * sizeof(Npp32f), height);
    if (cudaResult != cudaSuccess) return false;
    
    cudaResult = cudaMallocPitch((void**)&dstDev, &dstPitch, width * sizeof(Npp32f), height);
    if (cudaResult != cudaSuccess) {
        cudaFree(srcDev);
        return false;
    }
    
    cudaResult = cudaMemcpy2D(srcDev, srcPitch, srcHost.data(), width * sizeof(Npp32f),
                             width * sizeof(Npp32f), height, cudaMemcpyHostToDevice);
    if (cudaResult != cudaSuccess) {
        cudaFree(srcDev);
        cudaFree(dstDev);
        return false;
    }
    
    NppiSize roiSize = {width, height};
    NppStatus status = openFunc(srcDev, (int)srcPitch, constantValue, dstDev, (int)dstPitch, roiSize);
    
    bool testPassed = false;
    if (status == NPP_NO_ERROR) {
        cudaResult = cudaMemcpy2D(dstHost.data(), width * sizeof(Npp32f), dstDev, dstPitch,
                                 width * sizeof(Npp32f), height, cudaMemcpyDeviceToHost);
        
        if (cudaResult == cudaSuccess) {
            testPassed = true;
            for (int i = 0; i < dataSize && testPassed; ++i) {
                if (std::abs(dstHost[i] - expectedResult) > 1e-6f) {
                    std::cout << "结果不匹配在索引 " << i << ": 得到 " << dstHost[i] 
                              << ", 期望 " << expectedResult << std::endl;
                    testPassed = false;
                }
            }
            
            if (testPassed) {
                std::cout << opName << ": 通过 ✓" << std::endl;
            }
        }
    } else {
        std::cout << opName << " 执行失败，状态码: " << status << std::endl;
    }
    
    cudaFree(srcDev);
    cudaFree(dstDev);
    
    return testPassed;
}

int main() {
    std::cout << "=== NPP 全面算术运算验证测试 ===" << std::endl;
    
    // 初始化CUDA
    cudaError_t cudaResult = cudaSetDevice(0);
    if (cudaResult != cudaSuccess) {
        std::cout << "CUDA设备初始化失败" << std::endl;
        return 1;
    }
    
    TestStats stats;
    
    // 测试8位无符号整数运算
    std::cout << "\n--- 8位无符号整数运算 ---" << std::endl;
    stats.addResult(testArithmeticOperation<Npp8u>("nppiAddC_8u_C1RSfs", nppiAddC_8u_C1RSfs, 50, 25, 1, (50 + 25) >> 1));
    stats.addResult(testArithmeticOperation<Npp8u>("nppiSubC_8u_C1RSfs", nppiSubC_8u_C1RSfs, 100, 25, 1, (100 - 25) >> 1));
    stats.addResult(testArithmeticOperation<Npp8u>("nppiMulC_8u_C1RSfs", nppiMulC_8u_C1RSfs, 20, 3, 2, (20 * 3) >> 2));
    stats.addResult(testArithmeticOperation<Npp8u>("nppiDivC_8u_C1RSfs", nppiDivC_8u_C1RSfs, 100, 4, 2, (100 / 4) << 2));
    
    // 测试16位无符号整数运算
    std::cout << "\n--- 16位无符号整数运算 ---" << std::endl;
    stats.addResult(testArithmeticOperation<Npp16u>("nppiAddC_16u_C1RSfs", nppiAddC_16u_C1RSfs, 1000, 500, 2, (1000 + 500) >> 2));
    stats.addResult(testArithmeticOperation<Npp16u>("nppiSubC_16u_C1RSfs", nppiSubC_16u_C1RSfs, 2000, 500, 1, (2000 - 500) >> 1));
    stats.addResult(testArithmeticOperation<Npp16u>("nppiMulC_16u_C1RSfs", nppiMulC_16u_C1RSfs, 100, 5, 3, (100 * 5) >> 3));
    stats.addResult(testArithmeticOperation<Npp16u>("nppiDivC_16u_C1RSfs", nppiDivC_16u_C1RSfs, 1000, 8, 2, (1000 / 8) << 2));
    
    // 测试16位有符号整数运算
    std::cout << "\n--- 16位有符号整数运算 ---" << std::endl;
    stats.addResult(testArithmeticOperation<Npp16s>("nppiAddC_16s_C1RSfs", nppiAddC_16s_C1RSfs, 1000, -300, 1, (1000 + (-300)) >> 1));
    stats.addResult(testArithmeticOperation<Npp16s>("nppiSubC_16s_C1RSfs", nppiSubC_16s_C1RSfs, 1000, 200, 1, (1000 - 200) >> 1));
    stats.addResult(testArithmeticOperation<Npp16s>("nppiMulC_16s_C1RSfs", nppiMulC_16s_C1RSfs, 50, 4, 2, (50 * 4) >> 2));
    stats.addResult(testArithmeticOperation<Npp16s>("nppiDivC_16s_C1RSfs", nppiDivC_16s_C1RSfs, 800, 4, 1, (800 / 4) << 1));
    
    // 测试32位浮点数运算
    std::cout << "\n--- 32位浮点数运算 ---" << std::endl;
    stats.addResult(testArithmeticOperation32f("nppiAddC_32f_C1R", nppiAddC_32f_C1R, 50.5f, 15.5f, 66.0f));
    stats.addResult(testArithmeticOperation32f("nppiSubC_32f_C1R", nppiSubC_32f_C1R, 100.5f, 25.5f, 75.0f));
    stats.addResult(testArithmeticOperation32f("nppiMulC_32f_C1R", nppiMulC_32f_C1R, 25.0f, 2.5f, 62.5f));
    stats.addResult(testArithmeticOperation32f("nppiDivC_32f_C1R", nppiDivC_32f_C1R, 100.0f, 4.0f, 25.0f));
    
    // 打印总结
    stats.printSummary();
    
    return stats.allPassed() ? 0 : 1;
}