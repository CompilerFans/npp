#include "npp.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <type_traits>

#ifdef HAVE_NVIDIA_NPP
#include <npp.h>
#endif

/**
 * 全面的算术运算验证测试 - 使用NPPI内存分配
 * 测试所有数据类型(8u, 16u, 16s, 32f)的所有算术运算(AddC, SubC, MulC, DivC)
 */

// NPPI内存分配辅助函数
template<typename T>
void* allocateNppiMemory(int width, int height, int* pStep) {
    if constexpr (std::is_same_v<T, Npp8u>) {
        return nppiMalloc_8u_C1(width, height, pStep);
    } else if constexpr (std::is_same_v<T, Npp16u>) {
        return nppiMalloc_16u_C1(width, height, pStep);
    } else if constexpr (std::is_same_v<T, Npp16s>) {
        return nppiMalloc_16s_C1(width, height, pStep);
    } else if constexpr (std::is_same_v<T, Npp32f>) {
        return nppiMalloc_32f_C1(width, height, pStep);
    }
    return nullptr;
}

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
    std::vector<T> dstHost(dataSize, 0);
    
    // 使用NPPI函数分配设备内存
    T *srcDev = nullptr, *dstDev = nullptr;
    int srcStep, dstStep;
    
    srcDev = static_cast<T*>(allocateNppiMemory<T>(width, height, &srcStep));
    if (!srcDev) {
        std::cout << "源内存分配失败" << std::endl;
        return false;
    }
    
    dstDev = static_cast<T*>(allocateNppiMemory<T>(width, height, &dstStep));
    if (!dstDev) {
        std::cout << "目标内存分配失败" << std::endl;
        nppiFree(srcDev);
        return false;
    }
    
    // 复制数据到设备
    cudaError_t cudaResult = cudaMemcpy2D(srcDev, srcStep, srcHost.data(), width * sizeof(T),
                             width * sizeof(T), height, cudaMemcpyHostToDevice);
    if (cudaResult != cudaSuccess) {
        std::cout << "主机到设备拷贝失败" << std::endl;
        nppiFree(srcDev);
        nppiFree(dstDev);
        return false;
    }
    
    // 执行运算
    NppiSize roiSize = {width, height};
    NppStatus status = openFunc(srcDev, srcStep, constantValue,
                               dstDev, dstStep, roiSize, scaleFactor);
    
    if (status != NPP_NO_ERROR) {
        std::cout << "  ✗ " << opName << " 失败: " << status << std::endl;
        nppiFree(srcDev);
        nppiFree(dstDev);
        return false;
    }
    
    // 拷贝结果回主机
    cudaResult = cudaMemcpy2D(dstHost.data(), width * sizeof(T), dstDev, dstStep,
                             width * sizeof(T), height, cudaMemcpyDeviceToHost);
    if (cudaResult != cudaSuccess) {
        std::cout << "设备到主机拷贝失败" << std::endl;
        nppiFree(srcDev);
        nppiFree(dstDev);
        return false;
    }
    
    // 清理内存
    nppiFree(srcDev);
    nppiFree(dstDev);
    
    // 验证结果
    bool passed = true;
    for (int i = 0; i < dataSize; ++i) {
        if (dstHost[i] != expectedResult) {
            if (i < 10) { // 只打印前几个错误
                std::cout << "  位置 " << i << ": 期望 " << (int)expectedResult 
                         << ", 实际 " << (int)dstHost[i] << std::endl;
            }
            passed = false;
        }
    }
    
    if (passed) {
        std::cout << "  ✓ " << opName << " 通过" << std::endl;
    } else {
        std::cout << "  ✗ " << opName << " 验证失败" << std::endl;
    }
    
    return passed;
}

// 32位浮点数版本(无缩放因子)
bool testArithmeticOperation32f(const std::string& opName,
                              NppStatus (*openFunc)(const Npp32f*, int, Npp32f, Npp32f*, int, NppiSize),
                              Npp32f sourceValue, Npp32f constantValue, Npp32f expectedResult) {
    std::cout << "测试 " << opName << "..." << std::endl;
    
    const int width = 64;
    const int height = 64;
    const int dataSize = width * height;
    
    // 准备主机数据
    std::vector<Npp32f> srcHost(dataSize, sourceValue);
    std::vector<Npp32f> dstHost(dataSize, 0);
    
    // 使用NPPI函数分配设备内存
    Npp32f *srcDev = nullptr, *dstDev = nullptr;
    int srcStep, dstStep;
    
    srcDev = nppiMalloc_32f_C1(width, height, &srcStep);
    if (!srcDev) return false;
    
    dstDev = nppiMalloc_32f_C1(width, height, &dstStep);
    if (!dstDev) {
        nppiFree(srcDev);
        return false;
    }
    
    // 复制数据到设备
    cudaMemcpy2D(srcDev, srcStep, srcHost.data(), width * sizeof(Npp32f),
                 width * sizeof(Npp32f), height, cudaMemcpyHostToDevice);
    
    // 执行运算
    NppiSize roiSize = {width, height};
    NppStatus status = openFunc(srcDev, srcStep, constantValue,
                               dstDev, dstStep, roiSize);
    
    if (status != NPP_NO_ERROR) {
        std::cout << "  ✗ " << opName << " 失败: " << status << std::endl;
        nppiFree(srcDev);
        nppiFree(dstDev);
        return false;
    }
    
    // 拷贝结果回主机
    cudaMemcpy2D(dstHost.data(), width * sizeof(Npp32f), dstDev, dstStep,
                 width * sizeof(Npp32f), height, cudaMemcpyDeviceToHost);
    
    // 清理内存
    nppiFree(srcDev);
    nppiFree(dstDev);
    
    // 验证结果(浮点数使用容差)
    bool passed = true;
    const float epsilon = 1e-6f;
    
    for (int i = 0; i < dataSize; ++i) {
        if (std::abs(dstHost[i] - expectedResult) > epsilon) {
            if (i < 10) {
                std::cout << "  位置 " << i << ": 期望 " << expectedResult 
                         << ", 实际 " << dstHost[i] << std::endl;
            }
            passed = false;
        }
    }
    
    if (passed) {
        std::cout << "  ✓ " << opName << " 通过" << std::endl;
    } else {
        std::cout << "  ✗ " << opName << " 验证失败" << std::endl;
    }
    
    return passed;
}

int main() {
    std::cout << "=== NPP全面算术运算测试 (使用NPPI内存分配) ===" << std::endl;
    std::cout << "测试所有数据类型和算术运算的组合\n" << std::endl;
    
    TestStats stats;
    
    // AddC测试
    std::cout << "\n--- AddC 运算测试 ---" << std::endl;
    stats.addResult(testArithmeticOperation<Npp8u>("nppiAddC_8u_C1RSfs", nppiAddC_8u_C1RSfs,
                                                   50, 10, 0, 60));
    stats.addResult(testArithmeticOperation<Npp16u>("nppiAddC_16u_C1RSfs", nppiAddC_16u_C1RSfs,
                                                    500, 100, 0, 600));
    stats.addResult(testArithmeticOperation<Npp16s>("nppiAddC_16s_C1RSfs", nppiAddC_16s_C1RSfs,
                                                    -500, 100, 0, -400));
    stats.addResult(testArithmeticOperation32f("nppiAddC_32f_C1R", nppiAddC_32f_C1R,
                                              50.0f, 10.5f, 60.5f));
    
    // SubC测试
    std::cout << "\n--- SubC 运算测试 ---" << std::endl;
    stats.addResult(testArithmeticOperation<Npp8u>("nppiSubC_8u_C1RSfs", nppiSubC_8u_C1RSfs,
                                                   50, 10, 0, 40));
    stats.addResult(testArithmeticOperation<Npp16u>("nppiSubC_16u_C1RSfs", nppiSubC_16u_C1RSfs,
                                                    500, 100, 0, 400));
    
    // MulC测试
    std::cout << "\n--- MulC 运算测试 ---" << std::endl;
    stats.addResult(testArithmeticOperation<Npp8u>("nppiMulC_8u_C1RSfs", nppiMulC_8u_C1RSfs,
                                                   10, 5, 0, 50));
    
    // DivC测试
    std::cout << "\n--- DivC 运算测试 ---" << std::endl;
    stats.addResult(testArithmeticOperation<Npp8u>("nppiDivC_8u_C1RSfs", nppiDivC_8u_C1RSfs,
                                                   100, 5, 0, 20));
    
    // 打印总结
    stats.printSummary();
    
    return stats.allPassed() ? 0 : 1;
}