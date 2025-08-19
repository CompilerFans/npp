/**
 * @file dual_library_test_base.h
 * @brief 双库动态加载对比测试基类
 * 
 * 提供同时动态加载OpenNPP和NVIDIA NPP的测试基类，
 * 实现真正独立的两个库对比测试，避免符号冲突
 */

#ifndef TEST_FRAMEWORK_DUAL_LIBRARY_TEST_BASE_H
#define TEST_FRAMEWORK_DUAL_LIBRARY_TEST_BASE_H

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <chrono>
#include "opennpp_loader.h"
#include "nvidia_npp_loader.h"

namespace test_framework {

/**
 * @brief 双库动态加载对比测试基类
 * 
 * 所有需要同时对比OpenNPP和NVIDIA NPP的测试都应该继承这个类
 * 这个类通过动态加载避免符号冲突，实现真正独立的库对比
 */
class DualLibraryTestBase : public ::testing::Test {
protected:
    // 设置测试环境
    void SetUp() override {
        // 设置CUDA设备
        cudaError_t err = cudaSetDevice(0);
        ASSERT_EQ(err, cudaSuccess) << "Failed to set CUDA device";
        
        // 获取加载器实例
        openNppLoader_ = &OpenNppLoader::getInstance();
        nvidiaLoader_ = &NvidiaNppLoader::getInstance();
        
        hasOpenNpp_ = openNppLoader_->isAvailable();
        hasNvidiaNpp_ = nvidiaLoader_->isAvailable();
        
        if (!hasOpenNpp_) {
            std::cout << "OpenNPP not available: " << openNppLoader_->getErrorMessage() << std::endl;
        }
        
        if (!hasNvidiaNpp_) {
            std::cout << "NVIDIA NPP not available: " << nvidiaLoader_->getErrorMessage() << std::endl;
        }
        
        // 如果两个库都不可用，跳过测试
        if (!hasOpenNpp_ && !hasNvidiaNpp_) {
            GTEST_SKIP() << "Both OpenNPP and NVIDIA NPP are not available";
        }
    }
    
    void TearDown() override {
        cudaDeviceSynchronize();
    }
    
    // 检查库可用性
    bool hasOpenNpp() const { return hasOpenNpp_; }
    bool hasNvidiaNpp() const { return hasNvidiaNpp_; }
    bool hasBothLibraries() const { return hasOpenNpp_ && hasNvidiaNpp_; }
    
    // 跳过需要两个库的测试
    void skipIfMissingLibraries(const std::string& reason = "Both libraries required") {
        if (!hasBothLibraries()) {
            GTEST_SKIP() << reason << " (OpenNPP: " << hasOpenNpp_ << ", NVIDIA: " << hasNvidiaNpp_ << ")";
        }
    }
    
    // 跳过需要OpenNPP的测试
    void skipIfNoOpenNpp(const std::string& reason = "OpenNPP not available") {
        if (!hasOpenNpp_) {
            GTEST_SKIP() << reason;
        }
    }
    
    // 跳过需要NVIDIA NPP的测试
    void skipIfNoNvidiaNpp(const std::string& reason = "NVIDIA NPP not available") {
        if (!hasNvidiaNpp_) {
            GTEST_SKIP() << reason;
        }
    }
    
    // ==================== 数据比较辅助函数 ====================
    
    // 比较两个整数数组
    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type
    compareArrays(const T* arr1, const T* arr2, size_t size, 
                 int maxDiff = 1, int* diffCount = nullptr, 
                 const std::string& arr1Name = "Array1", 
                 const std::string& arr2Name = "Array2") {
        int localDiffCount = 0;
        bool allMatch = true;
        
        for (size_t i = 0; i < size; i++) {
            int diff = std::abs(static_cast<int>(arr1[i]) - static_cast<int>(arr2[i]));
            if (diff > maxDiff) {
                if (localDiffCount < 10) { // 只打印前10个不匹配
                    std::cout << "Mismatch at index " << i << ": " 
                              << arr1Name << "=" << static_cast<int>(arr1[i]) << " vs " 
                              << arr2Name << "=" << static_cast<int>(arr2[i]) 
                              << " (diff=" << diff << ")" << std::endl;
                }
                localDiffCount++;
                allMatch = false;
            }
        }
        
        if (diffCount) *diffCount = localDiffCount;
        
        if (localDiffCount > 0) {
            std::cout << "Total mismatches: " << localDiffCount << "/" << size 
                      << " (" << (localDiffCount * 100.0 / size) << "%)" << std::endl;
        }
        
        return allMatch;
    }
    
    // 比较两个浮点数组
    template<typename T>
    typename std::enable_if<std::is_floating_point<T>::value, bool>::type
    compareArrays(const T* arr1, const T* arr2, size_t size, 
                 T epsilon = 1e-5, int* diffCount = nullptr,
                 const std::string& arr1Name = "Array1", 
                 const std::string& arr2Name = "Array2") {
        int localDiffCount = 0;
        bool allMatch = true;
        T maxDiff = 0;
        
        for (size_t i = 0; i < size; i++) {
            T diff = std::abs(arr1[i] - arr2[i]);
            if (diff > epsilon) {
                if (localDiffCount < 10) { // 只打印前10个不匹配
                    std::cout << "Mismatch at index " << i << ": " 
                              << arr1Name << "=" << arr1[i] << " vs " 
                              << arr2Name << "=" << arr2[i] 
                              << " (diff=" << diff << ")" << std::endl;
                }
                localDiffCount++;
                allMatch = false;
                maxDiff = std::max(maxDiff, diff);
            }
        }
        
        if (diffCount) *diffCount = localDiffCount;
        
        if (localDiffCount > 0) {
            std::cout << "Total mismatches: " << localDiffCount << "/" << size 
                      << " (" << (localDiffCount * 100.0 / size) << "%)" << std::endl;
            std::cout << "Max difference: " << maxDiff << std::endl;
        }
        
        return allMatch;
    }
    
    // ==================== 性能测试辅助函数 ====================
    
    // 性能测试结果结构
    struct PerformanceResult {
        double openNppTime;      // OpenNPP执行时间(ms)
        double nvidiaNppTime;    // NVIDIA NPP执行时间(ms)
        double speedup;          // 加速比
        size_t dataSize;         // 数据大小
    };
    
    // 测量函数执行时间
    template<typename Func>
    double measureExecutionTime(Func func, int iterations = 100) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        // 预热
        for (int i = 0; i < 10; i++) {
            func();
        }
        cudaDeviceSynchronize();
        
        // 正式测量
        cudaEventRecord(start);
        for (int i = 0; i < iterations; i++) {
            func();
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        return milliseconds / iterations;
    }
    
    // 打印性能对比结果
    void printPerformanceComparison(const PerformanceResult& result, 
                                   const std::string& testName) {
        std::cout << "\n=== Performance Comparison: " << testName << " ===" << std::endl;
        std::cout << "Data size: " << result.dataSize << " elements" << std::endl;
        std::cout << "OpenNPP time: " << std::fixed << std::setprecision(3) 
                  << result.openNppTime << " ms" << std::endl;
        std::cout << "NVIDIA NPP time: " << std::fixed << std::setprecision(3) 
                  << result.nvidiaNppTime << " ms" << std::endl;
        
        if (result.nvidiaNppTime > 0) {
            double speedup = result.openNppTime / result.nvidiaNppTime;
            std::string comparison;
            if (speedup > 1.0) {
                comparison = "NVIDIA NPP is " + std::to_string(speedup) + "x faster";
            } else if (speedup < 1.0) {
                comparison = "OpenNPP is " + std::to_string(1.0/speedup) + "x faster";
            } else {
                comparison = "Same performance";
            }
            std::cout << "Speedup: " << std::fixed << std::setprecision(2) 
                      << speedup << "x (" << comparison << ")" << std::endl;
        }
        std::cout << std::string(50, '=') << std::endl;
    }
    
    // ==================== 一致性测试辅助函数 ====================
    
    // 一致性测试结果结构
    struct ConsistencyResult {
        bool passed;             // 是否通过测试
        int mismatchCount;       // 不匹配数量
        double mismatchRate;     // 不匹配率
        size_t totalElements;    // 总元素数
        std::string details;     // 详细信息
    };
    
    // 打印一致性测试结果
    void printConsistencyResult(const ConsistencyResult& result, 
                               const std::string& testName) {
        std::cout << "\n=== Consistency Test: " << testName << " ===" << std::endl;
        std::cout << "Total elements: " << result.totalElements << std::endl;
        std::cout << "Mismatches: " << result.mismatchCount << std::endl;
        std::cout << "Mismatch rate: " << std::fixed << std::setprecision(2) 
                  << result.mismatchRate << "%" << std::endl;
        std::cout << "Result: " << (result.passed ? "PASSED ✓" : "FAILED ✗") << std::endl;
        if (!result.details.empty()) {
            std::cout << "Details: " << result.details << std::endl;
        }
        std::cout << std::string(50, '=') << std::endl;
    }
    
    // ==================== 内存分配辅助函数 ====================
    
    // 使用OpenNPP分配内存
    template<typename T>
    T* allocateWithOpenNpp(int width, int height, int* step) {
        if (!hasOpenNpp_) return nullptr;
        
        T* devPtr = nullptr;
        
        if constexpr (std::is_same_v<T, Npp8u>) {
            if (openNppLoader_->op_nppiMalloc_8u_C1) {
                devPtr = openNppLoader_->op_nppiMalloc_8u_C1(width, height, step);
            }
        } else if constexpr (std::is_same_v<T, Npp16u>) {
            if (openNppLoader_->op_nppiMalloc_16u_C1) {
                devPtr = openNppLoader_->op_nppiMalloc_16u_C1(width, height, step);
            }
        } else if constexpr (std::is_same_v<T, Npp32f>) {
            if (openNppLoader_->op_nppiMalloc_32f_C1) {
                devPtr = openNppLoader_->op_nppiMalloc_32f_C1(width, height, step);
            }
        }
        
        return devPtr;
    }
    
    // 使用NVIDIA NPP分配内存
    template<typename T>
    T* allocateWithNvidia(int width, int height, int* step) {
        if (!hasNvidiaNpp_) return nullptr;
        
        T* devPtr = nullptr;
        
        if constexpr (std::is_same_v<T, Npp8u>) {
            if (nvidiaLoader_->nv_nppiMalloc_8u_C1) {
                devPtr = nvidiaLoader_->nv_nppiMalloc_8u_C1(width, height, step);
            }
        } else if constexpr (std::is_same_v<T, Npp16u>) {
            if (nvidiaLoader_->nv_nppiMalloc_16u_C1) {
                devPtr = nvidiaLoader_->nv_nppiMalloc_16u_C1(width, height, step);
            }
        } else if constexpr (std::is_same_v<T, Npp32f>) {
            if (nvidiaLoader_->nv_nppiMalloc_32f_C1) {
                devPtr = nvidiaLoader_->nv_nppiMalloc_32f_C1(width, height, step);
            }
        }
        
        return devPtr;
    }
    
    // 释放OpenNPP内存
    void freeOpenNppMemory(void* ptr) {
        if (!ptr || !hasOpenNpp_) return;
        
        if (openNppLoader_->op_nppiFree) {
            openNppLoader_->op_nppiFree(ptr);
        }
    }
    
    // 释放NVIDIA NPP内存
    void freeNvidiaMemory(void* ptr) {
        if (!ptr || !hasNvidiaNpp_) return;
        
        if (nvidiaLoader_->nv_nppiFree) {
            nvidiaLoader_->nv_nppiFree(ptr);
        }
    }
    
    // 初始化数据到设备内存
    template<typename T>
    void initializeDeviceData(T* devPtr, int step, int width, int height, const std::vector<T>& hostData) {
        if (devPtr && !hostData.empty()) {
            cudaMemcpy2D(devPtr, step, hostData.data(), width * sizeof(T),
                        width * sizeof(T), height, cudaMemcpyHostToDevice);
        }
    }
    
protected:
    OpenNppLoader* openNppLoader_ = nullptr;
    NvidiaNppLoader* nvidiaLoader_ = nullptr;
    bool hasOpenNpp_ = false;
    bool hasNvidiaNpp_ = false;
};

} // namespace test_framework

#endif // TEST_FRAMEWORK_DUAL_LIBRARY_TEST_BASE_H