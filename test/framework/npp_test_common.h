/**
 * OpenNPP测试框架 - 通用头文件
 * 包含所有测试需要的基础设施
 */
#pragma once

#include "npp.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <cstring>
#include <random>
#include <type_traits>
#include <chrono>
#include <string>
#include <sstream>

// 如果系统中有NVIDIA NPP，包含它
#ifdef HAVE_NVIDIA_NPP
#include <npp.h>
#endif

namespace opennpp {
namespace test {

/**
 * 测试配置
 */
struct TestConfig {
    bool verbose = true;
    bool compareWithNvidia = true;
    bool compareWithCpu = true;
    double floatTolerance = 1e-6;
    double integerTolerance = 1.0;
    int defaultWidth = 127;  // 使用非对齐宽度
    int defaultHeight = 64;
};

/**
 * 测试结果
 */
struct TestResult {
    std::string testName;
    bool passed = false;
    double executionTime = 0.0;  // 毫秒
    double maxDifference = 0.0;
    std::string errorMessage;
    
    TestResult(const std::string& name = "") : testName(name) {}
};

/**
 * 测试统计
 */
class TestStatistics {
private:
    int totalTests_ = 0;
    int passedTests_ = 0;
    std::vector<TestResult> results_;
    
public:
    void addResult(const TestResult& result) {
        results_.push_back(result);
        totalTests_++;
        if (result.passed) passedTests_++;
    }
    
    void printSummary() const {
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "测试统计汇总" << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        std::cout << "总测试数: " << totalTests_ << std::endl;
        std::cout << "通过: " << passedTests_ << std::endl;
        std::cout << "失败: " << (totalTests_ - passedTests_) << std::endl;
        std::cout << "成功率: " << std::fixed << std::setprecision(1) 
                  << (totalTests_ > 0 ? (100.0 * passedTests_ / totalTests_) : 0.0) 
                  << "%" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
    }
    
    void printDetails() const {
        std::cout << "\n详细测试结果:" << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        std::cout << std::left << std::setw(40) << "测试名称"
                  << std::setw(10) << "状态"
                  << std::setw(15) << "耗时(ms)"
                  << std::setw(15) << "最大差异" << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        
        for (const auto& result : results_) {
            std::cout << std::left << std::setw(40) << result.testName
                      << std::setw(10) << (result.passed ? "✓ 通过" : "✗ 失败")
                      << std::setw(15) << std::fixed << std::setprecision(3) << result.executionTime
                      << std::setw(15) << std::scientific << std::setprecision(2) << result.maxDifference;
            if (!result.errorMessage.empty()) {
                std::cout << " [" << result.errorMessage << "]";
            }
            std::cout << std::endl;
        }
        std::cout << std::string(80, '-') << std::endl;
    }
    
    bool allPassed() const {
        return passedTests_ == totalTests_;
    }
    
    int getPassedCount() const { return passedTests_; }
    int getTotalCount() const { return totalTests_; }
};

/**
 * 计时器
 */
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_;
    
public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    
    double elapsedMilliseconds() const {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
        return duration.count() / 1000.0;
    }
};

/**
 * NPPI内存分配辅助模板
 */
template<typename T>
T* nppiMallocHelper(int width, int height, int channels, int* pStep) {
    if constexpr (std::is_same_v<T, Npp8u>) {
        switch(channels) {
            case 1: return reinterpret_cast<T*>(nppiMalloc_8u_C1(width, height, pStep));
            case 3: return reinterpret_cast<T*>(nppiMalloc_8u_C3(width, height, pStep));
            case 4: return reinterpret_cast<T*>(nppiMalloc_8u_C4(width, height, pStep));
        }
    } else if constexpr (std::is_same_v<T, Npp16u>) {
        switch(channels) {
            case 1: return reinterpret_cast<T*>(nppiMalloc_16u_C1(width, height, pStep));
            case 3: return reinterpret_cast<T*>(nppiMalloc_16u_C3(width, height, pStep));
            case 4: return reinterpret_cast<T*>(nppiMalloc_16u_C4(width, height, pStep));
        }
    } else if constexpr (std::is_same_v<T, Npp16s>) {
        switch(channels) {
            case 1: return reinterpret_cast<T*>(nppiMalloc_16s_C1(width, height, pStep));
            case 4: return reinterpret_cast<T*>(nppiMalloc_16s_C4(width, height, pStep));
        }
    } else if constexpr (std::is_same_v<T, Npp32f>) {
        switch(channels) {
            case 1: return reinterpret_cast<T*>(nppiMalloc_32f_C1(width, height, pStep));
            case 3: return reinterpret_cast<T*>(nppiMalloc_32f_C3(width, height, pStep));
            case 4: return reinterpret_cast<T*>(nppiMalloc_32f_C4(width, height, pStep));
        }
    }
    
    // 后备方案
    size_t pitch;
    void* devPtr = nullptr;
    cudaError_t result = cudaMallocPitch(&devPtr, &pitch, width * channels * sizeof(T), height);
    if (result == cudaSuccess) {
        *pStep = static_cast<int>(pitch);
        return static_cast<T*>(devPtr);
    }
    return nullptr;
}

/**
 * 数据类型名称获取
 */
template<typename T>
std::string getTypeName() {
    if constexpr (std::is_same_v<T, Npp8u>) return "8u";
    else if constexpr (std::is_same_v<T, Npp16u>) return "16u";
    else if constexpr (std::is_same_v<T, Npp16s>) return "16s";
    else if constexpr (std::is_same_v<T, Npp32f>) return "32f";
    else return "unknown";
}

/**
 * 获取通道字符串
 */
inline std::string getChannelString(int channels) {
    return "C" + std::to_string(channels);
}

/**
 * 生成测试数据模式
 */
enum class TestPattern {
    CONSTANT,      // 常数值
    GRADIENT,      // 渐变
    RANDOM,        // 随机
    CHECKERBOARD,  // 棋盘格
    EDGE_CASE      // 边界值测试
};

} // namespace test
} // namespace opennpp