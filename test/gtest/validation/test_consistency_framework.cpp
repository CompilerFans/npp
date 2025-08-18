/**
 * @file test_consistency_framework.cpp
 * @brief OpenNPP与NVIDIA NPP一致性验证框架
 * 
 * 设计目标：
 * 1. 全面对比OpenNPP和NVIDIA NPP的计算结果
 * 2. 处理不同类型的容差（整数舍入、浮点精度）
 * 3. 提供详细的差异分析和报告
 * 4. 支持性能对比
 * 5. 自动化回归测试
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include "npp.h"

#ifdef HAVE_NVIDIA_NPP
#include <nppdefs.h>
#include <nppi_arithmetic_and_logical_operations.h>
#include <nppi_support_functions.h>
#endif

// ==================== 一致性验证配置 ====================
namespace ConsistencyConfig {
    // 容差配置
    struct ToleranceConfig {
        // 整数类型容差
        static constexpr int TOLERANCE_8U = 1;     // 8位无符号允许±1差异
        static constexpr int TOLERANCE_16U = 1;    // 16位无符号允许±1差异
        static constexpr int TOLERANCE_16S = 1;    // 16位有符号允许±1差异
        static constexpr int TOLERANCE_32S = 1;    // 32位有符号允许±1差异
        
        // 浮点类型容差
        static constexpr float TOLERANCE_32F_RELATIVE = 1e-5f;  // 相对误差
        static constexpr float TOLERANCE_32F_ABSOLUTE = 1e-6f;  // 绝对误差
        static constexpr double TOLERANCE_64F_RELATIVE = 1e-10; // 双精度相对误差
        
        // 特殊情况容差
        static constexpr float TOLERANCE_SCALE_FACTOR = 2.0f;   // 有缩放因子时增加容差
    };
    
    // 测试配置
    struct TestConfig {
        // 标准测试尺寸
        static constexpr int SMALL_WIDTH = 32;
        static constexpr int SMALL_HEIGHT = 32;
        static constexpr int MEDIUM_WIDTH = 512;
        static constexpr int MEDIUM_HEIGHT = 512;
        static constexpr int LARGE_WIDTH = 1920;
        static constexpr int LARGE_HEIGHT = 1080;
        static constexpr int HUGE_WIDTH = 4096;
        static constexpr int HUGE_HEIGHT = 2160;
        
        // 边界测试尺寸
        static constexpr int PRIME_WIDTH = 1021;   // 质数宽度
        static constexpr int PRIME_HEIGHT = 769;   // 质数高度
        static constexpr int UNALIGNED_WIDTH = 1023; // 非对齐宽度
        
        // 性能测试迭代次数
        static constexpr int PERF_ITERATIONS = 100;
        static constexpr int PERF_WARMUP = 10;
    };
}

// ==================== 差异统计类 ====================
class DifferenceStatistics {
public:
    void AddDifference(double diff) {
        differences.push_back(std::abs(diff));
    }
    
    void Calculate() {
        if (differences.empty()) return;
        
        // 最大差异
        maxDiff = *std::max_element(differences.begin(), differences.end());
        
        // 平均差异
        avgDiff = std::accumulate(differences.begin(), differences.end(), 0.0) / differences.size();
        
        // 标准差
        double variance = 0;
        for (double d : differences) {
            variance += (d - avgDiff) * (d - avgDiff);
        }
        stdDev = std::sqrt(variance / differences.size());
        
        // 差异分布
        std::sort(differences.begin(), differences.end());
        percentile95 = differences[static_cast<size_t>(differences.size() * 0.95)];
        percentile99 = differences[static_cast<size_t>(differences.size() * 0.99)];
        
        // 统计超过阈值的数量
        for (double threshold : {1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1.0}) {
            int count = std::count_if(differences.begin(), differences.end(),
                                     [threshold](double d) { return d > threshold; });
            if (count > 0) {
                exceedanceMap[threshold] = count;
            }
        }
    }
    
    void PrintReport(const std::string& testName) const {
        std::cout << "\n=== 差异统计报告: " << testName << " ===\n";
        std::cout << std::fixed << std::scientific << std::setprecision(6);
        std::cout << "样本数量: " << differences.size() << "\n";
        std::cout << "最大差异: " << maxDiff << "\n";
        std::cout << "平均差异: " << avgDiff << "\n";
        std::cout << "标准差: " << stdDev << "\n";
        std::cout << "95%分位: " << percentile95 << "\n";
        std::cout << "99%分位: " << percentile99 << "\n";
        
        if (!exceedanceMap.empty()) {
            std::cout << "\n超过阈值的样本分布:\n";
            for (const auto& [threshold, count] : exceedanceMap) {
                double percentage = (count * 100.0) / differences.size();
                std::cout << "  > " << threshold << ": " << count 
                         << " (" << std::fixed << std::setprecision(2) 
                         << percentage << "%)\n";
            }
        }
    }
    
    bool IsConsistent(double tolerance) const {
        return maxDiff <= tolerance;
    }
    
    double GetMaxDifference() const { return maxDiff; }
    double GetAverageDifference() const { return avgDiff; }
    size_t GetSampleCount() const { return differences.size(); }
    
private:
    std::vector<double> differences;
    double maxDiff = 0;
    double avgDiff = 0;
    double stdDev = 0;
    double percentile95 = 0;
    double percentile99 = 0;
    std::map<double, int> exceedanceMap;
};

// ==================== 一致性验证基类 ====================
class ConsistencyTestBase : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        ASSERT_EQ(err, cudaSuccess);
        
        nppGetStreamContext(&nppStreamCtx);
        
#ifdef HAVE_NVIDIA_NPP
        hasNvidiaNPP = true;
#else
        hasNvidiaNPP = false;
        GTEST_SKIP() << "NVIDIA NPP not available, skipping consistency test";
#endif
    }
    
    void TearDown() override {
        cudaDeviceSynchronize();
    }
    
    // 生成测试数据的不同模式
    enum DataPattern {
        RANDOM,           // 随机数据
        GRADIENT,         // 渐变数据
        CHECKERBOARD,     // 棋盘格
        CONSTANT,         // 常量
        EDGE_CASE         // 边界值（0, 最大值, 最小值）
    };
    
    // 生成测试数据
    template<typename T>
    void GenerateTestData(T* devicePtr, int width, int height, int step, 
                         DataPattern pattern = RANDOM) {
        std::vector<T> hostData(width * height);
        
        switch (pattern) {
            case RANDOM:
                GenerateRandomData(hostData);
                break;
            case GRADIENT:
                GenerateGradientData(hostData, width, height);
                break;
            case CHECKERBOARD:
                GenerateCheckerboardData(hostData, width, height);
                break;
            case CONSTANT:
                std::fill(hostData.begin(), hostData.end(), static_cast<T>(42));
                break;
            case EDGE_CASE:
                GenerateEdgeCaseData(hostData);
                break;
        }
        
        cudaMemcpy2D(devicePtr, step, hostData.data(), 
                    width * sizeof(T), width * sizeof(T), 
                    height, cudaMemcpyHostToDevice);
    }
    
    // 比较两个结果
    template<typename T>
    DifferenceStatistics CompareResults(T* result1, T* result2, 
                                       int width, int height, 
                                       int step1, int step2) {
        std::vector<T> host1(width * height);
        std::vector<T> host2(width * height);
        
        cudaMemcpy2D(host1.data(), width * sizeof(T), result1, step1,
                    width * sizeof(T), height, cudaMemcpyDeviceToHost);
        cudaMemcpy2D(host2.data(), width * sizeof(T), result2, step2,
                    width * sizeof(T), height, cudaMemcpyDeviceToHost);
        
        DifferenceStatistics stats;
        
        for (int i = 0; i < width * height; i++) {
            double diff;
            if constexpr (std::is_floating_point_v<T>) {
                diff = std::abs(static_cast<double>(host1[i] - host2[i]));
            } else {
                diff = std::abs(static_cast<int>(host1[i]) - static_cast<int>(host2[i]));
            }
            stats.AddDifference(diff);
        }
        
        stats.Calculate();
        return stats;
    }
    
    // 性能比较
    struct PerformanceResult {
        double openNppTime;
        double nvidiaNppTime;
        double speedupRatio;  // openNppTime / nvidiaNppTime
        
        void Print() const {
            std::cout << "\n性能对比:\n";
            std::cout << "  OpenNPP: " << openNppTime << " ms\n";
            std::cout << "  NVIDIA NPP: " << nvidiaNppTime << " ms\n";
            std::cout << "  速度比: " << speedupRatio << "x ";
            
            if (speedupRatio > 1.1) {
                std::cout << "(OpenNPP较慢)\n";
            } else if (speedupRatio < 0.9) {
                std::cout << "(OpenNPP较快)\n";
            } else {
                std::cout << "(性能相当)\n";
            }
        }
    };
    
    template<typename Func>
    double MeasurePerformance(Func func, int iterations = 100) {
        // 预热
        for (int i = 0; i < 10; i++) {
            func();
        }
        cudaDeviceSynchronize();
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            func();
        }
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return duration.count() / (1000.0 * iterations); // 返回毫秒
    }
    
private:
    template<typename T>
    void GenerateRandomData(std::vector<T>& data) {
        std::random_device rd;
        std::mt19937 gen(rd());
        
        if constexpr (std::is_floating_point_v<T>) {
            std::uniform_real_distribution<float> dist(0.0f, 100.0f);
            for (auto& val : data) {
                val = static_cast<T>(dist(gen));
            }
        } else {
            // 使用较小的值范围避免溢出
            int maxVal = std::min(100, static_cast<int>(std::numeric_limits<T>::max() / 4));
            std::uniform_int_distribution<int> dist(0, maxVal);
            for (auto& val : data) {
                val = static_cast<T>(dist(gen));
            }
        }
    }
    
    template<typename T>
    void GenerateGradientData(std::vector<T>& data, int width, int height) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if constexpr (std::is_floating_point_v<T>) {
                    data[y * width + x] = static_cast<T>((x + y) / 2.0f);
                } else {
                    data[y * width + x] = static_cast<T>((x + y) / 2);
                }
            }
        }
    }
    
    template<typename T>
    void GenerateCheckerboardData(std::vector<T>& data, int width, int height) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                data[y * width + x] = static_cast<T>(((x / 8) + (y / 8)) % 2 ? 100 : 0);
            }
        }
    }
    
    template<typename T>
    void GenerateEdgeCaseData(std::vector<T>& data) {
        // 填充边界值
        size_t third = data.size() / 3;
        std::fill(data.begin(), data.begin() + third, std::numeric_limits<T>::min());
        std::fill(data.begin() + third, data.begin() + 2 * third, static_cast<T>(0));
        std::fill(data.begin() + 2 * third, data.end(), std::numeric_limits<T>::max());
    }
    
protected:
    NppStreamContext nppStreamCtx;
    bool hasNvidiaNPP;
};

// ==================== 算术运算一致性测试 ====================
class ArithmeticConsistencyTest : public ConsistencyTestBase {
protected:
    template<typename T>
    void TestAddCConsistency(int width, int height, T constant, 
                            int scaleFactor = 0, DataPattern pattern = RANDOM) {
        int srcStep, dstOpenStep, dstNvidiaStep;
        
        // 分配内存
        Npp8u* src = nppiMalloc_8u_C1(width * sizeof(T), height, &srcStep);
        Npp8u* dstOpen = nppiMalloc_8u_C1(width * sizeof(T), height, &dstOpenStep);
        Npp8u* dstNvidia = nppiMalloc_8u_C1(width * sizeof(T), height, &dstNvidiaStep);
        
        ASSERT_NE(src, nullptr);
        ASSERT_NE(dstOpen, nullptr);
        ASSERT_NE(dstNvidia, nullptr);
        
        T* typedSrc = reinterpret_cast<T*>(src);
        T* typedDstOpen = reinterpret_cast<T*>(dstOpen);
        T* typedDstNvidia = reinterpret_cast<T*>(dstNvidia);
        
        // 生成测试数据
        GenerateTestData(typedSrc, width, height, srcStep, pattern);
        
        NppiSize roi = {width, height};
        
        // 执行OpenNPP
        NppStatus statusOpen = CallOpenNppAddC(typedSrc, srcStep, constant, 
                                              typedDstOpen, dstOpenStep, 
                                              roi, scaleFactor);
        
        // 执行NVIDIA NPP
        NppStatus statusNvidia = CallNvidiaNppAddC(typedSrc, srcStep, constant,
                                                  typedDstNvidia, dstNvidiaStep,
                                                  roi, scaleFactor);
        
        EXPECT_EQ(statusOpen, NPP_SUCCESS) << "OpenNPP AddC failed";
        EXPECT_EQ(statusNvidia, NPP_SUCCESS) << "NVIDIA NPP AddC failed";
        
        // 比较结果
        auto stats = CompareResults(typedDstOpen, typedDstNvidia, 
                                   width, height, dstOpenStep, dstNvidiaStep);
        
        // 确定容差
        double tolerance = GetTolerance<T>(scaleFactor > 0);
        
        // 打印报告
        std::stringstream testName;
        testName << "AddC_" << TypeName<T>() << "_" << width << "x" << height;
        stats.PrintReport(testName.str());
        
        // 验证一致性
        EXPECT_TRUE(stats.IsConsistent(tolerance)) 
            << "Results not consistent, max diff: " << stats.GetMaxDifference()
            << ", tolerance: " << tolerance;
        
        // 性能对比
        if (width >= ConsistencyConfig::TestConfig::MEDIUM_WIDTH) {
            auto perfOpen = MeasurePerformance([&]() {
                CallOpenNppAddC(typedSrc, srcStep, constant, 
                              typedDstOpen, dstOpenStep, roi, scaleFactor);
            });
            
            auto perfNvidia = MeasurePerformance([&]() {
                CallNvidiaNppAddC(typedSrc, srcStep, constant,
                                typedDstNvidia, dstNvidiaStep, roi, scaleFactor);
            });
            
            PerformanceResult perf = {perfOpen, perfNvidia, perfOpen / perfNvidia};
            perf.Print();
        }
        
        nppiFree(src);
        nppiFree(dstOpen);
        nppiFree(dstNvidia);
    }
    
private:
    template<typename T>
    NppStatus CallOpenNppAddC(T* src, int srcStep, T constant, 
                             T* dst, int dstStep, NppiSize roi, int scaleFactor) {
        if constexpr (std::is_same_v<T, Npp8u>) {
            return nppiAddC_8u_C1RSfs(src, srcStep, constant, dst, dstStep, roi, scaleFactor);
        } else if constexpr (std::is_same_v<T, Npp16u>) {
            return nppiAddC_16u_C1RSfs(src, srcStep, constant, dst, dstStep, roi, scaleFactor);
        } else if constexpr (std::is_same_v<T, Npp16s>) {
            return nppiAddC_16s_C1RSfs(src, srcStep, constant, dst, dstStep, roi, scaleFactor);
        } else if constexpr (std::is_same_v<T, Npp32f>) {
            return nppiAddC_32f_C1R(src, srcStep, constant, dst, dstStep, roi);
        }
        return NPP_ERROR;
    }
    
#ifdef HAVE_NVIDIA_NPP
    template<typename T>
    NppStatus CallNvidiaNppAddC(T* src, int srcStep, T constant,
                               T* dst, int dstStep, NppiSize roi, int scaleFactor) {
        if constexpr (std::is_same_v<T, Npp8u>) {
            return ::nppiAddC_8u_C1RSfs(src, srcStep, constant, dst, dstStep, roi, scaleFactor);
        } else if constexpr (std::is_same_v<T, Npp16u>) {
            return ::nppiAddC_16u_C1RSfs(src, srcStep, constant, dst, dstStep, roi, scaleFactor);
        } else if constexpr (std::is_same_v<T, Npp16s>) {
            return ::nppiAddC_16s_C1RSfs(src, srcStep, constant, dst, dstStep, roi, scaleFactor);
        } else if constexpr (std::is_same_v<T, Npp32f>) {
            return ::nppiAddC_32f_C1R(src, srcStep, constant, dst, dstStep, roi);
        }
        return NPP_ERROR;
    }
#else
    template<typename T>
    NppStatus CallNvidiaNppAddC(T*, int, T, T*, int, NppiSize, int) {
        return NPP_ERROR;
    }
#endif
    
    template<typename T>
    double GetTolerance(bool hasScaleFactor) {
        if constexpr (std::is_same_v<T, Npp8u>) {
            return hasScaleFactor ? 2.0 : ConsistencyConfig::ToleranceConfig::TOLERANCE_8U;
        } else if constexpr (std::is_same_v<T, Npp16u>) {
            return hasScaleFactor ? 2.0 : ConsistencyConfig::ToleranceConfig::TOLERANCE_16U;
        } else if constexpr (std::is_same_v<T, Npp16s>) {
            return hasScaleFactor ? 2.0 : ConsistencyConfig::ToleranceConfig::TOLERANCE_16S;
        } else if constexpr (std::is_same_v<T, Npp32f>) {
            return ConsistencyConfig::ToleranceConfig::TOLERANCE_32F_ABSOLUTE;
        }
        return 1.0;
    }
    
    template<typename T>
    std::string TypeName() {
        if constexpr (std::is_same_v<T, Npp8u>) return "8u";
        else if constexpr (std::is_same_v<T, Npp16u>) return "16u";
        else if constexpr (std::is_same_v<T, Npp16s>) return "16s";
        else if constexpr (std::is_same_v<T, Npp32s>) return "32s";
        else if constexpr (std::is_same_v<T, Npp32f>) return "32f";
        return "unknown";
    }
};

// ==================== 具体测试用例 ====================

// 基本一致性测试
TEST_F(ArithmeticConsistencyTest, AddC_8u_Basic) {
    TestAddCConsistency<Npp8u>(512, 512, 50, 0, RANDOM);
}

TEST_F(ArithmeticConsistencyTest, AddC_16u_Basic) {
    TestAddCConsistency<Npp16u>(512, 512, 1000, 0, RANDOM);
}

TEST_F(ArithmeticConsistencyTest, AddC_32f_Basic) {
    TestAddCConsistency<Npp32f>(512, 512, 3.14159f, 0, RANDOM);
}

// 不同数据模式测试
TEST_F(ArithmeticConsistencyTest, AddC_8u_Gradient) {
    TestAddCConsistency<Npp8u>(256, 256, 25, 0, GRADIENT);
}

TEST_F(ArithmeticConsistencyTest, AddC_8u_Checkerboard) {
    TestAddCConsistency<Npp8u>(256, 256, 50, 0, CHECKERBOARD);
}

TEST_F(ArithmeticConsistencyTest, AddC_8u_EdgeCase) {
    TestAddCConsistency<Npp8u>(128, 128, 128, 0, EDGE_CASE);
}

// 带缩放因子的测试
TEST_F(ArithmeticConsistencyTest, AddC_8u_WithScaling) {
    TestAddCConsistency<Npp8u>(256, 256, 100, 1, RANDOM);  // 右移1位
}

// 非对齐尺寸测试
TEST_F(ArithmeticConsistencyTest, AddC_8u_UnalignedSize) {
    TestAddCConsistency<Npp8u>(1023, 769, 50, 0, RANDOM);
}

// 大尺寸性能对比
TEST_F(ArithmeticConsistencyTest, AddC_32f_LargeImage) {
    TestAddCConsistency<Npp32f>(1920, 1080, 2.5f, 0, RANDOM);
}

// ==================== 回归测试套件 ====================
class RegressionTest : public ConsistencyTestBase {
protected:
    struct RegressionCase {
        std::string name;
        int width;
        int height;
        double expectedMaxDiff;
        
        bool Validate(double actualMaxDiff) const {
            return actualMaxDiff <= expectedMaxDiff * 1.1; // 允许10%的误差
        }
    };
    
    void RunRegressionSuite() {
        std::vector<RegressionCase> cases = {
            {"Small_32x32", 32, 32, 1.0},
            {"Medium_512x512", 512, 512, 1.0},
            {"HD_1280x720", 1280, 720, 1.0},
            {"FullHD_1920x1080", 1920, 1080, 1.0},
            {"Prime_1021x769", 1021, 769, 1.0},
            {"Unaligned_1023x767", 1023, 767, 1.0}
        };
        
        std::cout << "\n=== 回归测试套件 ===\n";
        int passed = 0;
        
        for (const auto& testCase : cases) {
            // 运行测试...
            // double maxDiff = RunTest(testCase);
            // if (testCase.Validate(maxDiff)) passed++;
        }
        
        std::cout << "回归测试结果: " << passed << "/" << cases.size() << " 通过\n";
    }
};

// ==================== 批量验证测试 ====================
class BatchValidationTest : public ConsistencyTestBase {
protected:
    void ValidateAllDataTypes() {
        std::cout << "\n=== 批量数据类型验证 ===\n";
        
        struct TypeTest {
            std::string typeName;
            std::function<void()> testFunc;
        };
        
        std::vector<TypeTest> tests = {
            {"8u", [this]() { TestAddCConsistency<Npp8u>(256, 256, 50, 0, RANDOM); }},
            {"16u", [this]() { TestAddCConsistency<Npp16u>(256, 256, 1000, 0, RANDOM); }},
            {"16s", [this]() { TestAddCConsistency<Npp16s>(256, 256, -500, 0, RANDOM); }},
            {"32f", [this]() { TestAddCConsistency<Npp32f>(256, 256, 2.5f, 0, RANDOM); }}
        };
        
        for (const auto& test : tests) {
            std::cout << "测试 " << test.typeName << "...\n";
            test.testFunc();
        }
    }
    
    void ValidateAllSizes() {
        std::cout << "\n=== 批量尺寸验证 ===\n";
        
        std::vector<std::pair<int, int>> sizes = {
            {1, 1},
            {16, 16},
            {32, 32},
            {64, 64},
            {128, 128},
            {256, 256},
            {512, 512},
            {1024, 1024},
            {1920, 1080},
            {3840, 2160}
        };
        
        for (const auto& [width, height] : sizes) {
            std::cout << "测试尺寸 " << width << "x" << height << "...\n";
            TestAddCConsistency<Npp8u>(width, height, 50, 0, RANDOM);
        }
    }
    
private:
    // 从ArithmeticConsistencyTest继承TestAddCConsistency
    template<typename T>
    void TestAddCConsistency(int width, int height, T constant, 
                            int scaleFactor, DataPattern pattern) {
        // 实现省略，与ArithmeticConsistencyTest相同
    }
};

// ==================== 主函数由gtest_main提供 ====================