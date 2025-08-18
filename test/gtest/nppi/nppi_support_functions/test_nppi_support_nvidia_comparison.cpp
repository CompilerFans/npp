/**
 * @file test_nppi_support_nvidia_comparison.cpp
 * @brief OpenNPP与NVIDIA NPP支持函数对比测试
 * 
 * 测试内容：
 * - 内存分配函数的一致性
 * - Step/Pitch值的对比
 * - 不同数据类型和通道数的对比
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include "npp.h"

// 仅在有NVIDIA NPP时编译这些测试
#ifdef HAVE_NVIDIA_NPP

#include <nppi.h>

// 重命名避免冲突
namespace nv {
    using namespace std;
}

class NPPISupportComparisonTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        ASSERT_EQ(err, cudaSuccess);
    }
    
    void TearDown() override {
        cudaDeviceSynchronize();
    }
    
    // 辅助函数：比较step值
    void CompareStepValues(int openStep, int nvStep, const std::string& testName) {
        EXPECT_EQ(openStep, nvStep) 
            << testName << ": Step values differ - OpenNPP: " << openStep 
            << ", NVIDIA: " << nvStep;
        
        // 即使值不同，也应该满足基本对齐要求
        if (openStep != nvStep) {
            EXPECT_EQ(openStep % 32, 0) << "OpenNPP step not aligned";
            EXPECT_EQ(nvStep % 32, 0) << "NVIDIA step not aligned";
            
            // 记录差异百分比
            double diff = std::abs((double)(openStep - nvStep) / nvStep * 100);
            std::cout << testName << " step difference: " << diff << "%" << std::endl;
        }
    }
};

// ==================== 8-bit unsigned 测试 ====================

TEST_F(NPPISupportComparisonTest, Malloc_8u_C1_Comparison) {
    const int testCases[][2] = {
        {1, 1},       // 最小图像
        {64, 64},     // 小图像
        {640, 480},   // VGA
        {1920, 1080}, // Full HD
        {1023, 769},  // 非对齐尺寸
        {4096, 2160}  // 4K
    };
    
    for (const auto& tc : testCases) {
        int width = tc[0];
        int height = tc[1];
        int openStep, nvStep;
        
        // OpenNPP分配
        Npp8u* openPtr = nppiMalloc_8u_C1(width, height, &openStep);
        ASSERT_NE(openPtr, nullptr) << "OpenNPP allocation failed for " 
                                    << width << "x" << height;
        
        // NVIDIA NPP分配
        Npp8u* nvPtr = ::nppiMalloc_8u_C1(width, height, &nvStep);
        ASSERT_NE(nvPtr, nullptr) << "NVIDIA allocation failed for " 
                                  << width << "x" << height;
        
        // 比较step值
        std::string testName = "8u_C1_" + std::to_string(width) + "x" + std::to_string(height);
        CompareStepValues(openStep, nvStep, testName);
        
        // 清理
        nppiFree(openPtr);
        ::nppiFree(nvPtr);
    }
}

TEST_F(NPPISupportComparisonTest, Malloc_8u_C3_Comparison) {
    const int widths[] = {100, 640, 1920, 1023};
    const int heights[] = {100, 480, 1080, 769};
    
    for (int i = 0; i < 4; i++) {
        int width = widths[i];
        int height = heights[i];
        int openStep, nvStep;
        
        // OpenNPP分配
        Npp8u* openPtr = nppiMalloc_8u_C3(width, height, &openStep);
        ASSERT_NE(openPtr, nullptr);
        
        // NVIDIA NPP分配
        Npp8u* nvPtr = ::nppiMalloc_8u_C3(width, height, &nvStep);
        ASSERT_NE(nvPtr, nullptr);
        
        // 比较step值
        std::string testName = "8u_C3_" + std::to_string(width) + "x" + std::to_string(height);
        CompareStepValues(openStep, nvStep, testName);
        
        // 验证最小要求
        EXPECT_GE(openStep, width * 3) << "OpenNPP step less than minimum";
        EXPECT_GE(nvStep, width * 3) << "NVIDIA step less than minimum";
        
        // 清理
        nppiFree(openPtr);
        ::nppiFree(nvPtr);
    }
}

TEST_F(NPPISupportComparisonTest, Malloc_8u_C4_Comparison) {
    const int testSizes[] = {1, 31, 32, 33, 64, 127, 128, 129, 256, 512, 1024, 2048};
    
    for (int size : testSizes) {
        int openStep, nvStep;
        
        // OpenNPP分配
        Npp8u* openPtr = nppiMalloc_8u_C4(size, size, &openStep);
        ASSERT_NE(openPtr, nullptr);
        
        // NVIDIA NPP分配
        Npp8u* nvPtr = ::nppiMalloc_8u_C4(size, size, &nvStep);
        ASSERT_NE(nvPtr, nullptr);
        
        // 比较step值
        std::string testName = "8u_C4_" + std::to_string(size) + "x" + std::to_string(size);
        CompareStepValues(openStep, nvStep, testName);
        
        // 清理
        nppiFree(openPtr);
        ::nppiFree(nvPtr);
    }
}

// ==================== 16-bit unsigned 测试 ====================

TEST_F(NPPISupportComparisonTest, Malloc_16u_C1_Comparison) {
    const int widths[] = {512, 1024, 1920, 777};
    const int heights[] = {512, 768, 1080, 555};
    
    for (int i = 0; i < 4; i++) {
        int width = widths[i];
        int height = heights[i];
        int openStep, nvStep;
        
        // OpenNPP分配
        Npp16u* openPtr = nppiMalloc_16u_C1(width, height, &openStep);
        ASSERT_NE(openPtr, nullptr);
        
        // NVIDIA NPP分配
        Npp16u* nvPtr = ::nppiMalloc_16u_C1(width, height, &nvStep);
        ASSERT_NE(nvPtr, nullptr);
        
        // 比较step值
        std::string testName = "16u_C1_" + std::to_string(width) + "x" + std::to_string(height);
        CompareStepValues(openStep, nvStep, testName);
        
        // 验证最小要求
        EXPECT_GE(openStep, width * sizeof(Npp16u));
        EXPECT_GE(nvStep, width * sizeof(Npp16u));
        
        // 清理
        nppiFree(openPtr);
        ::nppiFree(nvPtr);
    }
}

// ==================== 32-bit float 测试 ====================

TEST_F(NPPISupportComparisonTest, Malloc_32f_C1_Comparison) {
    const int testCases[][2] = {
        {256, 256},
        {640, 480},
        {1920, 1080},
        {333, 444}  // 非对齐尺寸
    };
    
    for (const auto& tc : testCases) {
        int width = tc[0];
        int height = tc[1];
        int openStep, nvStep;
        
        // OpenNPP分配
        Npp32f* openPtr = nppiMalloc_32f_C1(width, height, &openStep);
        ASSERT_NE(openPtr, nullptr);
        
        // NVIDIA NPP分配
        Npp32f* nvPtr = ::nppiMalloc_32f_C1(width, height, &nvStep);
        ASSERT_NE(nvPtr, nullptr);
        
        // 比较step值
        std::string testName = "32f_C1_" + std::to_string(width) + "x" + std::to_string(height);
        CompareStepValues(openStep, nvStep, testName);
        
        // 验证最小要求
        EXPECT_GE(openStep, width * sizeof(Npp32f));
        EXPECT_GE(nvStep, width * sizeof(Npp32f));
        
        // 清理
        nppiFree(openPtr);
        ::nppiFree(nvPtr);
    }
}

TEST_F(NPPISupportComparisonTest, Malloc_32f_C3_Comparison) {
    const int sizes[] = {64, 128, 256, 512, 1024};
    
    for (int size : sizes) {
        int openStep, nvStep;
        
        // OpenNPP分配
        Npp32f* openPtr = nppiMalloc_32f_C3(size, size, &openStep);
        ASSERT_NE(openPtr, nullptr);
        
        // NVIDIA NPP分配
        Npp32f* nvPtr = ::nppiMalloc_32f_C3(size, size, &nvStep);
        ASSERT_NE(nvPtr, nullptr);
        
        // 比较step值
        std::string testName = "32f_C3_" + std::to_string(size);
        CompareStepValues(openStep, nvStep, testName);
        
        // 清理
        nppiFree(openPtr);
        ::nppiFree(nvPtr);
    }
}

// ==================== 混合测试 ====================

TEST_F(NPPISupportComparisonTest, MixedDataTypes_Comparison) {
    const int width = 512;
    const int height = 512;
    
    struct TestCase {
        std::string name;
        int openStep;
        int nvStep;
    };
    
    std::vector<TestCase> results;
    
    // 8u_C1
    {
        int openStep, nvStep;
        Npp8u* openPtr = nppiMalloc_8u_C1(width, height, &openStep);
        Npp8u* nvPtr = ::nppiMalloc_8u_C1(width, height, &nvStep);
        results.push_back({"8u_C1", openStep, nvStep});
        nppiFree(openPtr);
        ::nppiFree(nvPtr);
    }
    
    // 8u_C3
    {
        int openStep, nvStep;
        Npp8u* openPtr = nppiMalloc_8u_C3(width, height, &openStep);
        Npp8u* nvPtr = ::nppiMalloc_8u_C3(width, height, &nvStep);
        results.push_back({"8u_C3", openStep, nvStep});
        nppiFree(openPtr);
        ::nppiFree(nvPtr);
    }
    
    // 16u_C1
    {
        int openStep, nvStep;
        Npp16u* openPtr = nppiMalloc_16u_C1(width, height, &openStep);
        Npp16u* nvPtr = ::nppiMalloc_16u_C1(width, height, &nvStep);
        results.push_back({"16u_C1", openStep, nvStep});
        nppiFree(openPtr);
        ::nppiFree(nvPtr);
    }
    
    // 32f_C1
    {
        int openStep, nvStep;
        Npp32f* openPtr = nppiMalloc_32f_C1(width, height, &openStep);
        Npp32f* nvPtr = ::nppiMalloc_32f_C1(width, height, &nvStep);
        results.push_back({"32f_C1", openStep, nvStep});
        nppiFree(openPtr);
        ::nppiFree(nvPtr);
    }
    
    // 32f_C4
    {
        int openStep, nvStep;
        Npp32f* openPtr = nppiMalloc_32f_C4(width, height, &openStep);
        Npp32f* nvPtr = ::nppiMalloc_32f_C4(width, height, &nvStep);
        results.push_back({"32f_C4", openStep, nvStep});
        nppiFree(openPtr);
        ::nppiFree(nvPtr);
    }
    
    // 打印对比结果
    std::cout << "\n=== Step Values Comparison (512x512) ===" << std::endl;
    std::cout << "Type\t\tOpenNPP\tNVIDIA\tMatch" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    int matchCount = 0;
    for (const auto& r : results) {
        bool match = (r.openStep == r.nvStep);
        if (match) matchCount++;
        
        std::cout << r.name << "\t\t" 
                  << r.openStep << "\t" 
                  << r.nvStep << "\t"
                  << (match ? "✓" : "✗") << std::endl;
    }
    
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Match rate: " << matchCount << "/" << results.size() 
              << " (" << (matchCount * 100.0 / results.size()) << "%)" << std::endl;
}

// ==================== 边界条件测试 ====================

TEST_F(NPPISupportComparisonTest, BoundaryConditions_Comparison) {
    struct BoundaryTest {
        int width;
        int height;
        std::string description;
    };
    
    std::vector<BoundaryTest> tests = {
        {1, 1, "Minimum size"},
        {1, 1000, "Single column"},
        {1000, 1, "Single row"},
        {31, 31, "Just below 32"},
        {32, 32, "Exactly 32"},
        {33, 33, "Just above 32"},
        {255, 255, "Just below 256"},
        {256, 256, "Exactly 256"},
        {257, 257, "Just above 256"},
        {511, 511, "Just below 512"},
        {512, 512, "Exactly 512"},
        {513, 513, "Just above 512"}
    };
    
    std::cout << "\n=== Boundary Conditions Step Comparison ===" << std::endl;
    std::cout << "Size\t\tDescription\t\tOpenNPP\tNVIDIA\tMatch" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;
    
    for (const auto& test : tests) {
        int openStep, nvStep;
        
        Npp8u* openPtr = nppiMalloc_8u_C1(test.width, test.height, &openStep);
        Npp8u* nvPtr = ::nppiMalloc_8u_C1(test.width, test.height, &nvStep);
        
        bool match = (openStep == nvStep);
        
        std::cout << test.width << "x" << test.height << "\t"
                  << test.description << "\t\t"
                  << openStep << "\t"
                  << nvStep << "\t"
                  << (match ? "✓" : "✗") << std::endl;
        
        nppiFree(openPtr);
        ::nppiFree(nvPtr);
    }
}

// ==================== 性能对比测试 ====================

TEST_F(NPPISupportComparisonTest, AllocationPerformance_Comparison) {
    const int iterations = 1000;
    const int width = 1920;
    const int height = 1080;
    
    // OpenNPP性能测试
    cudaEvent_t startOpen, stopOpen;
    cudaEventCreate(&startOpen);
    cudaEventCreate(&stopOpen);
    
    cudaEventRecord(startOpen);
    for (int i = 0; i < iterations; i++) {
        int step;
        Npp8u* ptr = nppiMalloc_8u_C3(width, height, &step);
        nppiFree(ptr);
    }
    cudaEventRecord(stopOpen);
    cudaEventSynchronize(stopOpen);
    
    float openTime;
    cudaEventElapsedTime(&openTime, startOpen, stopOpen);
    
    // NVIDIA NPP性能测试
    cudaEvent_t startNv, stopNv;
    cudaEventCreate(&startNv);
    cudaEventCreate(&stopNv);
    
    cudaEventRecord(startNv);
    for (int i = 0; i < iterations; i++) {
        int step;
        Npp8u* ptr = ::nppiMalloc_8u_C3(width, height, &step);
        ::nppiFree(ptr);
    }
    cudaEventRecord(stopNv);
    cudaEventSynchronize(stopNv);
    
    float nvTime;
    cudaEventElapsedTime(&nvTime, startNv, stopNv);
    
    // 打印性能对比
    std::cout << "\n=== Allocation Performance (1920x1080 RGB, " 
              << iterations << " iterations) ===" << std::endl;
    std::cout << "OpenNPP: " << openTime << " ms" << std::endl;
    std::cout << "NVIDIA: " << nvTime << " ms" << std::endl;
    std::cout << "Speed ratio: " << (nvTime / openTime) << "x" << std::endl;
    
    // 性能应该相近（都是调用cudaMallocPitch）
    double ratio = openTime / nvTime;
    EXPECT_GT(ratio, 0.5) << "OpenNPP too slow compared to NVIDIA";
    EXPECT_LT(ratio, 2.0) << "OpenNPP unexpectedly faster than NVIDIA";
    
    // 清理
    cudaEventDestroy(startOpen);
    cudaEventDestroy(stopOpen);
    cudaEventDestroy(startNv);
    cudaEventDestroy(stopNv);
}

#else // !HAVE_NVIDIA_NPP

TEST(NPPISupportComparisonTest, DISABLED_NoNvidiaNPP) {
    GTEST_SKIP() << "NVIDIA NPP not available, skipping comparison tests";
}

#endif // HAVE_NVIDIA_NPP

// Main函数由gtest_main提供