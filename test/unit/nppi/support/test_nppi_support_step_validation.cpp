/**
 * @file test_nppi_support_step_validation.cpp
 * @brief 全面验证OpenNPP与NVIDIA NPP内存分配step一致性
 * 
 * 测试目标：
 * 1. 验证所有数据类型的step对齐
 * 2. 比较OpenNPP与NVIDIA NPP的step值
 * 3. 测试各种宽度下的对齐行为
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <map>
#include "npp.h"

#ifdef HAVE_NVIDIA_NPP
#include <nppi_support_functions.h>
#endif

class NPPIStepValidationTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        ASSERT_EQ(err, cudaSuccess);
        
#ifndef HAVE_NVIDIA_NPP
        GTEST_SKIP() << "NVIDIA NPP not available for comparison";
#endif
    }
    
    void TearDown() override {
        cudaDeviceSynchronize();
    }
    
    // 测试结果结构
    struct StepComparison {
        int openNppStep;
        int nvidiaNppStep;
        bool match;
        double deviation; // 百分比偏差
    };
    
    // 执行step比较
    template<typename AllocFunc>
    StepComparison CompareSteps(AllocFunc openNppAlloc, AllocFunc nvidiaNppAlloc,
                                int width, int height) {
        StepComparison result;
        int openStep, nvidiaStep;
        
        // OpenNPP分配
        void* openPtr = openNppAlloc(width, height, &openStep);
        EXPECT_NE(openPtr, nullptr) << "OpenNPP allocation failed";
        
        // NVIDIA NPP分配
        void* nvidiaPtr = nvidiaNppAlloc(width, height, &nvidiaStep);
        EXPECT_NE(nvidiaPtr, nullptr) << "NVIDIA NPP allocation failed";
        
        result.openNppStep = openStep;
        result.nvidiaNppStep = nvidiaStep;
        result.match = (openStep == nvidiaStep);
        
        if (nvidiaStep > 0) {
            result.deviation = std::abs(openStep - nvidiaStep) * 100.0 / nvidiaStep;
        } else {
            result.deviation = 0;
        }
        
        // 清理
        if (openPtr) nppiFree(openPtr);
        if (nvidiaPtr) ::nppiFree(nvidiaPtr);
        
        return result;
    }
    
    // 批量测试不同宽度
    void TestMultipleWidths(const std::string& typeName, int elementSize, int channels) {
        std::cout << "\n=== " << typeName << " Step对齐测试 ===\n";
        std::cout << std::setw(10) << "Width" 
                  << std::setw(15) << "Expected Min"
                  << std::setw(15) << "OpenNPP Step"
                  << std::setw(15) << "NVIDIA Step"
                  << std::setw(10) << "Match"
                  << std::setw(15) << "Deviation(%)\n";
        std::cout << std::string(80, '-') << "\n";
        
        // 测试各种宽度
        std::vector<int> testWidths = {
            1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128,
            255, 256, 511, 512, 1023, 1024, 1920, 2047, 2048, 4095, 4096
        };
        
        int totalTests = 0;
        int matchCount = 0;
        double maxDeviation = 0;
        
        for (int width : testWidths) {
            int expectedMin = width * elementSize * channels;
            
            // 这里需要根据实际类型调用相应的分配函数
            // 由于模板限制，这里只是示例
            
            totalTests++;
            // 实际比较会在具体测试中进行
        }
        
        std::cout << "\n统计: " << matchCount << "/" << totalTests 
                  << " 匹配 (最大偏差: " << maxDeviation << "%)\n";
    }
};

// ==================== 8位无符号测试 ====================
#ifdef HAVE_NVIDIA_NPP

TEST_F(NPPIStepValidationTest, Compare_8u_C1_Steps) {
    std::cout << "\n=== Npp8u_C1 Step对齐验证 ===\n";
    std::cout << std::setw(10) << "Width" 
              << std::setw(15) << "Expected Min"
              << std::setw(15) << "OpenNPP Step"
              << std::setw(15) << "NVIDIA Step"
              << std::setw(10) << "Match\n";
    std::cout << std::string(65, '-') << "\n";
    
    std::vector<int> testWidths = {
        1, 7, 8, 31, 32, 63, 64, 127, 128, 255, 256, 511, 512,
        1023, 1024, 1920, 2047, 2048, 4095, 4096
    };
    
    int height = 100;
    int matchCount = 0;
    
    for (int width : testWidths) {
        int openStep, nvidiaStep;
        
        // OpenNPP
        Npp8u* openPtr = nppiMalloc_8u_C1(width, height, &openStep);
        ASSERT_NE(openPtr, nullptr);
        
        // NVIDIA NPP
        Npp8u* nvidiaPtr = ::nppiMalloc_8u_C1(width, height, &nvidiaStep);
        ASSERT_NE(nvidiaPtr, nullptr);
        
        bool match = (openStep == nvidiaStep);
        if (match) matchCount++;
        
        std::cout << std::setw(10) << width
                  << std::setw(15) << width  // Expected minimum
                  << std::setw(15) << openStep
                  << std::setw(15) << nvidiaStep
                  << std::setw(10) << (match ? "✓" : "✗") << "\n";
        
        nppiFree(openPtr);
        ::nppiFree(nvidiaPtr);
    }
    
    std::cout << "\n结果: " << matchCount << "/" << testWidths.size() << " 完全匹配\n";
    EXPECT_EQ(matchCount, testWidths.size()) << "Step values should match NVIDIA NPP";
}

TEST_F(NPPIStepValidationTest, Compare_8u_C3_Steps) {
    std::cout << "\n=== Npp8u_C3 Step对齐验证 ===\n";
    std::cout << std::setw(10) << "Width" 
              << std::setw(15) << "Expected Min"
              << std::setw(15) << "OpenNPP Step"
              << std::setw(15) << "NVIDIA Step"
              << std::setw(10) << "Match\n";
    std::cout << std::string(65, '-') << "\n";
    
    std::vector<int> testWidths = {1, 8, 32, 64, 128, 256, 512, 1024, 1920, 4096};
    int height = 100;
    int matchCount = 0;
    
    for (int width : testWidths) {
        int openStep, nvidiaStep;
        
        Npp8u* openPtr = nppiMalloc_8u_C3(width, height, &openStep);
        ASSERT_NE(openPtr, nullptr);
        
        Npp8u* nvidiaPtr = ::nppiMalloc_8u_C3(width, height, &nvidiaStep);
        ASSERT_NE(nvidiaPtr, nullptr);
        
        bool match = (openStep == nvidiaStep);
        if (match) matchCount++;
        
        std::cout << std::setw(10) << width
                  << std::setw(15) << (width * 3)  // Expected minimum for C3
                  << std::setw(15) << openStep
                  << std::setw(15) << nvidiaStep
                  << std::setw(10) << (match ? "✓" : "✗") << "\n";
        
        nppiFree(openPtr);
        ::nppiFree(nvidiaPtr);
    }
    
    std::cout << "\n结果: " << matchCount << "/" << testWidths.size() << " 完全匹配\n";
    EXPECT_EQ(matchCount, testWidths.size());
}

TEST_F(NPPIStepValidationTest, Compare_8u_C4_Steps) {
    std::cout << "\n=== Npp8u_C4 Step对齐验证 ===\n";
    
    std::vector<int> testWidths = {1, 8, 32, 64, 128, 256, 512, 1024, 1920};
    int height = 100;
    int matchCount = 0;
    
    for (int width : testWidths) {
        int openStep, nvidiaStep;
        
        Npp8u* openPtr = nppiMalloc_8u_C4(width, height, &openStep);
        ASSERT_NE(openPtr, nullptr);
        
        Npp8u* nvidiaPtr = ::nppiMalloc_8u_C4(width, height, &nvidiaStep);
        ASSERT_NE(nvidiaPtr, nullptr);
        
        bool match = (openStep == nvidiaStep);
        if (match) matchCount++;
        
        EXPECT_EQ(openStep, nvidiaStep) 
            << "Width " << width << ": OpenNPP=" << openStep 
            << " vs NVIDIA=" << nvidiaStep;
        
        nppiFree(openPtr);
        ::nppiFree(nvidiaPtr);
    }
    
    EXPECT_EQ(matchCount, testWidths.size());
}

// ==================== 16位测试 ====================
TEST_F(NPPIStepValidationTest, Compare_16u_C1_Steps) {
    std::cout << "\n=== Npp16u_C1 Step对齐验证 ===\n";
    
    std::vector<int> testWidths = {1, 8, 32, 64, 128, 256, 512, 1024, 1920};
    int height = 100;
    int matchCount = 0;
    
    for (int width : testWidths) {
        int openStep, nvidiaStep;
        
        Npp16u* openPtr = nppiMalloc_16u_C1(width, height, &openStep);
        ASSERT_NE(openPtr, nullptr);
        
        Npp16u* nvidiaPtr = ::nppiMalloc_16u_C1(width, height, &nvidiaStep);
        ASSERT_NE(nvidiaPtr, nullptr);
        
        bool match = (openStep == nvidiaStep);
        if (match) matchCount++;
        
        EXPECT_EQ(openStep, nvidiaStep)
            << "Width " << width << ": OpenNPP=" << openStep 
            << " vs NVIDIA=" << nvidiaStep;
        
        nppiFree(openPtr);
        ::nppiFree(nvidiaPtr);
    }
    
    EXPECT_EQ(matchCount, testWidths.size());
}

TEST_F(NPPIStepValidationTest, Compare_16s_C1_Steps) {
    std::cout << "\n=== Npp16s_C1 Step对齐验证 ===\n";
    
    std::vector<int> testWidths = {1, 8, 32, 64, 128, 256, 512, 1024};
    int height = 100;
    int matchCount = 0;
    
    for (int width : testWidths) {
        int openStep, nvidiaStep;
        
        Npp16s* openPtr = nppiMalloc_16s_C1(width, height, &openStep);
        ASSERT_NE(openPtr, nullptr);
        
        Npp16s* nvidiaPtr = ::nppiMalloc_16s_C1(width, height, &nvidiaStep);
        ASSERT_NE(nvidiaPtr, nullptr);
        
        bool match = (openStep == nvidiaStep);
        if (match) matchCount++;
        
        EXPECT_EQ(openStep, nvidiaStep);
        
        nppiFree(openPtr);
        ::nppiFree(nvidiaPtr);
    }
    
    EXPECT_EQ(matchCount, testWidths.size());
}

// ==================== 32位测试 ====================
TEST_F(NPPIStepValidationTest, Compare_32s_C1_Steps) {
    std::cout << "\n=== Npp32s_C1 Step对齐验证 ===\n";
    
    std::vector<int> testWidths = {1, 8, 32, 64, 128, 256, 512, 1024};
    int height = 100;
    int matchCount = 0;
    
    for (int width : testWidths) {
        int openStep, nvidiaStep;
        
        Npp32s* openPtr = nppiMalloc_32s_C1(width, height, &openStep);
        ASSERT_NE(openPtr, nullptr);
        
        Npp32s* nvidiaPtr = ::nppiMalloc_32s_C1(width, height, &nvidiaStep);
        ASSERT_NE(nvidiaPtr, nullptr);
        
        bool match = (openStep == nvidiaStep);
        if (match) matchCount++;
        
        EXPECT_EQ(openStep, nvidiaStep)
            << "Width " << width << ": OpenNPP=" << openStep 
            << " vs NVIDIA=" << nvidiaStep;
        
        nppiFree(openPtr);
        ::nppiFree(nvidiaPtr);
    }
    
    EXPECT_EQ(matchCount, testWidths.size());
}

TEST_F(NPPIStepValidationTest, Compare_32f_C1_Steps) {
    std::cout << "\n=== Npp32f_C1 Step对齐验证 ===\n";
    
    std::vector<int> testWidths = {1, 8, 32, 64, 128, 256, 512, 1024, 1920};
    int height = 100;
    int matchCount = 0;
    
    for (int width : testWidths) {
        int openStep, nvidiaStep;
        
        Npp32f* openPtr = nppiMalloc_32f_C1(width, height, &openStep);
        ASSERT_NE(openPtr, nullptr);
        
        Npp32f* nvidiaPtr = ::nppiMalloc_32f_C1(width, height, &nvidiaStep);
        ASSERT_NE(nvidiaPtr, nullptr);
        
        bool match = (openStep == nvidiaStep);
        if (match) matchCount++;
        
        EXPECT_EQ(openStep, nvidiaStep)
            << "Width " << width << ": OpenNPP=" << openStep 
            << " vs NVIDIA=" << nvidiaStep;
        
        nppiFree(openPtr);
        ::nppiFree(nvidiaPtr);
    }
    
    EXPECT_EQ(matchCount, testWidths.size());
}

// ==================== 复数类型测试 ====================
TEST_F(NPPIStepValidationTest, Compare_32fc_C1_Steps) {
    std::cout << "\n=== Npp32fc_C1 (复数) Step对齐验证 ===\n";
    
    std::vector<int> testWidths = {1, 8, 32, 64, 128, 256, 512, 1024};
    int height = 100;
    int matchCount = 0;
    
    for (int width : testWidths) {
        int openStep, nvidiaStep;
        
        Npp32fc* openPtr = nppiMalloc_32fc_C1(width, height, &openStep);
        ASSERT_NE(openPtr, nullptr);
        
        Npp32fc* nvidiaPtr = ::nppiMalloc_32fc_C1(width, height, &nvidiaStep);
        ASSERT_NE(nvidiaPtr, nullptr);
        
        bool match = (openStep == nvidiaStep);
        if (match) matchCount++;
        
        // 复数类型每个元素是8字节（2个float）
        int expectedMin = width * sizeof(Npp32fc);
        std::cout << "Width " << width 
                  << ": Expected>=" << expectedMin
                  << ", OpenNPP=" << openStep 
                  << ", NVIDIA=" << nvidiaStep 
                  << " " << (match ? "✓" : "✗") << "\n";
        
        EXPECT_EQ(openStep, nvidiaStep);
        
        nppiFree(openPtr);
        ::nppiFree(nvidiaPtr);
    }
    
    EXPECT_EQ(matchCount, testWidths.size());
}

// ==================== 极端情况测试 ====================
TEST_F(NPPIStepValidationTest, ExtremeWidthTests) {
    std::cout << "\n=== 极端宽度测试 ===\n";
    
    struct TestCase {
        int width;
        int height;
        const char* description;
    };
    
    std::vector<TestCase> testCases = {
        {1, 1, "最小图像"},
        {1, 10000, "极窄图像"},
        {10000, 1, "极宽图像"},
        {4096, 2160, "4K分辨率"},
        {7680, 4320, "8K分辨率"},
        {1921, 1081, "非标准HD"},
        {2047, 2047, "接近2的幂"},
        {2049, 2049, "略超2的幂"}
    };
    
    for (const auto& tc : testCases) {
        int openStep, nvidiaStep;
        
        std::cout << tc.description << " (" << tc.width << "x" << tc.height << "): ";
        
        Npp8u* openPtr = nppiMalloc_8u_C1(tc.width, tc.height, &openStep);
        Npp8u* nvidiaPtr = ::nppiMalloc_8u_C1(tc.width, tc.height, &nvidiaStep);
        
        if (openPtr && nvidiaPtr) {
            bool match = (openStep == nvidiaStep);
            std::cout << "OpenNPP=" << openStep 
                      << ", NVIDIA=" << nvidiaStep 
                      << " " << (match ? "✓" : "✗") << "\n";
            
            EXPECT_EQ(openStep, nvidiaStep) 
                << "Mismatch for " << tc.description;
        } else {
            std::cout << "分配失败\n";
        }
        
        if (openPtr) nppiFree(openPtr);
        if (nvidiaPtr) ::nppiFree(nvidiaPtr);
    }
}

// ==================== 对齐模式分析 ====================
TEST_F(NPPIStepValidationTest, AlignmentPatternAnalysis) {
    std::cout << "\n=== 对齐模式分析 ===\n";
    std::cout << "测试CUDA内存对齐模式...\n\n";
    
    // 分析不同数据类型的对齐模式
    std::map<std::string, std::vector<std::pair<int, int>>> alignmentData;
    
    // 测试8u_C1
    for (int width = 1; width <= 256; width++) {
        int openStep, nvidiaStep;
        Npp8u* openPtr = nppiMalloc_8u_C1(width, 10, &openStep);
        Npp8u* nvidiaPtr = ::nppiMalloc_8u_C1(width, 10, &nvidiaStep);
        
        if (openPtr && nvidiaPtr && openStep == nvidiaStep) {
            int alignment = openStep - width;
            if (alignment > 0) {
                alignmentData["8u_C1"].push_back({width, alignment});
            }
        }
        
        if (openPtr) nppiFree(openPtr);
        if (nvidiaPtr) ::nppiFree(nvidiaPtr);
    }
    
    // 分析对齐规律
    std::cout << "发现的对齐规律:\n";
    for (const auto& [type, data] : alignmentData) {
        if (!data.empty()) {
            std::cout << type << ": ";
            
            // 查找最小对齐
            int minAlign = INT_MAX;
            for (const auto& [width, align] : data) {
                if (align < minAlign && align > 0) {
                    minAlign = align;
                }
            }
            
            // 检查是否是固定对齐值
            bool consistent = true;
            int commonAlign = data[0].second;
            for (const auto& [width, align] : data) {
                if (std::abs(align - commonAlign) > width * 0.1) {
                    consistent = false;
                    break;
                }
            }
            
            if (consistent) {
                std::cout << "固定填充模式\n";
            } else {
                std::cout << "动态对齐模式\n";
            }
        }
    }
}

// ==================== 性能影响测试 ====================
TEST_F(NPPIStepValidationTest, StepPerformanceImpact) {
    std::cout << "\n=== Step对性能的影响 ===\n";
    
    // 测试不同step值对内存访问性能的影响
    std::vector<int> testWidths = {1920, 1921, 1984, 2048};
    int height = 1080;
    
    std::cout << "Width\tStep\tAlignment\tExpected Performance\n";
    std::cout << std::string(50, '-') << "\n";
    
    for (int width : testWidths) {
        int step;
        Npp8u* ptr = nppiMalloc_8u_C1(width, height, &step);
        
        if (ptr) {
            int alignment = step - width;
            std::string performance;
            
            if (step % 128 == 0) {
                performance = "最优（128字节对齐）";
            } else if (step % 64 == 0) {
                performance = "良好（64字节对齐）";
            } else if (step % 32 == 0) {
                performance = "可接受（32字节对齐）";
            } else {
                performance = "较差（非优化对齐）";
            }
            
            std::cout << width << "\t" << step << "\t" 
                      << alignment << "\t\t" << performance << "\n";
            
            nppiFree(ptr);
        }
    }
}

#endif // HAVE_NVIDIA_NPP

// ==================== 主函数 ====================
// Main函数由gtest_main提供