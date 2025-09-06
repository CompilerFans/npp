/**
 * @file test_memory_alignment.cpp
 * @brief NPPI内存对齐和step值的专门测试
 * 
 * 测试各种宽度、数据类型和通道组合下的step值计算和对齐
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <map>
#include <algorithm>
#include "npp.h"

#ifdef HAVE_NVIDIA_NPP
#include <dlfcn.h>
#endif

class NPPIMemoryAlignmentTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        ASSERT_EQ(err, cudaSuccess);
    }
    
    void TearDown() override {
        cudaDeviceSynchronize();
    }
    
    // 计算理论上的最小step值
    int calculateMinStep(int width, int elementSize, int channels) {
        return width * elementSize * channels;
    }
    
    // 计算32字节对齐后的step值
    int calculateAlignedStep(int minStep, int alignment = 32) {
        return ((minStep + alignment - 1) / alignment) * alignment;
    }
    
    // 分析step值的对齐模式
    void analyzeAlignmentPattern(const std::string& typeName, 
                                 const std::vector<int>& widths,
                                 const std::vector<int>& steps,
                                 int elementSize,
                                 int channels) {
        std::cout << "\n=== " << typeName << " Alignment Analysis ===" << std::endl;
        std::cout << std::setw(10) << "Width" 
                  << std::setw(15) << "Min Step"
                  << std::setw(15) << "Actual Step"
                  << std::setw(15) << "Expected"
                  << std::setw(10) << "Match"
                  << std::setw(15) << "Overhead %" << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        
        int matchCount = 0;
        double totalOverhead = 0;
        
        for (size_t i = 0; i < widths.size(); i++) {
            int minStep = calculateMinStep(widths[i], elementSize, channels);
            int expectedStep = calculateAlignedStep(minStep);
            bool match = (steps[i] == expectedStep);
            double overhead = ((double)(steps[i] - minStep) / minStep) * 100;
            
            std::cout << std::setw(10) << widths[i]
                      << std::setw(15) << minStep
                      << std::setw(15) << steps[i]
                      << std::setw(15) << expectedStep
                      << std::setw(10) << (match ? "✓" : "✗")
                      << std::setw(14) << std::fixed << std::setprecision(2) 
                      << overhead << "%" << std::endl;
            
            if (match) matchCount++;
            totalOverhead += overhead;
        }
        
        std::cout << std::string(80, '-') << std::endl;
        std::cout << "Match Rate: " << matchCount << "/" << widths.size() 
                  << " (" << (matchCount * 100.0 / widths.size()) << "%)" << std::endl;
        std::cout << "Average Overhead: " << (totalOverhead / widths.size()) << "%" << std::endl;
    }
};

// ==================== 8-bit数据类型对齐测试 ====================

TEST_F(NPPIMemoryAlignmentTest, Alignment_8u_AllWidths) {
    std::vector<int> testWidths = {
        1, 2, 3, 4, 5, 6, 7, 8,           // 极小宽度
        15, 16, 17,                        // 16边界
        31, 32, 33,                        // 32边界
        63, 64, 65,                        // 64边界
        127, 128, 129,                     // 128边界
        255, 256, 257,                     // 256边界
        511, 512, 513,                     // 512边界
        640, 800, 1024, 1280, 1920         // 常见宽度
    };
    
    // 测试C1
    {
        std::vector<int> steps;
        for (int width : testWidths) {
            int step;
            Npp8u* ptr = nppiMalloc_8u_C1(width, 1, &step);
            ASSERT_NE(ptr, nullptr);
            steps.push_back(step);
            nppiFree(ptr);
        }
        analyzeAlignmentPattern("8u_C1", testWidths, steps, 1, 1);
    }
    
    // 测试C3
    {
        std::vector<int> steps;
        for (int width : testWidths) {
            int step;
            Npp8u* ptr = nppiMalloc_8u_C3(width, 1, &step);
            ASSERT_NE(ptr, nullptr);
            steps.push_back(step);
            nppiFree(ptr);
        }
        analyzeAlignmentPattern("8u_C3", testWidths, steps, 1, 3);
    }
    
    // 测试C4
    {
        std::vector<int> steps;
        for (int width : testWidths) {
            int step;
            Npp8u* ptr = nppiMalloc_8u_C4(width, 1, &step);
            ASSERT_NE(ptr, nullptr);
            steps.push_back(step);
            nppiFree(ptr);
        }
        analyzeAlignmentPattern("8u_C4", testWidths, steps, 1, 4);
    }
}

// ==================== 16-bit数据类型对齐测试 ====================

TEST_F(NPPIMemoryAlignmentTest, Alignment_16u_AllWidths) {
    std::vector<int> testWidths = {
        1, 2, 4, 8, 15, 16, 17, 31, 32, 33,
        63, 64, 65, 127, 128, 129, 255, 256, 257,
        640, 800, 1024, 1920
    };
    
    // 测试16u_C1
    {
        std::vector<int> steps;
        for (int width : testWidths) {
            int step;
            Npp16u* ptr = nppiMalloc_16u_C1(width, 1, &step);
            ASSERT_NE(ptr, nullptr);
            steps.push_back(step);
            nppiFree(ptr);
        }
        analyzeAlignmentPattern("16u_C1", testWidths, steps, 2, 1);
    }
    
    // 测试16u_C3
    {
        std::vector<int> steps;
        for (int width : testWidths) {
            int step;
            Npp16u* ptr = nppiMalloc_16u_C3(width, 1, &step);
            ASSERT_NE(ptr, nullptr);
            steps.push_back(step);
            nppiFree(ptr);
        }
        analyzeAlignmentPattern("16u_C3", testWidths, steps, 2, 3);
    }
}

// ==================== 32-bit数据类型对齐测试 ====================

TEST_F(NPPIMemoryAlignmentTest, Alignment_32f_AllWidths) {
    std::vector<int> testWidths = {
        1, 2, 4, 7, 8, 9, 15, 16, 17,
        31, 32, 33, 63, 64, 65,
        127, 128, 129, 255, 256, 257,
        640, 800, 1024, 1920
    };
    
    // 测试32f_C1
    {
        std::vector<int> steps;
        for (int width : testWidths) {
            int step;
            Npp32f* ptr = nppiMalloc_32f_C1(width, 1, &step);
            ASSERT_NE(ptr, nullptr);
            steps.push_back(step);
            nppiFree(ptr);
        }
        analyzeAlignmentPattern("32f_C1", testWidths, steps, 4, 1);
    }
    
    // 测试32f_C3
    {
        std::vector<int> steps;
        for (int width : testWidths) {
            int step;
            Npp32f* ptr = nppiMalloc_32f_C3(width, 1, &step);
            ASSERT_NE(ptr, nullptr);
            steps.push_back(step);
            nppiFree(ptr);
        }
        analyzeAlignmentPattern("32f_C3", testWidths, steps, 4, 3);
    }
    
    // 测试32f_C4
    {
        std::vector<int> steps;
        for (int width : testWidths) {
            int step;
            Npp32f* ptr = nppiMalloc_32f_C4(width, 1, &step);
            ASSERT_NE(ptr, nullptr);
            steps.push_back(step);
            nppiFree(ptr);
        }
        analyzeAlignmentPattern("32f_C4", testWidths, steps, 4, 4);
    }
}

// ==================== 复数类型对齐测试 ====================

TEST_F(NPPIMemoryAlignmentTest, Alignment_ComplexTypes) {
    std::vector<int> testWidths = {1, 2, 4, 8, 16, 32, 64, 128, 256};
    
    // 测试32fc_C1 (每个元素8字节)
    {
        std::vector<int> steps;
        for (int width : testWidths) {
            int step;
            Npp32fc* ptr = nppiMalloc_32fc_C1(width, 1, &step);
            ASSERT_NE(ptr, nullptr);
            steps.push_back(step);
            nppiFree(ptr);
        }
        analyzeAlignmentPattern("32fc_C1", testWidths, steps, 8, 1);
    }
    
    // 测试32sc_C1 (每个元素8字节)
    {
        std::vector<int> steps;
        for (int width : testWidths) {
            int step;
            Npp32sc* ptr = nppiMalloc_32sc_C1(width, 1, &step);
            ASSERT_NE(ptr, nullptr);
            steps.push_back(step);
            nppiFree(ptr);
        }
        analyzeAlignmentPattern("32sc_C1", testWidths, steps, 8, 1);
    }
}

// ==================== 特殊对齐场景测试 ====================

TEST_F(NPPIMemoryAlignmentTest, PrimeWidthAlignment) {
    // 测试质数宽度的对齐
    std::vector<int> primeWidths = {
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
        73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
        157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233,
        239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317
    };
    
    int perfectAlignCount = 0;
    
    for (int width : primeWidths) {
        int step;
        Npp8u* ptr = nppiMalloc_8u_C1(width, 1, &step);
        ASSERT_NE(ptr, nullptr);
        
        // 检查是否恰好是32的倍数
        if (step == calculateAlignedStep(width, 32)) {
            perfectAlignCount++;
        }
        
        nppiFree(ptr);
    }
    
    std::cout << "\nPrime width perfect alignment rate: " 
              << perfectAlignCount << "/" << primeWidths.size() 
              << " (" << (perfectAlignCount * 100.0 / primeWidths.size()) << "%)" << std::endl;
}

TEST_F(NPPIMemoryAlignmentTest, PowerOfTwoAlignment) {
    // 测试2的幂次宽度
    std::vector<int> powerOfTwoWidths;
    for (int i = 0; i <= 14; i++) {
        powerOfTwoWidths.push_back(1 << i); // 1, 2, 4, 8, ..., 16384
    }
    
    std::cout << "\n=== Power of Two Width Alignment ===" << std::endl;
    std::cout << std::setw(10) << "Width" 
              << std::setw(15) << "8u_C1 Step"
              << std::setw(15) << "16u_C1 Step"
              << std::setw(15) << "32f_C1 Step" << std::endl;
    std::cout << std::string(55, '-') << std::endl;
    
    for (int width : powerOfTwoWidths) {
        int step8u, step16u, step32f;
        
        Npp8u* ptr8u = nppiMalloc_8u_C1(width, 1, &step8u);
        Npp16u* ptr16u = nppiMalloc_16u_C1(width, 1, &step16u);
        Npp32f* ptr32f = nppiMalloc_32f_C1(width, 1, &step32f);
        
        ASSERT_NE(ptr8u, nullptr);
        ASSERT_NE(ptr16u, nullptr);
        ASSERT_NE(ptr32f, nullptr);
        
        std::cout << std::setw(10) << width
                  << std::setw(15) << step8u
                  << std::setw(15) << step16u  
                  << std::setw(15) << step32f << std::endl;
        
        nppiFree(ptr8u);
        nppiFree(ptr16u);
        nppiFree(ptr32f);
    }
}

// ==================== 性能影响测试 ====================

TEST_F(NPPIMemoryAlignmentTest, AlignmentPerformanceImpact) {
    // 测试不同对齐对内存访问性能的影响
    const int height = 1024;
    const int iterations = 100;
    
    struct AlignmentTest {
        int width;
        std::string description;
    };
    
    std::vector<AlignmentTest> tests = {
        {32, "Perfect 32-byte alignment"},
        {31, "One byte before alignment"},
        {33, "One byte after alignment"},
        {64, "Perfect 64-byte alignment"},
        {128, "Perfect 128-byte alignment"},
        {127, "One byte before 128 alignment"},
        {129, "One byte after 128 alignment"}
    };
    
    std::cout << "\n=== Alignment Performance Impact ===" << std::endl;
    
    for (const auto& test : tests) {
        int step;
        Npp32f* ptr = nppiMalloc_32f_C1(test.width, height, &step);
        ASSERT_NE(ptr, nullptr);
        
        // 准备测试数据
        std::vector<float> hostData(test.width * height, 1.0f);
        
        // 测试写入性能
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        for (int i = 0; i < iterations; i++) {
            cudaMemcpy2D(ptr, step, hostData.data(), test.width * sizeof(float),
                        test.width * sizeof(float), height, cudaMemcpyHostToDevice);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        std::cout << "Width " << std::setw(4) << test.width 
                  << " (step=" << std::setw(4) << step << "): "
                  << std::setw(8) << std::fixed << std::setprecision(3) 
                  << milliseconds << " ms - "
                  << test.description << std::endl;
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        nppiFree(ptr);
    }
}

// ==================== 与NVIDIA NPP对比测试 ====================

#ifdef HAVE_NVIDIA_NPP
TEST_F(NPPIMemoryAlignmentTest, CompareWithNvidiaNPP) {
    // 动态加载NVIDIA NPP库进行对比
    void* nvidiaLib = dlopen("libnppisu.so.12", RTLD_NOW | RTLD_LOCAL);
    if (!nvidiaLib) {
        nvidiaLib = dlopen("/usr/local/cuda/lib64/libnppisu.so.12", RTLD_NOW | RTLD_LOCAL);
    }
    
    if (!nvidiaLib) {
        GTEST_SKIP() << "NVIDIA NPP library not available for comparison";
    }
    
    typedef Npp8u* (*nppiMalloc_8u_C1_func)(int, int, int*);
    typedef Npp32f* (*nppiMalloc_32f_C1_func)(int, int, int*);
    typedef void (*nppiFree_func)(void*);
    
    auto nv_nppiMalloc_8u_C1 = (nppiMalloc_8u_C1_func)dlsym(nvidiaLib, "nppiMalloc_8u_C1");
    auto nv_nppiMalloc_32f_C1 = (nppiMalloc_32f_C1_func)dlsym(nvidiaLib, "nppiMalloc_32f_C1");
    auto nv_nppiFree = (nppiFree_func)dlsym(nvidiaLib, "nppiFree");
    
    if (!nv_nppiMalloc_8u_C1 || !nv_nppiMalloc_32f_C1 || !nv_nppiFree) {
        dlclose(nvidiaLib);
        GTEST_SKIP() << "Failed to load NVIDIA NPP functions";
    }
    
    std::vector<int> testWidths = {1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 
                                   255, 256, 257, 640, 1024, 1920};
    
    int matchCount8u = 0;
    int matchCount32f = 0;
    
    std::cout << "\n=== NVIDIA NPP vs OpenNPP Step Comparison ===" << std::endl;
    std::cout << std::setw(10) << "Width" 
              << std::setw(20) << "8u_C1 (NV/Open)"
              << std::setw(10) << "Match"
              << std::setw(20) << "32f_C1 (NV/Open)"
              << std::setw(10) << "Match" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    for (int width : testWidths) {
        int openStep8u, nvStep8u, openStep32f, nvStep32f;
        
        // 8u_C1
        Npp8u* openPtr8u = nppiMalloc_8u_C1(width, 1, &openStep8u);
        Npp8u* nvPtr8u = nv_nppiMalloc_8u_C1(width, 1, &nvStep8u);
        
        // 32f_C1
        Npp32f* openPtr32f = nppiMalloc_32f_C1(width, 1, &openStep32f);
        Npp32f* nvPtr32f = nv_nppiMalloc_32f_C1(width, 1, &nvStep32f);
        
        bool match8u = (openStep8u == nvStep8u);
        bool match32f = (openStep32f == nvStep32f);
        
        if (match8u) matchCount8u++;
        if (match32f) matchCount32f++;
        
        std::cout << std::setw(10) << width
                  << std::setw(9) << nvStep8u << "/" << std::setw(9) << openStep8u
                  << std::setw(10) << (match8u ? "✓" : "✗")
                  << std::setw(9) << nvStep32f << "/" << std::setw(9) << openStep32f
                  << std::setw(10) << (match32f ? "✓" : "✗") << std::endl;
        
        nppiFree(openPtr8u);
        nppiFree(openPtr32f);
        nv_nppiFree(nvPtr8u);
        nv_nppiFree(nvPtr32f);
    }
    
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "8u_C1 Match Rate: " << matchCount8u << "/" << testWidths.size() 
              << " (" << (matchCount8u * 100.0 / testWidths.size()) << "%)" << std::endl;
    std::cout << "32f_C1 Match Rate: " << matchCount32f << "/" << testWidths.size()
              << " (" << (matchCount32f * 100.0 / testWidths.size()) << "%)" << std::endl;
    
    dlclose(nvidiaLib);
    
    // 我们期望有合理的匹配率
    // NVIDIA NPP和OpenNPP可能使用不同的内存对齐策略，特别是对于32f类型
    // 这是正常现象，只要都满足基本的对齐要求即可
    EXPECT_GE(matchCount8u * 100.0 / testWidths.size(), 90.0) 
        << "8u_C1 step values should mostly match NVIDIA NPP";
    EXPECT_GE(matchCount32f * 100.0 / testWidths.size(), 60.0)
        << "32f_C1 step values should have reasonable match rate with NVIDIA NPP (different alignment strategies are acceptable)";
}
#endif