/**
 * @file test_nppi_support_complete_coverage.cpp
 * @brief 完整覆盖NPPI支持函数的测试
 * 
 * 测试所有数据类型和通道组合：
 * - 8u: C1, C2, C3, C4
 * - 16u: C1, C2, C3, C4
 * - 16s: C1, C2, C3, C4
 * - 16sc: C1, C2, C3, C4
 * - 32s: C1, C2, C3, C4
 * - 32sc: C1, C2, C3, C4
 * - 32f: C1, C2, C3, C4
 * - 32fc: C1, C2, C3, C4
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstring>
#include "npp.h"

#ifdef HAVE_NVIDIA_NPP
#include <nppi.h>
#endif

class NPPISupportCompleteCoverageTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        ASSERT_EQ(err, cudaSuccess);
    }
    
    void TearDown() override {
        cudaDeviceSynchronize();
    }
    
    // 通用测试函数模板
    template<typename T>
    void TestMemoryAllocation(
        T* (*allocFunc)(int, int, int*),
        const std::string& typeName,
        int channels,
        int expectedMinStep) {
        
        const int width = 640;
        const int height = 480;
        int step;
        
        // 分配内存
        T* ptr = allocFunc(width, height, &step);
        
        // 验证分配成功
        ASSERT_NE(ptr, nullptr) << typeName << "_C" << channels 
                                << " allocation failed";
        
        // 验证step值
        EXPECT_GE(step, expectedMinStep) << typeName << "_C" << channels 
                                         << " step less than minimum";
        
        // 验证对齐
        EXPECT_EQ(step % 32, 0) << typeName << "_C" << channels 
                                << " step not aligned";
        
        // 测试内存写入读取
        size_t dataSize = width * channels * sizeof(T);
        std::vector<T> hostData(width * height * channels);
        
        // 填充测试数据
        for (size_t i = 0; i < hostData.size(); i++) {
            if (sizeof(T) == 1) {
                reinterpret_cast<Npp8u*>(hostData.data())[i] = i % 256;
            } else if (sizeof(T) == 2) {
                reinterpret_cast<Npp16u*>(hostData.data())[i] = i % 65536;
            } else if (sizeof(T) == 4) {
                reinterpret_cast<Npp32f*>(hostData.data())[i] = static_cast<float>(i);
            }
        }
        
        // 复制到设备
        cudaError_t err = cudaMemcpy2D(
            ptr, step,
            hostData.data(), dataSize,
            dataSize, height,
            cudaMemcpyHostToDevice
        );
        EXPECT_EQ(err, cudaSuccess) << typeName << "_C" << channels 
                                    << " copy to device failed";
        
        // 复制回主机
        std::vector<T> hostDataBack(hostData.size());
        err = cudaMemcpy2D(
            hostDataBack.data(), dataSize,
            ptr, step,
            dataSize, height,
            cudaMemcpyDeviceToHost
        );
        EXPECT_EQ(err, cudaSuccess) << typeName << "_C" << channels 
                                    << " copy from device failed";
        
        // 验证数据一致性
        bool dataMatch = (memcmp(hostData.data(), hostDataBack.data(), 
                                 hostData.size() * sizeof(T)) == 0);
        EXPECT_TRUE(dataMatch) << typeName << "_C" << channels 
                               << " data mismatch after copy";
        
        // 清理
        nppiFree(ptr);
    }
    
#ifdef HAVE_NVIDIA_NPP
    // 与NVIDIA NPP对比的通用测试函数
    template<typename T>
    void CompareWithNvidia(
        T* (*openFunc)(int, int, int*),
        T* (*nvFunc)(int, int, int*),
        const std::string& typeName,
        int channels) {
        
        const int testSizes[][2] = {
            {64, 64}, {640, 480}, {1920, 1080}, {333, 444}
        };
        
        int matchCount = 0;
        
        for (const auto& size : testSizes) {
            int width = size[0];
            int height = size[1];
            int openStep, nvStep;
            
            // OpenNPP分配
            T* openPtr = openFunc(width, height, &openStep);
            ASSERT_NE(openPtr, nullptr);
            
            // NVIDIA NPP分配
            T* nvPtr = nvFunc(width, height, &nvStep);
            ASSERT_NE(nvPtr, nullptr);
            
            // 比较step值
            if (openStep == nvStep) {
                matchCount++;
            }
            
            // 清理
            nppiFree(openPtr);
            ::nppiFree(nvPtr);
        }
        
        // 报告匹配率
        double matchRate = matchCount * 100.0 / 4;
        std::cout << typeName << "_C" << channels 
                  << " step match rate: " << matchRate << "%" << std::endl;
    }
#endif
};

// ==================== 8-bit unsigned 测试 ====================

TEST_F(NPPISupportCompleteCoverageTest, Malloc_8u_AllChannels) {
    TestMemoryAllocation<Npp8u>(nppiMalloc_8u_C1, "8u", 1, 640);
    TestMemoryAllocation<Npp8u>(nppiMalloc_8u_C2, "8u", 2, 640 * 2);
    TestMemoryAllocation<Npp8u>(nppiMalloc_8u_C3, "8u", 3, 640 * 3);
    TestMemoryAllocation<Npp8u>(nppiMalloc_8u_C4, "8u", 4, 640 * 4);
}

// ==================== 16-bit unsigned 测试 ====================

TEST_F(NPPISupportCompleteCoverageTest, Malloc_16u_AllChannels) {
    TestMemoryAllocation<Npp16u>(nppiMalloc_16u_C1, "16u", 1, 640 * 2);
    TestMemoryAllocation<Npp16u>(nppiMalloc_16u_C2, "16u", 2, 640 * 4);
    TestMemoryAllocation<Npp16u>(nppiMalloc_16u_C3, "16u", 3, 640 * 6);
    TestMemoryAllocation<Npp16u>(nppiMalloc_16u_C4, "16u", 4, 640 * 8);
}

// ==================== 16-bit signed 测试 ====================

TEST_F(NPPISupportCompleteCoverageTest, Malloc_16s_AllChannels) {
    TestMemoryAllocation<Npp16s>(nppiMalloc_16s_C1, "16s", 1, 640 * 2);
    TestMemoryAllocation<Npp16s>(nppiMalloc_16s_C2, "16s", 2, 640 * 4);
    // 16s_C3 not in original API
    TestMemoryAllocation<Npp16s>(nppiMalloc_16s_C4, "16s", 4, 640 * 8);
}

// ==================== 16-bit signed complex 测试 ====================

TEST_F(NPPISupportCompleteCoverageTest, Malloc_16sc_AllChannels) {
    TestMemoryAllocation<Npp16sc>(nppiMalloc_16sc_C1, "16sc", 1, 640 * 4);
    TestMemoryAllocation<Npp16sc>(nppiMalloc_16sc_C2, "16sc", 2, 640 * 8);
    TestMemoryAllocation<Npp16sc>(nppiMalloc_16sc_C3, "16sc", 3, 640 * 12);
    TestMemoryAllocation<Npp16sc>(nppiMalloc_16sc_C4, "16sc", 4, 640 * 16);
}

// ==================== 32-bit signed 测试 ====================

TEST_F(NPPISupportCompleteCoverageTest, Malloc_32s_AllChannels) {
    TestMemoryAllocation<Npp32s>(nppiMalloc_32s_C1, "32s", 1, 640 * 4);
    // 32s_C2 not in original API
    TestMemoryAllocation<Npp32s>(nppiMalloc_32s_C3, "32s", 3, 640 * 12);
    TestMemoryAllocation<Npp32s>(nppiMalloc_32s_C4, "32s", 4, 640 * 16);
}

// ==================== 32-bit signed complex 测试 ====================

TEST_F(NPPISupportCompleteCoverageTest, Malloc_32sc_AllChannels) {
    TestMemoryAllocation<Npp32sc>(nppiMalloc_32sc_C1, "32sc", 1, 640 * 8);
    TestMemoryAllocation<Npp32sc>(nppiMalloc_32sc_C2, "32sc", 2, 640 * 16);
    TestMemoryAllocation<Npp32sc>(nppiMalloc_32sc_C3, "32sc", 3, 640 * 24);
    TestMemoryAllocation<Npp32sc>(nppiMalloc_32sc_C4, "32sc", 4, 640 * 32);
}

// ==================== 32-bit float 测试 ====================

TEST_F(NPPISupportCompleteCoverageTest, Malloc_32f_AllChannels) {
    TestMemoryAllocation<Npp32f>(nppiMalloc_32f_C1, "32f", 1, 640 * 4);
    TestMemoryAllocation<Npp32f>(nppiMalloc_32f_C2, "32f", 2, 640 * 8);
    TestMemoryAllocation<Npp32f>(nppiMalloc_32f_C3, "32f", 3, 640 * 12);
    TestMemoryAllocation<Npp32f>(nppiMalloc_32f_C4, "32f", 4, 640 * 16);
}

// ==================== 32-bit float complex 测试 ====================

TEST_F(NPPISupportCompleteCoverageTest, Malloc_32fc_AllChannels) {
    TestMemoryAllocation<Npp32fc>(nppiMalloc_32fc_C1, "32fc", 1, 640 * 8);
    TestMemoryAllocation<Npp32fc>(nppiMalloc_32fc_C2, "32fc", 2, 640 * 16);
    TestMemoryAllocation<Npp32fc>(nppiMalloc_32fc_C3, "32fc", 3, 640 * 24);
    TestMemoryAllocation<Npp32fc>(nppiMalloc_32fc_C4, "32fc", 4, 640 * 32);
}

// ==================== NVIDIA NPP对比测试 ====================

#ifdef HAVE_NVIDIA_NPP

TEST_F(NPPISupportCompleteCoverageTest, CompareAllTypes_WithNvidia) {
    std::cout << "\n=== Complete Coverage Comparison with NVIDIA NPP ===" << std::endl;
    
    // 8-bit unsigned
    CompareWithNvidia<Npp8u>(nppiMalloc_8u_C1, ::nppiMalloc_8u_C1, "8u", 1);
    CompareWithNvidia<Npp8u>(nppiMalloc_8u_C2, ::nppiMalloc_8u_C2, "8u", 2);
    CompareWithNvidia<Npp8u>(nppiMalloc_8u_C3, ::nppiMalloc_8u_C3, "8u", 3);
    CompareWithNvidia<Npp8u>(nppiMalloc_8u_C4, ::nppiMalloc_8u_C4, "8u", 4);
    
    // 16-bit unsigned
    CompareWithNvidia<Npp16u>(nppiMalloc_16u_C1, ::nppiMalloc_16u_C1, "16u", 1);
    CompareWithNvidia<Npp16u>(nppiMalloc_16u_C2, ::nppiMalloc_16u_C2, "16u", 2);
    CompareWithNvidia<Npp16u>(nppiMalloc_16u_C3, ::nppiMalloc_16u_C3, "16u", 3);
    CompareWithNvidia<Npp16u>(nppiMalloc_16u_C4, ::nppiMalloc_16u_C4, "16u", 4);
    
    // 16-bit signed
    CompareWithNvidia<Npp16s>(nppiMalloc_16s_C1, ::nppiMalloc_16s_C1, "16s", 1);
    CompareWithNvidia<Npp16s>(nppiMalloc_16s_C2, ::nppiMalloc_16s_C2, "16s", 2);
    // Note: NVIDIA NPP可能没有16s_C3，需要检查
    CompareWithNvidia<Npp16s>(nppiMalloc_16s_C4, ::nppiMalloc_16s_C4, "16s", 4);
    
    // 32-bit signed
    CompareWithNvidia<Npp32s>(nppiMalloc_32s_C1, ::nppiMalloc_32s_C1, "32s", 1);
    // Note: 检查NVIDIA NPP是否有32s_C2
    CompareWithNvidia<Npp32s>(nppiMalloc_32s_C3, ::nppiMalloc_32s_C3, "32s", 3);
    CompareWithNvidia<Npp32s>(nppiMalloc_32s_C4, ::nppiMalloc_32s_C4, "32s", 4);
    
    // 32-bit float
    CompareWithNvidia<Npp32f>(nppiMalloc_32f_C1, ::nppiMalloc_32f_C1, "32f", 1);
    CompareWithNvidia<Npp32f>(nppiMalloc_32f_C2, ::nppiMalloc_32f_C2, "32f", 2);
    CompareWithNvidia<Npp32f>(nppiMalloc_32f_C3, ::nppiMalloc_32f_C3, "32f", 3);
    CompareWithNvidia<Npp32f>(nppiMalloc_32f_C4, ::nppiMalloc_32f_C4, "32f", 4);
    
    // Complex types
    CompareWithNvidia<Npp16sc>(nppiMalloc_16sc_C1, ::nppiMalloc_16sc_C1, "16sc", 1);
    CompareWithNvidia<Npp32sc>(nppiMalloc_32sc_C1, ::nppiMalloc_32sc_C1, "32sc", 1);
    CompareWithNvidia<Npp32fc>(nppiMalloc_32fc_C1, ::nppiMalloc_32fc_C1, "32fc", 1);
}

#endif // HAVE_NVIDIA_NPP

// ==================== 压力测试 ====================

TEST_F(NPPISupportCompleteCoverageTest, StressTest_ManyAllocations) {
    const int iterations = 100;
    std::vector<void*> pointers;
    std::vector<int> steps;
    
    // 分配多个不同类型和大小的内存
    for (int i = 0; i < iterations; i++) {
        int width = 100 + (i * 10);
        int height = 100 + (i * 5);
        int step;
        
        // 交替分配不同类型
        void* ptr = nullptr;
        switch (i % 4) {
            case 0:
                ptr = nppiMalloc_8u_C1(width, height, &step);
                break;
            case 1:
                ptr = nppiMalloc_16u_C3(width, height, &step);
                break;
            case 2:
                ptr = nppiMalloc_32f_C1(width, height, &step);
                break;
            case 3:
                ptr = nppiMalloc_32f_C4(width, height, &step);
                break;
        }
        
        ASSERT_NE(ptr, nullptr) << "Allocation failed at iteration " << i;
        pointers.push_back(ptr);
        steps.push_back(step);
    }
    
    // 验证所有step值都对齐
    for (int step : steps) {
        EXPECT_EQ(step % 32, 0) << "Step not aligned";
    }
    
    // 释放所有内存
    for (void* ptr : pointers) {
        nppiFree(ptr);
    }
    
    SUCCEED() << "Stress test completed with " << iterations << " allocations";
}

// ==================== 边界条件测试 ====================

TEST_F(NPPISupportCompleteCoverageTest, EdgeCases_VariousSizes) {
    struct TestCase {
        int width;
        int height;
        std::string description;
    };
    
    std::vector<TestCase> testCases = {
        {1, 1, "1x1 minimum"},
        {1, 10000, "1x10000 tall"},
        {10000, 1, "10000x1 wide"},
        {15, 15, "15x15 non-aligned"},
        {16, 16, "16x16 half-aligned"},
        {17, 17, "17x17 just over"},
        {31, 31, "31x31 just under 32"},
        {32, 32, "32x32 aligned"},
        {33, 33, "33x33 just over 32"},
        {4095, 4095, "4095x4095 just under 4K"},
        {4096, 4096, "4096x4096 exactly 4K"}
    };
    
    for (const auto& tc : testCases) {
        int step;
        
        // 测试8u_C1
        Npp8u* ptr8u = nppiMalloc_8u_C1(tc.width, tc.height, &step);
        ASSERT_NE(ptr8u, nullptr) << "8u_C1 failed for " << tc.description;
        EXPECT_GE(step, tc.width) << "8u_C1 step too small for " << tc.description;
        EXPECT_EQ(step % 32, 0) << "8u_C1 not aligned for " << tc.description;
        nppiFree(ptr8u);
        
        // 测试32f_C4（最大的基本类型）
        Npp32f* ptr32f = nppiMalloc_32f_C4(tc.width, tc.height, &step);
        ASSERT_NE(ptr32f, nullptr) << "32f_C4 failed for " << tc.description;
        EXPECT_GE(step, tc.width * 16) << "32f_C4 step too small for " << tc.description;
        EXPECT_EQ(step % 32, 0) << "32f_C4 not aligned for " << tc.description;
        nppiFree(ptr32f);
    }
}

// Main函数由gtest_main提供