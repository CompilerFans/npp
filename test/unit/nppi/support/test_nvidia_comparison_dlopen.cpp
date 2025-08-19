/**
 * @file test_nvidia_comparison_dlopen.cpp
 * @brief 使用动态加载来真正对比OpenNPP与NVIDIA NPP
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <iostream>
#include <vector>
#include "npp.h"

class NPPComparisonDlopenTest : public ::testing::Test {
protected:
    void* nvidiaLib = nullptr;
    
    // NVIDIA函数指针
    typedef Npp8u* (*nppiMalloc_8u_C1_func)(int, int, int*);
    typedef Npp16u* (*nppiMalloc_16u_C1_func)(int, int, int*);
    typedef Npp32f* (*nppiMalloc_32f_C1_func)(int, int, int*);
    typedef void (*nppiFree_func)(void*);
    
    nppiMalloc_8u_C1_func nv_nppiMalloc_8u_C1 = nullptr;
    nppiMalloc_16u_C1_func nv_nppiMalloc_16u_C1 = nullptr;
    nppiMalloc_32f_C1_func nv_nppiMalloc_32f_C1 = nullptr;
    nppiFree_func nv_nppiFree = nullptr;
    
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        ASSERT_EQ(err, cudaSuccess);
        
        // 尝试加载NVIDIA NPP支持函数库 (内存分配函数在nppisu中)
        nvidiaLib = dlopen("libnppisu.so.12", RTLD_NOW | RTLD_GLOBAL);
        if (!nvidiaLib) {
            nvidiaLib = dlopen("/usr/local/cuda/lib64/libnppisu.so.12", RTLD_NOW | RTLD_GLOBAL);
        }
        if (!nvidiaLib) {
            // 尝试不带版本号
            nvidiaLib = dlopen("/usr/local/cuda/lib64/libnppisu.so", RTLD_NOW | RTLD_GLOBAL);
        }
        
        if (!nvidiaLib) {
            const char* error = dlerror();
            std::cout << "Failed to load NVIDIA NPP library: " << (error ? error : "unknown error") << std::endl;
        } else {
            std::cout << "Successfully loaded NVIDIA NPP library" << std::endl;
            
            // 加载函数符号
            nv_nppiMalloc_8u_C1 = (nppiMalloc_8u_C1_func)dlsym(nvidiaLib, "nppiMalloc_8u_C1");
            if (!nv_nppiMalloc_8u_C1) {
                std::cout << "Failed to load nppiMalloc_8u_C1: " << dlerror() << std::endl;
            }
            
            nv_nppiMalloc_16u_C1 = (nppiMalloc_16u_C1_func)dlsym(nvidiaLib, "nppiMalloc_16u_C1");
            nv_nppiMalloc_32f_C1 = (nppiMalloc_32f_C1_func)dlsym(nvidiaLib, "nppiMalloc_32f_C1");
            nv_nppiFree = (nppiFree_func)dlsym(nvidiaLib, "nppiFree");
        }
    }
    
    void TearDown() override {
        cudaDeviceSynchronize();
        if (nvidiaLib) {
            dlclose(nvidiaLib);
        }
    }
    
    bool HasNvidiaLib() const {
        return nvidiaLib != nullptr && nv_nppiMalloc_8u_C1 != nullptr;
    }
};

TEST_F(NPPComparisonDlopenTest, VerifyDifferentImplementations) {
    if (!HasNvidiaLib()) {
        GTEST_SKIP() << "NVIDIA NPP library not available";
    }
    
    // 验证函数地址不同
    void* our_func = (void*)&nppiMalloc_8u_C1;
    void* nv_func = (void*)nv_nppiMalloc_8u_C1;
    
    EXPECT_NE(our_func, nv_func) 
        << "Functions should have different addresses";
    
    std::cout << "\nOpenNPP function address: " << our_func << std::endl;
    std::cout << "NVIDIA function address: " << nv_func << std::endl;
}

TEST_F(NPPComparisonDlopenTest, CompareStepValues_8u_C1) {
    if (!HasNvidiaLib()) {
        GTEST_SKIP() << "NVIDIA NPP library not available";
    }
    
    // Clear GPU memory first
    cudaDeviceSynchronize();
    cudaError_t resetResult = cudaDeviceReset();
    if (resetResult != cudaSuccess) {
        GTEST_SKIP() << "Failed to reset GPU device: " << cudaGetErrorString(resetResult);
    }
    
    const int testSizes[][2] = {
        {64, 64}, {640, 480}, {1920, 1080}, {333, 444}
    };
    
    int matchCount = 0;
    int totalTests = sizeof(testSizes) / sizeof(testSizes[0]);
    int skippedTests = 0;
    
    std::cout << "\n=== 8u_C1 Step Comparison ===" << std::endl;
    
    for (const auto& size : testSizes) {
        int width = size[0];
        int height = size[1];
        int openStep, nvStep;
        
        // OpenNPP分配
        Npp8u* openPtr = nppiMalloc_8u_C1(width, height, &openStep);
        if (!openPtr) {
            std::cout << "Size " << width << "x" << height << ": OpenNPP allocation failed - SKIPPED" << std::endl;
            skippedTests++;
            continue;
        }
        
        // NVIDIA NPP分配
        Npp8u* nvPtr = nv_nppiMalloc_8u_C1(width, height, &nvStep);
        if (!nvPtr) {
            std::cout << "Size " << width << "x" << height << ": NVIDIA NPP allocation failed - SKIPPED" << std::endl;
            nppiFree(openPtr);
            skippedTests++;
            continue;
        }
        
        std::cout << "Size " << width << "x" << height 
                  << ": OpenNPP=" << openStep 
                  << ", NVIDIA=" << nvStep;
        
        if (openStep == nvStep) {
            std::cout << " ✓ MATCH" << std::endl;
            matchCount++;
        } else {
            std::cout << " ✗ DIFFER (diff=" << (openStep - nvStep) << ")" << std::endl;
        }
        
        // 清理
        nppiFree(openPtr);
        nv_nppiFree(nvPtr);
    }
    
    int testedCases = totalTests - skippedTests;
    if (testedCases > 0) {
        double matchRate = (matchCount * 100.0) / testedCases;
        std::cout << "Match rate: " << matchRate << "% (tested " << testedCases << "/" << totalTests << " cases)" << std::endl;
        
        // 我们期望有很高的匹配率（都应该是32字节对齐）
        EXPECT_GE(matchRate, 75.0) << "Step values should mostly match";
    } else {
        std::cout << "All test cases skipped due to memory allocation failures" << std::endl;
        GTEST_SKIP() << "No successful allocations to compare";
    }
}

TEST_F(NPPComparisonDlopenTest, CompareStepValues_16u_C1) {
    if (!HasNvidiaLib() || !nv_nppiMalloc_16u_C1) {
        GTEST_SKIP() << "NVIDIA NPP 16u function not available";
    }
    
    // Clear GPU memory first
    cudaDeviceSynchronize();
    
    const int testSizes[][2] = {
        {64, 64}, {640, 480}, {1920, 1080}, {333, 444}
    };
    
    int matchCount = 0;
    int totalTests = sizeof(testSizes) / sizeof(testSizes[0]);
    int skippedTests = 0;
    
    std::cout << "\n=== 16u_C1 Step Comparison ===" << std::endl;
    
    for (const auto& size : testSizes) {
        int width = size[0];
        int height = size[1];
        int openStep, nvStep;
        
        // OpenNPP分配
        Npp16u* openPtr = nppiMalloc_16u_C1(width, height, &openStep);
        if (!openPtr) {
            std::cout << "Size " << width << "x" << height << ": OpenNPP allocation failed - SKIPPED" << std::endl;
            skippedTests++;
            continue;
        }
        
        // NVIDIA NPP分配
        Npp16u* nvPtr = nv_nppiMalloc_16u_C1(width, height, &nvStep);
        if (!nvPtr) {
            std::cout << "Size " << width << "x" << height << ": NVIDIA NPP allocation failed - SKIPPED" << std::endl;
            nppiFree(openPtr);
            skippedTests++;
            continue;
        }
        
        std::cout << "Size " << width << "x" << height 
                  << ": OpenNPP=" << openStep 
                  << ", NVIDIA=" << nvStep;
        
        if (openStep == nvStep) {
            std::cout << " ✓ MATCH" << std::endl;
            matchCount++;
        } else {
            std::cout << " ✗ DIFFER (diff=" << (openStep - nvStep) << ")" << std::endl;
        }
        
        // 清理
        nppiFree(openPtr);
        nv_nppiFree(nvPtr);
    }
    
    int testedCases = totalTests - skippedTests;
    if (testedCases > 0) {
        double matchRate = (matchCount * 100.0) / testedCases;
        std::cout << "Match rate: " << matchRate << "% (tested " << testedCases << "/" << totalTests << " cases)" << std::endl;
        
        // 我们期望有很高的匹配率（都应该是32字节对齐）
        EXPECT_GE(matchRate, 75.0) << "Step values should mostly match";
    } else {
        std::cout << "All test cases skipped due to memory allocation failures" << std::endl;
        GTEST_SKIP() << "No successful allocations to compare";
    }
}

TEST_F(NPPComparisonDlopenTest, CompareStepValues_32f_C1) {
    if (!HasNvidiaLib() || !nv_nppiMalloc_32f_C1) {
        GTEST_SKIP() << "NVIDIA NPP 32f function not available";
    }
    
    // Clear GPU memory first
    cudaDeviceSynchronize();
    
    const int testSizes[][2] = {
        {64, 64}, {640, 480}, {1920, 1080}, {333, 444}
    };
    
    int matchCount = 0;
    int totalTests = sizeof(testSizes) / sizeof(testSizes[0]);
    int skippedTests = 0;
    
    std::cout << "\n=== 32f_C1 Step Comparison ===" << std::endl;
    
    for (const auto& size : testSizes) {
        int width = size[0];
        int height = size[1];
        int openStep, nvStep;
        
        // OpenNPP分配
        Npp32f* openPtr = nppiMalloc_32f_C1(width, height, &openStep);
        if (!openPtr) {
            std::cout << "Size " << width << "x" << height << ": OpenNPP allocation failed - SKIPPED" << std::endl;
            skippedTests++;
            continue;
        }
        
        // NVIDIA NPP分配
        Npp32f* nvPtr = nv_nppiMalloc_32f_C1(width, height, &nvStep);
        if (!nvPtr) {
            std::cout << "Size " << width << "x" << height << ": NVIDIA NPP allocation failed - SKIPPED" << std::endl;
            nppiFree(openPtr);
            skippedTests++;
            continue;
        }
        
        std::cout << "Size " << width << "x" << height 
                  << ": OpenNPP=" << openStep 
                  << ", NVIDIA=" << nvStep;
        
        if (openStep == nvStep) {
            std::cout << " ✓ MATCH" << std::endl;
            matchCount++;
        } else {
            std::cout << " ✗ DIFFER (diff=" << (openStep - nvStep) << ")" << std::endl;
        }
        
        // 清理
        nppiFree(openPtr);
        nv_nppiFree(nvPtr);
    }
    
    int testedCases = totalTests - skippedTests;
    if (testedCases > 0) {
        double matchRate = (matchCount * 100.0) / testedCases;
        std::cout << "Match rate: " << matchRate << "% (tested " << testedCases << "/" << totalTests << " cases)" << std::endl;
        
        // 我们期望有很高的匹配率（都应该是32字节对齐）
        EXPECT_GE(matchRate, 75.0) << "Step values should mostly match";
    } else {
        std::cout << "All test cases skipped due to memory allocation failures" << std::endl;
        GTEST_SKIP() << "No successful allocations to compare";
    }
}

// 测试算术运算结果对比
TEST_F(NPPComparisonDlopenTest, CompareArithmeticResults) {
    // 这需要更多的动态加载工作，暂时跳过
    // 但这展示了如何真正对比两个不同的实现
    GTEST_SKIP() << "Arithmetic comparison requires more dlopen setup";
}