/**
 * @file test_nppcore_cuda_comparison.cpp
 * @brief NPP Core与CUDA Runtime API对比测试
 * 
 * 测试内容：
 * - 设备属性查询对比
 * - Stream管理对比
 * - Context信息对比
 * - 性能基准测试
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstring>
#include "npp.h"

class NPPCoreCudaComparisonTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        ASSERT_EQ(err, cudaSuccess);
        
        // 获取CUDA设备属性
        err = cudaGetDeviceProperties(&cudaProp, 0);
        ASSERT_EQ(err, cudaSuccess);
    }
    
    void TearDown() override {
        cudaDeviceSynchronize();
    }
    
    cudaDeviceProp cudaProp;
};

// ==================== 设备属性对比测试 ====================

TEST_F(NPPCoreCudaComparisonTest, DeviceProperties_NumSMs) {
    // NPP API
    int nppNumSMs = nppGetGpuNumSMs();
    
    // CUDA API
    int cudaNumSMs = cudaProp.multiProcessorCount;
    
    EXPECT_EQ(nppNumSMs, cudaNumSMs) 
        << "NPP and CUDA should report the same number of SMs";
    
    std::cout << "\nNumber of SMs: NPP=" << nppNumSMs 
              << ", CUDA=" << cudaNumSMs << std::endl;
}

TEST_F(NPPCoreCudaComparisonTest, DeviceProperties_MaxThreadsPerBlock) {
    // NPP API
    int nppMaxThreads = nppGetMaxThreadsPerBlock();
    
    // CUDA API
    int cudaMaxThreads = cudaProp.maxThreadsPerBlock;
    
    EXPECT_EQ(nppMaxThreads, cudaMaxThreads)
        << "NPP and CUDA should report the same max threads per block";
    
    std::cout << "Max Threads per Block: NPP=" << nppMaxThreads 
              << ", CUDA=" << cudaMaxThreads << std::endl;
}

TEST_F(NPPCoreCudaComparisonTest, DeviceProperties_MaxThreadsPerSM) {
    // NPP API
    int nppMaxThreadsPerSM = nppGetMaxThreadsPerSM();
    
    // CUDA API
    int cudaMaxThreadsPerSM = cudaProp.maxThreadsPerMultiProcessor;
    
    EXPECT_EQ(nppMaxThreadsPerSM, cudaMaxThreadsPerSM)
        << "NPP and CUDA should report the same max threads per SM";
    
    std::cout << "Max Threads per SM: NPP=" << nppMaxThreadsPerSM 
              << ", CUDA=" << cudaMaxThreadsPerSM << std::endl;
}

TEST_F(NPPCoreCudaComparisonTest, DeviceProperties_DeviceName) {
    // NPP API
    const char* nppDeviceName = nppGetGpuName();
    
    // CUDA API
    const char* cudaDeviceName = cudaProp.name;
    
    EXPECT_STREQ(nppDeviceName, cudaDeviceName)
        << "NPP and CUDA should report the same device name";
    
    std::cout << "Device Name: NPP='" << nppDeviceName 
              << "', CUDA='" << cudaDeviceName << "'" << std::endl;
}

TEST_F(NPPCoreCudaComparisonTest, DeviceProperties_ComputeCapability) {
    // No nppGetGpuComputeCapability in original API
    // Just verify we can get it from CUDA
    int cudaComputeCap = cudaProp.major * 10 + cudaProp.minor;
    
    EXPECT_GT(cudaComputeCap, 0)
        << "Should be able to get compute capability from CUDA";
    
    std::cout << "Compute Capability: " << cudaProp.major << "." << cudaProp.minor << std::endl;
}

TEST_F(NPPCoreCudaComparisonTest, DeviceProperties_AllAtOnce) {
    // NPP API - 一次获取多个属性
    int nppMaxThreadsPerSM, nppMaxThreadsPerBlock, nppNumSMs;
    int result = nppGetGpuDeviceProperties(&nppMaxThreadsPerSM, 
                                          &nppMaxThreadsPerBlock, 
                                          &nppNumSMs);
    EXPECT_EQ(result, 0) << "nppGetGpuDeviceProperties should succeed";
    
    // 与CUDA属性对比
    EXPECT_EQ(nppMaxThreadsPerSM, cudaProp.maxThreadsPerMultiProcessor);
    EXPECT_EQ(nppMaxThreadsPerBlock, cudaProp.maxThreadsPerBlock);
    EXPECT_EQ(nppNumSMs, cudaProp.multiProcessorCount);
}

// ==================== Stream Context 对比测试 ====================

TEST_F(NPPCoreCudaComparisonTest, StreamContext_DeviceID) {
    NppStreamContext ctx;
    NppStatus status = nppGetStreamContext(&ctx);
    ASSERT_EQ(status, NPP_NO_ERROR);
    
    int cudaDeviceId;
    cudaGetDevice(&cudaDeviceId);
    
    EXPECT_EQ(ctx.nCudaDeviceId, cudaDeviceId)
        << "Stream context should have correct device ID";
}

TEST_F(NPPCoreCudaComparisonTest, StreamContext_Properties) {
    NppStreamContext ctx;
    NppStatus status = nppGetStreamContext(&ctx);
    ASSERT_EQ(status, NPP_NO_ERROR);
    
    // 验证所有属性与CUDA一致
    EXPECT_EQ(ctx.nMultiProcessorCount, cudaProp.multiProcessorCount);
    EXPECT_EQ(ctx.nMaxThreadsPerMultiProcessor, cudaProp.maxThreadsPerMultiProcessor);
    EXPECT_EQ(ctx.nMaxThreadsPerBlock, cudaProp.maxThreadsPerBlock);
    EXPECT_EQ(ctx.nSharedMemPerBlock, cudaProp.sharedMemPerBlock);
    EXPECT_EQ(ctx.nCudaDevAttrComputeCapabilityMajor, cudaProp.major);
    EXPECT_EQ(ctx.nCudaDevAttrComputeCapabilityMinor, cudaProp.minor);
    
    std::cout << "\n=== Stream Context Properties ===" << std::endl;
    std::cout << "Device ID: " << ctx.nCudaDeviceId << std::endl;
    std::cout << "SM Count: " << ctx.nMultiProcessorCount << std::endl;
    std::cout << "Max Threads/SM: " << ctx.nMaxThreadsPerMultiProcessor << std::endl;
    std::cout << "Max Threads/Block: " << ctx.nMaxThreadsPerBlock << std::endl;
    std::cout << "Shared Mem/Block: " << ctx.nSharedMemPerBlock << " bytes" << std::endl;
    std::cout << "Compute Capability: " << ctx.nCudaDevAttrComputeCapabilityMajor 
              << "." << ctx.nCudaDevAttrComputeCapabilityMinor << std::endl;
}

// ==================== Stream 管理对比测试 ====================

TEST_F(NPPCoreCudaComparisonTest, Stream_DefaultStream) {
    // NPP返回的默认stream
    cudaStream_t nppStream = nppGetStream();
    
    // 验证是legacy stream
    EXPECT_EQ(nppStream, cudaStreamLegacy)
        << "NPP should return legacy default stream";
}

TEST_F(NPPCoreCudaComparisonTest, Stream_CustomStreamValidation) {
    // 创建自定义stream
    cudaStream_t customStream;
    cudaError_t err = cudaStreamCreate(&customStream);
    ASSERT_EQ(err, cudaSuccess);
    
    // NPP验证stream有效性
    NppStatus status = nppSetStream(customStream);
    EXPECT_EQ(status, NPP_NO_ERROR)
        << "NPP should validate custom stream successfully";
    
    // 销毁stream
    cudaStreamDestroy(customStream);
}

TEST_F(NPPCoreCudaComparisonTest, Stream_InvalidStreamValidation) {
    // 使用无效的stream值（不会导致段错误）
    cudaStream_t invalidStream = (cudaStream_t)0xDEADBEEF;
    
    // 尝试使用无效stream
    NppStatus status = nppSetStream(invalidStream);
    EXPECT_NE(status, NPP_NO_ERROR)
        << "NPP should detect invalid stream";
}

// ==================== 多设备支持测试 ====================

TEST_F(NPPCoreCudaComparisonTest, MultiDevice_DeviceCount) {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    ASSERT_EQ(err, cudaSuccess);
    
    if (deviceCount > 1) {
        // 测试切换设备
        for (int i = 0; i < deviceCount; i++) {
            err = cudaSetDevice(i);
            ASSERT_EQ(err, cudaSuccess);
            
            // 获取NPP设备属性
            int numSMs = nppGetGpuNumSMs();
            const char* deviceName = nppGetGpuName();
            
            // 获取CUDA设备属性
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            
            // 对比
            EXPECT_EQ(numSMs, prop.multiProcessorCount)
                << "Device " << i << " SM count mismatch";
            EXPECT_STREQ(deviceName, prop.name)
                << "Device " << i << " name mismatch";
            
            std::cout << "Device " << i << ": " << deviceName 
                      << " (SMs=" << numSMs << ")" << std::endl;
        }
        
        // 恢复到设备0
        cudaSetDevice(0);
    } else {
        GTEST_SKIP() << "Only one GPU available, skipping multi-device test";
    }
}

// ==================== Context 切换测试 ====================

// nppSetStreamContext is not part of original NPP API
// This test is removed

// ==================== 错误处理对比测试 ====================

// Test removed - nppSetStreamContext not in original API

TEST_F(NPPCoreCudaComparisonTest, ErrorHandling_NullPointer) {
    // NPP API空指针处理
    NppStatus status = nppGetStreamContext(nullptr);
    EXPECT_EQ(status, NPP_NULL_POINTER_ERROR)
        << "Should return null pointer error";
    
    // nppSetStreamContext test removed - not in original API
    
    int result = nppGetGpuDeviceProperties(nullptr, nullptr, nullptr);
    EXPECT_EQ(result, -1)
        << "Should return error for null pointers";
}

// ==================== 性能基准测试 ====================

TEST_F(NPPCoreCudaComparisonTest, Performance_PropertyQueries) {
    const int iterations = 10000;
    
    // 测试NPP API性能
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        int numSMs = nppGetGpuNumSMs();
        (void)numSMs; // 避免未使用警告
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float nppTime;
    cudaEventElapsedTime(&nppTime, start, stop);
    
    // 测试CUDA API性能
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        int numSMs = prop.multiProcessorCount;
        (void)numSMs;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float cudaTime;
    cudaEventElapsedTime(&cudaTime, start, stop);
    
    std::cout << "\n=== Performance Comparison (" << iterations << " iterations) ===" << std::endl;
    std::cout << "NPP API: " << nppTime << " ms" << std::endl;
    std::cout << "CUDA API: " << cudaTime << " ms" << std::endl;
    std::cout << "NPP is " << (cudaTime / nppTime) << "x speed of CUDA" << std::endl;
    
    // NPP可能会缓存结果，所以可能更快
    // 不强制要求特定的性能比例，只是记录
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ==================== Stream 属性测试 ====================

TEST_F(NPPCoreCudaComparisonTest, StreamProperties_Flags) {
    // 创建不同标志的stream
    cudaStream_t streamDefault, streamNonBlocking;
    
    cudaStreamCreate(&streamDefault);
    cudaStreamCreateWithFlags(&streamNonBlocking, cudaStreamNonBlocking);
    
    // 验证默认stream
    NppStatus status = nppSetStream(streamDefault);
    EXPECT_EQ(status, NPP_NO_ERROR);
    
    // 验证非阻塞stream
    status = nppSetStream(streamNonBlocking);
    EXPECT_EQ(status, NPP_NO_ERROR);
    
    // 清理
    cudaStreamDestroy(streamDefault);
    cudaStreamDestroy(streamNonBlocking);
}

// ==================== 综合测试 ====================

TEST_F(NPPCoreCudaComparisonTest, Integration_CompleteWorkflow) {
    // 1. 获取设备信息
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    ASSERT_GT(numDevices, 0);
    
    // 2. 设置设备
    cudaSetDevice(0);
    
    // 3. 获取NPP context
    NppStreamContext ctx;
    NppStatus status = nppGetStreamContext(&ctx);
    ASSERT_EQ(status, NPP_NO_ERROR);
    
    // 4. 验证context信息
    EXPECT_EQ(ctx.nCudaDeviceId, 0);
    EXPECT_GT(ctx.nMultiProcessorCount, 0);
    EXPECT_GT(ctx.nMaxThreadsPerMultiProcessor, 0);
    EXPECT_GT(ctx.nMaxThreadsPerBlock, 0);
    
    // 5. 创建自定义stream
    cudaStream_t customStream;
    cudaStreamCreate(&customStream);
    
    // 6. 设置stream（仅验证）
    status = nppSetStream(customStream);
    EXPECT_EQ(status, NPP_NO_ERROR);
    
    // 7. 获取设备属性
    const char* deviceName = nppGetGpuName();
    int computeCap = cudaProp.major * 10 + cudaProp.minor;
    
    std::cout << "\n=== Integration Test Summary ===" << std::endl;
    std::cout << "Device: " << deviceName << std::endl;
    std::cout << "Compute Capability: " << (computeCap/10) << "." << (computeCap%10) << std::endl;
    std::cout << "SMs: " << ctx.nMultiProcessorCount << std::endl;
    std::cout << "Context validated: YES" << std::endl;
    std::cout << "Stream validated: YES" << std::endl;
    
    // 8. 清理
    cudaStreamDestroy(customStream);
}

// Main函数由gtest_main提供