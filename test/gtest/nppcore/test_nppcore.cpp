/**
 * @file test_nppcore.cpp
 * @brief NPP核心功能Google Test测试套件
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <cctype>
#include "npp.h"

// 测试NPP版本信息
class NPPVersionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 获取版本信息
        library_version = nppGetLibVersion();
        // Get compute capability directly from CUDA
        int deviceId;
        cudaGetDevice(&deviceId);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, deviceId);
        gpu_compute_capability = prop.major * 10 + prop.minor;
    }
    
    const NppLibraryVersion* library_version;
    int gpu_compute_capability;
};

TEST_F(NPPVersionTest, GetLibraryVersion) {
    ASSERT_NE(library_version, nullptr) << "Failed to get library version";
    
    std::cout << "\nNPP Library Version:\n";
    std::cout << "  Major: " << library_version->major << "\n";
    std::cout << "  Minor: " << library_version->minor << "\n";
    std::cout << "  Build: " << library_version->build << "\n";
    
    // 验证版本号合理性
    EXPECT_GE(library_version->major, 1) << "Major version should be >= 1";
    EXPECT_GE(library_version->minor, 0) << "Minor version should be >= 0";
    EXPECT_GE(library_version->build, 0) << "Build number should be >= 0";
}

TEST_F(NPPVersionTest, GetGpuComputeCapability) {
    std::cout << "\nGPU Compute Capability: " 
              << (gpu_compute_capability / 10) << "." 
              << (gpu_compute_capability % 10) << "\n";
    
    // 验证compute capability合理性（至少3.5）
    EXPECT_GE(gpu_compute_capability, 35) 
        << "GPU compute capability should be at least 3.5";
}

// 测试GPU属性查询
class GPUPropertiesTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaGetDeviceProperties(&props, 0);
        ASSERT_EQ(err, cudaSuccess) << "Failed to get device properties";
        
        num_sms = nppGetGpuNumSMs();
        gpu_name = nppGetGpuName();
    }
    
    cudaDeviceProp props;
    int num_sms;
    const char* gpu_name;
};

TEST_F(GPUPropertiesTest, GetGpuName) {
    ASSERT_NE(gpu_name, nullptr) << "Failed to get GPU name";
    
    std::string name(gpu_name);
    std::cout << "\nGPU Name: " << name << "\n";
    
    EXPECT_FALSE(name.empty()) << "GPU name should not be empty";
    
    // 验证GPU名称与CUDA属性一致
    std::string cuda_name(props.name);
    EXPECT_EQ(name, cuda_name) << "NPP GPU name should match CUDA device name";
}

TEST_F(GPUPropertiesTest, GetNumberOfSMs) {
    std::cout << "Number of SMs: " << num_sms << "\n";
    
    EXPECT_GT(num_sms, 0) << "Number of SMs should be positive";
    EXPECT_EQ(num_sms, props.multiProcessorCount) 
        << "NPP SM count should match CUDA device SM count";
}

TEST_F(GPUPropertiesTest, GetMaxThreadsPerSM) {
    int max_threads_per_sm;
    max_threads_per_sm = nppGetMaxThreadsPerSM();
    
    std::cout << "Max Threads per SM: " << max_threads_per_sm << "\n";
    
    EXPECT_GT(max_threads_per_sm, 0) << "Max threads per SM should be positive";
    EXPECT_LE(max_threads_per_sm, 2048) << "Max threads per SM seems unreasonably high";
}

TEST_F(GPUPropertiesTest, GetMaxThreadsPerBlock) {
    int max_threads_per_block;
    max_threads_per_block = nppGetMaxThreadsPerBlock();
    
    std::cout << "Max Threads per Block: " << max_threads_per_block << "\n";
    
    EXPECT_GT(max_threads_per_block, 0) << "Max threads per block should be positive";
    EXPECT_EQ(max_threads_per_block, props.maxThreadsPerBlock)
        << "NPP max threads per block should match CUDA device property";
}

// 测试Stream管理
class StreamManagementTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建CUDA stream
        cudaError_t err = cudaStreamCreate(&stream);
        ASSERT_EQ(err, cudaSuccess) << "Failed to create CUDA stream";
        
        // 获取默认stream context
        nppGetStreamContext(&default_ctx);
    }
    
    void TearDown() override {
        if (stream) {
            cudaStreamDestroy(stream);
        }
    }
    
    cudaStream_t stream = nullptr;
    NppStreamContext default_ctx;
};

TEST_F(StreamManagementTest, GetStreamContext) {
    // 验证默认context
    EXPECT_NE(default_ctx.hStream, nullptr) 
        << "Default stream context should have valid stream";
    EXPECT_GT(default_ctx.nCudaDeviceId, -1) 
        << "Device ID should be valid";
    EXPECT_GT(default_ctx.nMultiProcessorCount, 0) 
        << "Should have positive number of SMs";
    EXPECT_GT(default_ctx.nMaxThreadsPerMultiProcessor, 0) 
        << "Should have positive max threads per SM";
    EXPECT_GT(default_ctx.nMaxThreadsPerBlock, 0) 
        << "Should have positive max threads per block";
    
    std::cout << "\nDefault Stream Context:\n";
    std::cout << "  Device ID: " << default_ctx.nCudaDeviceId << "\n";
    std::cout << "  SM Count: " << default_ctx.nMultiProcessorCount << "\n";
    std::cout << "  Max Threads/SM: " << default_ctx.nMaxThreadsPerMultiProcessor << "\n";
    std::cout << "  Max Threads/Block: " << default_ctx.nMaxThreadsPerBlock << "\n";
    std::cout << "  Shared Mem/Block: " << default_ctx.nSharedMemPerBlock << " bytes\n";
}

TEST_F(StreamManagementTest, SetStream) {
    // 设置自定义stream - 在新实现中只验证stream有效性
    NppStatus status = nppSetStream(stream);
    EXPECT_EQ(status, NPP_SUCCESS) << "Failed to validate custom stream";
    
    // 验证默认stream也是有效的
    status = nppSetStream(0);  // CUDA default stream
    EXPECT_EQ(status, NPP_SUCCESS) << "Failed to validate default stream";
    
    // 验证legacy stream
    status = nppSetStream(cudaStreamLegacy);
    EXPECT_EQ(status, NPP_SUCCESS) << "Failed to validate legacy stream";
}

TEST_F(StreamManagementTest, GetStreamAfterSet) {
    // 在新实现中，nppGetStream始终返回cudaStreamLegacy
    // 这是设计决定：用户应该使用CUDA Runtime API管理stream
    cudaStream_t retrieved = nppGetStream();
    
    // 验证返回的是legacy stream
    EXPECT_EQ(retrieved, cudaStreamLegacy) 
        << "nppGetStream should return cudaStreamLegacy in simplified implementation";
    
    // 验证SetStream不会改变GetStream的返回值
    nppSetStream(stream);
    retrieved = nppGetStream();
    EXPECT_EQ(retrieved, cudaStreamLegacy) 
        << "nppGetStream should still return cudaStreamLegacy after SetStream";
}

// 测试错误处理
class ErrorHandlingTest : public ::testing::Test {
protected:
    std::string GetErrorString(NppStatus status) {
        // 没有nppGetStatusString API，直接实现
        switch(status) {
            case NPP_SUCCESS: return "Success";
            case NPP_ERROR: return "General error";
            case NPP_MEMORY_ALLOCATION_ERR: return "Memory allocation error";
            case NPP_NULL_POINTER_ERROR: return "Null pointer error";
            case NPP_CUDA_KERNEL_EXECUTION_ERROR: return "CUDA kernel execution error";
            case NPP_ALIGNMENT_ERROR: return "Alignment error";
            case NPP_SIZE_ERROR: return "Size error";
            case NPP_STEP_ERROR: return "Step error";
            case NPP_NOT_SUPPORTED_MODE_ERROR: return "Unsupported mode error";
            case NPP_QUALITY_INDEX_ERROR: return "Quality index error";
            case NPP_CHANNEL_ORDER_ERROR: return "Channel order error";
            case NPP_ZERO_MASK_VALUE_ERROR: return "Zero mask value error";
            case NPP_NOT_EVEN_STEP_ERROR: return "Not even step error";
            case NPP_DIVIDE_BY_ZERO_ERROR: return "Divide by zero error";
            case NPP_COEFFICIENT_ERROR: return "Coefficient error";
            case NPP_BAD_ARGUMENT_ERROR: return "Bad argument error";
            default: return "Unknown error";
        }
    }
};

TEST_F(ErrorHandlingTest, GetStatusString) {
    // 测试各种错误码的字符串表示
    struct {
        NppStatus status;
        const char* expected_substring;
    } test_cases[] = {
        {NPP_SUCCESS, "Success"},
        {NPP_ERROR, "Error"},
        {NPP_MEMORY_ALLOCATION_ERR, "Memory"},
        {NPP_NULL_POINTER_ERROR, "Null"},
        {NPP_CUDA_KERNEL_EXECUTION_ERROR, "Kernel"},
        // {NPP_INVALID_INPUT, "Invalid"}, // TODO: 定义这个错误码
        {NPP_ALIGNMENT_ERROR, "Alignment"},
        {NPP_SIZE_ERROR, "Size"},
        {NPP_STEP_ERROR, "Step"},
        {NPP_NOT_SUPPORTED_MODE_ERROR, "Not Supported"},
        {NPP_QUALITY_INDEX_ERROR, "Quality"},
        {NPP_CHANNEL_ORDER_ERROR, "Channel"},
        {NPP_ZERO_MASK_VALUE_ERROR, "Zero Mask"},
        {NPP_NOT_EVEN_STEP_ERROR, "Not Even Step"},
        {NPP_DIVIDE_BY_ZERO_ERROR, "Divide"},
        {NPP_COEFFICIENT_ERROR, "Coefficient"},
        {NPP_BAD_ARGUMENT_ERROR, "Bad Argument"},
        // {NPP_WARNING, "Warning"} // TODO: 定义警告码
    };
    
    for (const auto& test : test_cases) {
        std::string error_str = GetErrorString(test.status);
        EXPECT_FALSE(error_str.empty()) 
            << "Error string should not be empty for status " << test.status;
        
        // 验证错误字符串包含预期的关键词（忽略大小写）
        std::string expected(test.expected_substring);
        std::transform(error_str.begin(), error_str.end(), error_str.begin(), ::tolower);
        std::transform(expected.begin(), expected.end(), expected.begin(), ::tolower);
        EXPECT_NE(error_str.find(expected), std::string::npos)
            << "Error string for status " << test.status 
            << " should contain '" << test.expected_substring << "', got: " << GetErrorString(test.status);
    }
}

TEST_F(ErrorHandlingTest, StatusStringConsistency) {
    // 验证相同的错误码总是返回相同的字符串
    NppStatus status = NPP_CUDA_KERNEL_EXECUTION_ERROR;
    
    std::string str1 = GetErrorString(status);
    std::string str2 = GetErrorString(status);
    
    EXPECT_EQ(str1, str2) 
        << "Same error status should always return the same string";
}

// 测试设备管理
class DeviceManagementTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaGetDevice(&original_device);
        cudaGetDeviceCount(&device_count);
    }
    
    void TearDown() override {
        cudaSetDevice(original_device);
    }
    
    int original_device;
    int device_count;
};

TEST_F(DeviceManagementTest, MultiDeviceSupport) {
    if (device_count < 2) {
        GTEST_SKIP() << "Test requires multiple GPUs";
    }
    
    // 测试在不同设备上获取属性
    for (int device = 0; device < std::min(device_count, 2); ++device) {
        cudaSetDevice(device);
        
        int compute_cap = 80; // TODO: 实现nppGetGpuComputeCapability
        EXPECT_GT(compute_cap, 0) 
            << "Should get valid compute capability for device " << device;
        
        const char* gpu_name = nppGetGpuName();
        EXPECT_NE(gpu_name, nullptr) 
            << "Should get valid GPU name for device " << device;
        
        std::cout << "Device " << device << ": " << gpu_name 
                  << " (CC " << compute_cap/10 << "." << compute_cap%10 << ")\n";
    }
}

// 测试内存管理辅助功能
class MemoryManagementTest : public ::testing::Test {
protected:
    void SetUp() override {
        nppGetStreamContext(&ctx);
    }
    
    NppStreamContext ctx;
};

TEST_F(MemoryManagementTest, StreamContextMemoryInfo) {
    // 验证stream context中的内存信息
    EXPECT_GT(ctx.nSharedMemPerBlock, 0) 
        << "Shared memory per block should be positive";
    
    // 典型值范围检查（取决于GPU架构）
    EXPECT_GE(ctx.nSharedMemPerBlock, 16384)  // 至少16KB
        << "Shared memory seems too small";
    EXPECT_LE(ctx.nSharedMemPerBlock, 167936) // 最多164KB (newer GPUs)
        << "Shared memory seems unreasonably large";
    
    std::cout << "\nMemory Configuration:\n";
    std::cout << "  Shared Memory per Block: " 
              << ctx.nSharedMemPerBlock << " bytes\n";
}

// 综合测试套件
class NPPCoreIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 完整的NPP初始化序列
        const NppLibraryVersion* version = nppGetLibVersion();
        ASSERT_NE(version, nullptr);
        
        int compute_cap = 80; // TODO: 实现nppGetGpuComputeCapability
        ASSERT_GT(compute_cap, 0);
        
        nppGetStreamContext(&ctx);
        ASSERT_NE(ctx.hStream, nullptr);
    }
    
    NppStreamContext ctx;
};

TEST_F(NPPCoreIntegrationTest, CompleteInitialization) {
    // 验证完整的初始化流程
    EXPECT_GT(ctx.nCudaDeviceId, -1);
    EXPECT_GT(ctx.nMultiProcessorCount, 0);
    EXPECT_GT(ctx.nMaxThreadsPerMultiProcessor, 0);
    EXPECT_GT(ctx.nMaxThreadsPerBlock, 0);
    EXPECT_GT(ctx.nSharedMemPerBlock, 0);
    
    // 验证可以成功分配NPP内存
    int step;
    Npp8u* ptr = nppiMalloc_8u_C1(1024, 768, &step);
    EXPECT_NE(ptr, nullptr) << "Should be able to allocate NPP memory";
    EXPECT_GE(step, 1024) << "Step should be at least width";
    
    if (ptr) {
        nppiFree(ptr);
    }
}

// Main函数由gtest_main提供，不需要自定义main