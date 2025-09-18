/**
 * @file test_npp_core_functional.cpp
 * @brief NPP 核心功能纯功能单元测试
 * 
 * 测试NPP库的核心功能，包括版本信息、设备管理、内存分配等
 */

#include "../framework/npp_test_base.h"

using namespace npp_functional_test;

class NppCoreFunctionalTest : public NppTestBase {
protected:
    void SetUp() override {
        NppTestBase::SetUp();
    }
    
    void TearDown() override {
        NppTestBase::TearDown();
    }
};

// ==================== 版本信息测试 ====================

TEST_F(NppCoreFunctionalTest, GetLibraryVersion) {
    const NppLibraryVersion* version = nppGetLibVersion();
    ASSERT_NE(version, nullptr) << "Failed to get library version";
    
    // 验证版本号合理性
    EXPECT_GE(version->major, 1) << "Major version should be >= 1";
    EXPECT_GE(version->minor, 0) << "Minor version should be >= 0";
    EXPECT_GE(version->build, 0) << "Build number should be >= 0";
    
    // 打印版本信息用于调试
    std::cout << "NPP Library Version: " 
              << version->major << "." 
              << version->minor << "." 
              << version->build << std::endl;
}

// ==================== GPU属性测试 ====================

TEST_F(NppCoreFunctionalTest, GetGpuProperties) {
    // 测试GPU名称获取
    const char* gpu_name = nppGetGpuName();
    ASSERT_NE(gpu_name, nullptr) << "Failed to get GPU name";
    
    std::string name(gpu_name);
    EXPECT_FALSE(name.empty()) << "GPU name should not be empty";
    
    std::cout << "GPU Name: " << name << std::endl;
    
    // 测试SM数量获取
    int num_sms = nppGetGpuNumSMs();
    EXPECT_GT(num_sms, 0) << "Number of SMs should be positive";
    
    std::cout << "Number of SMs: " << num_sms << std::endl;
    
    // 测试最大线程数获取
    int max_threads_per_sm = nppGetMaxThreadsPerSM();
    EXPECT_GT(max_threads_per_sm, 0) << "Max threads per SM should be positive";
    EXPECT_LE(max_threads_per_sm, 2048) << "Max threads per SM seems unreasonably high";
    
    std::cout << "Max Threads per SM: " << max_threads_per_sm << std::endl;
    
    int max_threads_per_block = nppGetMaxThreadsPerBlock();
    EXPECT_GT(max_threads_per_block, 0) << "Max threads per block should be positive";
    
    std::cout << "Max Threads per Block: " << max_threads_per_block << std::endl;
}

// ==================== Stream管理测试 ====================

TEST_F(NppCoreFunctionalTest, StreamContext_DefaultContext) {
    NppStreamContext default_ctx;
    nppGetStreamContext(&default_ctx);
    
    // 验证默认context的有效性
    // 默认流通常是0 (nullptr)，这是合法的CUDA默认流
    // EXPECT_NE(default_ctx.hStream, nullptr) << "Default stream should be valid";
    EXPECT_GE(default_ctx.nCudaDeviceId, 0) << "Device ID should be valid";
    EXPECT_GT(default_ctx.nMultiProcessorCount, 0) << "Should have positive number of SMs";
    EXPECT_GT(default_ctx.nMaxThreadsPerMultiProcessor, 0) << "Should have positive max threads per SM";
    EXPECT_GT(default_ctx.nMaxThreadsPerBlock, 0) << "Should have positive max threads per block";
    EXPECT_GT(default_ctx.nSharedMemPerBlock, 0) << "Should have positive shared memory per block";
    
    std::cout << "Default Stream Context:" << std::endl;
    std::cout << "  Device ID: " << default_ctx.nCudaDeviceId << std::endl;
    std::cout << "  SM Count: " << default_ctx.nMultiProcessorCount << std::endl;
    std::cout << "  Max Threads/SM: " << default_ctx.nMaxThreadsPerMultiProcessor << std::endl;
    std::cout << "  Max Threads/Block: " << default_ctx.nMaxThreadsPerBlock << std::endl;
    std::cout << "  Shared Mem/Block: " << default_ctx.nSharedMemPerBlock << " bytes" << std::endl;
}

TEST_F(NppCoreFunctionalTest, StreamContext_CustomStream) {
    // 创建自定义CUDA stream
    cudaStream_t custom_stream;
    cudaError_t err = cudaStreamCreate(&custom_stream);
    ASSERT_EQ(err, cudaSuccess) << "Failed to create CUDA stream";
    
    // 设置自定义stream
    NppStatus status = nppSetStream(custom_stream);
    EXPECT_EQ(status, NPP_SUCCESS) << "Failed to set custom stream";
    
    // 验证可以设置默认stream
    status = nppSetStream(nullptr);  // CUDA default stream
    EXPECT_EQ(status, NPP_SUCCESS) << "Failed to set default stream";
    
    // 清理
    cudaStreamDestroy(custom_stream);
}

TEST_F(NppCoreFunctionalTest, StreamContext_GetStream) {
    // 在简化实现中，nppGetStream应该返回一致的值
    cudaStream_t retrieved_stream = nppGetStream();
    
    // 验证返回值的一致性
    cudaStream_t retrieved_stream2 = nppGetStream();
    EXPECT_EQ(retrieved_stream, retrieved_stream2) 
        << "nppGetStream should return consistent values";
}

// ==================== 内存分配测试 ====================

TEST_F(NppCoreFunctionalTest, Memory_BasicAllocation) {
    const int width = 256;
    const int height = 256;
    
    // 测试不同数据类型的内存分配
    struct TestCase {
        std::string name;
        std::function<void*()> allocator;
        std::function<void(void*)> deallocator;
    };
    
    std::vector<TestCase> test_cases = {
        {"8u_C1", [&]() {
            int step;
            return nppiMalloc_8u_C1(width, height, &step);
        }, [](void* ptr) {
            nppiFree(ptr);
        }},
        {"16u_C1", [&]() {
            int step;
            return nppiMalloc_16u_C1(width, height, &step);
        }, [](void* ptr) {
            nppiFree(ptr);
        }},
        {"32f_C1", [&]() {
            int step;
            return nppiMalloc_32f_C1(width, height, &step);
        }, [](void* ptr) {
            nppiFree(ptr);
        }},
        {"32f_C3", [&]() {
            int step;
            return nppiMalloc_32f_C3(width, height, &step);
        }, [](void* ptr) {
            nppiFree(ptr);
        }}
    };
    
    for (const auto& test_case : test_cases) {
        void* ptr = test_case.allocator();
        EXPECT_NE(ptr, nullptr) << "Failed to allocate " << test_case.name << " memory";
        
        if (ptr) {
            test_case.deallocator(ptr);
        }
    }
}

TEST_F(NppCoreFunctionalTest, Memory_AllocationStep) {
    const int width = 100;  // 非2的幂次，测试对齐
    const int height = 50;
    
    int step8u, step16u, step32f;
    
    // 分配不同类型的内存并检查step
    Npp8u* ptr8u = nppiMalloc_8u_C1(width, height, &step8u);
    Npp16u* ptr16u = nppiMalloc_16u_C1(width, height, &step16u);
    Npp32f* ptr32f = nppiMalloc_32f_C1(width, height, &step32f);
    
    ASSERT_NE(ptr8u, nullptr);
    ASSERT_NE(ptr16u, nullptr);
    ASSERT_NE(ptr32f, nullptr);
    
    // 验证step的合理性
    EXPECT_GE(step8u, width * sizeof(Npp8u)) << "8u step too small";
    EXPECT_GE(step16u, width * sizeof(Npp16u)) << "16u step too small";
    EXPECT_GE(step32f, width * sizeof(Npp32f)) << "32f step too small";
    
    // 验证对齐（通常step应该是某个值的倍数）
    EXPECT_EQ(step8u % sizeof(Npp8u), 0) << "8u step not aligned";
    EXPECT_EQ(step16u % sizeof(Npp16u), 0) << "16u step not aligned";
    EXPECT_EQ(step32f % sizeof(Npp32f), 0) << "32f step not aligned";
    
    std::cout << "Memory allocation steps for " << width << "x" << height << ":" << std::endl;
    std::cout << "  8u step: " << step8u << " bytes" << std::endl;
    std::cout << "  16u step: " << step16u << " bytes" << std::endl;
    std::cout << "  32f step: " << step32f << " bytes" << std::endl;
    
    // 清理
    nppiFree(ptr8u);
    nppiFree(ptr16u);
    nppiFree(ptr32f);
}

TEST_F(NppCoreFunctionalTest, Memory_AllocationLimits) {
    // 测试大内存分配
    const int large_width = 4096;
    const int large_height = 4096;
    
    int step;
    Npp32f* large_ptr = nppiMalloc_32f_C1(large_width, large_height, &step);
    
    if (large_ptr) {
        EXPECT_GE(step, large_width * sizeof(Npp32f));
        std::cout << "Successfully allocated large memory: " 
                  << large_width << "x" << large_height 
                  << " 32f, step=" << step << std::endl;
        nppiFree(large_ptr);
    } else {
        std::cout << "Large memory allocation failed (may be expected on some systems)" << std::endl;
    }
}

// ==================== 错误处理测试 ====================

TEST_F(NppCoreFunctionalTest, ErrorHandling_StatusCodes) {
    // 测试各种状态码的定义
    EXPECT_EQ(static_cast<int>(NPP_NO_ERROR), 0);
    EXPECT_LT(static_cast<int>(NPP_ERROR), 0);
    EXPECT_LT(static_cast<int>(NPP_NULL_POINTER_ERROR), 0);
    EXPECT_LT(static_cast<int>(NPP_SIZE_ERROR), 0);
    EXPECT_LT(static_cast<int>(NPP_BAD_ARGUMENT_ERROR), 0);
}

TEST_F(NppCoreFunctionalTest, ErrorHandling_NullPointerDetection) {
    // 模拟null pointer错误
    NppStreamContext ctx;
    nppGetStreamContext(&ctx);
    
    // 大部分NPP函数应该检测null pointer并返回相应错误
    // 这里我们测试stream设置的边界情况
    NppStatus status = nppSetStream(nullptr);  // 这应该是允许的（设置为默认stream）
    EXPECT_EQ(status, NPP_SUCCESS);
}

// ==================== 计算能力测试 ====================

TEST_F(NppCoreFunctionalTest, ComputeCapability_Detection) {
    // 验证我们能正确检测到合适的计算能力
    EXPECT_TRUE(hasComputeCapability(3, 0)) << "Should support at least compute capability 3.0";
    EXPECT_TRUE(hasComputeCapability(3, 5)) << "Should support at least compute capability 3.5";
    
    // 测试设备属性
    const auto& props = getDeviceProperties();
    EXPECT_GT(props.major, 0) << "Compute capability major should be positive";
    EXPECT_GE(props.minor, 0) << "Compute capability minor should be non-negative";
    
    std::cout << "Compute Capability: " << props.major << "." << props.minor << std::endl;
    std::cout << "GPU: " << props.name << std::endl;
    std::cout << "Global Memory: " << props.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    std::cout << "Shared Memory per Block: " << props.sharedMemPerBlock << " bytes" << std::endl;
    std::cout << "Max Threads per Block: " << props.maxThreadsPerBlock << std::endl;
    std::cout << "Multiprocessor Count: " << props.multiProcessorCount << std::endl;
}

// ==================== 综合集成测试 ====================

TEST_F(NppCoreFunctionalTest, Integration_CompleteWorkflow) {
    // 测试完整的NPP工作流程
    
    // 1. 获取版本信息
    const NppLibraryVersion* version = nppGetLibVersion();
    ASSERT_NE(version, nullptr);
    
    // 2. 获取stream context
    NppStreamContext ctx;
    nppGetStreamContext(&ctx);
    ASSERT_GT(ctx.nMultiProcessorCount, 0);
    
    // 3. 分配内存
    const int width = 128;
    const int height = 128;
    int step;
    Npp32f* test_ptr = nppiMalloc_32f_C1(width, height, &step);
    ASSERT_NE(test_ptr, nullptr);
    
    // 4. 执行简单的内存操作
    std::vector<Npp32f> test_data(width * height, 42.0f);
    cudaError_t err = cudaMemcpy2D(test_ptr, step, 
                                  test_data.data(), width * sizeof(Npp32f),
                                  width * sizeof(Npp32f), height,
                                  cudaMemcpyHostToDevice);
    EXPECT_EQ(err, cudaSuccess);
    
    // 5. 读回数据验证
    std::vector<Npp32f> result_data(width * height);
    err = cudaMemcpy2D(result_data.data(), width * sizeof(Npp32f),
                      test_ptr, step,
                      width * sizeof(Npp32f), height,
                      cudaMemcpyDeviceToHost);
    EXPECT_EQ(err, cudaSuccess);
    
    // 验证数据一致性
    for (size_t i = 0; i < result_data.size(); i++) {
        EXPECT_FLOAT_EQ(result_data[i], 42.0f) << "Data mismatch at index " << i;
    }
    
    // 6. 清理
    nppiFree(test_ptr);
    
    std::cout << "Complete workflow test passed" << std::endl;
}