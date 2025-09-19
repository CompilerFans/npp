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
  void SetUp() override { NppTestBase::SetUp(); }

  void TearDown() override { NppTestBase::TearDown(); }
};

// ==================== 版本信息测试 ====================

TEST_F(NppCoreFunctionalTest, GetLibraryVersion) {
  const NppLibraryVersion *version = nppGetLibVersion();
  ASSERT_NE(version, nullptr) << "Failed to get library version";

  // 验证版本号合理性
  EXPECT_GE(version->major, 1) << "Major version should be >= 1";
  EXPECT_GE(version->minor, 0) << "Minor version should be >= 0";
  EXPECT_GE(version->build, 0) << "Build number should be >= 0";

  // 打印版本信息用于调试
  std::cout << "NPP Library Version: " << version->major << "." << version->minor << "." << version->build << std::endl;
}

// ==================== GPU属性测试 ====================

TEST_F(NppCoreFunctionalTest, GetGpuProperties) {
  // 测试GPU名称获取
  const char *gpu_name = nppGetGpuName();
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
  status = nppSetStream(nullptr); // CUDA default stream
  EXPECT_EQ(status, NPP_SUCCESS) << "Failed to set default stream";

  // 清理
  cudaStreamDestroy(custom_stream);
}

TEST_F(NppCoreFunctionalTest, StreamContext_GetStream) {
  // 在简化实现中，nppGetStream应该返回一致的值
  cudaStream_t retrieved_stream = nppGetStream();

  // 验证返回值的一致性
  cudaStream_t retrieved_stream2 = nppGetStream();
  EXPECT_EQ(retrieved_stream, retrieved_stream2) << "nppGetStream should return consistent values";
}

// ==================== 内存分配测试 ====================

TEST_F(NppCoreFunctionalTest, Memory_BasicAllocation) {
  const int width = 256;
  const int height = 256;

  // 测试不同数据类型的内存分配
  struct TestCase {
    std::string name;
    std::function<void *()> allocator;
    std::function<void(void *)> deallocator;
  };

  std::vector<TestCase> test_cases = {{"8u_C1",
                                       [&]() {
                                         int step;
                                         return nppiMalloc_8u_C1(width, height, &step);
                                       },
                                       [](void *ptr) { nppiFree(ptr); }},
                                      {"16u_C1",
                                       [&]() {
                                         int step;
                                         return nppiMalloc_16u_C1(width, height, &step);
                                       },
                                       [](void *ptr) { nppiFree(ptr); }},
                                      {"32f_C1",
                                       [&]() {
                                         int step;
                                         return nppiMalloc_32f_C1(width, height, &step);
                                       },
                                       [](void *ptr) { nppiFree(ptr); }},
                                      {"32f_C3",
                                       [&]() {
                                         int step;
                                         return nppiMalloc_32f_C3(width, height, &step);
                                       },
                                       [](void *ptr) { nppiFree(ptr); }}};

  for (const auto &test_case : test_cases) {
    void *ptr = test_case.allocator();
    EXPECT_NE(ptr, nullptr) << "Failed to allocate " << test_case.name << " memory";

    if (ptr) {
      test_case.deallocator(ptr);
    }
  }
}

TEST_F(NppCoreFunctionalTest, Memory_AllocationStep) {
  const int width = 100; // 非2的幂次，测试对齐
  const int height = 50;

  int step8u, step16u, step32f;

  // 分配不同类型的内存并检查step
  Npp8u *ptr8u = nppiMalloc_8u_C1(width, height, &step8u);
  Npp16u *ptr16u = nppiMalloc_16u_C1(width, height, &step16u);
  Npp32f *ptr32f = nppiMalloc_32f_C1(width, height, &step32f);

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
  Npp32f *large_ptr = nppiMalloc_32f_C1(large_width, large_height, &step);

  if (large_ptr) {
    EXPECT_GE(step, large_width * sizeof(Npp32f));
    std::cout << "Successfully allocated large memory: " << large_width << "x" << large_height << " 32f, step=" << step
              << std::endl;
    nppiFree(large_ptr);
  } else {
    std::cout << "Large memory allocation failed (may be expected on some systems)" << std::endl;
  }
}

// ==================== 错误处理测试 ====================

// 测试NPP状态码定义是否正确
TEST_F(NppCoreFunctionalTest, ErrorHandling_StatusCodes) {
  // 测试各种状态码的定义
  EXPECT_EQ(static_cast<int>(NPP_NO_ERROR), 0) << "NPP_NO_ERROR should be 0";
  EXPECT_LT(static_cast<int>(NPP_ERROR), 0) << "NPP_ERROR should be negative";
  EXPECT_LT(static_cast<int>(NPP_NULL_POINTER_ERROR), 0) << "NPP_NULL_POINTER_ERROR should be negative";
  EXPECT_LT(static_cast<int>(NPP_SIZE_ERROR), 0) << "NPP_SIZE_ERROR should be negative";
  EXPECT_LT(static_cast<int>(NPP_BAD_ARGUMENT_ERROR), 0) << "NPP_BAD_ARGUMENT_ERROR should be negative";

  // 验证错误码不重复
  EXPECT_NE(static_cast<int>(NPP_NULL_POINTER_ERROR), static_cast<int>(NPP_SIZE_ERROR));
  EXPECT_NE(static_cast<int>(NPP_SIZE_ERROR), static_cast<int>(NPP_BAD_ARGUMENT_ERROR));
}

// 测试基础的stream操作行为
TEST_F(NppCoreFunctionalTest, ErrorHandling_StreamOperations) {
  // 获取当前stream上下文
  NppStreamContext ctx;
  NppStatus status = nppGetStreamContext(&ctx);
  EXPECT_EQ(status, NPP_NO_ERROR) << "nppGetStreamContext should succeed";

  // 设置默认stream应该是允许的
  status = nppSetStream(nullptr); // nullptr表示使用默认stream
  EXPECT_EQ(status, NPP_NO_ERROR) << "nppSetStream with nullptr should succeed";

  // 验证我们可以重新获取上下文
  NppStreamContext newCtx;
  status = nppGetStreamContext(&newCtx);
  EXPECT_EQ(status, NPP_NO_ERROR) << "nppGetStreamContext should work after nppSetStream";
}

// ==================== 计算能力测试 ====================

TEST_F(NppCoreFunctionalTest, ComputeCapability_Detection) {
  // 验证我们能正确检测到合适的计算能力
  EXPECT_TRUE(hasComputeCapability(3, 0)) << "Should support at least compute capability 3.0";
  EXPECT_TRUE(hasComputeCapability(3, 5)) << "Should support at least compute capability 3.5";

  // 测试设备属性
  const auto &props = getDeviceProperties();
  EXPECT_GT(props.major, 0) << "Compute capability major should be positive";
  EXPECT_GE(props.minor, 0) << "Compute capability minor should be non-negative";

  std::cout << "Compute Capability: " << props.major << "." << props.minor << std::endl;
  std::cout << "GPU: " << props.name << std::endl;
  std::cout << "Global Memory: " << props.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
  std::cout << "Shared Memory per Block: " << props.sharedMemPerBlock << " bytes" << std::endl;
  std::cout << "Max Threads per Block: " << props.maxThreadsPerBlock << std::endl;
  std::cout << "Multiprocessor Count: " << props.multiProcessorCount << std::endl;
}

// ==================== 综合集成测试 ====================

TEST_F(NppCoreFunctionalTest, Integration_CompleteWorkflow) {
  // 测试完整的NPP工作流程

  // 1. 获取版本信息
  const NppLibraryVersion *version = nppGetLibVersion();
  ASSERT_NE(version, nullptr);

  // 2. 获取stream context
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);
  ASSERT_GT(ctx.nMultiProcessorCount, 0);

  // 3. 分配内存
  const int width = 128;
  const int height = 128;
  int step;
  Npp32f *test_ptr = nppiMalloc_32f_C1(width, height, &step);
  ASSERT_NE(test_ptr, nullptr);

  // 4. 执行简单的内存操作
  std::vector<Npp32f> test_data(width * height, 42.0f);
  cudaError_t err = cudaMemcpy2D(test_ptr, step, test_data.data(), width * sizeof(Npp32f), width * sizeof(Npp32f),
                                 height, cudaMemcpyHostToDevice);
  EXPECT_EQ(err, cudaSuccess);

  // 5. 读回数据验证
  std::vector<Npp32f> result_data(width * height);
  err = cudaMemcpy2D(result_data.data(), width * sizeof(Npp32f), test_ptr, step, width * sizeof(Npp32f), height,
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

// ==================== 参数化内存分配测试 ====================

// Memory allocation size parameters
struct MemoryAllocParams {
  int width;
  int height;
  std::string name;
  size_t expected_min_bytes;
};

// Parameterized test class for memory allocation
class MemoryAllocSizeTest : public NppCoreFunctionalTest, public ::testing::WithParamInterface<MemoryAllocParams> {};

// Size combinations for testing
INSTANTIATE_TEST_SUITE_P(
    MemorySizes, MemoryAllocSizeTest,
    ::testing::Values(
        MemoryAllocParams{1, 1, "SinglePixel", 4}, MemoryAllocParams{2, 1, "TwoPixelRow", 8},
        MemoryAllocParams{1, 2, "TwoPixelCol", 4}, MemoryAllocParams{3, 3, "SmallSquare", 36},
        MemoryAllocParams{4, 4, "BasicSquare", 64}, MemoryAllocParams{8, 8, "SmallBlock", 256},
        MemoryAllocParams{16, 16, "MediumBlock", 1024}, MemoryAllocParams{32, 32, "StandardBlock", 4096},
        MemoryAllocParams{64, 64, "LargeBlock", 16384}, MemoryAllocParams{128, 128, "VeryLargeBlock", 65536},
        MemoryAllocParams{256, 256, "HighRes", 262144}, MemoryAllocParams{512, 512, "UltraHighRes", 1048576},
        MemoryAllocParams{1024, 1024, "ExtremeRes", 4194304}, MemoryAllocParams{5, 7, "OddSize1", 140},
        MemoryAllocParams{13, 17, "OddSize2", 884}, MemoryAllocParams{100, 50, "RectSize1", 20000},
        MemoryAllocParams{200, 100, "RectSize2", 80000}, MemoryAllocParams{1920, 1080, "HDSize", 8294400}),
    [](const ::testing::TestParamInfo<MemoryAllocParams> &info) { return info.param.name; });

TEST_P(MemoryAllocSizeTest, nppiMalloc_32f_C1_SizeParameterized) {
  auto params = GetParam();
  int step = 0;

  // Test allocation
  Npp32f *ptr = nppiMalloc_32f_C1(params.width, params.height, &step);

  // Verify allocation success
  EXPECT_NE(ptr, nullptr) << "Failed to allocate " << params.name << " memory of size " << params.width << "x"
                          << params.height;

  if (ptr != nullptr) {
    // Verify step validity
    EXPECT_GT(step, 0) << "Step should be positive for " << params.name;
    EXPECT_GE(step, params.width * sizeof(Npp32f)) << "Step too small for " << params.name;
    EXPECT_EQ(step % sizeof(Npp32f), 0) << "Step not aligned for " << params.name;

    // Test basic memory operations
    std::vector<Npp32f> test_data(params.width * params.height, 123.456f);
    cudaError_t err = cudaMemcpy2D(ptr, step, test_data.data(), params.width * sizeof(Npp32f),
                                   params.width * sizeof(Npp32f), params.height, cudaMemcpyHostToDevice);
    EXPECT_EQ(err, cudaSuccess) << "Failed to copy data to GPU for " << params.name;

    // Verify memory can be read back
    std::vector<Npp32f> result_data(params.width * params.height);
    err = cudaMemcpy2D(result_data.data(), params.width * sizeof(Npp32f), ptr, step, params.width * sizeof(Npp32f),
                       params.height, cudaMemcpyDeviceToHost);
    EXPECT_EQ(err, cudaSuccess) << "Failed to copy data from GPU for " << params.name;

    // Verify data integrity
    for (int i = 0; i < params.width * params.height; i++) {
      EXPECT_FLOAT_EQ(result_data[i], 123.456f) << "Data mismatch at index " << i << " for " << params.name;
    }

    // Test deallocation
    nppiFree(ptr);
    // Note: No direct way to verify free success, but subsequent allocations should work
  }
}

TEST_P(MemoryAllocSizeTest, nppiMalloc_32f_C3_SizeParameterized) {
  auto params = GetParam();
  int step = 0;

  // Test allocation for 3-channel
  Npp32f *ptr = nppiMalloc_32f_C3(params.width, params.height, &step);

  // Verify allocation success
  EXPECT_NE(ptr, nullptr) << "Failed to allocate " << params.name << " C3 memory of size " << params.width << "x"
                          << params.height;

  if (ptr != nullptr) {
    // Verify step validity for 3-channel
    EXPECT_GT(step, 0) << "Step should be positive for C3 " << params.name;
    EXPECT_GE(step, params.width * 3 * sizeof(Npp32f)) << "Step too small for C3 " << params.name;
    EXPECT_EQ(step % sizeof(Npp32f), 0) << "Step not aligned for C3 " << params.name;

    // Test memory operations with 3-channel data
    std::vector<Npp32f> test_data(params.width * params.height * 3);
    for (int i = 0; i < params.width * params.height * 3; i++) {
      test_data[i] = static_cast<Npp32f>(i % 256) / 255.0f; // Pattern data
    }

    cudaError_t err = cudaMemcpy2D(ptr, step, test_data.data(), params.width * 3 * sizeof(Npp32f),
                                   params.width * 3 * sizeof(Npp32f), params.height, cudaMemcpyHostToDevice);
    EXPECT_EQ(err, cudaSuccess) << "Failed to copy C3 data to GPU for " << params.name;

    // Verify memory can be read back
    std::vector<Npp32f> result_data(params.width * params.height * 3);
    err = cudaMemcpy2D(result_data.data(), params.width * 3 * sizeof(Npp32f), ptr, step,
                       params.width * 3 * sizeof(Npp32f), params.height, cudaMemcpyDeviceToHost);
    EXPECT_EQ(err, cudaSuccess) << "Failed to copy C3 data from GPU for " << params.name;

    // Verify data integrity
    for (int i = 0; i < params.width * params.height * 3; i++) {
      EXPECT_FLOAT_EQ(result_data[i], test_data[i]) << "C3 data mismatch at index " << i << " for " << params.name;
    }

    // Test deallocation
    nppiFree(ptr);
  }
}

// ==================== 内存分配模式测试 ====================

// Memory allocation pattern parameters
struct MemoryPatternParams {
  std::string pattern_name;
  std::function<std::vector<Npp32f>(int, int)> data_generator;
  std::function<bool(const std::vector<Npp32f> &, const std::vector<Npp32f> &)> validator;
};

class MemoryPatternTest : public NppCoreFunctionalTest, public ::testing::WithParamInterface<MemoryPatternParams> {};

INSTANTIATE_TEST_SUITE_P(
    MemoryPatterns, MemoryPatternTest,
    ::testing::Values(
        MemoryPatternParams{"ZeroPattern",
                            [](int w, int h) -> std::vector<Npp32f> { return std::vector<Npp32f>(w * h, 0.0f); },
                            [](const std::vector<Npp32f> &original, const std::vector<Npp32f> &result) -> bool {
                              (void)original; // Suppress unused parameter warning
                              return std::all_of(result.begin(), result.end(), [](Npp32f val) { return val == 0.0f; });
                            }},
        MemoryPatternParams{"OnePattern",
                            [](int w, int h) -> std::vector<Npp32f> { return std::vector<Npp32f>(w * h, 1.0f); },
                            [](const std::vector<Npp32f> &original, const std::vector<Npp32f> &result) -> bool {
                              (void)original; // Suppress unused parameter warning
                              return std::all_of(result.begin(), result.end(), [](Npp32f val) { return val == 1.0f; });
                            }},
        MemoryPatternParams{"RandomPattern",
                            [](int w, int h) -> std::vector<Npp32f> {
                              std::vector<Npp32f> data(w * h);
                              for (int i = 0; i < w * h; i++) {
                                data[i] = static_cast<Npp32f>(rand()) / RAND_MAX;
                              }
                              return data;
                            },
                            [](const std::vector<Npp32f> &original, const std::vector<Npp32f> &result) -> bool {
                              for (size_t i = 0; i < original.size(); i++) {
                                if (std::abs(original[i] - result[i]) > 1e-6f)
                                  return false;
                              }
                              return true;
                            }},
        MemoryPatternParams{"GradientPattern",
                            [](int w, int h) -> std::vector<Npp32f> {
                              std::vector<Npp32f> data(w * h);
                              for (int y = 0; y < h; y++) {
                                for (int x = 0; x < w; x++) {
                                  data[y * w + x] = static_cast<Npp32f>(x + y) / (w + h - 2);
                                }
                              }
                              return data;
                            },
                            [](const std::vector<Npp32f> &original, const std::vector<Npp32f> &result) -> bool {
                              for (size_t i = 0; i < original.size(); i++) {
                                if (std::abs(original[i] - result[i]) > 1e-6f)
                                  return false;
                              }
                              return true;
                            }},
        MemoryPatternParams{"CheckerboardPattern",
                            [](int w, int h) -> std::vector<Npp32f> {
                              std::vector<Npp32f> data(w * h);
                              for (int y = 0; y < h; y++) {
                                for (int x = 0; x < w; x++) {
                                  data[y * w + x] = ((x + y) % 2 == 0) ? 1.0f : 0.0f;
                                }
                              }
                              return data;
                            },
                            [](const std::vector<Npp32f> &original, const std::vector<Npp32f> &result) -> bool {
                              for (size_t i = 0; i < original.size(); i++) {
                                if (std::abs(original[i] - result[i]) > 1e-6f)
                                  return false;
                              }
                              return true;
                            }}),
    [](const ::testing::TestParamInfo<MemoryPatternParams> &info) { return info.param.pattern_name; });

TEST_P(MemoryPatternTest, MemoryPattern_32f_C1) {
  auto params = GetParam();
  const int width = 64;
  const int height = 64;
  int step = 0;

  // Allocate memory
  Npp32f *ptr = nppiMalloc_32f_C1(width, height, &step);
  ASSERT_NE(ptr, nullptr) << "Failed to allocate memory for " << params.pattern_name;

  // Generate test data
  auto test_data = params.data_generator(width, height);

  // Copy to GPU
  cudaError_t err = cudaMemcpy2D(ptr, step, test_data.data(), width * sizeof(Npp32f), width * sizeof(Npp32f), height,
                                 cudaMemcpyHostToDevice);
  EXPECT_EQ(err, cudaSuccess) << "Failed to copy " << params.pattern_name << " to GPU";

  // Copy back from GPU
  std::vector<Npp32f> result_data(width * height);
  err = cudaMemcpy2D(result_data.data(), width * sizeof(Npp32f), ptr, step, width * sizeof(Npp32f), height,
                     cudaMemcpyDeviceToHost);
  EXPECT_EQ(err, cudaSuccess) << "Failed to copy " << params.pattern_name << " from GPU";

  // Validate pattern
  EXPECT_TRUE(params.validator(test_data, result_data)) << "Pattern validation failed for " << params.pattern_name;

  // Free memory
  nppiFree(ptr);
}

TEST_P(MemoryPatternTest, MemoryPattern_32f_C3) {
  auto params = GetParam();
  const int width = 64;
  const int height = 64;
  int step = 0;

  // Allocate memory for 3-channel
  Npp32f *ptr = nppiMalloc_32f_C3(width, height, &step);
  ASSERT_NE(ptr, nullptr) << "Failed to allocate C3 memory for " << params.pattern_name;

  // Generate test data for 3 channels
  auto channel_data = params.data_generator(width, height);
  std::vector<Npp32f> test_data(width * height * 3);
  for (int i = 0; i < width * height; i++) {
    test_data[i * 3 + 0] = channel_data[i];         // R channel
    test_data[i * 3 + 1] = channel_data[i] * 0.5f;  // G channel (scaled)
    test_data[i * 3 + 2] = channel_data[i] * 0.25f; // B channel (scaled)
  }

  // Copy to GPU
  cudaError_t err = cudaMemcpy2D(ptr, step, test_data.data(), width * 3 * sizeof(Npp32f), width * 3 * sizeof(Npp32f),
                                 height, cudaMemcpyHostToDevice);
  EXPECT_EQ(err, cudaSuccess) << "Failed to copy C3 " << params.pattern_name << " to GPU";

  // Copy back from GPU
  std::vector<Npp32f> result_data(width * height * 3);
  err = cudaMemcpy2D(result_data.data(), width * 3 * sizeof(Npp32f), ptr, step, width * 3 * sizeof(Npp32f), height,
                     cudaMemcpyDeviceToHost);
  EXPECT_EQ(err, cudaSuccess) << "Failed to copy C3 " << params.pattern_name << " from GPU";

  // Validate each channel
  for (int i = 0; i < width * height; i++) {
    EXPECT_FLOAT_EQ(result_data[i * 3 + 0], test_data[i * 3 + 0])
        << "R channel mismatch at pixel " << i << " for " << params.pattern_name;
    EXPECT_FLOAT_EQ(result_data[i * 3 + 1], test_data[i * 3 + 1])
        << "G channel mismatch at pixel " << i << " for " << params.pattern_name;
    EXPECT_FLOAT_EQ(result_data[i * 3 + 2], test_data[i * 3 + 2])
        << "B channel mismatch at pixel " << i << " for " << params.pattern_name;
  }

  // Free memory
  nppiFree(ptr);
}

// ==================== 内存对齐和步长测试 ====================

// Step alignment test parameters
struct StepAlignmentParams {
  int width;
  int height;
  std::string test_name;
  int expected_alignment;
};

class StepAlignmentTest : public NppCoreFunctionalTest, public ::testing::WithParamInterface<StepAlignmentParams> {};

INSTANTIATE_TEST_SUITE_P(
    StepAlignments, StepAlignmentTest,
    ::testing::Values(StepAlignmentParams{1, 1, "MinSize", 4}, StepAlignmentParams{3, 1, "OddWidth", 4},
                      StepAlignmentParams{5, 1, "PrimeWidth", 4}, StepAlignmentParams{7, 1, "SevenWidth", 4},
                      StepAlignmentParams{15, 1, "FifteenWidth", 4}, StepAlignmentParams{17, 1, "SeventeenWidth", 4},
                      StepAlignmentParams{31, 1, "ThirtyOneWidth", 4},
                      StepAlignmentParams{33, 1, "ThirtyThreeWidth", 4},
                      StepAlignmentParams{63, 1, "SixtyThreeWidth", 4}, StepAlignmentParams{65, 1, "SixtyFiveWidth", 4},
                      StepAlignmentParams{127, 1, "HundredTwentySevenWidth", 4},
                      StepAlignmentParams{129, 1, "HundredTwentyNineWidth", 4}),
    [](const ::testing::TestParamInfo<StepAlignmentParams> &info) { return info.param.test_name; });

TEST_P(StepAlignmentTest, StepAlignment_32f_C1) {
  auto params = GetParam();
  int step = 0;

  Npp32f *ptr = nppiMalloc_32f_C1(params.width, params.height, &step);
  ASSERT_NE(ptr, nullptr) << "Failed to allocate memory for " << params.test_name;

  // Verify step alignment
  EXPECT_EQ(step % params.expected_alignment, 0)
      << "Step not aligned to " << params.expected_alignment << " bytes for " << params.test_name;

  // Verify minimum step size
  EXPECT_GE(step, params.width * sizeof(Npp32f)) << "Step too small for " << params.test_name;

  // Test memory access patterns at different alignments
  std::vector<Npp32f> test_data(params.width * params.height);
  for (int i = 0; i < params.width * params.height; i++) {
    test_data[i] = static_cast<Npp32f>(i % 256) / 255.0f;
  }

  cudaError_t err = cudaMemcpy2D(ptr, step, test_data.data(), params.width * sizeof(Npp32f),
                                 params.width * sizeof(Npp32f), params.height, cudaMemcpyHostToDevice);
  EXPECT_EQ(err, cudaSuccess) << "Failed to copy data for " << params.test_name;

  nppiFree(ptr);
}

TEST_P(StepAlignmentTest, StepAlignment_32f_C3) {
  auto params = GetParam();
  int step = 0;

  Npp32f *ptr = nppiMalloc_32f_C3(params.width, params.height, &step);
  ASSERT_NE(ptr, nullptr) << "Failed to allocate C3 memory for " << params.test_name;

  // Verify step alignment
  EXPECT_EQ(step % params.expected_alignment, 0)
      << "C3 step not aligned to " << params.expected_alignment << " bytes for " << params.test_name;

  // Verify minimum step size for 3-channel
  EXPECT_GE(step, params.width * 3 * sizeof(Npp32f)) << "C3 step too small for " << params.test_name;

  // Test memory access for 3-channel data
  std::vector<Npp32f> test_data(params.width * params.height * 3);
  for (int i = 0; i < params.width * params.height * 3; i++) {
    test_data[i] = static_cast<Npp32f>(i % 256) / 255.0f;
  }

  cudaError_t err = cudaMemcpy2D(ptr, step, test_data.data(), params.width * 3 * sizeof(Npp32f),
                                 params.width * 3 * sizeof(Npp32f), params.height, cudaMemcpyHostToDevice);
  EXPECT_EQ(err, cudaSuccess) << "Failed to copy C3 data for " << params.test_name;

  nppiFree(ptr);
}

// ==================== 内存泄漏和重复分配测试 ====================

TEST_F(NppCoreFunctionalTest, MemoryStress_MultipleAllocations) {
  const int num_allocations = 50;
  const int width = 128;
  const int height = 128;

  std::vector<Npp32f *> c1_ptrs;
  std::vector<Npp32f *> c3_ptrs;
  std::vector<int> c1_steps;
  std::vector<int> c3_steps;

  // Multiple C1 allocations
  for (int i = 0; i < num_allocations; i++) {
    int step = 0;
    Npp32f *ptr = nppiMalloc_32f_C1(width, height, &step);
    EXPECT_NE(ptr, nullptr) << "Failed C1 allocation " << i;
    if (ptr) {
      c1_ptrs.push_back(ptr);
      c1_steps.push_back(step);
    }
  }

  // Multiple C3 allocations
  for (int i = 0; i < num_allocations; i++) {
    int step = 0;
    Npp32f *ptr = nppiMalloc_32f_C3(width, height, &step);
    EXPECT_NE(ptr, nullptr) << "Failed C3 allocation " << i;
    if (ptr) {
      c3_ptrs.push_back(ptr);
      c3_steps.push_back(step);
    }
  }

  // Verify all pointers are unique
  std::set<void *> unique_ptrs;
  for (auto ptr : c1_ptrs) {
    EXPECT_TRUE(unique_ptrs.insert(ptr).second) << "Duplicate C1 pointer detected";
  }
  for (auto ptr : c3_ptrs) {
    EXPECT_TRUE(unique_ptrs.insert(ptr).second) << "Duplicate C3 pointer detected";
  }

  // Free all allocations
  for (auto ptr : c1_ptrs) {
    nppiFree(ptr);
  }
  for (auto ptr : c3_ptrs) {
    nppiFree(ptr);
  }

  std::cout << "Successfully allocated and freed " << num_allocations << " C1 and " << num_allocations
            << " C3 memory blocks" << std::endl;
}

TEST_F(NppCoreFunctionalTest, MemoryStress_AllocFreeCycle) {
  const int cycles = 100;
  const int width = 256;
  const int height = 256;

  for (int cycle = 0; cycle < cycles; cycle++) {
    int step_c1 = 0, step_c3 = 0;

    // Allocate
    Npp32f *ptr_c1 = nppiMalloc_32f_C1(width, height, &step_c1);
    Npp32f *ptr_c3 = nppiMalloc_32f_C3(width, height, &step_c3);

    EXPECT_NE(ptr_c1, nullptr) << "Failed C1 allocation in cycle " << cycle;
    EXPECT_NE(ptr_c3, nullptr) << "Failed C3 allocation in cycle " << cycle;

    if (ptr_c1 && ptr_c3) {
      // Quick memory test
      std::vector<Npp32f> test_data_c1(width * height, static_cast<Npp32f>(cycle));
      std::vector<Npp32f> test_data_c3(width * height * 3, static_cast<Npp32f>(cycle) * 0.5f);

      cudaMemcpy2D(ptr_c1, step_c1, test_data_c1.data(), width * sizeof(Npp32f), width * sizeof(Npp32f), height,
                   cudaMemcpyHostToDevice);
      cudaMemcpy2D(ptr_c3, step_c3, test_data_c3.data(), width * 3 * sizeof(Npp32f), width * 3 * sizeof(Npp32f), height,
                   cudaMemcpyHostToDevice);
    }

    // Free
    if (ptr_c1)
      nppiFree(ptr_c1);
    if (ptr_c3)
      nppiFree(ptr_c3);
  }

  std::cout << "Successfully completed " << cycles << " alloc/free cycles" << std::endl;
}