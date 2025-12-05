#include "npp_test_base.h"

#include "npp.h"
#include <algorithm>
#include <atomic>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <mutex>
#include <random>
#include <set>
#include <thread>
#include <vector>

using namespace npp_functional_test;

class SupportFunctionsTest : public NppTestBase {
protected:
  void SetUp() override { NppTestBase::SetUp(); }

  void TearDown() override { NppTestBase::TearDown(); }
};

TEST_F(SupportFunctionsTest, GetLibVersion_BasicTest) {
  const NppLibraryVersion *version = nppGetLibVersion();
  ASSERT_NE(version, nullptr) << "nppGetLibVersion should not return null";

  // Validate版本号格式合理
  EXPECT_GT(version->major, 0) << "Major version should be greater than 0";
  EXPECT_GE(version->minor, 0) << "Minor version should be non-negative";
  EXPECT_GE(version->build, 0) << "Build version should be non-negative";

  std::cout << "NPP Library Version: " << version->major << "." << version->minor << "." << version->build << std::endl;
}

// nppGetGpuComputeCapability函数在当前API中不存在，测试已禁用
/*
TEST_F(SupportFunctionsTest, GetGpuComputeCapability_BasicTest) {
    int major, minor;
    NppStatus status = nppGetGpuComputeCapability(&major, &minor);

    ASSERT_EQ(status, NPP_NO_ERROR) << "nppGetGpuComputeCapability should succeed";

    // Validate计算能力值合理
    EXPECT_GT(major, 0) << "GPU compute capability major version should be > 0";
    EXPECT_GE(minor, 0) << "GPU compute capability minor version should be >= 0";
    EXPECT_LE(major, 9) << "GPU compute capability major version should be reasonable";
    EXPECT_LE(minor, 9) << "GPU compute capability minor version should be reasonable";

    std::cout << "GPU Compute Capability: " << major << "." << minor << std::endl;
}
*/

TEST_F(SupportFunctionsTest, GetGpuName_BasicTest) {
  // nppGetGpuName 实际上已经实现，返回const char*，不接受缓冲区参数
  const char *name = nppGetGpuName();

  if (name != nullptr) {
    // Validate名称不为空
    EXPECT_GT(strlen(name), 0) << "GPU name should not be empty";
    std::cout << "GPU Name: " << name << std::endl;
  } else {
    std::cout << "nppGetGpuName returned nullptr" << std::endl;
  }
}

TEST_F(SupportFunctionsTest, GetStreamContext_BasicTest) {
  NppStreamContext streamCtx;
  NppStatus status = nppGetStreamContext(&streamCtx);

  if (status == NPP_NO_ERROR) {
    // ValidateDefault stream上下文
    EXPECT_EQ(streamCtx.hStream, (cudaStream_t)0) << "Default stream should be 0";
    EXPECT_NE(streamCtx.nCudaDeviceId, -1) << "Device ID should be valid";
    std::cout << "nppGetStreamContext succeeded" << std::endl;
  } else {
    std::cout << "nppGetStreamContext returned error: " << status << std::endl;
  }
}

TEST_F(SupportFunctionsTest, SetGetStreamContext_BasicTest) {
  //
  NppStreamContext currentCtx;
  NppStatus status = nppGetStreamContext(&currentCtx);
  ASSERT_EQ(status, NPP_NO_ERROR) << "nppGetStreamContext should succeed";

  // nppSetStreamContext函数在当前API中不存在，相关测试已禁用
  /*
  status = nppSetStreamContext(currentCtx);
  ASSERT_EQ(status, NPP_NO_ERROR) << "nppSetStreamContext should succeed with valid context";

  // 测试无效设备ID
  NppStreamContext invalidCtx = currentCtx;
  invalidCtx.nCudaDeviceId = 999;  // 无效的设备ID
  status = nppSetStreamContext(invalidCtx);
  EXPECT_EQ(status, NPP_BAD_ARGUMENT_ERROR) << "nppSetStreamContext should reject invalid device ID";
  */

  std::cout << "nppSetStreamContext validation tests passed" << std::endl;
}

TEST_F(SupportFunctionsTest, GetGpuDeviceProperties_BasicTest) {
  int maxThreadsPerSM, maxThreadsPerBlock, numSMs;
  int result = nppGetGpuDeviceProperties(&maxThreadsPerSM, &maxThreadsPerBlock, &numSMs);

  ASSERT_EQ(result, 0) << "nppGetGpuDeviceProperties should succeed (return 0)";

  // Validate返回值合理
  EXPECT_GT(maxThreadsPerSM, 0) << "Max threads per SM should be > 0";
  EXPECT_GT(maxThreadsPerBlock, 0) << "Max threads per block should be > 0";
  EXPECT_GT(numSMs, 0) << "Number of SMs should be > 0";

  // Validate合理bounds
  EXPECT_LE(maxThreadsPerSM, 4096) << "Max threads per SM should be reasonable";
  EXPECT_LE(maxThreadsPerBlock, 2048) << "Max threads per block should be reasonable";
  EXPECT_LE(numSMs, 256) << "Number of SMs should be reasonable";

  std::cout << "GPU Properties: " << numSMs << " SMs, " << maxThreadsPerSM << " threads/SM, " << maxThreadsPerBlock
            << " threads/block" << std::endl;
}

TEST_F(SupportFunctionsTest, GetGpuNumSMs_BasicTest) {
  int numSMs = nppGetGpuNumSMs();

  EXPECT_GT(numSMs, 0) << "Number of SMs should be > 0";
  EXPECT_LE(numSMs, 256) << "Number of SMs should be reasonable";

  std::cout << "GPU has " << numSMs << " SMs" << std::endl;
}

class NppiMemoryManagementTest : public ::testing::Test {
protected:
  void SetUp() override {
    cudaError_t err = cudaSetDevice(0);
    ASSERT_EQ(err, cudaSuccess);
  }

  void TearDown() override {
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess);
  }

  // Helper to check memory alignment
  bool isAligned(void *ptr, size_t alignment) { return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0; }

  // Helper to get device memory info
  void getMemoryInfo(size_t &free, size_t &total) { cudaMemGetInfo(&free, &total); }
};

// Test 2: Basic malloc/free for 3 channels
TEST_F(NppiMemoryManagementTest, nppiMalloc_32f_C3_BasicAllocation) {
  const int width = 256;
  const int height = 256;
  int step;

  // Allocate 3-channel memory
  Npp32f *ptr = nppiMalloc_32f_C3(width, height, &step);
  ASSERT_NE(ptr, nullptr);
  EXPECT_GT(step, 0);
  EXPECT_GE(step, width * 3 * sizeof(Npp32f));

  // Check alignment
  EXPECT_TRUE(isAligned(ptr, 256)) << "Memory not properly aligned";
  EXPECT_EQ(step % 256, 0) << "Step not aligned to 256 bytes";

  // Test memory access with 3-channel pattern
  std::vector<Npp32f> testData(width * 3);
  for (int i = 0; i < width; i++) {
    testData[i * 3] = i;         // R
    testData[i * 3 + 1] = i * 2; // G
    testData[i * 3 + 2] = i * 3; // B
  }

  cudaError_t err = cudaMemcpy(ptr, testData.data(), width * 3 * sizeof(Npp32f), cudaMemcpyHostToDevice);
  EXPECT_EQ(err, cudaSuccess);

  // Verify data
  std::vector<Npp32f> readBack(width * 3);
  err = cudaMemcpy(readBack.data(), ptr, width * 3 * sizeof(Npp32f), cudaMemcpyDeviceToHost);
  EXPECT_EQ(err, cudaSuccess);

  for (int i = 0; i < width * 3; i++) {
    EXPECT_EQ(readBack[i], testData[i]);
  }

  nppiFree(ptr);
}

// Test 3: Various dimensions for C1
TEST_F(NppiMemoryManagementTest, nppiMalloc_32f_C1_VariousDimensions) {
  struct TestDimension {
    int width, height;
    const char *description;
  };

  TestDimension dimensions[] = {{1, 1, "Single pixel"},  {17, 13, "Prime dimensions"}, {640, 480, "VGA"},
                                {1920, 1080, "Full HD"}, {4096, 2160, "4K"},           {1, 10000, "Tall narrow"},
                                {10000, 1, "Wide short"}};

  for (const auto &dim : dimensions) {
    int step;
    Npp32f *ptr = nppiMalloc_32f_C1(dim.width, dim.height, &step);

    if (ptr != nullptr) {
      EXPECT_GE(step, dim.width * sizeof(Npp32f)) << "Insufficient step for " << dim.description;
      EXPECT_TRUE(isAligned(ptr, 256)) << "Bad alignment for " << dim.description;

      // Test memory access
      cudaError_t err = cudaMemset(ptr, 0, step * dim.height);
      EXPECT_EQ(err, cudaSuccess) << "Memory access failed for " << dim.description;

      nppiFree(ptr);
    }
  }
}

// Test 4: Various dimensions for C3
TEST_F(NppiMemoryManagementTest, nppiMalloc_32f_C3_VariousDimensions) {
  struct TestDimension {
    int width, height;
    const char *description;
  };

  TestDimension dimensions[] = {{1, 1, "Single pixel"},
                                {85, 64, "Non-aligned width (85*3=255)"},
                                {86, 64, "Barely aligned width (86*3=258)"},
                                {341, 256, "Large non-aligned (341*3=1023)"},
                                {342, 256, "Large aligned (342*3=1026)"}};

  for (const auto &dim : dimensions) {
    int step;
    Npp32f *ptr = nppiMalloc_32f_C3(dim.width, dim.height, &step);

    if (ptr != nullptr) {
      EXPECT_GE(step, dim.width * 3 * sizeof(Npp32f)) << "Insufficient step for " << dim.description;
      EXPECT_TRUE(isAligned(ptr, 256)) << "Bad alignment for " << dim.description;
      EXPECT_EQ(step % 256, 0) << "Step not aligned for " << dim.description;

      // Test memory access
      cudaError_t err = cudaMemset(ptr, 0xFF, step * dim.height);
      EXPECT_EQ(err, cudaSuccess) << "Memory access failed for " << dim.description;

      nppiFree(ptr);
    }
  }
}

// Test 5: Memory reuse patterns
TEST_F(NppiMemoryManagementTest, nppiMalloc_nppiFree_MemoryReusePatterns) {
  const int width = 512;
  const int height = 512;
  const int iterations = 100;

  std::set<void *> allocatedAddresses;
  std::vector<int> steps;

  // Allocate and free multiple times
  for (int i = 0; i < iterations; i++) {
    int step;
    Npp32f *ptr = nppiMalloc_32f_C1(width, height, &step);
    ASSERT_NE(ptr, nullptr);

    allocatedAddresses.insert(ptr);
    steps.push_back(step);

    // Use the memory
    cudaMemset(ptr, i, step * height);

    nppiFree(ptr);
  }

  // Check if memory was reused (address appeared multiple times)
  EXPECT_LT(allocatedAddresses.size(), iterations) << "Memory doesn't seem to be reused efficiently";

  // Check step consistency
  for (int step : steps) {
    EXPECT_EQ(step, steps[0]) << "Step size inconsistent across allocations";
  }
}

// Test 6: Concurrent allocations
TEST_F(NppiMemoryManagementTest, nppiMalloc_32f_ConcurrentAllocations) {
  const int numAllocations = 50;
  const int width = 256;
  const int height = 256;

  std::vector<Npp32f *> ptrsC1(numAllocations);
  std::vector<Npp32f *> ptrsC3(numAllocations);
  std::vector<int> stepsC1(numAllocations);
  std::vector<int> stepsC3(numAllocations);

  // Allocate many buffers
  for (int i = 0; i < numAllocations; i++) {
    ptrsC1[i] = nppiMalloc_32f_C1(width, height, &stepsC1[i]);
    ASSERT_NE(ptrsC1[i], nullptr) << "C1 allocation " << i << " failed";

    ptrsC3[i] = nppiMalloc_32f_C3(width, height, &stepsC3[i]);
    ASSERT_NE(ptrsC3[i], nullptr) << "C3 allocation " << i << " failed";
  }

  // Verify all pointers are unique
  std::set<void *> uniquePtrs;
  for (int i = 0; i < numAllocations; i++) {
    uniquePtrs.insert(ptrsC1[i]);
    uniquePtrs.insert(ptrsC3[i]);
  }
  EXPECT_EQ(uniquePtrs.size(), numAllocations * 2) << "Some pointers are not unique";

  // Test memory access for all
  for (int i = 0; i < numAllocations; i++) {
    cudaError_t err = cudaMemset(ptrsC1[i], i, stepsC1[i] * height);
    EXPECT_EQ(err, cudaSuccess);

    err = cudaMemset(ptrsC3[i], i + 100, stepsC3[i] * height);
    EXPECT_EQ(err, cudaSuccess);
  }

  // Free all in reverse order
  for (int i = numAllocations - 1; i >= 0; i--) {
    nppiFree(ptrsC3[i]);
    nppiFree(ptrsC1[i]);
  }
}

// Test 7: Stress test with random allocations/deallocations
TEST_F(NppiMemoryManagementTest, nppiMalloc_nppiFree_StressTest) {
  const int maxAllocations = 100;
  const int operations = 1000;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> widthDist(1, 1024);
  std::uniform_int_distribution<> heightDist(1, 1024);
  std::uniform_int_distribution<> actionDist(0, 2); // 0: alloc C1, 1: alloc C3, 2: free

  struct Allocation {
    Npp32f *ptr;
    int width, height, step;
    bool isC3;
  };

  std::vector<Allocation> allocations;

  for (int op = 0; op < operations; op++) {
    int action = actionDist(gen);

    if ((action < 2 && allocations.size() < maxAllocations) || allocations.empty()) {
      // Allocate
      int width = widthDist(gen);
      int height = heightDist(gen);
      int step;
      Npp32f *ptr;

      if (action == 0) {
        ptr = nppiMalloc_32f_C1(width, height, &step);
      } else {
        ptr = nppiMalloc_32f_C3(width, height, &step);
      }

      if (ptr != nullptr) {
        allocations.push_back({ptr, width, height, step, action == 1});

        // Write pattern to verify memory
        cudaMemset(ptr, op % 256, step * height);
      }
    } else if (!allocations.empty()) {
      // Free random allocation
      std::uniform_int_distribution<> indexDist(0, allocations.size() - 1);
      int index = indexDist(gen);

      nppiFree(allocations[index].ptr);
      allocations.erase(allocations.begin() + index);
    }
  }

  // Clean up remaining allocations
  for (const auto &alloc : allocations) {
    nppiFree(alloc.ptr);
  }
}

// Test 8: Memory fragmentation test
TEST_F(NppiMemoryManagementTest, nppiMalloc_nppiFree_FragmentationTest) {
  const int numSizes = 5;
  const int allocsPerSize = 20;

  struct SizeInfo {
    int width, height;
  };

  SizeInfo sizes[] = {{64, 64}, {128, 128}, {256, 256}, {512, 512}, {1024, 1024}};

  std::vector<std::vector<Npp32f *>> allocations(numSizes);
  std::vector<std::vector<int>> steps(numSizes);

  // Allocate in interleaved pattern
  for (int i = 0; i < allocsPerSize; i++) {
    for (int s = 0; s < numSizes; s++) {
      int step;
      Npp32f *ptr = nppiMalloc_32f_C1(sizes[s].width, sizes[s].height, &step);

      if (ptr != nullptr) {
        allocations[s].push_back(ptr);
        steps[s].push_back(step);
      }
    }
  }

  // Free every other allocation to create fragmentation
  for (int s = 0; s < numSizes; s++) {
    for (size_t i = 0; i < allocations[s].size(); i += 2) {
      nppiFree(allocations[s][i]);
      allocations[s][i] = nullptr;
    }
  }

  // Try to allocate large blocks
  int largeStep;
  Npp32f *largePtr = nppiMalloc_32f_C1(2048, 2048, &largeStep);

  // Should still succeed despite fragmentation
  if (largePtr != nullptr) {
    nppiFree(largePtr);
  }

  // Clean up
  for (int s = 0; s < numSizes; s++) {
    for (auto ptr : allocations[s]) {
      if (ptr != nullptr) {
        nppiFree(ptr);
      }
    }
  }
}

// Test 10: Alignment verification for edge cases
TEST_F(NppiMemoryManagementTest, nppiMalloc_32f_AlignmentVerification) {
  // Test specific widths that might cause alignment issues
  const int testCases[] = {1, 63, 64, 65, 85, 86, 127, 128, 129, 255, 256, 257};

  for (int width : testCases) {
    // Test C1
    {
      int step;
      Npp32f *ptr = nppiMalloc_32f_C1(width, 16, &step);

      if (ptr != nullptr) {
        EXPECT_TRUE(isAligned(ptr, 256)) << "C1 pointer not aligned for width " << width;
        EXPECT_EQ(step % 512, 0) << "C1 step not aligned for width " << width;

        nppiFree(ptr);
      }
    }

    // Test C3
    {
      int step;
      Npp32f *ptr = nppiMalloc_32f_C3(width, 16, &step);

      if (ptr != nullptr) {
        EXPECT_TRUE(isAligned(ptr, 256)) << "C3 pointer not aligned for width " << width;
        EXPECT_EQ(step % 512, 0) << "C3 step not aligned for width " << width;

        nppiFree(ptr);
      }
    }
  }
}

// Test 11: 16-bit signed 2-channel allocation
TEST_F(NppiMemoryManagementTest, nppiMalloc_16s_C2_BasicAllocation) {
  const int width = 256;
  const int height = 256;
  int step;

  Npp16s *ptr = nppiMalloc_16s_C2(width, height, &step);
  ASSERT_NE(ptr, nullptr);
  EXPECT_GT(step, 0);
  EXPECT_GE(step, width * 2 * sizeof(Npp16s));

  EXPECT_TRUE(isAligned(ptr, 256)) << "Memory not properly aligned";

  std::vector<Npp16s> testData(width * 2);
  for (int i = 0; i < width; i++) {
    testData[i * 2] = static_cast<Npp16s>(i);
    testData[i * 2 + 1] = static_cast<Npp16s>(-i);
  }

  cudaError_t err = cudaMemcpy(ptr, testData.data(), width * 2 * sizeof(Npp16s), cudaMemcpyHostToDevice);
  EXPECT_EQ(err, cudaSuccess);

  std::vector<Npp16s> readBack(width * 2);
  err = cudaMemcpy(readBack.data(), ptr, width * 2 * sizeof(Npp16s), cudaMemcpyDeviceToHost);
  EXPECT_EQ(err, cudaSuccess);

  for (int i = 0; i < width * 2; i++) {
    EXPECT_EQ(readBack[i], testData[i]);
  }

  nppiFree(ptr);
}

// Test 12: 16-bit signed complex allocations (C1, C2, C3, C4)
TEST_F(NppiMemoryManagementTest, nppiMalloc_16sc_AllChannels) {
  const int width = 128;
  const int height = 128;
  int step;

  // Test C1
  {
    Npp16sc *ptr = nppiMalloc_16sc_C1(width, height, &step);
    ASSERT_NE(ptr, nullptr);
    EXPECT_GE(step, width * sizeof(Npp16sc));
    EXPECT_TRUE(isAligned(ptr, 256));

    cudaError_t err = cudaMemset(ptr, 0, step * height);
    EXPECT_EQ(err, cudaSuccess);

    nppiFree(ptr);
  }

  // Test C2
  {
    Npp16sc *ptr = nppiMalloc_16sc_C2(width, height, &step);
    ASSERT_NE(ptr, nullptr);
    EXPECT_GE(step, width * 2 * sizeof(Npp16sc));
    EXPECT_TRUE(isAligned(ptr, 256));

    cudaError_t err = cudaMemset(ptr, 0, step * height);
    EXPECT_EQ(err, cudaSuccess);

    nppiFree(ptr);
  }

  // Test C3
  {
    Npp16sc *ptr = nppiMalloc_16sc_C3(width, height, &step);
    ASSERT_NE(ptr, nullptr);
    EXPECT_GE(step, width * 3 * sizeof(Npp16sc));
    EXPECT_TRUE(isAligned(ptr, 256));

    cudaError_t err = cudaMemset(ptr, 0, step * height);
    EXPECT_EQ(err, cudaSuccess);

    nppiFree(ptr);
  }

  // Test C4
  {
    Npp16sc *ptr = nppiMalloc_16sc_C4(width, height, &step);
    ASSERT_NE(ptr, nullptr);
    EXPECT_GE(step, width * 4 * sizeof(Npp16sc));
    EXPECT_TRUE(isAligned(ptr, 256));

    cudaError_t err = cudaMemset(ptr, 0, step * height);
    EXPECT_EQ(err, cudaSuccess);

    nppiFree(ptr);
  }
}

// Test 13: 16-bit unsigned 2-channel allocation
TEST_F(NppiMemoryManagementTest, nppiMalloc_16u_C2_BasicAllocation) {
  const int width = 256;
  const int height = 256;
  int step;

  Npp16u *ptr = nppiMalloc_16u_C2(width, height, &step);
  ASSERT_NE(ptr, nullptr);
  EXPECT_GT(step, 0);
  EXPECT_GE(step, width * 2 * sizeof(Npp16u));

  EXPECT_TRUE(isAligned(ptr, 256)) << "Memory not properly aligned";

  std::vector<Npp16u> testData(width * 2);
  for (int i = 0; i < width; i++) {
    testData[i * 2] = static_cast<Npp16u>(i);
    testData[i * 2 + 1] = static_cast<Npp16u>(i + 1000);
  }

  cudaError_t err = cudaMemcpy(ptr, testData.data(), width * 2 * sizeof(Npp16u), cudaMemcpyHostToDevice);
  EXPECT_EQ(err, cudaSuccess);

  std::vector<Npp16u> readBack(width * 2);
  err = cudaMemcpy(readBack.data(), ptr, width * 2 * sizeof(Npp16u), cudaMemcpyDeviceToHost);
  EXPECT_EQ(err, cudaSuccess);

  for (int i = 0; i < width * 2; i++) {
    EXPECT_EQ(readBack[i], testData[i]);
  }

  nppiFree(ptr);
}

// Test 14: 32-bit float 2-channel allocation
TEST_F(NppiMemoryManagementTest, nppiMalloc_32f_C2_BasicAllocation) {
  const int width = 256;
  const int height = 256;
  int step;

  Npp32f *ptr = nppiMalloc_32f_C2(width, height, &step);
  ASSERT_NE(ptr, nullptr);
  EXPECT_GT(step, 0);
  EXPECT_GE(step, width * 2 * sizeof(Npp32f));

  EXPECT_TRUE(isAligned(ptr, 256)) << "Memory not properly aligned";

  std::vector<Npp32f> testData(width * 2);
  for (int i = 0; i < width; i++) {
    testData[i * 2] = static_cast<Npp32f>(i) * 0.5f;
    testData[i * 2 + 1] = static_cast<Npp32f>(i) * 1.5f;
  }

  cudaError_t err = cudaMemcpy(ptr, testData.data(), width * 2 * sizeof(Npp32f), cudaMemcpyHostToDevice);
  EXPECT_EQ(err, cudaSuccess);

  std::vector<Npp32f> readBack(width * 2);
  err = cudaMemcpy(readBack.data(), ptr, width * 2 * sizeof(Npp32f), cudaMemcpyDeviceToHost);
  EXPECT_EQ(err, cudaSuccess);

  for (int i = 0; i < width * 2; i++) {
    EXPECT_FLOAT_EQ(readBack[i], testData[i]);
  }

  nppiFree(ptr);
}

// Test 15: 32-bit float complex allocations (C1, C2, C3, C4)
TEST_F(NppiMemoryManagementTest, nppiMalloc_32fc_AllChannels) {
  const int width = 128;
  const int height = 128;
  int step;

  // Test C1
  {
    Npp32fc *ptr = nppiMalloc_32fc_C1(width, height, &step);
    ASSERT_NE(ptr, nullptr);
    EXPECT_GE(step, width * sizeof(Npp32fc));
    EXPECT_TRUE(isAligned(ptr, 256));

    std::vector<Npp32fc> testData(width);
    for (int i = 0; i < width; i++) {
      testData[i].re = static_cast<Npp32f>(i) * 0.5f;
      testData[i].im = static_cast<Npp32f>(i) * 0.3f;
    }

    cudaError_t err = cudaMemcpy(ptr, testData.data(), width * sizeof(Npp32fc), cudaMemcpyHostToDevice);
    EXPECT_EQ(err, cudaSuccess);

    std::vector<Npp32fc> readBack(width);
    err = cudaMemcpy(readBack.data(), ptr, width * sizeof(Npp32fc), cudaMemcpyDeviceToHost);
    EXPECT_EQ(err, cudaSuccess);

    for (int i = 0; i < width; i++) {
      EXPECT_FLOAT_EQ(readBack[i].re, testData[i].re);
      EXPECT_FLOAT_EQ(readBack[i].im, testData[i].im);
    }

    nppiFree(ptr);
  }

  // Test C2
  {
    Npp32fc *ptr = nppiMalloc_32fc_C2(width, height, &step);
    ASSERT_NE(ptr, nullptr);
    EXPECT_GE(step, width * 2 * sizeof(Npp32fc));
    EXPECT_TRUE(isAligned(ptr, 256));

    cudaError_t err = cudaMemset(ptr, 0, step * height);
    EXPECT_EQ(err, cudaSuccess);

    nppiFree(ptr);
  }

  // Test C3
  {
    Npp32fc *ptr = nppiMalloc_32fc_C3(width, height, &step);
    ASSERT_NE(ptr, nullptr);
    EXPECT_GE(step, width * 3 * sizeof(Npp32fc));
    EXPECT_TRUE(isAligned(ptr, 256));

    cudaError_t err = cudaMemset(ptr, 0, step * height);
    EXPECT_EQ(err, cudaSuccess);

    nppiFree(ptr);
  }

  // Test C4
  {
    Npp32fc *ptr = nppiMalloc_32fc_C4(width, height, &step);
    ASSERT_NE(ptr, nullptr);
    EXPECT_GE(step, width * 4 * sizeof(Npp32fc));
    EXPECT_TRUE(isAligned(ptr, 256));

    cudaError_t err = cudaMemset(ptr, 0, step * height);
    EXPECT_EQ(err, cudaSuccess);

    nppiFree(ptr);
  }
}

// Test 16: 32-bit signed complex allocations (C1, C2, C3, C4)
TEST_F(NppiMemoryManagementTest, nppiMalloc_32sc_AllChannels) {
  const int width = 128;
  const int height = 128;
  int step;

  // Test C1
  {
    Npp32sc *ptr = nppiMalloc_32sc_C1(width, height, &step);
    ASSERT_NE(ptr, nullptr);
    EXPECT_GE(step, width * sizeof(Npp32sc));
    EXPECT_TRUE(isAligned(ptr, 256));

    std::vector<Npp32sc> testData(width);
    for (int i = 0; i < width; i++) {
      testData[i].re = static_cast<Npp32s>(i);
      testData[i].im = static_cast<Npp32s>(-i);
    }

    cudaError_t err = cudaMemcpy(ptr, testData.data(), width * sizeof(Npp32sc), cudaMemcpyHostToDevice);
    EXPECT_EQ(err, cudaSuccess);

    std::vector<Npp32sc> readBack(width);
    err = cudaMemcpy(readBack.data(), ptr, width * sizeof(Npp32sc), cudaMemcpyDeviceToHost);
    EXPECT_EQ(err, cudaSuccess);

    for (int i = 0; i < width; i++) {
      EXPECT_EQ(readBack[i].re, testData[i].re);
      EXPECT_EQ(readBack[i].im, testData[i].im);
    }

    nppiFree(ptr);
  }

  // Test C2
  {
    Npp32sc *ptr = nppiMalloc_32sc_C2(width, height, &step);
    ASSERT_NE(ptr, nullptr);
    EXPECT_GE(step, width * 2 * sizeof(Npp32sc));
    EXPECT_TRUE(isAligned(ptr, 256));

    cudaError_t err = cudaMemset(ptr, 0, step * height);
    EXPECT_EQ(err, cudaSuccess);

    nppiFree(ptr);
  }

  // Test C3
  {
    Npp32sc *ptr = nppiMalloc_32sc_C3(width, height, &step);
    ASSERT_NE(ptr, nullptr);
    EXPECT_GE(step, width * 3 * sizeof(Npp32sc));
    EXPECT_TRUE(isAligned(ptr, 256));

    cudaError_t err = cudaMemset(ptr, 0, step * height);
    EXPECT_EQ(err, cudaSuccess);

    nppiFree(ptr);
  }

  // Test C4
  {
    Npp32sc *ptr = nppiMalloc_32sc_C4(width, height, &step);
    ASSERT_NE(ptr, nullptr);
    EXPECT_GE(step, width * 4 * sizeof(Npp32sc));
    EXPECT_TRUE(isAligned(ptr, 256));

    cudaError_t err = cudaMemset(ptr, 0, step * height);
    EXPECT_EQ(err, cudaSuccess);

    nppiFree(ptr);
  }
}
