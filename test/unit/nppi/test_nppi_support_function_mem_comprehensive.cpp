#include "npp.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <mutex>
#include <random>
#include <set>
#include <thread>
#include <vector>

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
  void getMemoryInfo(size_t &free, size_t &total) { gpuMemGetInfo(&free, &total); }
};

// Test 1: Basic malloc/free for single channel
TEST_F(NppiMemoryManagementTest, nppiMalloc_32f_C1_BasicAllocation) {
  const int width = 256;
  const int height = 256;
  int step;

  // Get initial memory state
  size_t freeBefore, total;
  getMemoryInfo(freeBefore, total);

  // Allocate memory
  Npp32f *ptr = nppiMalloc_32f_C1(width, height, &step);
  ASSERT_NE(ptr, nullptr);
  EXPECT_GT(step, 0);
  EXPECT_GE(step, width * sizeof(Npp32f));

  // Check alignment (NPP typically aligns to 256 bytes)
  EXPECT_TRUE(isAligned(ptr, 256)) << "Memory not properly aligned";
  EXPECT_EQ(step % 256, 0) << "Step not aligned to 256 bytes";

  // Get memory after allocation
  size_t freeAfter, totalAfter;
  getMemoryInfo(freeAfter, totalAfter);
  EXPECT_LT(freeAfter, freeBefore) << "Memory was not allocated";

  // Test memory access
  cudaError_t err = cudaMemset(ptr, 0x42, step * height);
  EXPECT_EQ(err, cudaSuccess);

  // Free memory
  nppiFree(ptr);

  // Verify memory is released
  size_t freeAfterFree, totalAfterFree;
  getMemoryInfo(freeAfterFree, totalAfterFree);
  EXPECT_GE(freeAfterFree, freeAfter) << "Memory was not freed";
}

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

// Test 9: Multi-threaded allocation/deallocation
TEST_F(NppiMemoryManagementTest, nppiMalloc_32f_MultiThreaded) {
  const int numThreads = 4;
  const int allocsPerThread = 25;
  const int width = 512;
  const int height = 512;

  std::vector<std::thread> threads;
  std::vector<std::vector<Npp32f *>> threadAllocations(numThreads);
  std::mutex resultMutex;
  std::atomic<int> successCount(0);

  auto threadFunc = [&](int threadId) {
    // Set device in thread
    cudaSetDevice(0);

    std::vector<Npp32f *> localAllocs;

    // Allocate memory
    for (int i = 0; i < allocsPerThread; i++) {
      int step;
      Npp32f *ptr = (i % 2 == 0) ? nppiMalloc_32f_C1(width, height, &step) : nppiMalloc_32f_C3(width, height, &step);

      if (ptr != nullptr) {
        localAllocs.push_back(ptr);

        // Test memory
        cudaError_t err = cudaMemset(ptr, threadId * 10 + i, step * height);
        if (err == cudaSuccess) {
          successCount++;
        }
      }
    }

    // Store allocations
    {
      std::lock_guard<std::mutex> lock(resultMutex);
      threadAllocations[threadId] = localAllocs;
    }
  };

  // Launch threads
  for (int i = 0; i < numThreads; i++) {
    threads.emplace_back(threadFunc, i);
  }

  // Wait for all threads
  for (auto &t : threads) {
    t.join();
  }

  // Verify and clean up
  int totalAllocs = 0;
  for (int i = 0; i < numThreads; i++) {
    totalAllocs += threadAllocations[i].size();
    for (auto ptr : threadAllocations[i]) {
      nppiFree(ptr);
    }
  }

  EXPECT_GT(totalAllocs, 0) << "No successful allocations";
  EXPECT_EQ(successCount.load(), totalAllocs) << "Some memory accesses failed";
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

// Test 11: Free null pointer (should be safe)
TEST_F(NppiMemoryManagementTest, DISABLED_nppiFree_NullPointer) {
  // This should not crash or cause errors
  nppiFree(nullptr);
}

// Test 12: Memory pattern preservation
TEST_F(NppiMemoryManagementTest, nppiMalloc_32f_MemoryPatternPreservation) {
  const int width = 256;
  const int height = 256;

  // Test C1
  {
    int step;
    Npp32f *ptr = nppiMalloc_32f_C1(width, height, &step);
    ASSERT_NE(ptr, nullptr);

    // Create and upload test pattern
    std::vector<Npp32f> pattern(width * height);
    for (int i = 0; i < width * height; i++) {
      pattern[i] = i * 0.1f;
    }

    cudaMemcpy2D(ptr, step, pattern.data(), width * sizeof(Npp32f), width * sizeof(Npp32f), height,
                 cudaMemcpyHostToDevice);

    // Read back and verify
    std::vector<Npp32f> readback(width * height);
    cudaMemcpy2D(readback.data(), width * sizeof(Npp32f), ptr, step, width * sizeof(Npp32f), height,
                 cudaMemcpyDeviceToHost);

    for (int i = 0; i < width * height; i++) {
      EXPECT_FLOAT_EQ(readback[i], pattern[i]);
    }

    nppiFree(ptr);
  }

  // Test C3
  {
    int step;
    Npp32f *ptr = nppiMalloc_32f_C3(width, height, &step);
    ASSERT_NE(ptr, nullptr);

    // Create and upload test pattern
    std::vector<Npp32f> pattern(width * height * 3);
    for (int i = 0; i < width * height; i++) {
      pattern[i * 3] = i * 0.1f;
      pattern[i * 3 + 1] = i * 0.2f;
      pattern[i * 3 + 2] = i * 0.3f;
    }

    cudaMemcpy2D(ptr, step, pattern.data(), width * 3 * sizeof(Npp32f), width * 3 * sizeof(Npp32f), height,
                 cudaMemcpyHostToDevice);

    // Read back and verify
    std::vector<Npp32f> readback(width * height * 3);
    cudaMemcpy2D(readback.data(), width * 3 * sizeof(Npp32f), ptr, step, width * 3 * sizeof(Npp32f), height,
                 cudaMemcpyDeviceToHost);

    for (int i = 0; i < width * height * 3; i++) {
      EXPECT_FLOAT_EQ(readback[i], pattern[i]);
    }

    nppiFree(ptr);
  }
}
