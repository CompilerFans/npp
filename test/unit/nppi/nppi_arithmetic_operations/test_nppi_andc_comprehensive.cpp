#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <vector>

#include "npp.h"

// Comprehensive test class for nppiAndC APIs
class NppiAndCComprehensiveTest : public ::testing::Test {
protected:
  void SetUp() override {
    cudaError_t err = cudaSetDevice(0);
    ASSERT_EQ(err, cudaSuccess);
  }

  void TearDown() override {
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess);
  }

  // Reference implementation for validation
  template <typename T>
  void computeAndCReference(const std::vector<T> &src, const std::vector<T> &constants, std::vector<T> &dst,
                            int channels) {
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); i++) {
      int ch = i % channels;
      dst[i] = src[i] & constants[ch];
    }
  }

  // Data generators
  template <typename T>
  std::vector<T> generateTestData(int width, int height, int channels, T minVal, T maxVal, int seed = 42) {
    std::vector<T> data(width * height * channels);
    std::mt19937 gen(seed);

    if (std::is_integral<T>::value) {
      std::uniform_int_distribution<int> dis(static_cast<int>(minVal), static_cast<int>(maxVal));
      for (auto &val : data) {
        val = static_cast<T>(dis(gen));
      }
    } else {
      std::uniform_real_distribution<double> dis(static_cast<double>(minVal), static_cast<double>(maxVal));
      for (auto &val : data) {
        val = static_cast<T>(dis(gen));
      }
    }
    return data;
  }

  template <typename T> std::vector<T> generateConstantData(int width, int height, int channels, T value) {
    return std::vector<T>(width * height * channels, value);
  }

  template <typename T> std::vector<T> generatePatternData(int width, int height, int channels) {
    std::vector<T> data(width * height * channels);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        for (int c = 0; c < channels; c++) {
          int idx = (y * width + x) * channels + c;
          data[idx] = static_cast<T>((x + y + c) % (1 << (sizeof(T) * 8)));
        }
      }
    }
    return data;
  }
};

// Test nppiAndC_8u_C3R_Ctx
TEST_F(NppiAndCComprehensiveTest, AndC_8u_C3R_Ctx_Comprehensive) {
  const int width = 64;
  const int height = 64;
  const int channels = 3;
  const NppiSize oSizeROI = {width, height};

  struct TestCase {
    std::string name;
    std::vector<Npp8u> srcData;
    std::vector<Npp8u> constants;
  };

  std::vector<TestCase> testCases;

  // Test case 1: Constant data
  testCases.push_back({"Constant_128", generateConstantData<Npp8u>(width, height, channels, 128), {0xFF, 0x0F, 0xF0}});

  // Test case 2: Random data
  testCases.push_back({"Random_Data", generateTestData<Npp8u>(width, height, channels, 0, 255), {0xAA, 0x55, 0xCC}});

  // Test case 3: Pattern data
  testCases.push_back({"Pattern_Data", generatePatternData<Npp8u>(width, height, channels), {0x3F, 0x7F, 0x1F}});

  // Test case 4: Extreme values
  testCases.push_back(
      {"Extreme_Values", generateTestData<Npp8u>(width, height, channels, 0, 255, 123), {0x00, 0xFF, 0x80}});

  for (const auto &testCase : testCases) {
    SCOPED_TRACE("Testing: " + testCase.name);

    // Compute reference result
    std::vector<Npp8u> referenceResult;
    computeAndCReference(testCase.srcData, testCase.constants, referenceResult, channels);

    // Test Context version
    {
      int srcStep = width * channels * sizeof(Npp8u);
      int dstStep = width * channels * sizeof(Npp8u);

      Npp8u *d_src = nullptr;
      Npp8u *d_dst = nullptr;

      ASSERT_EQ(cudaMalloc(&d_src, srcStep * height), cudaSuccess);
      ASSERT_EQ(cudaMalloc(&d_dst, dstStep * height), cudaSuccess);

      ASSERT_EQ(cudaMemcpy(d_src, testCase.srcData.data(), srcStep * height, cudaMemcpyHostToDevice), cudaSuccess);

      cudaStream_t stream;
      ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

      NppStreamContext nppStreamCtx;
      nppGetStreamContext(&nppStreamCtx);
      nppStreamCtx.hStream = stream;

      NppStatus status =
          nppiAndC_8u_C3R_Ctx(d_src, srcStep, testCase.constants.data(), d_dst, dstStep, oSizeROI, nppStreamCtx);
      ASSERT_EQ(status, NPP_SUCCESS);

      ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

      std::vector<Npp8u> result(width * height * channels);
      ASSERT_EQ(cudaMemcpy(result.data(), d_dst, dstStep * height, cudaMemcpyDeviceToHost), cudaSuccess);

      // Validate results
      for (int i = 0; i < width * height * channels; i++) {
        EXPECT_EQ(result[i], referenceResult[i]) << "Mismatch at index " << i << " for test case: " << testCase.name;
      }

      cudaStreamDestroy(stream);
      cudaFree(d_src);
      cudaFree(d_dst);
    }

    // Test non-context version
    {
      int srcStep = width * channels * sizeof(Npp8u);
      int dstStep = width * channels * sizeof(Npp8u);

      Npp8u *d_src = nullptr;
      Npp8u *d_dst = nullptr;

      ASSERT_EQ(cudaMalloc(&d_src, srcStep * height), cudaSuccess);
      ASSERT_EQ(cudaMalloc(&d_dst, dstStep * height), cudaSuccess);

      ASSERT_EQ(cudaMemcpy(d_src, testCase.srcData.data(), srcStep * height, cudaMemcpyHostToDevice), cudaSuccess);

      NppStatus status = nppiAndC_8u_C3R(d_src, srcStep, testCase.constants.data(), d_dst, dstStep, oSizeROI);
      ASSERT_EQ(status, NPP_SUCCESS);

      std::vector<Npp8u> result(width * height * channels);
      ASSERT_EQ(cudaMemcpy(result.data(), d_dst, dstStep * height, cudaMemcpyDeviceToHost), cudaSuccess);

      // Validate results match context version
      for (int i = 0; i < width * height * channels; i++) {
        EXPECT_EQ(result[i], referenceResult[i])
            << "Mismatch at index " << i << " for non-context version: " << testCase.name;
      }

      cudaFree(d_src);
      cudaFree(d_dst);
    }
  }
}

// Test nppiAndC_16u_C3R_Ctx
TEST_F(NppiAndCComprehensiveTest, AndC_16u_C3R_Ctx_Comprehensive) {
  const int width = 32;
  const int height = 32;
  const int channels = 3;
  const NppiSize oSizeROI = {width, height};

  // Test data
  auto srcData = generateTestData<Npp16u>(width, height, channels, 0, 65535);
  std::vector<Npp16u> constants = {0xFF00, 0x00FF, 0xF0F0};

  // Compute reference result
  std::vector<Npp16u> referenceResult;
  computeAndCReference(srcData, constants, referenceResult, channels);

  int srcStep = width * channels * sizeof(Npp16u);
  int dstStep = width * channels * sizeof(Npp16u);

  Npp16u *d_src = nullptr;
  Npp16u *d_dst = nullptr;

  ASSERT_EQ(cudaMalloc(&d_src, srcStep * height), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_dst, dstStep * height), cudaSuccess);

  ASSERT_EQ(cudaMemcpy(d_src, srcData.data(), srcStep * height, cudaMemcpyHostToDevice), cudaSuccess);

  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  nppStreamCtx.hStream = stream;

  NppStatus status = nppiAndC_16u_C3R_Ctx(d_src, srcStep, constants.data(), d_dst, dstStep, oSizeROI, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  std::vector<Npp16u> result(width * height * channels);
  ASSERT_EQ(cudaMemcpy(result.data(), d_dst, dstStep * height, cudaMemcpyDeviceToHost), cudaSuccess);

  // Validate results
  for (int i = 0; i < width * height * channels; i++) {
    EXPECT_EQ(result[i], referenceResult[i]) << "Mismatch at index " << i;
  }

  cudaStreamDestroy(stream);
  cudaFree(d_src);
  cudaFree(d_dst);
}

// Test nppiAndC_16u_C4R_Ctx
TEST_F(NppiAndCComprehensiveTest, AndC_16u_C4R_Ctx_Comprehensive) {
  const int width = 32;
  const int height = 32;
  const int channels = 4;
  const NppiSize oSizeROI = {width, height};

  // Test data with interesting bit patterns
  auto srcData = generatePatternData<Npp16u>(width, height, channels);
  std::vector<Npp16u> constants = {0xAAAA, 0x5555, 0xFF00, 0x00FF};

  // Compute reference result
  std::vector<Npp16u> referenceResult;
  computeAndCReference(srcData, constants, referenceResult, channels);

  int srcStep = width * channels * sizeof(Npp16u);
  int dstStep = width * channels * sizeof(Npp16u);

  Npp16u *d_src = nullptr;
  Npp16u *d_dst = nullptr;

  ASSERT_EQ(cudaMalloc(&d_src, srcStep * height), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_dst, dstStep * height), cudaSuccess);

  ASSERT_EQ(cudaMemcpy(d_src, srcData.data(), srcStep * height, cudaMemcpyHostToDevice), cudaSuccess);

  NppStreamContext nppStreamCtx;
  memset(&nppStreamCtx, 0, sizeof(NppStreamContext));
  NppStatus status = nppiAndC_16u_C4R_Ctx(d_src, srcStep, constants.data(), d_dst, dstStep, oSizeROI, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp16u> result(width * height * channels);
  ASSERT_EQ(cudaMemcpy(result.data(), d_dst, dstStep * height, cudaMemcpyDeviceToHost), cudaSuccess);

  // Validate results
  for (int i = 0; i < width * height * channels; i++) {
    EXPECT_EQ(result[i], referenceResult[i]) << "Mismatch at index " << i;
  }

  cudaFree(d_src);
  cudaFree(d_dst);
}

// Test nppiAndC_32s_C3R_Ctx
TEST_F(NppiAndCComprehensiveTest, AndC_32s_C3R_Ctx_Comprehensive) {
  const int width = 24;
  const int height = 24;
  const int channels = 3;
  const NppiSize oSizeROI = {width, height};

  // Test with signed integer data
  auto srcData = generateTestData<Npp32s>(width, height, channels, -32768, 32767);
  std::vector<Npp32s> constants = {static_cast<Npp32s>(0xFFFF0000), 0x0000FFFF, static_cast<Npp32s>(0xFF00FF00)};

  // Compute reference result
  std::vector<Npp32s> referenceResult;
  computeAndCReference(srcData, constants, referenceResult, channels);

  int srcStep = width * channels * sizeof(Npp32s);
  int dstStep = width * channels * sizeof(Npp32s);

  Npp32s *d_src = nullptr;
  Npp32s *d_dst = nullptr;

  ASSERT_EQ(cudaMalloc(&d_src, srcStep * height), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_dst, dstStep * height), cudaSuccess);

  ASSERT_EQ(cudaMemcpy(d_src, srcData.data(), srcStep * height, cudaMemcpyHostToDevice), cudaSuccess);

  NppStreamContext nppStreamCtx;
  memset(&nppStreamCtx, 0, sizeof(NppStreamContext));
  NppStatus status = nppiAndC_32s_C3R_Ctx(d_src, srcStep, constants.data(), d_dst, dstStep, oSizeROI, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32s> result(width * height * channels);
  ASSERT_EQ(cudaMemcpy(result.data(), d_dst, dstStep * height, cudaMemcpyDeviceToHost), cudaSuccess);

  // Validate results
  for (int i = 0; i < width * height * channels; i++) {
    EXPECT_EQ(result[i], referenceResult[i]) << "Mismatch at index " << i;
  }

  cudaFree(d_src);
  cudaFree(d_dst);
}

// Test nppiAndC_32s_C4R_Ctx
TEST_F(NppiAndCComprehensiveTest, AndC_32s_C4R_Ctx_Comprehensive) {
  const int width = 16;
  const int height = 16;
  const int channels = 4;
  const NppiSize oSizeROI = {width, height};

  // Test with extreme values
  auto srcData = generateTestData<Npp32s>(width, height, channels, -2147483648, 2147483647, 999);
  std::vector<Npp32s> constants = {0x12345678, static_cast<Npp32s>(0x87654321), static_cast<Npp32s>(0xAAAAAAAA),
                                   0x55555555};

  // Compute reference result
  std::vector<Npp32s> referenceResult;
  computeAndCReference(srcData, constants, referenceResult, channels);

  int srcStep = width * channels * sizeof(Npp32s);
  int dstStep = width * channels * sizeof(Npp32s);

  Npp32s *d_src = nullptr;
  Npp32s *d_dst = nullptr;

  ASSERT_EQ(cudaMalloc(&d_src, srcStep * height), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_dst, dstStep * height), cudaSuccess);

  ASSERT_EQ(cudaMemcpy(d_src, srcData.data(), srcStep * height, cudaMemcpyHostToDevice), cudaSuccess);

  NppStreamContext nppStreamCtx;
  memset(&nppStreamCtx, 0, sizeof(NppStreamContext));
  NppStatus status = nppiAndC_32s_C4R_Ctx(d_src, srcStep, constants.data(), d_dst, dstStep, oSizeROI, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32s> result(width * height * channels);
  ASSERT_EQ(cudaMemcpy(result.data(), d_dst, dstStep * height, cudaMemcpyDeviceToHost), cudaSuccess);

  // Validate results
  for (int i = 0; i < width * height * channels; i++) {
    EXPECT_EQ(result[i], referenceResult[i]) << "Mismatch at index " << i;
  }

  cudaFree(d_src);
  cudaFree(d_dst);
}

// Performance comparison test
TEST_F(NppiAndCComprehensiveTest, PerformanceComparison) {
  const int width = 1024;
  const int height = 1024;
  const int channels = 3;
  const NppiSize oSizeROI = {width, height};
  const int iterations = 10;

  auto srcData = generateTestData<Npp8u>(width, height, channels, 0, 255);
  std::vector<Npp8u> constants = {0xAA, 0x55, 0xCC};

  int srcStep = width * channels * sizeof(Npp8u);
  int dstStep = width * channels * sizeof(Npp8u);

  Npp8u *d_src = nullptr;
  Npp8u *d_dst = nullptr;

  ASSERT_EQ(cudaMalloc(&d_src, srcStep * height), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_dst, dstStep * height), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_src, srcData.data(), srcStep * height, cudaMemcpyHostToDevice), cudaSuccess);

  // Time context version
  NppStreamContext nppStreamCtx;
  memset(&nppStreamCtx, 0, sizeof(NppStreamContext));
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    NppStatus status = nppiAndC_8u_C3R_Ctx(d_src, srcStep, constants.data(), d_dst, dstStep, oSizeROI, nppStreamCtx);
    ASSERT_EQ(status, NPP_SUCCESS);
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  auto contextTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  // Time non-context version
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    NppStatus status = nppiAndC_8u_C3R(d_src, srcStep, constants.data(), d_dst, dstStep, oSizeROI);
    ASSERT_EQ(status, NPP_SUCCESS);
  }
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();
  auto regularTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  std::cout << "\\nPerformance Results (" << iterations << " iterations, " << width << "x" << height << "x" << channels
            << "):\\n";
  std::cout << "  Context version: " << contextTime << " μs (avg: " << contextTime / iterations << " μs)\\n";
  std::cout << "  Regular version: " << regularTime << " μs (avg: " << regularTime / iterations << " μs)\\n";

  cudaFree(d_src);
  cudaFree(d_dst);
}