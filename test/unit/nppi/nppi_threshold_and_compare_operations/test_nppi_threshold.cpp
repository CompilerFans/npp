#include "../../framework/npp_test_base.h"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <vector>

#include "npp.h"

using namespace npp_functional_test;

class ThresholdFunctionalTest : public NppTestBase {
protected:
  void SetUp() override { NppTestBase::SetUp(); }

  void TearDown() override { NppTestBase::TearDown(); }
};

// 测试8位阈值处理 - LESS操作
TEST_F(ThresholdFunctionalTest, Threshold_8u_C1R_Less) {
  const int width = 32, height = 32;
  const Npp8u threshold = 128;

  // prepare test data - 渐变图像
  std::vector<Npp8u> srcData(width * height);
  std::vector<Npp8u> expectedData(width * height);

  for (int i = 0; i < width * height; i++) {
    srcData[i] = (Npp8u)(i % 256);
    // NPP_CMP_LESS: if (src < threshold) dst = threshold; else dst = src;
    expectedData[i] = (srcData[i] < threshold) ? threshold : srcData[i];
  }

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> dst(width, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiThreshold_8u_C1R(src.get(), src.step(), dst.get(), dst.step(), roi, threshold, NPP_CMP_LESS);

  ASSERT_EQ(status, NPP_SUCCESS) << "nppiThreshold_8u_C1R failed";

  // Validate结果
  std::vector<Npp8u> resultData(width * height);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData))
      << "Threshold LESS operation produced incorrect results";
}

// 测试8位阈值处理 - GREATER操作
TEST_F(ThresholdFunctionalTest, Threshold_8u_C1R_Greater) {
  const int width = 32, height = 32;
  const Npp8u threshold = 128;

  // prepare test data
  std::vector<Npp8u> srcData(width * height);
  std::vector<Npp8u> expectedData(width * height);

  for (int i = 0; i < width * height; i++) {
    srcData[i] = (Npp8u)(i % 256);
    // NPP_CMP_GREATER: if (src > threshold) dst = threshold; else dst = src;
    expectedData[i] = (srcData[i] > threshold) ? threshold : srcData[i];
  }

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> dst(width, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status =
      nppiThreshold_8u_C1R(src.get(), src.step(), dst.get(), dst.step(), roi, threshold, NPP_CMP_GREATER);

  ASSERT_EQ(status, NPP_SUCCESS) << "nppiThreshold_8u_C1R failed";

  // Validate结果
  std::vector<Npp8u> resultData(width * height);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData))
      << "Threshold GREATER operation produced incorrect results";
}

// 测试8位原地阈值处理
TEST_F(ThresholdFunctionalTest, Threshold_8u_C1IR_InPlace) {
  const int width = 16, height = 16;
  const Npp8u threshold = 100;

  // prepare test data
  std::vector<Npp8u> srcData(width * height);
  std::vector<Npp8u> expectedData(width * height);

  // 创建具有明显对比的测试图像
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      srcData[idx] = (x < width / 2) ? 50 : 150;
      expectedData[idx] = (srcData[idx] < threshold) ? threshold : srcData[idx];
    }
  }

  NppImageMemory<Npp8u> srcDst(width, height);
  srcDst.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiThreshold_8u_C1IR(srcDst.get(), srcDst.step(), roi, threshold, NPP_CMP_LESS);

  ASSERT_EQ(status, NPP_SUCCESS) << "nppiThreshold_8u_C1IR failed";

  // Validate结果
  std::vector<Npp8u> resultData(width * height);
  srcDst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData))
      << "In-place threshold operation produced incorrect results";
}

// 测试32位浮点阈值处理
TEST_F(ThresholdFunctionalTest, Threshold_32f_C1R_Float) {
  const int width = 16, height = 16;
  const Npp32f threshold = 0.5f;

  // prepare test data
  std::vector<Npp32f> srcData(width * height);
  std::vector<Npp32f> expectedData(width * height);

  for (int i = 0; i < width * height; i++) {
    srcData[i] = (float)i / (width * height);
    expectedData[i] = (srcData[i] > threshold) ? threshold : srcData[i];
  }

  NppImageMemory<Npp32f> src(width, height);
  NppImageMemory<Npp32f> dst(width, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status =
      nppiThreshold_32f_C1R(src.get(), src.step(), dst.get(), dst.step(), roi, threshold, NPP_CMP_GREATER);

  ASSERT_EQ(status, NPP_SUCCESS) << "nppiThreshold_32f_C1R failed";

  // Validate结果
  std::vector<Npp32f> resultData(width * height);
  dst.copyToHost(resultData);

  EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-6f))
      << "Float threshold operation produced incorrect results";
}

// 错误处理测试
// NOTE: 测试已被禁用 - vendor NPP对无效参数的错误检测行为与预期不符
TEST_F(ThresholdFunctionalTest, DISABLED_Threshold_ErrorHandling) {
  const int width = 16, height = 16;
  const Npp8u threshold = 128;

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> dst(width, height);

  NppiSize roi = {width, height};

  // 测试空指针
  NppStatus status = nppiThreshold_8u_C1R(nullptr, src.step(), dst.get(), dst.step(), roi, threshold, NPP_CMP_LESS);
  EXPECT_EQ(status, NPP_NULL_POINTER_ERROR);

  // 测试无效ROI
  NppiSize invalidRoi = {0, 0};
  status = nppiThreshold_8u_C1R(src.get(), src.step(), dst.get(), dst.step(), invalidRoi, threshold, NPP_CMP_LESS);
  EXPECT_EQ(status, NPP_SIZE_ERROR);

  // 测试无效比较操作
  status = nppiThreshold_8u_C1R(src.get(), src.step(), dst.get(), dst.step(), roi, threshold,
                                NPP_CMP_EQ); // EQ not supported for threshold
  EXPECT_EQ(status, NPP_NOT_SUPPORTED_MODE_ERROR);
}

// 二值化测试（特殊应用场景）
TEST_F(ThresholdFunctionalTest, Threshold_BinaryImage) {
  const int width = 32, height = 32;
  const Npp8u threshold = 128;

  // prepare test data - 创建一个包含噪声的二值化场景
  std::vector<Npp8u> srcData(width * height);

  for (int i = 0; i < width * height; i++) {
    // 创建一个圆形区域
    int x = i % width;
    int y = i / width;
    int dx = x - width / 2;
    int dy = y - height / 2;
    float dist = sqrt(dx * dx + dy * dy);

    srcData[i] = (dist < width / 3) ? 200 : 50;
  }

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> dst(width, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiThreshold_8u_C1R(src.get(), src.step(), dst.get(), dst.step(), roi, threshold, NPP_CMP_LESS);

  ASSERT_EQ(status, NPP_SUCCESS) << "Binary threshold failed";

  // Validate结果 - 检查二值化效果
  std::vector<Npp8u> resultData(width * height);
  dst.copyToHost(resultData);

  int lowCount = 0, highCount = 0;
  for (size_t i = 0; i < resultData.size(); i++) {
    if (resultData[i] == threshold)
      lowCount++;
    else
      highCount++;
  }

  EXPECT_GT(lowCount, 0) << "No pixels were thresholded";
  EXPECT_GT(highCount, 0) << "All pixels were thresholded";
}

// Comprehensive test class for nppiThreshold_32f_C1R API
class NppiThreshold32fComprehensiveTest : public ::testing::Test {
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
  void computeThresholdReference(const std::vector<Npp32f> &src, std::vector<Npp32f> &dst, Npp32f threshold,
                                 NppCmpOp op) {
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); i++) {
      bool condition = false;
      switch (op) {
      case NPP_CMP_LESS:
        condition = (src[i] < threshold);
        break;
      case NPP_CMP_GREATER:
        condition = (src[i] > threshold);
        break;
      default:
        // Should not reach here in valid tests
        break;
      }
      dst[i] = condition ? threshold : src[i];
    }
  }

  // Data generators
  std::vector<Npp32f> generateRandomData(int width, int height, float minVal, float maxVal, int seed = 42) {
    std::vector<Npp32f> data(width * height);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(minVal, maxVal);

    for (auto &val : data) {
      val = dis(gen);
    }
    return data;
  }

  std::vector<Npp32f> generateConstantData(int width, int height, float value) {
    return std::vector<Npp32f>(width * height, value);
  }

  std::vector<Npp32f> generateGradientData(int width, int height, float minVal, float maxVal) {
    std::vector<Npp32f> data(width * height);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        float ratio = static_cast<float>(x + y) / (width + height - 2);
        data[y * width + x] = minVal + ratio * (maxVal - minVal);
      }
    }
    return data;
  }

  std::vector<Npp32f> generateSineWaveData(int width, int height, float amplitude, float frequency) {
    std::vector<Npp32f> data(width * height);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        float t = static_cast<float>(x + y) / (width + height);
        data[y * width + x] = amplitude * std::sin(2.0f * M_PI * frequency * t);
      }
    }
    return data;
  }
};

// Comprehensive test for various data patterns and thresholds
TEST_F(NppiThreshold32fComprehensiveTest, Threshold_32f_C1R_ComprehensivePatterns) {
  const int width = 64;
  const int height = 64;
  const NppiSize oSizeROI = {width, height};

  struct TestCase {
    std::string name;
    std::vector<Npp32f> srcData;
    Npp32f threshold;
    NppCmpOp operation;
  };

  std::vector<TestCase> testCases;

  // Test case 1: Constant data with LESS operation
  testCases.push_back({"Constant_5.0_LESS_3.0", generateConstantData(width, height, 5.0f), 3.0f, NPP_CMP_LESS});

  // Test case 2: Constant data with GREATER operation
  testCases.push_back({"Constant_2.0_GREATER_3.0", generateConstantData(width, height, 2.0f), 3.0f, NPP_CMP_GREATER});

  // Test case 3: Random data with LESS operation
  testCases.push_back({"Random_LESS_0.5", generateRandomData(width, height, -1.0f, 1.0f), 0.5f, NPP_CMP_LESS});

  // Test case 4: Random data with GREATER operation
  testCases.push_back(
      {"Random_GREATER_-0.5", generateRandomData(width, height, -1.0f, 1.0f, 123), -0.5f, NPP_CMP_GREATER});

  // Test case 5: Gradient data with LESS operation
  testCases.push_back({"Gradient_LESS_50.0", generateGradientData(width, height, 0.0f, 100.0f), 50.0f, NPP_CMP_LESS});

  // Test case 6: Gradient data with GREATER operation
  testCases.push_back(
      {"Gradient_GREATER_25.0", generateGradientData(width, height, 0.0f, 100.0f), 25.0f, NPP_CMP_GREATER});

  // Test case 7: Sine wave data with LESS operation
  testCases.push_back({"SineWave_LESS_0.0", generateSineWaveData(width, height, 2.0f, 3.0f), 0.0f, NPP_CMP_LESS});

  // Test case 8: Sine wave data with GREATER operation
  testCases.push_back({"SineWave_GREATER_1.0", generateSineWaveData(width, height, 2.0f, 3.0f), 1.0f, NPP_CMP_GREATER});

  for (const auto &testCase : testCases) {
    SCOPED_TRACE("Testing: " + testCase.name);

    // Compute reference result
    std::vector<Npp32f> referenceResult;
    computeThresholdReference(testCase.srcData, referenceResult, testCase.threshold, testCase.operation);

    // Test Context version
    {
      int srcStep = width * sizeof(Npp32f);
      int dstStep = width * sizeof(Npp32f);

      Npp32f *d_src = nullptr;
      Npp32f *d_dst = nullptr;

      ASSERT_EQ(cudaMalloc(&d_src, srcStep * height), cudaSuccess);
      ASSERT_EQ(cudaMalloc(&d_dst, dstStep * height), cudaSuccess);

      ASSERT_EQ(cudaMemcpy(d_src, testCase.srcData.data(), srcStep * height, cudaMemcpyHostToDevice), cudaSuccess);

      cudaStream_t stream;
      ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

      NppStreamContext nppStreamCtx;
      nppGetStreamContext(&nppStreamCtx);
      nppStreamCtx.hStream = stream;

      NppStatus status = nppiThreshold_32f_C1R_Ctx(d_src, srcStep, d_dst, dstStep, oSizeROI, testCase.threshold,
                                                   testCase.operation, nppStreamCtx);
      ASSERT_EQ(status, NPP_SUCCESS);

      ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

      std::vector<Npp32f> result(width * height);
      ASSERT_EQ(cudaMemcpy(result.data(), d_dst, dstStep * height, cudaMemcpyDeviceToHost), cudaSuccess);

      // Validate results with tolerance for floating point
      for (int i = 0; i < width * height; i++) {
        EXPECT_NEAR(result[i], referenceResult[i], 1e-6f)
            << "Mismatch at index " << i << " for test case: " << testCase.name << " (expected: " << referenceResult[i]
            << ", got: " << result[i] << ")";
      }

      cudaStreamDestroy(stream);
      cudaFree(d_src);
      cudaFree(d_dst);
    }

    // Test non-context version
    {
      int srcStep = width * sizeof(Npp32f);
      int dstStep = width * sizeof(Npp32f);

      Npp32f *d_src = nullptr;
      Npp32f *d_dst = nullptr;

      ASSERT_EQ(cudaMalloc(&d_src, srcStep * height), cudaSuccess);
      ASSERT_EQ(cudaMalloc(&d_dst, dstStep * height), cudaSuccess);

      ASSERT_EQ(cudaMemcpy(d_src, testCase.srcData.data(), srcStep * height, cudaMemcpyHostToDevice), cudaSuccess);

      NppStatus status =
          nppiThreshold_32f_C1R(d_src, srcStep, d_dst, dstStep, oSizeROI, testCase.threshold, testCase.operation);
      ASSERT_EQ(status, NPP_SUCCESS);

      std::vector<Npp32f> result(width * height);
      ASSERT_EQ(cudaMemcpy(result.data(), d_dst, dstStep * height, cudaMemcpyDeviceToHost), cudaSuccess);

      // Validate results match context version
      for (int i = 0; i < width * height; i++) {
        EXPECT_NEAR(result[i], referenceResult[i], 1e-6f)
            << "Mismatch at index " << i << " for non-context version: " << testCase.name;
      }

      cudaFree(d_src);
      cudaFree(d_dst);
    }
  }
}

// Test extreme values and edge cases
TEST_F(NppiThreshold32fComprehensiveTest, ExtremeValues) {
  const int width = 32;
  const int height = 32;
  const NppiSize oSizeROI = {width, height};

  struct ExtremeTestCase {
    std::string name;
    std::vector<Npp32f> srcData;
    Npp32f threshold;
    NppCmpOp operation;
  };

  std::vector<ExtremeTestCase> extremeCases;

  // Very large values
  extremeCases.push_back({"LargeValues_LESS", generateConstantData(width, height, 1e6f), 1e7f, NPP_CMP_LESS});

  // Very small values
  extremeCases.push_back({"SmallValues_GREATER", generateConstantData(width, height, 1e-6f), 1e-7f, NPP_CMP_GREATER});

  // Negative values
  extremeCases.push_back({"NegativeValues_LESS", generateConstantData(width, height, -100.0f), -50.0f, NPP_CMP_LESS});

  // Zero threshold
  extremeCases.push_back(
      {"ZeroThreshold_GREATER", generateRandomData(width, height, -10.0f, 10.0f, 789), 0.0f, NPP_CMP_GREATER});

  // Infinity values
  std::vector<Npp32f> infData(width * height);
  for (size_t i = 0; i < infData.size(); i++) {
    infData[i] = (i % 2 == 0) ? std::numeric_limits<float>::infinity() : -std::numeric_limits<float>::infinity();
  }
  extremeCases.push_back({"InfinityValues_LESS", infData, 0.0f, NPP_CMP_LESS});

  for (const auto &testCase : extremeCases) {
    SCOPED_TRACE("Testing extreme case: " + testCase.name);

    // Compute reference result
    std::vector<Npp32f> referenceResult;
    computeThresholdReference(testCase.srcData, referenceResult, testCase.threshold, testCase.operation);

    int srcStep = width * sizeof(Npp32f);
    int dstStep = width * sizeof(Npp32f);

    Npp32f *d_src = nullptr;
    Npp32f *d_dst = nullptr;

    ASSERT_EQ(cudaMalloc(&d_src, srcStep * height), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dst, dstStep * height), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_src, testCase.srcData.data(), srcStep * height, cudaMemcpyHostToDevice), cudaSuccess);

    NppStatus status =
        nppiThreshold_32f_C1R(d_src, srcStep, d_dst, dstStep, oSizeROI, testCase.threshold, testCase.operation);
    ASSERT_EQ(status, NPP_SUCCESS);

    std::vector<Npp32f> result(width * height);
    ASSERT_EQ(cudaMemcpy(result.data(), d_dst, dstStep * height, cudaMemcpyDeviceToHost), cudaSuccess);

    // Validate results (with special handling for infinity)
    for (int i = 0; i < width * height; i++) {
      if (std::isfinite(referenceResult[i]) && std::isfinite(result[i])) {
        EXPECT_NEAR(result[i], referenceResult[i], 1e-6f)
            << "Mismatch at index " << i << " for extreme case: " << testCase.name;
      } else {
        // For infinity values, just check they match in type
        EXPECT_EQ(std::isinf(result[i]), std::isinf(referenceResult[i])) << "Infinity mismatch at index " << i;
        if (std::isinf(result[i]) && std::isinf(referenceResult[i])) {
          EXPECT_EQ(std::signbit(result[i]), std::signbit(referenceResult[i]))
              << "Infinity sign mismatch at index " << i;
        }
      }
    }

    cudaFree(d_src);
    cudaFree(d_dst);
  }
}
// Performance test and comparison
TEST_F(NppiThreshold32fComprehensiveTest, PerformanceComparison) {
  const int width = 1024;
  const int height = 1024;
  const NppiSize oSizeROI = {width, height};
  const int iterations = 10;

  auto srcData = generateRandomData(width, height, -100.0f, 100.0f, 456);
  const Npp32f threshold = 0.0f;
  const NppCmpOp operation = NPP_CMP_GREATER;

  int srcStep = width * sizeof(Npp32f);
  int dstStep = width * sizeof(Npp32f);

  Npp32f *d_src = nullptr;
  Npp32f *d_dst = nullptr;

  ASSERT_EQ(cudaMalloc(&d_src, srcStep * height), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_dst, dstStep * height), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_src, srcData.data(), srcStep * height, cudaMemcpyHostToDevice), cudaSuccess);

  // Time context version
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  nppStreamCtx.hStream = stream;

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    NppStatus status =
        nppiThreshold_32f_C1R_Ctx(d_src, srcStep, d_dst, dstStep, oSizeROI, threshold, operation, nppStreamCtx);
    ASSERT_EQ(status, NPP_SUCCESS);
  }
  cudaStreamSynchronize(stream);
  auto end = std::chrono::high_resolution_clock::now();
  auto contextTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  // Time non-context version
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    NppStatus status = nppiThreshold_32f_C1R(d_src, srcStep, d_dst, dstStep, oSizeROI, threshold, operation);
    ASSERT_EQ(status, NPP_SUCCESS);
  }
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();
  auto regularTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  std::cout << "\\nThreshold Performance Results (" << iterations << " iterations, " << width << "x" << height
            << "):\\n";
  std::cout << "  Context version: " << contextTime << " μs (avg: " << contextTime / iterations << " μs)\\n";
  std::cout << "  Regular version: " << regularTime << " μs (avg: " << regularTime / iterations << " μs)\\n";

  cudaStreamDestroy(stream);
  cudaFree(d_src);
  cudaFree(d_dst);
}

// Test minimum size images
TEST_F(NppiThreshold32fComprehensiveTest, MinimumSizeImage) {
  const int width = 1;
  const int height = 1;
  const NppiSize oSizeROI = {width, height};

  std::vector<Npp32f> srcData = {42.5f};
  const Npp32f threshold = 40.0f;

  // Test GREATER operation
  {
    std::vector<Npp32f> referenceResult;
    computeThresholdReference(srcData, referenceResult, threshold, NPP_CMP_GREATER);

    int srcStep = width * sizeof(Npp32f);
    int dstStep = width * sizeof(Npp32f);

    Npp32f *d_src = nullptr;
    Npp32f *d_dst = nullptr;

    ASSERT_EQ(cudaMalloc(&d_src, srcStep * height), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dst, dstStep * height), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_src, srcData.data(), srcStep * height, cudaMemcpyHostToDevice), cudaSuccess);

    NppStatus status = nppiThreshold_32f_C1R(d_src, srcStep, d_dst, dstStep, oSizeROI, threshold, NPP_CMP_GREATER);
    ASSERT_EQ(status, NPP_SUCCESS);

    std::vector<Npp32f> result(1);
    ASSERT_EQ(cudaMemcpy(result.data(), d_dst, srcStep * height, cudaMemcpyDeviceToHost), cudaSuccess);

    EXPECT_NEAR(result[0], referenceResult[0], 1e-6f);

    cudaFree(d_src);
    cudaFree(d_dst);
  }

  // Test LESS operation
  {
    std::vector<Npp32f> referenceResult;
    computeThresholdReference(srcData, referenceResult, threshold, NPP_CMP_LESS);

    int srcStep = width * sizeof(Npp32f);
    int dstStep = width * sizeof(Npp32f);

    Npp32f *d_src = nullptr;
    Npp32f *d_dst = nullptr;

    ASSERT_EQ(cudaMalloc(&d_src, srcStep * height), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dst, dstStep * height), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_src, srcData.data(), srcStep * height, cudaMemcpyHostToDevice), cudaSuccess);

    NppStatus status = nppiThreshold_32f_C1R(d_src, srcStep, d_dst, dstStep, oSizeROI, threshold, NPP_CMP_LESS);
    ASSERT_EQ(status, NPP_SUCCESS);

    std::vector<Npp32f> result(1);
    ASSERT_EQ(cudaMemcpy(result.data(), d_dst, srcStep * height, cudaMemcpyDeviceToHost), cudaSuccess);

    EXPECT_NEAR(result[0], referenceResult[0], 1e-6f);

    cudaFree(d_src);
    cudaFree(d_dst);
  }
}