#include "npp.h"
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

class RGBToGrayFunctionalTest : public ::testing::Test {
protected:
  void SetUp() override {
    width = 16;
    height = 12;
    roi.width = width;
    roi.height = height;
  }

  int width, height;
  NppiSize roi;

  // Function to calculate expected grayscale value
  float calculateGrayValue(float r, float g, float b) { return 0.299f * r + 0.587f * g + 0.114f * b; }

  Npp8u calculateGrayValue8u(Npp8u r, Npp8u g, Npp8u b) {
    float gray = calculateGrayValue((float)r, (float)g, (float)b);
    return (Npp8u)(gray + 0.5f); // Round to nearest
  }
};

// Test 3-channel RGB to grayscale conversion (8-bit)
TEST_F(RGBToGrayFunctionalTest, RGBToGray_8u_C3C1R_BasicOperation) {
  std::vector<Npp8u> srcData(width * height * 3);
  std::vector<Npp8u> expectedData(width * height);

  // Create test RGB image with known values
  for (int i = 0; i < width * height; i++) {
    Npp8u r = (Npp8u)(i % 256);
    Npp8u g = (Npp8u)((i * 2) % 256);
    Npp8u b = (Npp8u)((i * 3) % 256);

    srcData[i * 3 + 0] = r;
    srcData[i * 3 + 1] = g;
    srcData[i * 3 + 2] = b;

    expectedData[i] = calculateGrayValue8u(r, g, b);
  }

  // Allocate GPU memory
  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C3(width, height, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C1(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy input data to GPU row by row to handle step properly
  for (int y = 0; y < height; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * width * 3, width * 3 * sizeof(Npp8u),
               cudaMemcpyHostToDevice);
  }

  // Initialize destination memory to a pattern to detect if kernel runs
  cudaMemset(d_dst, 255, height * dstStep);

  // Execute NPP function
  NppStatus status = nppiRGBToGray_8u_C3C1R(d_src, srcStep, d_dst, dstStep, roi);
  EXPECT_EQ(status, NPP_SUCCESS);

  // Copy result back to host row by row to handle step properly
  std::vector<Npp8u> resultData(width * height);
  for (int y = 0; y < height; y++) {
    cudaMemcpy(resultData.data() + y * width, (char *)d_dst + y * dstStep, width * sizeof(Npp8u),
               cudaMemcpyDeviceToHost);
  }

  // Verify results (allow small rounding differences)
  for (int i = 0; i < width * height; i++) {
    EXPECT_NEAR(resultData[i], expectedData[i], 1)
        << "Mismatch at pixel " << i << ": got " << (int)resultData[i] << ", expected " << (int)expectedData[i];
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Test 4-channel RGBA to grayscale conversion (8-bit)
TEST_F(RGBToGrayFunctionalTest, RGBToGray_8u_AC4C1R_BasicOperation) {
  std::vector<Npp8u> srcData(width * height * 4);
  std::vector<Npp8u> expectedData(width * height);

  // Create test RGBA image with known values
  for (int i = 0; i < width * height; i++) {
    Npp8u r = 100;
    Npp8u g = 150;
    Npp8u b = 200;
    Npp8u a = 255; // Alpha should be ignored

    srcData[i * 4 + 0] = r;
    srcData[i * 4 + 1] = g;
    srcData[i * 4 + 2] = b;
    srcData[i * 4 + 3] = a;

    expectedData[i] = calculateGrayValue8u(r, g, b);
  }

  // Allocate GPU memory
  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C4(width, height, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C1(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy input data to GPU row by row to handle step properly
  for (int y = 0; y < height; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * width * 4, width * 4 * sizeof(Npp8u),
               cudaMemcpyHostToDevice);
  }

  // Execute NPP function
  NppStatus status = nppiRGBToGray_8u_AC4C1R(d_src, srcStep, d_dst, dstStep, roi);
  EXPECT_EQ(status, NPP_SUCCESS);

  // Copy result back to host row by row to handle step properly
  std::vector<Npp8u> resultData(width * height);
  for (int y = 0; y < height; y++) {
    cudaMemcpy(resultData.data() + y * width, (char *)d_dst + y * dstStep, width * sizeof(Npp8u),
               cudaMemcpyDeviceToHost);
  }

  // Verify results
  for (int i = 0; i < width * height; i++) {
    EXPECT_NEAR(resultData[i], expectedData[i], 1) << "Mismatch at pixel " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Test 3-channel RGB to grayscale conversion (32-bit float)
TEST_F(RGBToGrayFunctionalTest, RGBToGray_32f_C3C1R_BasicOperation) {
  std::vector<Npp32f> srcData(width * height * 3);
  std::vector<Npp32f> expectedData(width * height);

  // Create test RGB image with float values
  for (int i = 0; i < width * height; i++) {
    Npp32f r = 0.8f;
    Npp32f g = 0.6f;
    Npp32f b = 0.4f;

    srcData[i * 3 + 0] = r;
    srcData[i * 3 + 1] = g;
    srcData[i * 3 + 2] = b;

    expectedData[i] = calculateGrayValue(r, g, b);
  }

  // Allocate GPU memory
  int srcStep, dstStep;
  Npp32f *d_src = nppiMalloc_32f_C3(width, height, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C1(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy input data to GPU row by row to handle step properly
  for (int y = 0; y < height; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * width * 3, width * 3 * sizeof(Npp32f),
               cudaMemcpyHostToDevice);
  }

  // Execute NPP function
  NppStatus status = nppiRGBToGray_32f_C3C1R(d_src, srcStep, d_dst, dstStep, roi);
  EXPECT_EQ(status, NPP_SUCCESS);

  // Copy result back to host row by row to handle step properly
  std::vector<Npp32f> resultData(width * height);
  for (int y = 0; y < height; y++) {
    cudaMemcpy(resultData.data() + y * width, (char *)d_dst + y * dstStep, width * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
  }

  // Verify results with floating-point precision
  for (int i = 0; i < width * height; i++) {
    EXPECT_NEAR(resultData[i], expectedData[i], 0.001f) << "Mismatch at pixel " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Test primary colors conversion
TEST_F(RGBToGrayFunctionalTest, RGBToGray_PrimaryColors) {
  const int testSize = 3;
  NppiSize testRoi = {testSize, 1};

  // Test pure red, green, blue
  std::vector<Npp8u> srcData = {
      255, 0,   0,  // Pure red
      0,   255, 0,  // Pure green
      0,   0,   255 // Pure blue
  };

  // Calculate expected values using standard weights
  std::vector<Npp8u> expectedData = {
      calculateGrayValue8u(255, 0, 0), // Red -> ~76
      calculateGrayValue8u(0, 255, 0), // Green -> ~150
      calculateGrayValue8u(0, 0, 255)  // Blue -> ~29
  };

  // Allocate GPU memory
  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C3(testSize, 1, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C1(testSize, 1, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy input data to GPU
  cudaMemcpy(d_src, srcData.data(), testSize * 3 * sizeof(Npp8u), cudaMemcpyHostToDevice);

  // Execute NPP function
  NppStatus status = nppiRGBToGray_8u_C3C1R(d_src, srcStep, d_dst, dstStep, testRoi);
  EXPECT_EQ(status, NPP_SUCCESS);

  // Copy result back to host
  std::vector<Npp8u> resultData(testSize);
  cudaMemcpy(resultData.data(), d_dst, testSize * sizeof(Npp8u), cudaMemcpyDeviceToHost);

  // Verify primary color conversion
  for (int i = 0; i < testSize; i++) {
    EXPECT_NEAR(resultData[i], expectedData[i], 1) << "Primary color " << i << " conversion failed";
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Test stream context version
TEST_F(RGBToGrayFunctionalTest, RGBToGray_StreamContext) {
  std::vector<Npp8u> srcData(width * height * 3, 128); // Gray color

  // Allocate GPU memory
  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C3(width, height, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C1(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy input data to GPU row by row to handle step properly
  for (int y = 0; y < height; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * width * 3, width * 3 * sizeof(Npp8u),
               cudaMemcpyHostToDevice);
  }

  // Create stream context
  NppStreamContext nppStreamCtx = {};
  nppStreamCtx.hStream = 0;

  // Execute NPP function with context
  NppStatus status = nppiRGBToGray_8u_C3C1R_Ctx(d_src, srcStep, d_dst, dstStep, roi, nppStreamCtx);
  EXPECT_EQ(status, NPP_SUCCESS);

  // Verify result - copy row by row to handle step properly
  std::vector<Npp8u> resultData(width * height);
  for (int y = 0; y < height; y++) {
    cudaMemcpy(resultData.data() + y * width, (char *)d_dst + y * dstStep, width * sizeof(Npp8u),
               cudaMemcpyDeviceToHost);
  }

  Npp8u expectedGray = calculateGrayValue8u(128, 128, 128);
  for (int i = 0; i < width * height; i++) {
    EXPECT_NEAR(resultData[i], expectedGray, 1);
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

namespace {
template <typename T>
void dumpExpected(const char *label, const std::vector<T> &values) {
  std::cout << label << " = {";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i) {
      std::cout << ", ";
    }
    std::cout << static_cast<long long>(values[i]);
  }
  std::cout << "};\n";
}
} // namespace

TEST_F(RGBToGrayFunctionalTest, RGBToGray_16u_C3C1R_ExpectedValues) {
  constexpr int testWidth = 4;
  constexpr int testHeight = 3;
  NppiSize testRoi{testWidth, testHeight};
  std::vector<Npp16u> srcData(testWidth * testHeight * 3);

  for (int i = 0; i < testWidth * testHeight; ++i) {
    Npp16u r = static_cast<Npp16u>(1000 + i * 11);
    Npp16u g = static_cast<Npp16u>(2000 + i * 7);
    Npp16u b = static_cast<Npp16u>(3000 + i * 5);
    srcData[i * 3 + 0] = r;
    srcData[i * 3 + 1] = g;
    srcData[i * 3 + 2] = b;
  }

  int srcStep = 0;
  int dstStep = 0;
  Npp16u *d_src = nppiMalloc_16u_C3(testWidth, testHeight, &srcStep);
  Npp16u *d_dst = nppiMalloc_16u_C1(testWidth, testHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  for (int y = 0; y < testHeight; ++y) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * testWidth * 3,
               testWidth * 3 * sizeof(Npp16u), cudaMemcpyHostToDevice);
  }

  NppStatus status = nppiRGBToGray_16u_C3C1R(d_src, srcStep, d_dst, dstStep, testRoi);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp16u> resultData(testWidth * testHeight);
  for (int y = 0; y < testHeight; ++y) {
    cudaMemcpy(resultData.data() + y * testWidth, (char *)d_dst + y * dstStep, testWidth * sizeof(Npp16u),
               cudaMemcpyDeviceToHost);
  }

  if (std::getenv("NPP_DUMP_EXPECTED")) {
    dumpExpected("kExpectedRGBToGray16uC3", resultData);
    GTEST_SKIP();
  }

  const Npp16u kExpectedRGBToGray16uC3[] = {1815, 1823, 1831, 1839, 1847, 1855,
                                            1863, 1871, 1879, 1887, 1895, 1903};
  ASSERT_EQ(resultData.size(), sizeof(kExpectedRGBToGray16uC3) / sizeof(kExpectedRGBToGray16uC3[0]));
  for (size_t i = 0; i < resultData.size(); ++i) {
    EXPECT_EQ(resultData[i], kExpectedRGBToGray16uC3[i]) << "Mismatch at " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(RGBToGrayFunctionalTest, RGBToGray_16u_AC4C1R_ExpectedValues) {
  constexpr int testWidth = 4;
  constexpr int testHeight = 3;
  NppiSize testRoi{testWidth, testHeight};
  std::vector<Npp16u> srcData(testWidth * testHeight * 4);

  for (int i = 0; i < testWidth * testHeight; ++i) {
    Npp16u r = static_cast<Npp16u>(1000 + i * 11);
    Npp16u g = static_cast<Npp16u>(2000 + i * 7);
    Npp16u b = static_cast<Npp16u>(3000 + i * 5);
    Npp16u a = static_cast<Npp16u>(4000 + i);
    srcData[i * 4 + 0] = r;
    srcData[i * 4 + 1] = g;
    srcData[i * 4 + 2] = b;
    srcData[i * 4 + 3] = a;
  }

  int srcStep = 0;
  int dstStep = 0;
  Npp16u *d_src = nppiMalloc_16u_C4(testWidth, testHeight, &srcStep);
  Npp16u *d_dst = nppiMalloc_16u_C1(testWidth, testHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  for (int y = 0; y < testHeight; ++y) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * testWidth * 4,
               testWidth * 4 * sizeof(Npp16u), cudaMemcpyHostToDevice);
  }

  NppStatus status = nppiRGBToGray_16u_AC4C1R(d_src, srcStep, d_dst, dstStep, testRoi);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp16u> resultData(testWidth * testHeight);
  for (int y = 0; y < testHeight; ++y) {
    cudaMemcpy(resultData.data() + y * testWidth, (char *)d_dst + y * dstStep, testWidth * sizeof(Npp16u),
               cudaMemcpyDeviceToHost);
  }

  if (std::getenv("NPP_DUMP_EXPECTED")) {
    dumpExpected("kExpectedRGBToGray16uAC4", resultData);
    GTEST_SKIP();
  }

  const Npp16u kExpectedRGBToGray16uAC4[] = {1815, 1823, 1831, 1839, 1847, 1855,
                                             1863, 1871, 1879, 1887, 1895, 1903};
  ASSERT_EQ(resultData.size(), sizeof(kExpectedRGBToGray16uAC4) / sizeof(kExpectedRGBToGray16uAC4[0]));
  for (size_t i = 0; i < resultData.size(); ++i) {
    EXPECT_EQ(resultData[i], kExpectedRGBToGray16uAC4[i]) << "Mismatch at " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(RGBToGrayFunctionalTest, RGBToGray_16s_C3C1R_ExpectedValues) {
  constexpr int testWidth = 4;
  constexpr int testHeight = 3;
  NppiSize testRoi{testWidth, testHeight};
  std::vector<Npp16s> srcData(testWidth * testHeight * 3);

  for (int i = 0; i < testWidth * testHeight; ++i) {
    Npp16s r = static_cast<Npp16s>(-80 + i * 9);
    Npp16s g = static_cast<Npp16s>(-40 + i * 5);
    Npp16s b = static_cast<Npp16s>(-20 + i * 7);
    srcData[i * 3 + 0] = r;
    srcData[i * 3 + 1] = g;
    srcData[i * 3 + 2] = b;
  }

  int srcStep = 0;
  int dstStep = 0;
  Npp16s *d_src = nullptr;
  Npp16s *d_dst = nullptr;
  size_t srcStepBytes = 0;
  size_t dstStepBytes = 0;
  cudaMallocPitch(reinterpret_cast<void **>(&d_src), &srcStepBytes, testWidth * sizeof(Npp16s) * 3, testHeight);
  cudaMallocPitch(reinterpret_cast<void **>(&d_dst), &dstStepBytes, testWidth * sizeof(Npp16s), testHeight);
  srcStep = static_cast<int>(srcStepBytes);
  dstStep = static_cast<int>(dstStepBytes);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, srcStep, srcData.data(), testWidth * sizeof(Npp16s) * 3, testWidth * sizeof(Npp16s) * 3,
               testHeight, cudaMemcpyHostToDevice);

  NppStatus status = nppiRGBToGray_16s_C3C1R(d_src, srcStep, d_dst, dstStep, testRoi);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp16s> resultData(testWidth * testHeight);
  cudaMemcpy2D(resultData.data(), testWidth * sizeof(Npp16s), d_dst, dstStep, testWidth * sizeof(Npp16s), testHeight,
               cudaMemcpyDeviceToHost);

  if (std::getenv("NPP_DUMP_EXPECTED")) {
    dumpExpected("kExpectedRGBToGray16sC3", resultData);
    GTEST_SKIP();
  }

  const Npp16s kExpectedRGBToGray16sC3[] = {-50, -43, -37, -30, -24, -18,
                                            -11, -5, 2, 8, 15, 21};
  ASSERT_EQ(resultData.size(), sizeof(kExpectedRGBToGray16sC3) / sizeof(kExpectedRGBToGray16sC3[0]));
  for (size_t i = 0; i < resultData.size(); ++i) {
    EXPECT_EQ(resultData[i], kExpectedRGBToGray16sC3[i]) << "Mismatch at " << i;
  }

  cudaFree(d_src);
  cudaFree(d_dst);
}

TEST_F(RGBToGrayFunctionalTest, RGBToGray_16s_AC4C1R_ExpectedValues) {
  constexpr int testWidth = 4;
  constexpr int testHeight = 3;
  NppiSize testRoi{testWidth, testHeight};
  std::vector<Npp16s> srcData(testWidth * testHeight * 4);

  for (int i = 0; i < testWidth * testHeight; ++i) {
    Npp16s r = static_cast<Npp16s>(-80 + i * 9);
    Npp16s g = static_cast<Npp16s>(-40 + i * 5);
    Npp16s b = static_cast<Npp16s>(-20 + i * 7);
    Npp16s a = static_cast<Npp16s>(300 + i);
    srcData[i * 4 + 0] = r;
    srcData[i * 4 + 1] = g;
    srcData[i * 4 + 2] = b;
    srcData[i * 4 + 3] = a;
  }

  int srcStep = 0;
  int dstStep = 0;
  Npp16s *d_src = nullptr;
  Npp16s *d_dst = nullptr;
  size_t srcStepBytes = 0;
  size_t dstStepBytes = 0;
  cudaMallocPitch(reinterpret_cast<void **>(&d_src), &srcStepBytes, testWidth * sizeof(Npp16s) * 4, testHeight);
  cudaMallocPitch(reinterpret_cast<void **>(&d_dst), &dstStepBytes, testWidth * sizeof(Npp16s), testHeight);
  srcStep = static_cast<int>(srcStepBytes);
  dstStep = static_cast<int>(dstStepBytes);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, srcStep, srcData.data(), testWidth * sizeof(Npp16s) * 4, testWidth * sizeof(Npp16s) * 4,
               testHeight, cudaMemcpyHostToDevice);

  NppStatus status = nppiRGBToGray_16s_AC4C1R(d_src, srcStep, d_dst, dstStep, testRoi);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp16s> resultData(testWidth * testHeight);
  cudaMemcpy2D(resultData.data(), testWidth * sizeof(Npp16s), d_dst, dstStep, testWidth * sizeof(Npp16s), testHeight,
               cudaMemcpyDeviceToHost);

  if (std::getenv("NPP_DUMP_EXPECTED")) {
    dumpExpected("kExpectedRGBToGray16sAC4", resultData);
    GTEST_SKIP();
  }

  const Npp16s kExpectedRGBToGray16sAC4[] = {-50, -43, -37, -30, -24, -18,
                                             -11, -5, 2, 8, 15, 21};
  ASSERT_EQ(resultData.size(), sizeof(kExpectedRGBToGray16sAC4) / sizeof(kExpectedRGBToGray16sAC4[0]));
  for (size_t i = 0; i < resultData.size(); ++i) {
    EXPECT_EQ(resultData[i], kExpectedRGBToGray16sAC4[i]) << "Mismatch at " << i;
  }

  cudaFree(d_src);
  cudaFree(d_dst);
}
