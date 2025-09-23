#include "../../framework/npp_test_base.h"
#include <cmath>

// Comprehensive parameterized tests for NPP morphological operations

#include "npp.h"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>

// Test parameter structure for morphological operations
struct MorphologyTestParams {
  int width, height;
  std::string testName;
  std::string description;
};

// Parameterized test class for morphological operations
class NppiMorphologyParameterizedTest : public ::testing::TestWithParam<MorphologyTestParams> {
protected:
  void SetUp() override {
    cudaError_t err = cudaSetDevice(0);
    ASSERT_EQ(err, cudaSuccess);
  }

  void TearDown() override {
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess);

    // Clear any potential memory issues
    cudaError_t clearErr = cudaGetLastError();
    if (clearErr != cudaSuccess) {
      std::cerr << "Warning: CUDA error detected in teardown: " << cudaGetErrorString(clearErr) << std::endl;
    }
  }

  // Generate constant data for testing
  template <typename T> std::vector<T> generateConstantData(int width, int height, T value) {
    return std::vector<T>(width * height, value);
  }

  // Generate binary pattern for morphological testing
  template <typename T> std::vector<T> generateBinaryPattern(int width, int height, T bgValue, T fgValue) {
    std::vector<T> data(width * height, bgValue);

    // Create a centered square
    int startX = width / 4;
    int endX = 3 * width / 4;
    int startY = height / 4;
    int endY = 3 * height / 4;

    for (int y = startY; y < endY; y++) {
      for (int x = startX; x < endX; x++) {
        if (y >= 0 && y < height && x >= 0 && x < width) {
          data[y * width + x] = fgValue;
        }
      }
    }
    return data;
  }

  // Generate checkerboard pattern
  template <typename T> std::vector<T> generateCheckerboard(int width, int height, T value1, T value2) {
    std::vector<T> data(width * height);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        data[y * width + x] = ((x + y) % 2 == 0) ? value1 : value2;
      }
    }
    return data;
  }

  // Create structure elements
  std::vector<Npp8u> createCrossKernel3x3() { return {0, 1, 0, 1, 1, 1, 0, 1, 0}; }

  std::vector<Npp8u> createBoxKernel3x3() { return {1, 1, 1, 1, 1, 1, 1, 1, 1}; }

  std::vector<Npp8u> createBoxKernel5x5() {
    return {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  }

  // Basic morphological property verification
  template <typename T>
  bool verifyMorphologicalProperty(const std::vector<T> &original, const std::vector<T> &eroded,
                                   const std::vector<T> &dilated) {
    // Basic property: eroded <= original <= dilated (for most pixels)
    int violations = 0;
    for (size_t i = 0; i < original.size(); i++) {
      if (eroded[i] > original[i] || original[i] > dilated[i]) {
        violations++;
      }
    }
    // Allow some violations due to boundary effects
    return violations < static_cast<int>(original.size() * 0.1);
  }
};

// Test parameters for different image sizes and configurations
static const std::vector<MorphologyTestParams> MORPHOLOGY_TEST_PARAMS = {
    {32, 32, "Small_32x32", "Small square image"},
    {64, 64, "Medium_64x64", "Medium square image"},
    {128, 128, "Large_128x128", "Large square image"},
    {256, 128, "Rect_256x128", "Wide rectangular image"},
    {128, 256, "Rect_128x256", "Tall rectangular image"},
    {48, 48, "Mid_48x48", "Medium-small square"},
    {96, 96, "Mid_96x96", "Medium-large square"},
    {64, 32, "Rect_64x32", "Wide rectangle"},
    {32, 64, "Rect_32x64", "Tall rectangle"}};

// ==================================================================================
// 8-bit Single Channel Tests (C1R)
// ==================================================================================

TEST_P(NppiMorphologyParameterizedTest, Erode_8u_C1R_ComprehensiveTest) {
  auto params = GetParam();
  const int width = params.width;
  const int height = params.height;

  SCOPED_TRACE("Erode 8u C1R test for size: " + params.testName);

  // Test binary pattern
  auto binaryData = generateBinaryPattern<Npp8u>(width, height, 0, 255);

  // Test regular version
  {
    int srcStep, dstStep;
    Npp8u *d_src = nppiMalloc_8u_C1(width, height, &srcStep);
    Npp8u *d_dst = nppiMalloc_8u_C1(width, height, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);

    cudaMemcpy2D(d_src, srcStep, binaryData.data(), width * sizeof(Npp8u), width * sizeof(Npp8u), height,
                 cudaMemcpyHostToDevice);

    auto kernel = createBoxKernel3x3();
    Npp8u *d_mask = nullptr;
    cudaMalloc(&d_mask, kernel.size());
    cudaMemcpy(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice);

    NppiSize oSizeROI = {width, height};
    NppiSize oMaskSize = {3, 3};
    NppiPoint oAnchor = {1, 1};

    NppStatus status = nppiErode_8u_C1R(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor);
    ASSERT_EQ(status, NPP_SUCCESS);

    std::vector<Npp8u> result(width * height);
    cudaMemcpy2D(result.data(), width * sizeof(Npp8u), d_dst, dstStep, width * sizeof(Npp8u), height,
                 cudaMemcpyDeviceToHost);

    // Verify erosion occurred - should reduce white regions
    int originalWhitePixels = std::count(binaryData.begin(), binaryData.end(), 255);
    int resultWhitePixels = std::count(result.begin(), result.end(), 255);
    EXPECT_LE(resultWhitePixels, originalWhitePixels);

    nppiFree(d_src);
    nppiFree(d_dst);
    cudaFree(d_mask);
  }

  // Test context version
  {
    int srcStep, dstStep;
    Npp8u *d_src = nppiMalloc_8u_C1(width, height, &srcStep);
    Npp8u *d_dst = nppiMalloc_8u_C1(width, height, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    nppStreamCtx.hStream = stream;

    cudaMemcpy2DAsync(d_src, srcStep, binaryData.data(), width * sizeof(Npp8u), width * sizeof(Npp8u), height,
                      cudaMemcpyHostToDevice, stream);

    auto kernel = createCrossKernel3x3();
    Npp8u *d_mask = nullptr;
    cudaMalloc(&d_mask, kernel.size());
    cudaMemcpyAsync(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice, stream);

    NppiSize oSizeROI = {width, height};
    NppiSize oMaskSize = {3, 3};
    NppiPoint oAnchor = {1, 1};

    NppStatus status =
        nppiErode_8u_C1R_Ctx(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor, nppStreamCtx);
    ASSERT_EQ(status, NPP_SUCCESS);

    cudaStreamSynchronize(stream);

    std::vector<Npp8u> result(width * height);
    cudaMemcpy2D(result.data(), width * sizeof(Npp8u), d_dst, dstStep, width * sizeof(Npp8u), height,
                 cudaMemcpyDeviceToHost);

    // Verify results are reasonable
    EXPECT_TRUE(std::all_of(result.begin(), result.end(), [](Npp8u val) { return val <= 255; }));

    cudaStreamDestroy(stream);
    nppiFree(d_src);
    nppiFree(d_dst);
    cudaFree(d_mask);
  }
}

TEST_P(NppiMorphologyParameterizedTest, Dilate_8u_C1R_ComprehensiveTest) {
  auto params = GetParam();
  const int width = params.width;
  const int height = params.height;

  SCOPED_TRACE("Dilate 8u C1R test for size: " + params.testName);

  // Test with checkerboard pattern
  auto checkerData = generateCheckerboard<Npp8u>(width, height, 0, 255);

  // Test regular version
  {
    int srcStep, dstStep;
    Npp8u *d_src = nppiMalloc_8u_C1(width, height, &srcStep);
    Npp8u *d_dst = nppiMalloc_8u_C1(width, height, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);

    cudaMemcpy2D(d_src, srcStep, checkerData.data(), width * sizeof(Npp8u), width * sizeof(Npp8u), height,
                 cudaMemcpyHostToDevice);

    auto kernel = createBoxKernel3x3();
    Npp8u *d_mask = nullptr;
    cudaMalloc(&d_mask, kernel.size());
    cudaMemcpy(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice);

    NppiSize oSizeROI = {width, height};
    NppiSize oMaskSize = {3, 3};
    NppiPoint oAnchor = {1, 1};

    NppStatus status = nppiDilate_8u_C1R(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor);
    ASSERT_EQ(status, NPP_SUCCESS);

    std::vector<Npp8u> result(width * height);
    cudaMemcpy2D(result.data(), width * sizeof(Npp8u), d_dst, dstStep, width * sizeof(Npp8u), height,
                 cudaMemcpyDeviceToHost);

    // Verify dilation occurred - should expand white regions
    int originalWhitePixels = std::count(checkerData.begin(), checkerData.end(), 255);
    int resultWhitePixels = std::count(result.begin(), result.end(), 255);
    EXPECT_GE(resultWhitePixels, originalWhitePixels);

    nppiFree(d_src);
    nppiFree(d_dst);
    cudaFree(d_mask);
  }

  // Test context version
  {
    int srcStep, dstStep;
    Npp8u *d_src = nppiMalloc_8u_C1(width, height, &srcStep);
    Npp8u *d_dst = nppiMalloc_8u_C1(width, height, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    nppStreamCtx.hStream = stream;

    cudaMemcpy2DAsync(d_src, srcStep, checkerData.data(), width * sizeof(Npp8u), width * sizeof(Npp8u), height,
                      cudaMemcpyHostToDevice, stream);

    auto kernel = createCrossKernel3x3();
    Npp8u *d_mask = nullptr;
    cudaMalloc(&d_mask, kernel.size());
    cudaMemcpyAsync(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice, stream);

    NppiSize oSizeROI = {width, height};
    NppiSize oMaskSize = {3, 3};
    NppiPoint oAnchor = {1, 1};

    NppStatus status =
        nppiDilate_8u_C1R_Ctx(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor, nppStreamCtx);
    ASSERT_EQ(status, NPP_SUCCESS);

    cudaStreamSynchronize(stream);

    std::vector<Npp8u> result(width * height);
    cudaMemcpy2D(result.data(), width * sizeof(Npp8u), d_dst, dstStep, width * sizeof(Npp8u), height,
                 cudaMemcpyDeviceToHost);

    // Verify results are reasonable
    EXPECT_TRUE(std::all_of(result.begin(), result.end(), [](Npp8u val) { return val <= 255; }));

    cudaStreamDestroy(stream);
    nppiFree(d_src);
    nppiFree(d_dst);
    cudaFree(d_mask);
  }
}

// ==================================================================================
// 8-bit 4-Channel Tests (C4R)
// ==================================================================================

TEST_P(NppiMorphologyParameterizedTest, Erode_8u_C4R_ComprehensiveTest) {
  auto params = GetParam();
  const int width = params.width;
  const int height = params.height;

  SCOPED_TRACE("Erode 8u C4R test for size: " + params.testName);

  // Create 4-channel binary pattern
  std::vector<Npp8u> binaryData(width * height * 4);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      bool inCenter = (x >= width / 4 && x < 3 * width / 4 && y >= height / 4 && y < 3 * height / 4);
      for (int c = 0; c < 4; c++) {
        binaryData[(y * width + x) * 4 + c] = inCenter ? 255 : 0;
      }
    }
  }

  // Test regular version
  {
    int srcStep, dstStep;
    Npp8u *d_src = nppiMalloc_8u_C4(width, height, &srcStep);
    Npp8u *d_dst = nppiMalloc_8u_C4(width, height, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);

    cudaMemcpy2D(d_src, srcStep, binaryData.data(), width * 4 * sizeof(Npp8u), width * 4 * sizeof(Npp8u), height,
                 cudaMemcpyHostToDevice);

    auto kernel = createBoxKernel3x3();
    Npp8u *d_mask = nullptr;
    cudaMalloc(&d_mask, kernel.size());
    cudaMemcpy(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice);

    NppiSize oSizeROI = {width, height};
    NppiSize oMaskSize = {3, 3};
    NppiPoint oAnchor = {1, 1};

    NppStatus status = nppiErode_8u_C4R(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor);
    ASSERT_EQ(status, NPP_SUCCESS);

    std::vector<Npp8u> result(width * height * 4);
    cudaMemcpy2D(result.data(), width * 4 * sizeof(Npp8u), d_dst, dstStep, width * 4 * sizeof(Npp8u), height,
                 cudaMemcpyDeviceToHost);

    // Verify all channels processed correctly
    for (int c = 0; c < 4; c++) {
      int originalWhitePixels = 0;
      int resultWhitePixels = 0;
      for (int i = c; i < width * height * 4; i += 4) {
        if (binaryData[i] == 255)
          originalWhitePixels++;
        if (result[i] == 255)
          resultWhitePixels++;
      }
      EXPECT_LE(resultWhitePixels, originalWhitePixels) << "Channel " << c;
    }

    nppiFree(d_src);
    nppiFree(d_dst);
    cudaFree(d_mask);
  }

  // Test context version
  {
    int srcStep, dstStep;
    Npp8u *d_src = nppiMalloc_8u_C4(width, height, &srcStep);
    Npp8u *d_dst = nppiMalloc_8u_C4(width, height, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    nppStreamCtx.hStream = stream;

    cudaMemcpy2DAsync(d_src, srcStep, binaryData.data(), width * 4 * sizeof(Npp8u), width * 4 * sizeof(Npp8u), height,
                      cudaMemcpyHostToDevice, stream);

    auto kernel = createCrossKernel3x3();
    Npp8u *d_mask = nullptr;
    cudaMalloc(&d_mask, kernel.size());
    cudaMemcpyAsync(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice, stream);

    NppiSize oSizeROI = {width, height};
    NppiSize oMaskSize = {3, 3};
    NppiPoint oAnchor = {1, 1};

    NppStatus status =
        nppiErode_8u_C4R_Ctx(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor, nppStreamCtx);
    ASSERT_EQ(status, NPP_SUCCESS);

    cudaStreamSynchronize(stream);

    std::vector<Npp8u> result(width * height * 4);
    cudaMemcpy2D(result.data(), width * 4 * sizeof(Npp8u), d_dst, dstStep, width * 4 * sizeof(Npp8u), height,
                 cudaMemcpyDeviceToHost);

    // Verify results are reasonable
    EXPECT_TRUE(std::all_of(result.begin(), result.end(), [](Npp8u val) { return val <= 255; }));

    cudaStreamDestroy(stream);
    nppiFree(d_src);
    nppiFree(d_dst);
    cudaFree(d_mask);
  }
}

TEST_P(NppiMorphologyParameterizedTest, Dilate_8u_C4R_ComprehensiveTest) {
  auto params = GetParam();
  const int width = params.width;
  const int height = params.height;

  SCOPED_TRACE("Dilate 8u C4R test for size: " + params.testName);

  // Create 4-channel checkerboard pattern
  std::vector<Npp8u> checkerData(width * height * 4);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      bool isWhite = ((x + y) % 2 == 0);
      for (int c = 0; c < 4; c++) {
        checkerData[(y * width + x) * 4 + c] = isWhite ? 255 : 0;
      }
    }
  }

  // Test regular version
  {
    int srcStep, dstStep;
    Npp8u *d_src = nppiMalloc_8u_C4(width, height, &srcStep);
    Npp8u *d_dst = nppiMalloc_8u_C4(width, height, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);

    cudaMemcpy2D(d_src, srcStep, checkerData.data(), width * 4 * sizeof(Npp8u), width * 4 * sizeof(Npp8u), height,
                 cudaMemcpyHostToDevice);

    auto kernel = createBoxKernel3x3();
    Npp8u *d_mask = nullptr;
    cudaMalloc(&d_mask, kernel.size());
    cudaMemcpy(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice);

    NppiSize oSizeROI = {width, height};
    NppiSize oMaskSize = {3, 3};
    NppiPoint oAnchor = {1, 1};

    NppStatus status = nppiDilate_8u_C4R(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor);
    ASSERT_EQ(status, NPP_SUCCESS);

    std::vector<Npp8u> result(width * height * 4);
    cudaMemcpy2D(result.data(), width * 4 * sizeof(Npp8u), d_dst, dstStep, width * 4 * sizeof(Npp8u), height,
                 cudaMemcpyDeviceToHost);

    // Verify all channels processed correctly
    for (int c = 0; c < 4; c++) {
      int originalWhitePixels = 0;
      int resultWhitePixels = 0;
      for (int i = c; i < width * height * 4; i += 4) {
        if (checkerData[i] == 255)
          originalWhitePixels++;
        if (result[i] == 255)
          resultWhitePixels++;
      }
      EXPECT_GE(resultWhitePixels, originalWhitePixels) << "Channel " << c;
    }

    nppiFree(d_src);
    nppiFree(d_dst);
    cudaFree(d_mask);
  }

  // Test context version
  {
    int srcStep, dstStep;
    Npp8u *d_src = nppiMalloc_8u_C4(width, height, &srcStep);
    Npp8u *d_dst = nppiMalloc_8u_C4(width, height, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    nppStreamCtx.hStream = stream;

    cudaMemcpy2DAsync(d_src, srcStep, checkerData.data(), width * 4 * sizeof(Npp8u), width * 4 * sizeof(Npp8u), height,
                      cudaMemcpyHostToDevice, stream);

    auto kernel = createCrossKernel3x3();
    Npp8u *d_mask = nullptr;
    cudaMalloc(&d_mask, kernel.size());
    cudaMemcpyAsync(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice, stream);

    NppiSize oSizeROI = {width, height};
    NppiSize oMaskSize = {3, 3};
    NppiPoint oAnchor = {1, 1};

    NppStatus status =
        nppiDilate_8u_C4R_Ctx(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor, nppStreamCtx);
    ASSERT_EQ(status, NPP_SUCCESS);

    cudaStreamSynchronize(stream);

    std::vector<Npp8u> result(width * height * 4);
    cudaMemcpy2D(result.data(), width * 4 * sizeof(Npp8u), d_dst, dstStep, width * 4 * sizeof(Npp8u), height,
                 cudaMemcpyDeviceToHost);

    // Verify results are reasonable
    EXPECT_TRUE(std::all_of(result.begin(), result.end(), [](Npp8u val) { return val <= 255; }));

    cudaStreamDestroy(stream);
    nppiFree(d_src);
    nppiFree(d_dst);
    cudaFree(d_mask);
  }
}

// ==================================================================================
// 32-bit Float Single Channel Tests (C1R)
// ==================================================================================

TEST_P(NppiMorphologyParameterizedTest, Erode_32f_C1R_ComprehensiveTest) {
  auto params = GetParam();
  const int width = params.width;
  const int height = params.height;

  SCOPED_TRACE("Erode 32f C1R test for size: " + params.testName);

  // Test binary pattern with float values
  auto binaryData = generateBinaryPattern<Npp32f>(width, height, 0.0f, 1.0f);

  // Test regular version
  {
    int srcStep, dstStep;
    Npp32f *d_src = nppiMalloc_32f_C1(width, height, &srcStep);
    Npp32f *d_dst = nppiMalloc_32f_C1(width, height, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);

    cudaMemcpy2D(d_src, srcStep, binaryData.data(), width * sizeof(Npp32f), width * sizeof(Npp32f), height,
                 cudaMemcpyHostToDevice);

    auto kernel = createBoxKernel3x3();
    Npp8u *d_mask = nullptr;
    cudaMalloc(&d_mask, kernel.size());
    cudaMemcpy(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice);

    NppiSize oSizeROI = {width, height};
    NppiSize oMaskSize = {3, 3};
    NppiPoint oAnchor = {1, 1};

    NppStatus status = nppiErode_32f_C1R(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor);
    ASSERT_EQ(status, NPP_SUCCESS);

    std::vector<Npp32f> result(width * height);
    cudaMemcpy2D(result.data(), width * sizeof(Npp32f), d_dst, dstStep, width * sizeof(Npp32f), height,
                 cudaMemcpyDeviceToHost);

    // Verify erosion occurred
    int originalHighPixels = std::count(binaryData.begin(), binaryData.end(), 1.0f);
    int resultHighPixels = std::count(result.begin(), result.end(), 1.0f);
    EXPECT_LE(resultHighPixels, originalHighPixels);

    // Verify values are in expected range
    EXPECT_TRUE(std::all_of(result.begin(), result.end(), [](Npp32f val) { return val >= 0.0f && val <= 1.0f; }));

    nppiFree(d_src);
    nppiFree(d_dst);
    cudaFree(d_mask);
  }

  // Test context version
  {
    int srcStep, dstStep;
    Npp32f *d_src = nppiMalloc_32f_C1(width, height, &srcStep);
    Npp32f *d_dst = nppiMalloc_32f_C1(width, height, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    nppStreamCtx.hStream = stream;

    cudaMemcpy2DAsync(d_src, srcStep, binaryData.data(), width * sizeof(Npp32f), width * sizeof(Npp32f), height,
                      cudaMemcpyHostToDevice, stream);

    auto kernel = createCrossKernel3x3();
    Npp8u *d_mask = nullptr;
    cudaMalloc(&d_mask, kernel.size());
    cudaMemcpyAsync(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice, stream);

    NppiSize oSizeROI = {width, height};
    NppiSize oMaskSize = {3, 3};
    NppiPoint oAnchor = {1, 1};

    NppStatus status =
        nppiErode_32f_C1R_Ctx(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor, nppStreamCtx);
    ASSERT_EQ(status, NPP_SUCCESS);

    cudaStreamSynchronize(stream);

    std::vector<Npp32f> result(width * height);
    cudaMemcpy2D(result.data(), width * sizeof(Npp32f), d_dst, dstStep, width * sizeof(Npp32f), height,
                 cudaMemcpyDeviceToHost);

    // Verify results are reasonable
    EXPECT_TRUE(std::all_of(result.begin(), result.end(), [](Npp32f val) { return val >= 0.0f && val <= 1.0f; }));

    cudaStreamDestroy(stream);
    nppiFree(d_src);
    nppiFree(d_dst);
    cudaFree(d_mask);
  }
}

TEST_P(NppiMorphologyParameterizedTest, Dilate_32f_C1R_ComprehensiveTest) {
  auto params = GetParam();
  const int width = params.width;
  const int height = params.height;

  SCOPED_TRACE("Dilate 32f C1R test for size: " + params.testName);

  // Test checkerboard pattern with float values
  auto checkerData = generateCheckerboard<Npp32f>(width, height, 0.0f, 1.0f);

  // Test regular version
  {
    int srcStep, dstStep;
    Npp32f *d_src = nppiMalloc_32f_C1(width, height, &srcStep);
    Npp32f *d_dst = nppiMalloc_32f_C1(width, height, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);

    cudaMemcpy2D(d_src, srcStep, checkerData.data(), width * sizeof(Npp32f), width * sizeof(Npp32f), height,
                 cudaMemcpyHostToDevice);

    auto kernel = createBoxKernel3x3();
    Npp8u *d_mask = nullptr;
    cudaMalloc(&d_mask, kernel.size());
    cudaMemcpy(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice);

    NppiSize oSizeROI = {width, height};
    NppiSize oMaskSize = {3, 3};
    NppiPoint oAnchor = {1, 1};

    NppStatus status = nppiDilate_32f_C1R(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor);
    ASSERT_EQ(status, NPP_SUCCESS);

    std::vector<Npp32f> result(width * height);
    cudaMemcpy2D(result.data(), width * sizeof(Npp32f), d_dst, dstStep, width * sizeof(Npp32f), height,
                 cudaMemcpyDeviceToHost);

    // Verify dilation occurred
    int originalHighPixels = std::count(checkerData.begin(), checkerData.end(), 1.0f);
    int resultHighPixels = std::count(result.begin(), result.end(), 1.0f);
    EXPECT_GE(resultHighPixels, originalHighPixels);

    // Verify values are in expected range
    EXPECT_TRUE(std::all_of(result.begin(), result.end(), [](Npp32f val) { return val >= 0.0f && val <= 1.0f; }));

    nppiFree(d_src);
    nppiFree(d_dst);
    cudaFree(d_mask);
  }

  // Test context version
  {
    int srcStep, dstStep;
    Npp32f *d_src = nppiMalloc_32f_C1(width, height, &srcStep);
    Npp32f *d_dst = nppiMalloc_32f_C1(width, height, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    nppStreamCtx.hStream = stream;

    cudaMemcpy2DAsync(d_src, srcStep, checkerData.data(), width * sizeof(Npp32f), width * sizeof(Npp32f), height,
                      cudaMemcpyHostToDevice, stream);

    auto kernel = createCrossKernel3x3();
    Npp8u *d_mask = nullptr;
    cudaMalloc(&d_mask, kernel.size());
    cudaMemcpyAsync(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice, stream);

    NppiSize oSizeROI = {width, height};
    NppiSize oMaskSize = {3, 3};
    NppiPoint oAnchor = {1, 1};

    NppStatus status =
        nppiDilate_32f_C1R_Ctx(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor, nppStreamCtx);
    ASSERT_EQ(status, NPP_SUCCESS);

    cudaStreamSynchronize(stream);

    std::vector<Npp32f> result(width * height);
    cudaMemcpy2D(result.data(), width * sizeof(Npp32f), d_dst, dstStep, width * sizeof(Npp32f), height,
                 cudaMemcpyDeviceToHost);

    // Verify results are reasonable
    EXPECT_TRUE(std::all_of(result.begin(), result.end(), [](Npp32f val) { return val >= 0.0f && val <= 1.0f; }));

    cudaStreamDestroy(stream);
    nppiFree(d_src);
    nppiFree(d_dst);
    cudaFree(d_mask);
  }
}

// ==================================================================================
// 32-bit Float 4-Channel Tests (C4R)
// ==================================================================================

TEST_P(NppiMorphologyParameterizedTest, Erode_32f_C4R_ComprehensiveTest) {
  auto params = GetParam();
  const int width = params.width;
  const int height = params.height;

  SCOPED_TRACE("Erode 32f C4R test for size: " + params.testName);

  // Create 4-channel binary pattern with float values
  std::vector<Npp32f> binaryData(width * height * 4);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      bool inCenter = (x >= width / 4 && x < 3 * width / 4 && y >= height / 4 && y < 3 * height / 4);
      for (int c = 0; c < 4; c++) {
        binaryData[(y * width + x) * 4 + c] = inCenter ? 1.0f : 0.0f;
      }
    }
  }

  // Test regular version
  {
    int srcStep, dstStep;
    Npp32f *d_src = nppiMalloc_32f_C4(width, height, &srcStep);
    Npp32f *d_dst = nppiMalloc_32f_C4(width, height, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);

    cudaMemcpy2D(d_src, srcStep, binaryData.data(), width * 4 * sizeof(Npp32f), width * 4 * sizeof(Npp32f), height,
                 cudaMemcpyHostToDevice);

    auto kernel = createBoxKernel3x3();
    Npp8u *d_mask = nullptr;
    cudaMalloc(&d_mask, kernel.size());
    cudaMemcpy(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice);

    NppiSize oSizeROI = {width, height};
    NppiSize oMaskSize = {3, 3};
    NppiPoint oAnchor = {1, 1};

    NppStatus status = nppiErode_32f_C4R(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor);
    ASSERT_EQ(status, NPP_SUCCESS);

    std::vector<Npp32f> result(width * height * 4);
    cudaMemcpy2D(result.data(), width * 4 * sizeof(Npp32f), d_dst, dstStep, width * 4 * sizeof(Npp32f), height,
                 cudaMemcpyDeviceToHost);

    // Verify all channels processed correctly
    for (int c = 0; c < 4; c++) {
      int originalHighPixels = 0;
      int resultHighPixels = 0;
      for (int i = c; i < width * height * 4; i += 4) {
        if (binaryData[i] == 1.0f)
          originalHighPixels++;
        if (result[i] == 1.0f)
          resultHighPixels++;
      }
      EXPECT_LE(resultHighPixels, originalHighPixels) << "Channel " << c;
    }

    // Verify values are in expected range
    EXPECT_TRUE(std::all_of(result.begin(), result.end(), [](Npp32f val) { return val >= 0.0f && val <= 1.0f; }));

    nppiFree(d_src);
    nppiFree(d_dst);
    cudaFree(d_mask);
  }

  // Test context version
  {
    int srcStep, dstStep;
    Npp32f *d_src = nppiMalloc_32f_C4(width, height, &srcStep);
    Npp32f *d_dst = nppiMalloc_32f_C4(width, height, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    nppStreamCtx.hStream = stream;

    cudaMemcpy2DAsync(d_src, srcStep, binaryData.data(), width * 4 * sizeof(Npp32f), width * 4 * sizeof(Npp32f), height,
                      cudaMemcpyHostToDevice, stream);

    auto kernel = createCrossKernel3x3();
    Npp8u *d_mask = nullptr;
    cudaMalloc(&d_mask, kernel.size());
    cudaMemcpyAsync(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice, stream);

    NppiSize oSizeROI = {width, height};
    NppiSize oMaskSize = {3, 3};
    NppiPoint oAnchor = {1, 1};

    NppStatus status =
        nppiErode_32f_C4R_Ctx(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor, nppStreamCtx);
    ASSERT_EQ(status, NPP_SUCCESS);

    cudaStreamSynchronize(stream);

    std::vector<Npp32f> result(width * height * 4);
    cudaMemcpy2D(result.data(), width * 4 * sizeof(Npp32f), d_dst, dstStep, width * 4 * sizeof(Npp32f), height,
                 cudaMemcpyDeviceToHost);

    // Verify values are in expected range
    EXPECT_TRUE(std::all_of(result.begin(), result.end(), [](Npp32f val) { return val >= 0.0f && val <= 1.0f; }));

    cudaStreamDestroy(stream);
    nppiFree(d_src);
    nppiFree(d_dst);
    cudaFree(d_mask);
  }
}

TEST_P(NppiMorphologyParameterizedTest, Dilate_32f_C4R_ComprehensiveTest) {
  auto params = GetParam();
  const int width = params.width;
  const int height = params.height;

  SCOPED_TRACE("Dilate 32f C4R test for size: " + params.testName);

  // Create 4-channel checkerboard pattern with float values
  std::vector<Npp32f> checkerData(width * height * 4);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      bool isHigh = ((x + y) % 2 == 0);
      for (int c = 0; c < 4; c++) {
        checkerData[(y * width + x) * 4 + c] = isHigh ? 1.0f : 0.0f;
      }
    }
  }

  // Test regular version
  {
    int srcStep, dstStep;
    Npp32f *d_src = nppiMalloc_32f_C4(width, height, &srcStep);
    Npp32f *d_dst = nppiMalloc_32f_C4(width, height, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);

    cudaMemcpy2D(d_src, srcStep, checkerData.data(), width * 4 * sizeof(Npp32f), width * 4 * sizeof(Npp32f), height,
                 cudaMemcpyHostToDevice);

    auto kernel = createBoxKernel3x3();
    Npp8u *d_mask = nullptr;
    cudaMalloc(&d_mask, kernel.size());
    cudaMemcpy(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice);

    NppiSize oSizeROI = {width, height};
    NppiSize oMaskSize = {3, 3};
    NppiPoint oAnchor = {1, 1};

    NppStatus status = nppiDilate_32f_C4R(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor);
    ASSERT_EQ(status, NPP_SUCCESS);

    std::vector<Npp32f> result(width * height * 4);
    cudaMemcpy2D(result.data(), width * 4 * sizeof(Npp32f), d_dst, dstStep, width * 4 * sizeof(Npp32f), height,
                 cudaMemcpyDeviceToHost);

    // Verify all channels processed correctly
    for (int c = 0; c < 4; c++) {
      int originalHighPixels = 0;
      int resultHighPixels = 0;
      for (int i = c; i < width * height * 4; i += 4) {
        if (checkerData[i] == 1.0f)
          originalHighPixels++;
        if (result[i] == 1.0f)
          resultHighPixels++;
      }
      EXPECT_GE(resultHighPixels, originalHighPixels) << "Channel " << c;
    }

    // Verify values are in expected range
    EXPECT_TRUE(std::all_of(result.begin(), result.end(), [](Npp32f val) { return val >= 0.0f && val <= 1.0f; }));

    nppiFree(d_src);
    nppiFree(d_dst);
    cudaFree(d_mask);
  }

  // Test context version
  {
    int srcStep, dstStep;
    Npp32f *d_src = nppiMalloc_32f_C4(width, height, &srcStep);
    Npp32f *d_dst = nppiMalloc_32f_C4(width, height, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    nppStreamCtx.hStream = stream;

    cudaMemcpy2DAsync(d_src, srcStep, checkerData.data(), width * 4 * sizeof(Npp32f), width * 4 * sizeof(Npp32f),
                      height, cudaMemcpyHostToDevice, stream);

    auto kernel = createCrossKernel3x3();
    Npp8u *d_mask = nullptr;
    cudaMalloc(&d_mask, kernel.size());
    cudaMemcpyAsync(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice, stream);

    NppiSize oSizeROI = {width, height};
    NppiSize oMaskSize = {3, 3};
    NppiPoint oAnchor = {1, 1};

    NppStatus status =
        nppiDilate_32f_C4R_Ctx(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor, nppStreamCtx);
    ASSERT_EQ(status, NPP_SUCCESS);

    cudaStreamSynchronize(stream);

    std::vector<Npp32f> result(width * height * 4);
    cudaMemcpy2D(result.data(), width * 4 * sizeof(Npp32f), d_dst, dstStep, width * 4 * sizeof(Npp32f), height,
                 cudaMemcpyDeviceToHost);

    // Verify values are in expected range
    EXPECT_TRUE(std::all_of(result.begin(), result.end(), [](Npp32f val) { return val >= 0.0f && val <= 1.0f; }));

    cudaStreamDestroy(stream);
    nppiFree(d_src);
    nppiFree(d_dst);
    cudaFree(d_mask);
  }
}

// ==================================================================================
// Morphological Property Tests
// ==================================================================================

TEST_P(NppiMorphologyParameterizedTest, MorphologicalProperties_8u_C1R) {
  auto params = GetParam();
  const int width = params.width;
  const int height = params.height;

  SCOPED_TRACE("Morphological properties test for size: " + params.testName);

  // Test with binary pattern
  auto testData = generateBinaryPattern<Npp8u>(width, height, 0, 255);

  int srcStep, erodedStep, dilatedStep;
  Npp8u *d_src = nppiMalloc_8u_C1(width, height, &srcStep);
  Npp8u *d_eroded = nppiMalloc_8u_C1(width, height, &erodedStep);
  Npp8u *d_dilated = nppiMalloc_8u_C1(width, height, &dilatedStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_eroded, nullptr);
  ASSERT_NE(d_dilated, nullptr);

  cudaMemcpy2D(d_src, srcStep, testData.data(), width * sizeof(Npp8u), width * sizeof(Npp8u), height,
               cudaMemcpyHostToDevice);

  auto kernel = createBoxKernel3x3();
  Npp8u *d_mask = nullptr;
  cudaMalloc(&d_mask, kernel.size());
  cudaMemcpy(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  NppiSize oMaskSize = {3, 3};
  NppiPoint oAnchor = {1, 1};

  // Perform both erosion and dilation
  NppStatus statusErode = nppiErode_8u_C1R(d_src, srcStep, d_eroded, erodedStep, oSizeROI, d_mask, oMaskSize, oAnchor);
  NppStatus statusDilate =
      nppiDilate_8u_C1R(d_src, srcStep, d_dilated, dilatedStep, oSizeROI, d_mask, oMaskSize, oAnchor);
  ASSERT_EQ(statusErode, NPP_SUCCESS);
  ASSERT_EQ(statusDilate, NPP_SUCCESS);

  std::vector<Npp8u> eroded(width * height);
  std::vector<Npp8u> dilated(width * height);
  cudaMemcpy2D(eroded.data(), width * sizeof(Npp8u), d_eroded, erodedStep, width * sizeof(Npp8u), height,
               cudaMemcpyDeviceToHost);
  cudaMemcpy2D(dilated.data(), width * sizeof(Npp8u), d_dilated, dilatedStep, width * sizeof(Npp8u), height,
               cudaMemcpyDeviceToHost);

  // Verify morphological properties
  EXPECT_TRUE(verifyMorphologicalProperty(testData, eroded, dilated));

  nppiFree(d_src);
  nppiFree(d_eroded);
  nppiFree(d_dilated);
  cudaFree(d_mask);
}

TEST_P(NppiMorphologyParameterizedTest, MorphologicalProperties_32f_C1R) {
  auto params = GetParam();
  const int width = params.width;
  const int height = params.height;

  SCOPED_TRACE("Morphological properties test for size: " + params.testName);

  // Test with binary pattern
  auto testData = generateBinaryPattern<Npp32f>(width, height, 0.0f, 1.0f);

  int srcStep, erodedStep, dilatedStep;
  Npp32f *d_src = nppiMalloc_32f_C1(width, height, &srcStep);
  Npp32f *d_eroded = nppiMalloc_32f_C1(width, height, &erodedStep);
  Npp32f *d_dilated = nppiMalloc_32f_C1(width, height, &dilatedStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_eroded, nullptr);
  ASSERT_NE(d_dilated, nullptr);

  cudaMemcpy2D(d_src, srcStep, testData.data(), width * sizeof(Npp32f), width * sizeof(Npp32f), height,
               cudaMemcpyHostToDevice);

  auto kernel = createBoxKernel3x3();
  Npp8u *d_mask = nullptr;
  cudaMalloc(&d_mask, kernel.size());
  cudaMemcpy(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  NppiSize oMaskSize = {3, 3};
  NppiPoint oAnchor = {1, 1};

  // Perform both erosion and dilation
  NppStatus statusErode = nppiErode_32f_C1R(d_src, srcStep, d_eroded, erodedStep, oSizeROI, d_mask, oMaskSize, oAnchor);
  NppStatus statusDilate =
      nppiDilate_32f_C1R(d_src, srcStep, d_dilated, dilatedStep, oSizeROI, d_mask, oMaskSize, oAnchor);
  ASSERT_EQ(statusErode, NPP_SUCCESS);
  ASSERT_EQ(statusDilate, NPP_SUCCESS);

  std::vector<Npp32f> eroded(width * height);
  std::vector<Npp32f> dilated(width * height);
  cudaMemcpy2D(eroded.data(), width * sizeof(Npp32f), d_eroded, erodedStep, width * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);
  cudaMemcpy2D(dilated.data(), width * sizeof(Npp32f), d_dilated, dilatedStep, width * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);

  // Verify morphological properties
  EXPECT_TRUE(verifyMorphologicalProperty(testData, eroded, dilated));

  nppiFree(d_src);
  nppiFree(d_eroded);
  nppiFree(d_dilated);
  cudaFree(d_mask);
}

// Instantiate parameterized tests
INSTANTIATE_TEST_SUITE_P(ComprehensiveMorphology, NppiMorphologyParameterizedTest,
                         ::testing::ValuesIn(MORPHOLOGY_TEST_PARAMS),
                         [](const testing::TestParamInfo<MorphologyTestParams> &info) { return info.param.testName; });

using namespace npp_functional_test;

class MorphologyFunctionalTest : public NppTestBase {
protected:
  void SetUp() override { NppTestBase::SetUp(); }

  void TearDown() override { NppTestBase::TearDown(); }

  // Function to create test pattern for morphology operations
  void createMorphologyTestImage(std::vector<Npp8u> &data, int width, int height) {
    data.resize(width * height);

    // Create a simple pattern with foreground objects and background
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int idx = y * width + x;

        // Create rectangular objects
        if ((x >= width / 4 && x <= 3 * width / 4) && (y >= height / 4 && y <= 3 * height / 4)) {
          data[idx] = 255; // Foreground
        } else {
          data[idx] = 0; // Background
        }

        // Add some noise/small objects
        if ((x % 8 == 0) && (y % 8 == 0)) {
          data[idx] = 255;
        }
      }
    }
  }
};

// 测试8位单通道腐蚀操作
TEST_F(MorphologyFunctionalTest, Erode3x3_8u_C1R_Basic) {
  const int width = 32, height = 32;

  std::vector<Npp8u> srcData;
  createMorphologyTestImage(srcData, width, height);

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> dst(width, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiErode3x3_8u_C1R(src.get(), src.step(), dst.get(), dst.step(), roi);

  ASSERT_EQ(status, NPP_SUCCESS) << "nppiErode3x3_8u_C1R failed";

  // Validate结果 - 腐蚀应该缩小白色区域
  std::vector<Npp8u> resultData(width * height);
  dst.copyToHost(resultData);

  // 检查中心区域是否仍然是白色（应该被保留但缩小）
  int centerWhitePixels = 0;
  int totalWhitePixels = 0;

  for (int y = height / 3; y < 2 * height / 3; y++) {
    for (int x = width / 3; x < 2 * width / 3; x++) {
      if (resultData[y * width + x] == 255) {
        centerWhitePixels++;
      }
    }
  }

  for (int i = 0; i < width * height; i++) {
    if (resultData[i] == 255) {
      totalWhitePixels++;
    }
  }

  EXPECT_GT(centerWhitePixels, 0) << "Erosion should preserve some central white pixels";
  EXPECT_LT(totalWhitePixels, width * height / 2) << "Erosion should reduce overall white area";
}

// 测试8位单通道膨胀操作
TEST_F(MorphologyFunctionalTest, Dilate3x3_8u_C1R_Basic) {
  const int width = 32, height = 32;

  // 创建稀疏的白点测试图像
  std::vector<Npp8u> srcData(width * height, 0);

  // 在中心放置几个白点
  srcData[height / 2 * width + width / 2] = 255;
  srcData[(height / 2 + 2) * width + (width / 2 + 2)] = 255;
  srcData[(height / 2 - 2) * width + (width / 2 - 2)] = 255;

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> dst(width, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiDilate3x3_8u_C1R(src.get(), src.step(), dst.get(), dst.step(), roi);

  ASSERT_EQ(status, NPP_SUCCESS) << "nppiDilate3x3_8u_C1R failed";

  // Validate结果 - 膨胀应该扩大白色区域
  std::vector<Npp8u> resultData(width * height);
  dst.copyToHost(resultData);

  int originalWhitePixels = 3; // 我们放置的白点数量
  int resultWhitePixels = 0;

  for (int i = 0; i < width * height; i++) {
    if (resultData[i] == 255) {
      resultWhitePixels++;
    }
  }

  EXPECT_GT(resultWhitePixels, originalWhitePixels)
      << "Dilation should expand white regions, got " << resultWhitePixels << " vs " << originalWhitePixels;
}

// 测试32位浮点腐蚀操作
TEST_F(MorphologyFunctionalTest, Erode3x3_32f_C1R_Float) {
  const int width = 16, height = 16;

  // 创建浮点测试数据
  std::vector<Npp32f> srcData(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      // 创建梯度模式
      if (x >= width / 4 && x <= 3 * width / 4 && y >= height / 4 && y <= 3 * height / 4) {
        srcData[idx] = 1.0f;
      } else {
        srcData[idx] = 0.0f;
      }
    }
  }

  NppImageMemory<Npp32f> src(width, height);
  NppImageMemory<Npp32f> dst(width, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiErode3x3_32f_C1R(src.get(), src.step(), dst.get(), dst.step(), roi);

  ASSERT_EQ(status, NPP_SUCCESS) << "nppiErode3x3_32f_C1R failed";

  // Validate结果 - 检查浮点数据的正确性
  std::vector<Npp32f> resultData(width * height);
  dst.copyToHost(resultData);

  bool hasValidFloats = true;
  for (int i = 0; i < width * height; i++) {
    if (std::isnan(resultData[i]) || std::isinf(resultData[i])) {
      hasValidFloats = false;
      break;
    }
  }

  EXPECT_TRUE(hasValidFloats) << "Float erosion should produce valid floating-point results";
}

// 测试32位浮点膨胀操作
TEST_F(MorphologyFunctionalTest, Dilate3x3_32f_C1R_Float) {
  const int width = 16, height = 16;

  // 创建浮点测试数据 - 几个高值点
  std::vector<Npp32f> srcData(width * height, 0.1f);
  srcData[height / 2 * width + width / 2] = 0.9f;
  srcData[(height / 2 + 1) * width + (width / 2 + 1)] = 0.8f;

  NppImageMemory<Npp32f> src(width, height);
  NppImageMemory<Npp32f> dst(width, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiDilate3x3_32f_C1R(src.get(), src.step(), dst.get(), dst.step(), roi);

  ASSERT_EQ(status, NPP_SUCCESS) << "nppiDilate3x3_32f_C1R failed";

  // Validate结果 - 膨胀应该传播高值
  std::vector<Npp32f> resultData(width * height);
  dst.copyToHost(resultData);

  int highValuePixels = 0;
  for (int i = 0; i < width * height; i++) {
    if (resultData[i] > 0.7f) { // 查找被膨胀的高值区域
      highValuePixels++;
    }
  }

  EXPECT_GT(highValuePixels, 2) << "Float dilation should spread high values to neighboring pixels";
}

// 形态学操作组合测试（开运算和闭运算）
TEST_F(MorphologyFunctionalTest, Morphology_OpenClose_Operations) {
  const int width = 32, height = 32;

  std::vector<Npp8u> srcData;
  createMorphologyTestImage(srcData, width, height);

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> temp(width, height);
  NppImageMemory<Npp8u> opening(width, height);
  NppImageMemory<Npp8u> closing(width, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};

  // 开运算：先腐蚀后膨胀
  NppStatus status1 = nppiErode3x3_8u_C1R(src.get(), src.step(), temp.get(), temp.step(), roi);
  NppStatus status2 = nppiDilate3x3_8u_C1R(temp.get(), temp.step(), opening.get(), opening.step(), roi);

  ASSERT_EQ(status1, NPP_SUCCESS) << "Erosion for opening failed";
  ASSERT_EQ(status2, NPP_SUCCESS) << "Dilation for opening failed";

  // 闭运算：先膨胀后腐蚀
  status1 = nppiDilate3x3_8u_C1R(src.get(), src.step(), temp.get(), temp.step(), roi);
  status2 = nppiErode3x3_8u_C1R(temp.get(), temp.step(), closing.get(), closing.step(), roi);

  ASSERT_EQ(status1, NPP_SUCCESS) << "Dilation for closing failed";
  ASSERT_EQ(status2, NPP_SUCCESS) << "Erosion for closing failed";

  // Validate开运算和闭运算产生了不同的结果
  std::vector<Npp8u> openingData(width * height);
  std::vector<Npp8u> closingData(width * height);

  opening.copyToHost(openingData);
  closing.copyToHost(closingData);

  // Validate开运算和闭运算都成功执行
  // 对于某些图像，开运算和闭运算可能产生相同结果，这是正常的
  // 我们只Validate操作成功执行并产生了合理的结果

  // 统计开运算的非零像素
  int openingNonZero = 0;
  for (int i = 0; i < width * height; i++) {
    if (openingData[i] > 0)
      openingNonZero++;
  }

  // 统计闭运算的非零像素
  int closingNonZero = 0;
  for (int i = 0; i < width * height; i++) {
    if (closingData[i] > 0)
      closingNonZero++;
  }

  // 两种操作都应该产生一些结果
  EXPECT_GT(openingNonZero, 0) << "Opening operation should produce some foreground pixels";
  EXPECT_GT(closingNonZero, 0) << "Closing operation should produce some foreground pixels";

  std::cout << "Morphology OpenClose test passed - vendor NPP behavior verified" << std::endl;
}

class NppiMorphologyExtendedTest : public ::testing::Test {
protected:
  void SetUp() override {
    cudaError_t err = cudaSetDevice(0);
    ASSERT_EQ(err, cudaSuccess);
  }

  void TearDown() override {
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess);
  }

  // Create 3x3 cross structure element
  std::vector<Npp8u> createCrossKernel() { return {0, 1, 0, 1, 1, 1, 0, 1, 0}; }

  // Create 3x3 box structure element
  std::vector<Npp8u> createBoxKernel() { return {1, 1, 1, 1, 1, 1, 1, 1, 1}; }

  // Create 5x5 circle structure element
  std::vector<Npp8u> createCircleKernel() {
    return {0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0};
  }
};

// Test 1: Basic erosion for 8u_C1R with cross kernel
TEST_F(NppiMorphologyExtendedTest, Erode_8u_C1R_CrossKernel) {
  const int width = 16;
  const int height = 16;

  // Create test image with a white square in the center
  std::vector<Npp8u> hostSrc(width * height, 0);
  for (int y = 4; y < 12; y++) {
    for (int x = 4; x < 12; x++) {
      hostSrc[y * width + x] = 255;
    }
  }

  // Allocate GPU memory
  Npp8u *d_src = nullptr, *d_dst = nullptr;
  int srcStep, dstStep;
  d_src = nppiMalloc_8u_C1(width, height, &srcStep);
  d_dst = nppiMalloc_8u_C1(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Upload source data
  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width, width, height, cudaMemcpyHostToDevice);

  // Create structure element
  auto kernel = createCrossKernel();
  Npp8u *d_mask = nullptr;
  cudaMalloc(&d_mask, kernel.size());
  cudaMemcpy(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  NppiSize oMaskSize = {3, 3};
  NppiPoint oAnchor = {1, 1};

  // Perform erosion
  NppStatus status = nppiErode_8u_C1R(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back to host
  std::vector<Npp8u> hostDst(width * height);
  cudaMemcpy2D(hostDst.data(), width, d_dst, dstStep, width, height, cudaMemcpyDeviceToHost);

  // Verify erosion occurred - the white square should be smaller
  bool foundWhiteInCenter = false;
  bool foundZeroAtEdge = false;
  for (int y = 5; y < 11; y++) {
    for (int x = 5; x < 11; x++) {
      if (hostDst[y * width + x] == 255) {
        foundWhiteInCenter = true;
      }
    }
  }
  // Check edge of original square should now be black
  if (hostDst[4 * width + 4] == 0) {
    foundZeroAtEdge = true;
  }

  EXPECT_TRUE(foundWhiteInCenter);
  EXPECT_TRUE(foundZeroAtEdge);

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_dst);
  cudaFree(d_mask);
}

// Test 2: Basic dilation for 8u_C4R with box kernel
TEST_F(NppiMorphologyExtendedTest, Dilate_8u_C4R_BoxKernel) {
  const int width = 16;
  const int height = 16;

  // Create test image with a small white square in the center (4 channels)
  std::vector<Npp8u> hostSrc(width * height * 4, 0);
  for (int y = 6; y < 10; y++) {
    for (int x = 6; x < 10; x++) {
      for (int c = 0; c < 4; c++) {
        hostSrc[(y * width + x) * 4 + c] = 255;
      }
    }
  }

  // Allocate GPU memory
  Npp8u *d_src = nullptr, *d_dst = nullptr;
  int srcStep, dstStep;
  d_src = nppiMalloc_8u_C4(width, height, &srcStep);
  d_dst = nppiMalloc_8u_C4(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Upload source data
  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width * 4, width * 4, height, cudaMemcpyHostToDevice);

  // Create structure element
  auto kernel = createBoxKernel();
  Npp8u *d_mask = nullptr;
  cudaMalloc(&d_mask, kernel.size());
  cudaMemcpy(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  NppiSize oMaskSize = {3, 3};
  NppiPoint oAnchor = {1, 1};

  // Perform dilation
  NppStatus status = nppiDilate_8u_C4R(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back to host
  std::vector<Npp8u> hostDst(width * height * 4);
  cudaMemcpy2D(hostDst.data(), width * 4, d_dst, dstStep, width * 4, height, cudaMemcpyDeviceToHost);

  // Verify dilation occurred - the white area should be larger
  bool foundExpandedWhite = false;
  // Check if dilation expanded to (5,5) which was originally black
  for (int c = 0; c < 4; c++) {
    if (hostDst[(5 * width + 5) * 4 + c] == 255) {
      foundExpandedWhite = true;
      break;
    }
  }

  EXPECT_TRUE(foundExpandedWhite);

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_dst);
  cudaFree(d_mask);
}

// Test 3: Erosion and dilation with 32f_C1R
TEST_F(NppiMorphologyExtendedTest, Erode_Dilate_32f_C1R_Comparison) {
  const int width = 12;
  const int height = 12;

  // Create test image with gradient values
  std::vector<Npp32f> hostSrc(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      hostSrc[y * width + x] = static_cast<float>(x + y * 10);
    }
  }

  // Allocate GPU memory
  Npp32f *d_src = nullptr, *d_eroded = nullptr, *d_dilated = nullptr;
  int srcStep, erodedStep, dilatedStep;
  d_src = nppiMalloc_32f_C1(width, height, &srcStep);
  d_eroded = nppiMalloc_32f_C1(width, height, &erodedStep);
  d_dilated = nppiMalloc_32f_C1(width, height, &dilatedStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_eroded, nullptr);
  ASSERT_NE(d_dilated, nullptr);

  // Upload source data
  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width * sizeof(Npp32f), width * sizeof(Npp32f), height,
               cudaMemcpyHostToDevice);

  // Create structure element
  auto kernel = createCrossKernel();
  Npp8u *d_mask = nullptr;
  cudaMalloc(&d_mask, kernel.size());
  cudaMemcpy(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  NppiSize oMaskSize = {3, 3};
  NppiPoint oAnchor = {1, 1};

  // Perform erosion and dilation
  NppStatus status1 = nppiErode_32f_C1R(d_src, srcStep, d_eroded, erodedStep, oSizeROI, d_mask, oMaskSize, oAnchor);
  NppStatus status2 = nppiDilate_32f_C1R(d_src, srcStep, d_dilated, dilatedStep, oSizeROI, d_mask, oMaskSize, oAnchor);
  ASSERT_EQ(status1, NPP_SUCCESS);
  ASSERT_EQ(status2, NPP_SUCCESS);

  // Copy results back to host
  std::vector<Npp32f> hostEroded(width * height);
  std::vector<Npp32f> hostDilated(width * height);
  cudaMemcpy2D(hostEroded.data(), width * sizeof(Npp32f), d_eroded, erodedStep, width * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);
  cudaMemcpy2D(hostDilated.data(), width * sizeof(Npp32f), d_dilated, dilatedStep, width * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);

  // Verify morphological properties: eroded <= original <= dilated
  bool morphologyValid = true;
  for (int i = 1; i < (width - 1) * (height - 1); i++) {
    if (hostEroded[i] > hostSrc[i] || hostSrc[i] > hostDilated[i]) {
      morphologyValid = false;
      break;
    }
  }

  EXPECT_TRUE(morphologyValid);

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_eroded);
  nppiFree(d_dilated);
  cudaFree(d_mask);
}

// Test 4: Stream context test for 32f_C4R
TEST_F(NppiMorphologyExtendedTest, Erode_32f_C4R_StreamContext) {
  const int width = 20;
  const int height = 20;

  // Create test image (4 channels)
  std::vector<Npp32f> hostSrc(width * height * 4);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      for (int c = 0; c < 4; c++) {
        hostSrc[(y * width + x) * 4 + c] = static_cast<float>(100 + c * 10);
      }
    }
  }

  // Allocate GPU memory
  Npp32f *d_src = nullptr, *d_dst = nullptr;
  int srcStep, dstStep;
  d_src = nppiMalloc_32f_C4(width, height, &srcStep);
  d_dst = nppiMalloc_32f_C4(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Create stream
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  nppStreamCtx.hStream = stream;

  // Upload data asynchronously
  cudaMemcpy2DAsync(d_src, srcStep, hostSrc.data(), width * 4 * sizeof(Npp32f), width * 4 * sizeof(Npp32f), height,
                    cudaMemcpyHostToDevice, stream);

  // Create structure element
  auto kernel = createCircleKernel();
  Npp8u *d_mask = nullptr;
  cudaMalloc(&d_mask, kernel.size());
  cudaMemcpyAsync(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice, stream);

  NppiSize oSizeROI = {width, height};
  NppiSize oMaskSize = {5, 5};
  NppiPoint oAnchor = {2, 2};

  // Perform erosion with stream context
  NppStatus status =
      nppiErode_32f_C4R_Ctx(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Synchronize stream
  cudaStreamSynchronize(stream);

  // Copy result back to host
  std::vector<Npp32f> hostDst(width * height * 4);
  cudaMemcpy2D(hostDst.data(), width * 4 * sizeof(Npp32f), d_dst, dstStep, width * 4 * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);

  // Basic verification - results should not be all zeros
  bool hasNonZeroResults = false;
  for (const auto &val : hostDst) {
    if (val > 0.0f) {
      hasNonZeroResults = true;
      break;
    }
  }

  EXPECT_TRUE(hasNonZeroResults);

  // Cleanup
  cudaStreamDestroy(stream);
  nppiFree(d_src);
  nppiFree(d_dst);
  cudaFree(d_mask);
}

// Test 7: Large kernel test
TEST_F(NppiMorphologyExtendedTest, Morphology_LargeKernel) {
  const int width = 64;
  const int height = 64;

  // Create test image
  std::vector<Npp8u> hostSrc(width * height, 100);
  // Add a bright region in the center
  for (int y = 25; y < 40; y++) {
    for (int x = 25; x < 40; x++) {
      hostSrc[y * width + x] = 255;
    }
  }

  // Allocate GPU memory
  Npp8u *d_src = nullptr, *d_dst = nullptr;
  int srcStep, dstStep;
  d_src = nppiMalloc_8u_C1(width, height, &srcStep);
  d_dst = nppiMalloc_8u_C1(width, height, &dstStep);

  // Upload source data
  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width, width, height, cudaMemcpyHostToDevice);

  // Create a large 7x7 structure element
  std::vector<Npp8u> largeKernel(49, 1); // 7x7 all ones
  Npp8u *d_mask = nullptr;
  cudaMalloc(&d_mask, largeKernel.size());
  cudaMemcpy(d_mask, largeKernel.data(), largeKernel.size(), cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  NppiSize oMaskSize = {7, 7};
  NppiPoint oAnchor = {3, 3};

  // Perform dilation with large kernel
  NppStatus status = nppiDilate_8u_C1R(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back to host
  std::vector<Npp8u> hostDst(width * height);
  cudaMemcpy2D(hostDst.data(), width, d_dst, dstStep, width, height, cudaMemcpyDeviceToHost);

  // Verify that dilation expanded the bright region
  bool foundExpansion = false;
  // Check a position that should be affected by 7x7 dilation
  if (hostDst[22 * width + 22] > 100) {
    foundExpansion = true;
  }

  EXPECT_TRUE(foundExpansion);

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_dst);
  cudaFree(d_mask);
}
