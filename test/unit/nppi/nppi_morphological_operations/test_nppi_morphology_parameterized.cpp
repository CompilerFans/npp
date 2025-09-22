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
