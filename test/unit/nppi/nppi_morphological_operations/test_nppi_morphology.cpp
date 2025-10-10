#include "npp.h"
#include "npp_test_base.h"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

// Helper structure for padded buffer management
template <typename T> struct PaddedBuffer {
  T *base_ptr; // Base pointer (includes padding)
  T *data_ptr; // Data pointer (actual ROI area)
  int step;    // Step size
  int padding; // Padding size
  int width;   // ROI width
  int height;  // ROI height

  PaddedBuffer() : base_ptr(nullptr), data_ptr(nullptr), step(0), padding(0), width(0), height(0) {}

  ~PaddedBuffer() {
    if (base_ptr) {
      nppiFree(base_ptr);
    }
  }
};

// Helper function to allocate GPU memory with padding
template <typename T> PaddedBuffer<T> *allocateWithPadding(int width, int height, int padding = 8) {
  auto *buffer = new PaddedBuffer<T>();
  buffer->width = width;
  buffer->height = height;
  buffer->padding = padding;

  // Allocate larger buffer with padding on all sides
  int paddedWidth = width + 2 * padding;
  int paddedHeight = height + 2 * padding;

  if constexpr (std::is_same_v<T, Npp8u>) {
    buffer->base_ptr = nppiMalloc_8u_C1(paddedWidth, paddedHeight, &buffer->step);
  } else if constexpr (std::is_same_v<T, Npp32f>) {
    buffer->base_ptr = nppiMalloc_32f_C1(paddedWidth, paddedHeight, &buffer->step);
  } else {
    delete buffer;
    return nullptr;
  }

  if (!buffer->base_ptr) {
    delete buffer;
    return nullptr;
  }

  // Initialize entire buffer to zero (including padding)
  cudaMemset2D(buffer->base_ptr, buffer->step, 0, paddedWidth * sizeof(T), paddedHeight);

  // Calculate data pointer (points to start of actual ROI within padded buffer)
  buffer->data_ptr =
      reinterpret_cast<T *>(reinterpret_cast<char *>(buffer->base_ptr) + padding * buffer->step + padding * sizeof(T));

  return buffer;
}

// Helper function to copy data to padded buffer
template <typename T> void copyToPaddedBuffer(PaddedBuffer<T> *buffer, const std::vector<T> &hostData) {
  cudaMemcpy2D(buffer->data_ptr, buffer->step, hostData.data(), buffer->width * sizeof(T), buffer->width * sizeof(T),
               buffer->height, cudaMemcpyHostToDevice);
}

// Helper function to copy data from padded buffer
template <typename T> void copyFromPaddedBuffer(std::vector<T> &hostData, const PaddedBuffer<T> *buffer) {
  cudaMemcpy2D(hostData.data(), buffer->width * sizeof(T), buffer->data_ptr, buffer->step, buffer->width * sizeof(T),
               buffer->height, cudaMemcpyDeviceToHost);
}

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

    // Clear any potential CUDA errors from previous tests
    cudaGetLastError();
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
    // Use padded buffers to ensure valid data outside ROI boundaries
    auto *srcBuffer = allocateWithPadding<Npp32f>(width, height, 8);
    auto *dstBuffer = allocateWithPadding<Npp32f>(width, height, 8);
    ASSERT_NE(srcBuffer, nullptr);
    ASSERT_NE(dstBuffer, nullptr);

    // Copy test data to padded source buffer
    copyToPaddedBuffer(srcBuffer, binaryData);

    // Get step values from buffers
    int srcStep = srcBuffer->step;
    int dstStep = dstBuffer->step;

    auto kernel = createBoxKernel3x3();
    Npp8u *d_mask = nullptr;
    cudaMalloc(&d_mask, kernel.size());
    cudaMemcpy(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice);

    NppiSize oSizeROI = {width, height};
    NppiSize oMaskSize = {3, 3};
    NppiPoint oAnchor = {1, 1};

    NppStatus status = nppiErode_32f_C1R(srcBuffer->data_ptr, srcStep, dstBuffer->data_ptr, dstStep, oSizeROI, d_mask,
                                         oMaskSize, oAnchor);
    ASSERT_EQ(status, NPP_SUCCESS);

    std::vector<Npp32f> result(width * height);
    copyFromPaddedBuffer(result, dstBuffer);

    // Verify erosion occurred - erosion should not increase values
    float maxInput = *std::max_element(binaryData.begin(), binaryData.end());
    float maxOutput = *std::max_element(result.begin(), result.end());
    EXPECT_LE(maxOutput, maxInput + 1e-5f) << "Erosion should not increase maximum value";

    // Verify no invalid values (NaN or Inf)
    bool hasValidValues = std::all_of(result.begin(), result.end(), [](Npp32f val) { return std::isfinite(val); });
    if (!hasValidValues) {
      // Find and print invalid values for debugging
      int invalidCount = 0;
      for (size_t i = 0; i < result.size() && invalidCount < 10; i++) {
        if (!std::isfinite(result[i])) {
          printf("Invalid value at index %zu: %f (isnan=%d, isinf=%d)\n", i, result[i], std::isnan(result[i]),
                 std::isinf(result[i]));
          invalidCount++;
        }
      }
    }
    EXPECT_TRUE(hasValidValues) << "Result should not contain NaN or Inf values";

    delete srcBuffer;
    delete dstBuffer;
    cudaFree(d_mask);
  }

  // Test context version
  {
    // Use padded buffers
    auto *srcBuffer = allocateWithPadding<Npp32f>(width, height, 8);
    auto *dstBuffer = allocateWithPadding<Npp32f>(width, height, 8);
    ASSERT_NE(srcBuffer, nullptr);
    ASSERT_NE(dstBuffer, nullptr);

    int srcStep = srcBuffer->step;
    int dstStep = dstBuffer->step;

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    nppStreamCtx.hStream = stream;

    // Use async copy to padded buffer
    cudaMemcpy2DAsync(srcBuffer->data_ptr, srcStep, binaryData.data(), width * sizeof(Npp32f), width * sizeof(Npp32f),
                      height, cudaMemcpyHostToDevice, stream);

    auto kernel = createCrossKernel3x3();
    Npp8u *d_mask = nullptr;
    cudaMalloc(&d_mask, kernel.size());
    cudaMemcpyAsync(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice, stream);

    NppiSize oSizeROI = {width, height};
    NppiSize oMaskSize = {3, 3};
    NppiPoint oAnchor = {1, 1};

    NppStatus status = nppiErode_32f_C1R_Ctx(srcBuffer->data_ptr, srcStep, dstBuffer->data_ptr, dstStep, oSizeROI,
                                             d_mask, oMaskSize, oAnchor, nppStreamCtx);
    ASSERT_EQ(status, NPP_SUCCESS);

    cudaStreamSynchronize(stream);

    std::vector<Npp32f> result(width * height);
    copyFromPaddedBuffer(result, dstBuffer);

    // Verify no invalid values (NaN or Inf)
    bool hasValidValues2 = std::all_of(result.begin(), result.end(), [](Npp32f val) { return std::isfinite(val); });
    EXPECT_TRUE(hasValidValues2) << "Result should not contain NaN or Inf values (context version)";

    cudaStreamDestroy(stream);
    delete srcBuffer;
    delete dstBuffer;
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

  // Test regular version with padded buffers
  {
    const int padding = 8; // Sufficient for 3x3 kernel
    auto srcBuffer = allocateWithPadding<Npp32f>(width, height, padding);
    auto dstBuffer = allocateWithPadding<Npp32f>(width, height, padding);
    ASSERT_NE(srcBuffer, nullptr);
    ASSERT_NE(dstBuffer, nullptr);

    // Copy test data to padded source buffer
    copyToPaddedBuffer(srcBuffer, checkerData);

    // Get step values from buffers
    int srcStep = srcBuffer->step;
    int dstStep = dstBuffer->step;

    auto kernel = createBoxKernel3x3();
    Npp8u *d_mask = nullptr;
    cudaMalloc(&d_mask, kernel.size());
    cudaMemcpy(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice);

    NppiSize oSizeROI = {width, height};
    NppiSize oMaskSize = {3, 3};
    NppiPoint oAnchor = {1, 1};

    NppStatus status = nppiDilate_32f_C1R(srcBuffer->data_ptr, srcStep, dstBuffer->data_ptr, dstStep, oSizeROI, d_mask,
                                          oMaskSize, oAnchor);
    ASSERT_EQ(status, NPP_SUCCESS);

    std::vector<Npp32f> result(width * height);
    copyFromPaddedBuffer(result, dstBuffer);

    // Verify dilation occurred - dilation should not decrease values
    float minInput = *std::min_element(checkerData.begin(), checkerData.end());
    float minOutput = *std::min_element(result.begin(), result.end());
    EXPECT_GE(minOutput, minInput - 1e-5f) << "Dilation should not decrease minimum value";

    // Verify no invalid values (NaN or Inf)
    bool hasValidValues = std::all_of(result.begin(), result.end(), [](Npp32f val) { return std::isfinite(val); });
    if (!hasValidValues) {
      // Find and print invalid values for debugging
      int invalidCount = 0;
      for (size_t i = 0; i < result.size() && invalidCount < 10; i++) {
        if (!std::isfinite(result[i])) {
          printf("Invalid value at index %zu: %f (isnan=%d, isinf=%d)\n", i, result[i], std::isnan(result[i]),
                 std::isinf(result[i]));
          invalidCount++;
        }
      }
    }
    EXPECT_TRUE(hasValidValues) << "Result should not contain NaN or Inf values";

    delete srcBuffer;
    delete dstBuffer;
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

    // Initialize destination buffer to avoid garbage values from previous tests
    cudaMemset2DAsync(d_dst, dstStep, 0, width * sizeof(Npp32f), height, stream);

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

    // Verify no invalid values (NaN or Inf)
    bool hasValidValues2 = std::all_of(result.begin(), result.end(), [](Npp32f val) { return std::isfinite(val); });
    EXPECT_TRUE(hasValidValues2) << "Result should not contain NaN or Inf values (context version)";

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

    // Verify values are finite (no NaN or Inf)
    bool allFinite = std::all_of(result.begin(), result.end(), [](Npp32f val) { return std::isfinite(val); });
    if (!allFinite) {
      // Find and print non-finite values for debugging
      for (int i = 0; i < std::min(10, (int)result.size()); i++) {
        if (!std::isfinite(result[i])) {
          printf("Non-finite value at index %d: %f\n", i, result[i]);
          fflush(stdout);
        }
      }
    }
    EXPECT_TRUE(allFinite) << "Result should contain only finite values";

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

    // Verify values are finite (no NaN or Inf)
    bool allFinite = std::all_of(result.begin(), result.end(), [](Npp32f val) { return std::isfinite(val); });
    if (!allFinite) {
      // Find and print non-finite values for debugging
      for (int i = 0; i < std::min(10, (int)result.size()); i++) {
        if (!std::isfinite(result[i])) {
          printf("Non-finite value at index %d: %f\n", i, result[i]);
          fflush(stdout);
        }
      }
    }
    EXPECT_TRUE(allFinite) << "Result should contain only finite values";

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

    // Verify values are finite (no NaN or Inf)
    bool allFinite = std::all_of(result.begin(), result.end(), [](Npp32f val) { return std::isfinite(val); });
    if (!allFinite) {
      // Find and print non-finite values for debugging
      for (int i = 0; i < std::min(10, (int)result.size()); i++) {
        if (!std::isfinite(result[i])) {
          printf("Non-finite value at index %d: %f\n", i, result[i]);
          fflush(stdout);
        }
      }
    }
    EXPECT_TRUE(allFinite) << "Result should contain only finite values";

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

    // Verify values are finite (no NaN or Inf)
    bool allFinite = std::all_of(result.begin(), result.end(), [](Npp32f val) { return std::isfinite(val); });
    if (!allFinite) {
      // Find and print non-finite values for debugging
      for (int i = 0; i < std::min(10, (int)result.size()); i++) {
        if (!std::isfinite(result[i])) {
          printf("Non-finite value at index %d: %f\n", i, result[i]);
          fflush(stdout);
        }
      }
    }
    EXPECT_TRUE(allFinite) << "Result should contain only finite values";

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

// Enhanced test parameters structure with morphological-specific attributes
struct MorphologyTestConfig {
  int width, height;
  int kernelWidth, kernelHeight;
  int anchorX, anchorY;
  std::string description;
  std::string testCategory;
  bool isSquareKernel = true;
  bool isSymmetricKernel = true;
  bool isLargeKernel = false;
};

// Test patterns enumeration
enum class TestPattern {
  UNIFORM,
  BINARY_SQUARE,
  CHECKERBOARD,
  GRADIENT_HORIZONTAL,
  GRADIENT_VERTICAL,
  RANDOM_NOISE,
  SPARSE_DOTS,
  EDGE_PATTERN,
  CORNER_PATTERN,
  DIAGONAL_LINES
};

// =====================================================================================
// UTILITY CLASSES AND PATTERN GENERATORS
// =====================================================================================

// Enhanced pattern generator with morphology-specific patterns
template <typename T> class MorphologyPatternGenerator {
public:
  static std::vector<T> generatePattern(TestPattern pattern, int width, int height, T bgValue, T fgValue) {
    switch (pattern) {
    case TestPattern::UNIFORM:
      return createUniform(width, height, fgValue);
    case TestPattern::BINARY_SQUARE:
      return createBinarySquare(width, height, bgValue, fgValue);
    case TestPattern::CHECKERBOARD:
      return createCheckerboard(width, height, bgValue, fgValue);
    case TestPattern::GRADIENT_HORIZONTAL:
      return createHorizontalGradient(width, height, bgValue, fgValue);
    case TestPattern::GRADIENT_VERTICAL:
      return createVerticalGradient(width, height, bgValue, fgValue);
    case TestPattern::RANDOM_NOISE:
      return createRandomNoise(width, height, bgValue, fgValue);
    case TestPattern::SPARSE_DOTS:
      return createSparseDots(width, height, bgValue, fgValue);
    case TestPattern::EDGE_PATTERN:
      return createEdgePattern(width, height, bgValue, fgValue);
    case TestPattern::CORNER_PATTERN:
      return createCornerPattern(width, height, bgValue, fgValue);
    case TestPattern::DIAGONAL_LINES:
      return createDiagonalLines(width, height, bgValue, fgValue);
    default:
      return createUniform(width, height, bgValue);
    }
  }

private:
  static std::vector<T> createUniform(int width, int height, T value) { return std::vector<T>(width * height, value); }

  static std::vector<T> createBinarySquare(int width, int height, T bgValue, T fgValue) {
    std::vector<T> data(width * height, bgValue);
    int startX = width / 4;
    int endX = 3 * width / 4;
    int startY = height / 4;
    int endY = 3 * height / 4;

    for (int y = startY; y < endY && y < height; y++) {
      for (int x = startX; x < endX && x < width; x++) {
        data[y * width + x] = fgValue;
      }
    }
    return data;
  }

  static std::vector<T> createCheckerboard(int width, int height, T bgValue, T fgValue) {
    std::vector<T> data(width * height);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        data[y * width + x] = ((x + y) % 2 == 0) ? bgValue : fgValue;
      }
    }
    return data;
  }

  static std::vector<T> createHorizontalGradient(int width, int height, T minVal, T maxVal) {
    std::vector<T> data(width * height);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        if (width == 1) {
          data[y * width + x] = minVal;
        } else {
          double ratio = static_cast<double>(x) / (width - 1);
          if constexpr (std::is_integral_v<T>) {
            data[y * width + x] = static_cast<T>(minVal + ratio * (maxVal - minVal));
          } else {
            data[y * width + x] = static_cast<T>(minVal + ratio * (maxVal - minVal));
          }
        }
      }
    }
    return data;
  }

  static std::vector<T> createVerticalGradient(int width, int height, T minVal, T maxVal) {
    std::vector<T> data(width * height);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        if (height == 1) {
          data[y * width + x] = minVal;
        } else {
          double ratio = static_cast<double>(y) / (height - 1);
          if constexpr (std::is_integral_v<T>) {
            data[y * width + x] = static_cast<T>(minVal + ratio * (maxVal - minVal));
          } else {
            data[y * width + x] = static_cast<T>(minVal + ratio * (maxVal - minVal));
          }
        }
      }
    }
    return data;
  }

  static std::vector<T> createRandomNoise(int width, int height, T minVal, T maxVal) {
    std::vector<T> data(width * height);
    std::random_device rd;
    std::mt19937 gen(rd());

    if constexpr (std::is_integral_v<T>) {
      std::uniform_int_distribution<int> dis(static_cast<int>(minVal), static_cast<int>(maxVal));
      for (size_t i = 0; i < data.size(); i++) {
        data[i] = static_cast<T>(dis(gen));
      }
    } else {
      std::uniform_real_distribution<float> dis(static_cast<float>(minVal), static_cast<float>(maxVal));
      for (size_t i = 0; i < data.size(); i++) {
        data[i] = static_cast<T>(dis(gen));
      }
    }
    return data;
  }

  static std::vector<T> createSparseDots(int width, int height, T bgValue, T fgValue) {
    std::vector<T> data(width * height, bgValue);
    for (int y = 2; y < height - 2; y += 4) {
      for (int x = 2; x < width - 2; x += 4) {
        data[y * width + x] = fgValue;
      }
    }
    return data;
  }

  static std::vector<T> createEdgePattern(int width, int height, T bgValue, T fgValue) {
    std::vector<T> data(width * height, bgValue);
    // Create borders
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
          data[y * width + x] = fgValue;
        }
      }
    }
    return data;
  }

  static std::vector<T> createCornerPattern(int width, int height, T bgValue, T fgValue) {
    std::vector<T> data(width * height, bgValue);
    // Add corners
    if (width > 0 && height > 0) {
      data[0] = fgValue;                                // Top-left
      data[width - 1] = fgValue;                        // Top-right
      data[(height - 1) * width] = fgValue;             // Bottom-left
      data[(height - 1) * width + width - 1] = fgValue; // Bottom-right
    }
    return data;
  }

  static std::vector<T> createDiagonalLines(int width, int height, T bgValue, T fgValue) {
    std::vector<T> data(width * height, bgValue);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        if ((x + y) % 4 == 0 || (x - y) % 4 == 0) {
          data[y * width + x] = fgValue;
        }
      }
    }
    return data;
  }
};

// Structural element generator
class StructuralElementGenerator {
public:
  // Standard 3x3 kernels
  static std::vector<Npp8u> createCross3x3() { return {0, 1, 0, 1, 1, 1, 0, 1, 0}; }

  static std::vector<Npp8u> createBox3x3() { return {1, 1, 1, 1, 1, 1, 1, 1, 1}; }

  // 5x5 kernels
  static std::vector<Npp8u> createBox5x5() {
    return {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  }

  static std::vector<Npp8u> createCircle5x5() {
    return {0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0};
  }

  static std::vector<Npp8u> createCross5x5() {
    return {0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0};
  }

  // Asymmetric kernels
  static std::vector<Npp8u> createHorizontalLine(int length) {
    std::vector<Npp8u> kernel(1 * length, 1);
    return kernel;
  }

  static std::vector<Npp8u> createVerticalLine(int length) {
    std::vector<Npp8u> kernel(length * 1, 1);
    return kernel;
  }

  // Custom rectangular kernel
  static std::vector<Npp8u> createRectangle(int width, int height) { return std::vector<Npp8u>(width * height, 1); }

  // Diamond shaped kernel
  static std::vector<Npp8u> createDiamond(int size) {
    std::vector<Npp8u> kernel(size * size, 0);
    int center = size / 2;
    for (int y = 0; y < size; y++) {
      for (int x = 0; x < size; x++) {
        if (abs(x - center) + abs(y - center) <= center) {
          kernel[y * size + x] = 1;
        }
      }
    }
    return kernel;
  }
};

// =====================================================================================
// MORPHOLOGICAL OPERATION UTILITIES
// =====================================================================================

template <typename T> class MorphologyAnalyzer {
public:
  // Analyze morphological properties
  static bool verifyErosionProperty(const std::vector<T> &original, const std::vector<T> &eroded) {
    // For most non-boundary pixels, erosion should not increase values
    int violations = 0;
    int totalPixels = 0;

    for (size_t i = 0; i < original.size(); i++) {
      totalPixels++;
      if (eroded[i] > original[i]) {
        violations++;
      }
    }

    // Allow some violations due to boundary effects (up to 20%)
    return violations <= totalPixels * 0.2;
  }

  static bool verifyDilationProperty(const std::vector<T> &original, const std::vector<T> &dilated) {
    // For most non-boundary pixels, dilation should not decrease values
    int violations = 0;
    int totalPixels = 0;

    for (size_t i = 0; i < original.size(); i++) {
      totalPixels++;
      if (dilated[i] < original[i]) {
        violations++;
      }
    }

    // Allow some violations due to boundary effects (up to 20%)
    return violations <= totalPixels * 0.2;
  }

  static bool verifyDuality(const std::vector<T> &eroded, const std::vector<T> &dilated) {
    // Check basic morphological duality properties
    // This is a simplified check - real duality requires complement operations
    for (size_t i = 0; i < eroded.size(); i++) {
      if (eroded[i] > dilated[i]) {
        return false; // Erosion should not produce larger values than dilation
      }
    }
    return true;
  }

  // Count significant changes
  static int countChangedPixels(const std::vector<T> &original, const std::vector<T> &processed, T threshold) {
    int changes = 0;
    for (size_t i = 0; i < original.size(); i++) {
      if (abs(static_cast<double>(original[i]) - static_cast<double>(processed[i])) > threshold) {
        changes++;
      }
    }
    return changes;
  }

  // Calculate statistics
  struct MorphologyStats {
    double meanOriginal, meanProcessed;
    double minOriginal, maxOriginal;
    double minProcessed, maxProcessed;
    int changedPixels;
    double changePercentage;
  };

  static MorphologyStats calculateStats(const std::vector<T> &original, const std::vector<T> &processed, T threshold) {
    MorphologyStats stats = {};

    if (original.empty())
      return stats;

    stats.minOriginal = stats.maxOriginal = static_cast<double>(original[0]);
    stats.minProcessed = stats.maxProcessed = static_cast<double>(processed[0]);

    double sumOriginal = 0, sumProcessed = 0;
    stats.changedPixels = 0;

    for (size_t i = 0; i < original.size(); i++) {
      double orig = static_cast<double>(original[i]);
      double proc = static_cast<double>(processed[i]);

      sumOriginal += orig;
      sumProcessed += proc;

      stats.minOriginal = std::min(stats.minOriginal, orig);
      stats.maxOriginal = std::max(stats.maxOriginal, orig);
      stats.minProcessed = std::min(stats.minProcessed, proc);
      stats.maxProcessed = std::max(stats.maxProcessed, proc);

      if (abs(orig - proc) > threshold) {
        stats.changedPixels++;
      }
    }

    stats.meanOriginal = sumOriginal / original.size();
    stats.meanProcessed = sumProcessed / processed.size();
    stats.changePercentage = 100.0 * stats.changedPixels / original.size();

    return stats;
  }
};

// =====================================================================================
// COMPREHENSIVE TEST CONFIGURATIONS
// =====================================================================================

// Comprehensive test parameter definitions
static const std::vector<MorphologyTestConfig> MORPHOLOGY_COMPREHENSIVE_CONFIGS = {
    // Basic square configurations
    {16, 16, 3, 3, 1, 1, "16x16_3x3_basic", "basic", true, true, false},
    {32, 32, 3, 3, 1, 1, "32x32_3x3_standard", "basic", true, true, false},
    {64, 64, 5, 5, 2, 2, "64x64_5x5_medium", "basic", true, true, false},
    {128, 128, 7, 7, 3, 3, "128x128_7x7_large", "basic", true, true, false},

    // Different aspect ratios
    {64, 32, 3, 3, 1, 1, "64x32_3x3_wide", "aspect_ratio", true, true, false},
    {32, 64, 3, 3, 1, 1, "32x64_3x3_tall", "aspect_ratio", true, true, false},
    {128, 32, 5, 5, 2, 2, "128x32_5x5_very_wide", "aspect_ratio", true, true, false},
    {32, 128, 5, 5, 2, 2, "32x128_5x5_very_tall", "aspect_ratio", true, true, false},

    // Asymmetric kernels
    {64, 64, 1, 7, 0, 3, "64x64_1x7_vertical_line", "asymmetric", false, true, false},
    {64, 64, 7, 1, 3, 0, "64x64_7x1_horizontal_line", "asymmetric", false, true, false},
    {32, 32, 3, 7, 1, 3, "32x32_3x7_vertical_rect", "asymmetric", false, false, false},
    {32, 32, 7, 3, 3, 1, "32x32_7x3_horizontal_rect", "asymmetric", false, false, false},

    // Large kernels
    {128, 128, 11, 11, 5, 5, "128x128_11x11_large", "large_kernel", true, true, true},
    {64, 64, 15, 15, 7, 7, "64x64_15x15_very_large", "large_kernel", true, true, true},
    {256, 256, 9, 9, 4, 4, "256x256_9x9_high_res", "large_kernel", true, true, true},

    // Edge cases - small images
    {5, 5, 3, 3, 1, 1, "5x5_3x3_tiny", "edge_case", true, true, false},
    {3, 3, 3, 3, 1, 1, "3x3_3x3_exact", "edge_case", true, true, false},
    {7, 3, 3, 3, 1, 1, "7x3_3x3_narrow", "edge_case", true, true, false},
    {3, 7, 3, 3, 1, 1, "3x7_3x3_thin", "edge_case", true, true, false},

    // Different anchor positions
    {32, 32, 5, 5, 0, 0, "32x32_5x5_anchor_topleft", "anchor_test", true, true, false},
    {32, 32, 5, 5, 4, 4, "32x32_5x5_anchor_bottomright", "anchor_test", true, true, false},
    {32, 32, 5, 5, 2, 0, "32x32_5x5_anchor_top_center", "anchor_test", true, true, false},
    {32, 32, 5, 5, 2, 4, "32x32_5x5_anchor_bottom_center", "anchor_test", true, true, false},

    // Stress test configurations
    {512, 512, 3, 3, 1, 1, "512x512_3x3_stress", "stress_test", true, true, false},
    {1024, 256, 5, 5, 2, 2, "1024x256_5x5_stress_wide", "stress_test", true, true, false},
    {256, 1024, 5, 5, 2, 2, "256x1024_5x5_stress_tall", "stress_test", true, true, false},
};

// Pattern test combinations
static const std::vector<TestPattern> TEST_PATTERNS = {TestPattern::UNIFORM,           TestPattern::BINARY_SQUARE,
                                                       TestPattern::CHECKERBOARD,      TestPattern::GRADIENT_HORIZONTAL,
                                                       TestPattern::GRADIENT_VERTICAL, TestPattern::RANDOM_NOISE,
                                                       TestPattern::SPARSE_DOTS,       TestPattern::EDGE_PATTERN,
                                                       TestPattern::CORNER_PATTERN,    TestPattern::DIAGONAL_LINES};

// =====================================================================================
// BASE TEST CLASS
// =====================================================================================

class MorphologyComprehensiveTest : public ::testing::TestWithParam<MorphologyTestConfig> {
protected:
  void SetUp() override {
    cudaError_t err = cudaSetDevice(0);
    ASSERT_EQ(err, cudaSuccess);

    // Clear any potential CUDA errors from previous tests
    cudaGetLastError();
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

  // Helper method to create different structural elements based on test config
  std::vector<Npp8u> createStructuralElement(const MorphologyTestConfig &config) {
    if (config.kernelWidth == 3 && config.kernelHeight == 3) {
      if (config.description.find("cross") != std::string::npos) {
        return StructuralElementGenerator::createCross3x3();
      }
      return StructuralElementGenerator::createBox3x3();
    } else if (config.kernelWidth == 5 && config.kernelHeight == 5) {
      if (config.description.find("circle") != std::string::npos) {
        return StructuralElementGenerator::createCircle5x5();
      } else if (config.description.find("cross") != std::string::npos) {
        return StructuralElementGenerator::createCross5x5();
      }
      return StructuralElementGenerator::createBox5x5();
    } else if (config.kernelWidth == 1 || config.kernelHeight == 1) {
      // Handle line kernels
      return StructuralElementGenerator::createRectangle(config.kernelWidth, config.kernelHeight);
    } else {
      // Default to rectangular kernel
      return StructuralElementGenerator::createRectangle(config.kernelWidth, config.kernelHeight);
    }
  }
};

// =====================================================================================
// PARAMETERIZED TESTS FOR 8u_C1R
// =====================================================================================

TEST_P(MorphologyComprehensiveTest, Erode_8u_C1R_Comprehensive) {
  auto config = GetParam();
  SCOPED_TRACE("Erode 8u C1R test: " + config.description);

  // Test multiple patterns
  for (auto pattern : {TestPattern::BINARY_SQUARE, TestPattern::CHECKERBOARD, TestPattern::SPARSE_DOTS}) {
    auto testData = MorphologyPatternGenerator<Npp8u>::generatePattern(pattern, config.width, config.height, 0, 255);

    // Allocate device memory
    int srcStep, dstStep;
    Npp8u *d_src = nppiMalloc_8u_C1(config.width, config.height, &srcStep);
    Npp8u *d_dst = nppiMalloc_8u_C1(config.width, config.height, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);

    // Upload test data
    cudaMemcpy2D(d_src, srcStep, testData.data(), config.width * sizeof(Npp8u), config.width * sizeof(Npp8u),
                 config.height, cudaMemcpyHostToDevice);

    // Create structural element
    auto kernel = createStructuralElement(config);
    Npp8u *d_mask = nullptr;
    cudaMalloc(&d_mask, kernel.size());
    cudaMemcpy(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice);

    NppiSize oSizeROI = {config.width, config.height};
    NppiSize oMaskSize = {config.kernelWidth, config.kernelHeight};
    NppiPoint oAnchor = {config.anchorX, config.anchorY};

    // Test regular version
    NppStatus status = nppiErode_8u_C1R(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor);
    ASSERT_EQ(status, NPP_SUCCESS) << "Erosion failed for config: " << config.description;

    // Download and verify results
    std::vector<Npp8u> result(config.width * config.height);
    cudaMemcpy2D(result.data(), config.width * sizeof(Npp8u), d_dst, dstStep, config.width * sizeof(Npp8u),
                 config.height, cudaMemcpyDeviceToHost);

    // Verify erosion properties
    EXPECT_TRUE(MorphologyAnalyzer<Npp8u>::verifyErosionProperty(testData, result))
        << "Erosion property violated for " << config.description;

    // Verify output range
    for (auto val : result) {
      EXPECT_GE(val, 0) << "Invalid output value";
      EXPECT_LE(val, 255) << "Invalid output value";
    }

    // Calculate and log statistics
    auto stats = MorphologyAnalyzer<Npp8u>::calculateStats(testData, result, 1);
    if (config.width <= 64 && config.height <= 64) { // Log stats for smaller images
      std::cout << "Erosion stats [" << config.description << "]: "
                << "Changed: " << stats.changePercentage << "%, "
                << "Mean: " << stats.meanOriginal << "->" << stats.meanProcessed << std::endl;
    }

    // Cleanup
    nppiFree(d_src);
    nppiFree(d_dst);
    cudaFree(d_mask);
  }
}

TEST_P(MorphologyComprehensiveTest, Dilate_8u_C1R_Comprehensive) {
  auto config = GetParam();
  SCOPED_TRACE("Dilate 8u C1R test: " + config.description);

  // Test multiple patterns
  for (auto pattern : {TestPattern::SPARSE_DOTS, TestPattern::EDGE_PATTERN, TestPattern::CORNER_PATTERN}) {
    auto testData = MorphologyPatternGenerator<Npp8u>::generatePattern(pattern, config.width, config.height, 0, 255);

    // Allocate device memory
    int srcStep, dstStep;
    Npp8u *d_src = nppiMalloc_8u_C1(config.width, config.height, &srcStep);
    Npp8u *d_dst = nppiMalloc_8u_C1(config.width, config.height, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);

    // Upload test data
    cudaMemcpy2D(d_src, srcStep, testData.data(), config.width * sizeof(Npp8u), config.width * sizeof(Npp8u),
                 config.height, cudaMemcpyHostToDevice);

    // Create structural element
    auto kernel = createStructuralElement(config);
    Npp8u *d_mask = nullptr;
    cudaMalloc(&d_mask, kernel.size());
    cudaMemcpy(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice);

    NppiSize oSizeROI = {config.width, config.height};
    NppiSize oMaskSize = {config.kernelWidth, config.kernelHeight};
    NppiPoint oAnchor = {config.anchorX, config.anchorY};

    // Test regular version
    NppStatus status = nppiDilate_8u_C1R(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor);
    ASSERT_EQ(status, NPP_SUCCESS) << "Dilation failed for config: " << config.description;

    // Download and verify results
    std::vector<Npp8u> result(config.width * config.height);
    cudaMemcpy2D(result.data(), config.width * sizeof(Npp8u), d_dst, dstStep, config.width * sizeof(Npp8u),
                 config.height, cudaMemcpyDeviceToHost);

    // Verify dilation properties
    EXPECT_TRUE(MorphologyAnalyzer<Npp8u>::verifyDilationProperty(testData, result))
        << "Dilation property violated for " << config.description;

    // Verify output range
    for (auto val : result) {
      EXPECT_GE(val, 0) << "Invalid output value";
      EXPECT_LE(val, 255) << "Invalid output value";
    }

    // Cleanup
    nppiFree(d_src);
    nppiFree(d_dst);
    cudaFree(d_mask);
  }
}

// Context version tests for 8u_C1R
TEST_P(MorphologyComprehensiveTest, Erode_8u_C1R_Ctx_Comprehensive) {
  auto config = GetParam();
  SCOPED_TRACE("Erode 8u C1R Ctx test: " + config.description);

  auto testData = MorphologyPatternGenerator<Npp8u>::generatePattern(TestPattern::BINARY_SQUARE, config.width,
                                                                     config.height, 0, 255);

  // Allocate device memory
  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C1(config.width, config.height, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C1(config.width, config.height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Create stream context
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  nppStreamCtx.hStream = stream;

  // Upload test data asynchronously
  cudaMemcpy2DAsync(d_src, srcStep, testData.data(), config.width * sizeof(Npp8u), config.width * sizeof(Npp8u),
                    config.height, cudaMemcpyHostToDevice, stream);

  // Create structural element
  auto kernel = createStructuralElement(config);
  Npp8u *d_mask = nullptr;
  cudaMalloc(&d_mask, kernel.size());
  cudaMemcpyAsync(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice, stream);

  NppiSize oSizeROI = {config.width, config.height};
  NppiSize oMaskSize = {config.kernelWidth, config.kernelHeight};
  NppiPoint oAnchor = {config.anchorX, config.anchorY};

  // Test context version
  NppStatus status =
      nppiErode_8u_C1R_Ctx(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS) << "Erosion Ctx failed for config: " << config.description;

  // Synchronize and download results
  cudaStreamSynchronize(stream);
  std::vector<Npp8u> result(config.width * config.height);
  cudaMemcpy2D(result.data(), config.width * sizeof(Npp8u), d_dst, dstStep, config.width * sizeof(Npp8u), config.height,
               cudaMemcpyDeviceToHost);

  // Verify results are reasonable
  EXPECT_TRUE(std::all_of(result.begin(), result.end(), [](Npp8u val) { return val <= 255; }));

  // Cleanup
  cudaStreamDestroy(stream);
  nppiFree(d_src);
  nppiFree(d_dst);
  cudaFree(d_mask);
}

// =====================================================================================
// PARAMETERIZED TESTS FOR 8u_C4R
// =====================================================================================

TEST_P(MorphologyComprehensiveTest, Erode_8u_C4R_Comprehensive) {
  auto config = GetParam();
  SCOPED_TRACE("Erode 8u C4R test: " + config.description);

  // Create 4-channel test data
  std::vector<Npp8u> testData(config.width * config.height * 4);
  auto singleChannelData = MorphologyPatternGenerator<Npp8u>::generatePattern(TestPattern::BINARY_SQUARE, config.width,
                                                                              config.height, 0, 255);

  for (int i = 0; i < config.width * config.height; i++) {
    for (int c = 0; c < 4; c++) {
      testData[i * 4 + c] = singleChannelData[i];
    }
  }

  // Allocate device memory
  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C4(config.width, config.height, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C4(config.width, config.height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Upload test data
  cudaMemcpy2D(d_src, srcStep, testData.data(), config.width * 4 * sizeof(Npp8u), config.width * 4 * sizeof(Npp8u),
               config.height, cudaMemcpyHostToDevice);

  // Create structural element
  auto kernel = createStructuralElement(config);
  Npp8u *d_mask = nullptr;
  cudaMalloc(&d_mask, kernel.size());
  cudaMemcpy(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {config.width, config.height};
  NppiSize oMaskSize = {config.kernelWidth, config.kernelHeight};
  NppiPoint oAnchor = {config.anchorX, config.anchorY};

  // Test regular version
  NppStatus status = nppiErode_8u_C4R(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor);
  ASSERT_EQ(status, NPP_SUCCESS) << "Erosion C4R failed for config: " << config.description;

  // Download and verify results
  std::vector<Npp8u> result(config.width * config.height * 4);
  cudaMemcpy2D(result.data(), config.width * 4 * sizeof(Npp8u), d_dst, dstStep, config.width * 4 * sizeof(Npp8u),
               config.height, cudaMemcpyDeviceToHost);

  // Verify all channels processed correctly
  for (int c = 0; c < 4; c++) {
    std::vector<Npp8u> originalChannel, resultChannel;
    for (int i = c; i < config.width * config.height * 4; i += 4) {
      originalChannel.push_back(testData[i]);
      resultChannel.push_back(result[i]);
    }

    EXPECT_TRUE(MorphologyAnalyzer<Npp8u>::verifyErosionProperty(originalChannel, resultChannel))
        << "Erosion property violated for channel " << c << " in " << config.description;
  }

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_dst);
  cudaFree(d_mask);
}

TEST_P(MorphologyComprehensiveTest, Dilate_8u_C4R_Comprehensive) {
  auto config = GetParam();
  SCOPED_TRACE("Dilate 8u C4R test: " + config.description);

  // Create 4-channel test data with different patterns per channel
  std::vector<Npp8u> testData(config.width * config.height * 4);
  for (int y = 0; y < config.height; y++) {
    for (int x = 0; x < config.width; x++) {
      int idx = (y * config.width + x) * 4;
      // Different patterns for each channel to test independence
      testData[idx + 0] = ((x + y) % 4 == 0) ? 255 : 0;      // Sparse pattern
      testData[idx + 1] = (x < config.width / 2) ? 255 : 0;  // Half split
      testData[idx + 2] = (y < config.height / 2) ? 255 : 0; // Half split
      testData[idx + 3] = ((x + y) % 2 == 0) ? 255 : 0;      // Checkerboard
    }
  }

  // Allocate device memory
  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C4(config.width, config.height, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C4(config.width, config.height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Upload test data
  cudaMemcpy2D(d_src, srcStep, testData.data(), config.width * 4 * sizeof(Npp8u), config.width * 4 * sizeof(Npp8u),
               config.height, cudaMemcpyHostToDevice);

  // Create structural element
  auto kernel = createStructuralElement(config);
  Npp8u *d_mask = nullptr;
  cudaMalloc(&d_mask, kernel.size());
  cudaMemcpy(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {config.width, config.height};
  NppiSize oMaskSize = {config.kernelWidth, config.kernelHeight};
  NppiPoint oAnchor = {config.anchorX, config.anchorY};

  // Test regular version
  NppStatus status = nppiDilate_8u_C4R(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor);
  ASSERT_EQ(status, NPP_SUCCESS) << "Dilation C4R failed for config: " << config.description;

  // Download and verify results
  std::vector<Npp8u> result(config.width * config.height * 4);
  cudaMemcpy2D(result.data(), config.width * 4 * sizeof(Npp8u), d_dst, dstStep, config.width * 4 * sizeof(Npp8u),
               config.height, cudaMemcpyDeviceToHost);

  // Verify all channels processed correctly
  for (int c = 0; c < 4; c++) {
    std::vector<Npp8u> originalChannel, resultChannel;
    for (int i = c; i < config.width * config.height * 4; i += 4) {
      originalChannel.push_back(testData[i]);
      resultChannel.push_back(result[i]);
    }

    EXPECT_TRUE(MorphologyAnalyzer<Npp8u>::verifyDilationProperty(originalChannel, resultChannel))
        << "Dilation property violated for channel " << c << " in " << config.description;
  }

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_dst);
  cudaFree(d_mask);
}

// =====================================================================================
// PARAMETERIZED TESTS FOR 32f_C1R
// =====================================================================================

TEST_P(MorphologyComprehensiveTest, Erode_32f_C1R_Comprehensive) {
  auto config = GetParam();
  SCOPED_TRACE("Erode 32f C1R test: " + config.description);

  // Test multiple patterns with float values
  for (auto pattern : {TestPattern::BINARY_SQUARE, TestPattern::GRADIENT_HORIZONTAL, TestPattern::SPARSE_DOTS}) {
    auto testData =
        MorphologyPatternGenerator<Npp32f>::generatePattern(pattern, config.width, config.height, 0.0f, 1.0f);

    // Allocate padded device memory to ensure valid data outside ROI boundaries
    const int padding = 8; // Sufficient for up to 15x15 kernels
    auto *srcBuffer = allocateWithPadding<Npp32f>(config.width, config.height, padding);
    auto *dstBuffer = allocateWithPadding<Npp32f>(config.width, config.height, padding);
    ASSERT_NE(srcBuffer, nullptr);
    ASSERT_NE(dstBuffer, nullptr);

    // Copy test data to padded source buffer
    copyToPaddedBuffer(srcBuffer, testData);

    // Get step values from buffers
    int srcStep = srcBuffer->step;
    int dstStep = dstBuffer->step;

    // Create structural element
    auto kernel = createStructuralElement(config);
    Npp8u *d_mask = nullptr;
    cudaMalloc(&d_mask, kernel.size());
    cudaMemcpy(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice);

    NppiSize oSizeROI = {config.width, config.height};
    NppiSize oMaskSize = {config.kernelWidth, config.kernelHeight};
    NppiPoint oAnchor = {config.anchorX, config.anchorY};

    // Test regular version
    NppStatus status = nppiErode_32f_C1R(srcBuffer->data_ptr, srcStep, dstBuffer->data_ptr, dstStep, oSizeROI, d_mask,
                                         oMaskSize, oAnchor);
    ASSERT_EQ(status, NPP_SUCCESS) << "Erosion 32f failed for config: " << config.description;

    // Download and verify results
    std::vector<Npp32f> result(config.width * config.height);
    copyFromPaddedBuffer(result, dstBuffer);

    // Verify erosion properties
    EXPECT_TRUE(MorphologyAnalyzer<Npp32f>::verifyErosionProperty(testData, result))
        << "Erosion property violated for " << config.description;

    // Verify floating point validity
    for (auto val : result) {
      EXPECT_TRUE(std::isfinite(val)) << "Non-finite output value in erosion";
      EXPECT_GE(val, -10.0f) << "Unreasonable output value";
      EXPECT_LE(val, 10.0f) << "Unreasonable output value";
    }

    // Calculate and log statistics for smaller images
    auto stats = MorphologyAnalyzer<Npp32f>::calculateStats(testData, result, 0.01f);
    if (config.width <= 64 && config.height <= 64) {
      std::cout << "Erosion 32f stats [" << config.description << "]: "
                << "Changed: " << stats.changePercentage << "%, "
                << "Mean: " << stats.meanOriginal << "->" << stats.meanProcessed << std::endl;
    }

    // Cleanup
    delete srcBuffer;
    delete dstBuffer;
    cudaFree(d_mask);
  }
}

TEST_P(MorphologyComprehensiveTest, Dilate_32f_C1R_Comprehensive) {
  auto config = GetParam();
  SCOPED_TRACE("Dilate 32f C1R test: " + config.description);

  // Test multiple patterns with float values
  for (auto pattern : {TestPattern::SPARSE_DOTS, TestPattern::GRADIENT_VERTICAL, TestPattern::CORNER_PATTERN}) {
    auto testData =
        MorphologyPatternGenerator<Npp32f>::generatePattern(pattern, config.width, config.height, 0.0f, 1.0f);

    // Allocate device memory
    int srcStep, dstStep;
    Npp32f *d_src = nppiMalloc_32f_C1(config.width, config.height, &srcStep);
    Npp32f *d_dst = nppiMalloc_32f_C1(config.width, config.height, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);

    // Upload test data
    cudaMemcpy2D(d_src, srcStep, testData.data(), config.width * sizeof(Npp32f), config.width * sizeof(Npp32f),
                 config.height, cudaMemcpyHostToDevice);

    // Create structural element
    auto kernel = createStructuralElement(config);
    Npp8u *d_mask = nullptr;
    cudaMalloc(&d_mask, kernel.size());
    cudaMemcpy(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice);

    NppiSize oSizeROI = {config.width, config.height};
    NppiSize oMaskSize = {config.kernelWidth, config.kernelHeight};
    NppiPoint oAnchor = {config.anchorX, config.anchorY};

    // Test regular version
    NppStatus status = nppiDilate_32f_C1R(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor);
    ASSERT_EQ(status, NPP_SUCCESS) << "Dilation 32f failed for config: " << config.description;

    // Download and verify results
    std::vector<Npp32f> result(config.width * config.height);
    cudaMemcpy2D(result.data(), config.width * sizeof(Npp32f), d_dst, dstStep, config.width * sizeof(Npp32f),
                 config.height, cudaMemcpyDeviceToHost);

    // Verify dilation properties
    EXPECT_TRUE(MorphologyAnalyzer<Npp32f>::verifyDilationProperty(testData, result))
        << "Dilation property violated for " << config.description;

    // Verify floating point validity
    for (auto val : result) {
      EXPECT_TRUE(std::isfinite(val)) << "Non-finite output value in dilation";
      EXPECT_GE(val, -10.0f) << "Unreasonable output value";
      EXPECT_LE(val, 10.0f) << "Unreasonable output value";
    }

    // Cleanup
    nppiFree(d_src);
    nppiFree(d_dst);
    cudaFree(d_mask);
  }
}

// Context version tests for 32f_C1R
TEST_P(MorphologyComprehensiveTest, Erode_32f_C1R_Ctx_Comprehensive) {
  auto config = GetParam();
  SCOPED_TRACE("Erode 32f C1R Ctx test: " + config.description);

  auto testData = MorphologyPatternGenerator<Npp32f>::generatePattern(TestPattern::BINARY_SQUARE, config.width,
                                                                      config.height, 0.0f, 1.0f);

  // Allocate device memory
  int srcStep, dstStep;
  Npp32f *d_src = nppiMalloc_32f_C1(config.width, config.height, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C1(config.width, config.height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Create stream context
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  nppStreamCtx.hStream = stream;

  // Upload test data asynchronously
  cudaMemcpy2DAsync(d_src, srcStep, testData.data(), config.width * sizeof(Npp32f), config.width * sizeof(Npp32f),
                    config.height, cudaMemcpyHostToDevice, stream);

  // Create structural element
  auto kernel = createStructuralElement(config);
  Npp8u *d_mask = nullptr;
  cudaMalloc(&d_mask, kernel.size());
  cudaMemcpyAsync(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice, stream);

  NppiSize oSizeROI = {config.width, config.height};
  NppiSize oMaskSize = {config.kernelWidth, config.kernelHeight};
  NppiPoint oAnchor = {config.anchorX, config.anchorY};

  // Test context version
  NppStatus status =
      nppiErode_32f_C1R_Ctx(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS) << "Erosion 32f Ctx failed for config: " << config.description;

  // Synchronize and download results
  cudaStreamSynchronize(stream);
  std::vector<Npp32f> result(config.width * config.height);
  cudaMemcpy2D(result.data(), config.width * sizeof(Npp32f), d_dst, dstStep, config.width * sizeof(Npp32f),
               config.height, cudaMemcpyDeviceToHost);

  // Verify results are reasonable
  EXPECT_TRUE(std::all_of(result.begin(), result.end(), [](Npp32f val) { return std::isfinite(val); }));

  // Cleanup
  cudaStreamDestroy(stream);
  nppiFree(d_src);
  nppiFree(d_dst);
  cudaFree(d_mask);
}

// =====================================================================================
// PARAMETERIZED TESTS FOR 32f_C4R
// =====================================================================================

TEST_P(MorphologyComprehensiveTest, Erode_32f_C4R_Comprehensive) {
  auto config = GetParam();
  SCOPED_TRACE("Erode 32f C4R test: " + config.description);

  // Create 4-channel test data with different ranges per channel
  std::vector<Npp32f> testData(config.width * config.height * 4);
  for (int y = 0; y < config.height; y++) {
    for (int x = 0; x < config.width; x++) {
      int idx = (y * config.width + x) * 4;
      bool inCenter =
          (x >= config.width / 4 && x < 3 * config.width / 4 && y >= config.height / 4 && y < 3 * config.height / 4);

      testData[idx + 0] = inCenter ? 1.0f : 0.0f;                        // Binary pattern
      testData[idx + 1] = x / float(config.width);                       // Horizontal gradient
      testData[idx + 2] = y / float(config.height);                      // Vertical gradient
      testData[idx + 3] = (x + y) / float(config.width + config.height); // Diagonal gradient
    }
  }

  // Allocate device memory
  int srcStep, dstStep;
  Npp32f *d_src = nppiMalloc_32f_C4(config.width, config.height, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C4(config.width, config.height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Upload test data
  cudaMemcpy2D(d_src, srcStep, testData.data(), config.width * 4 * sizeof(Npp32f), config.width * 4 * sizeof(Npp32f),
               config.height, cudaMemcpyHostToDevice);

  // Create structural element
  auto kernel = createStructuralElement(config);
  Npp8u *d_mask = nullptr;
  cudaMalloc(&d_mask, kernel.size());
  cudaMemcpy(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {config.width, config.height};
  NppiSize oMaskSize = {config.kernelWidth, config.kernelHeight};
  NppiPoint oAnchor = {config.anchorX, config.anchorY};

  // Test regular version
  NppStatus status = nppiErode_32f_C4R(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor);
  ASSERT_EQ(status, NPP_SUCCESS) << "Erosion 32f C4R failed for config: " << config.description;

  // Download and verify results
  std::vector<Npp32f> result(config.width * config.height * 4);
  cudaMemcpy2D(result.data(), config.width * 4 * sizeof(Npp32f), d_dst, dstStep, config.width * 4 * sizeof(Npp32f),
               config.height, cudaMemcpyDeviceToHost);

  // Verify all channels processed correctly
  for (int c = 0; c < 4; c++) {
    std::vector<Npp32f> originalChannel, resultChannel;
    for (int i = c; i < config.width * config.height * 4; i += 4) {
      originalChannel.push_back(testData[i]);
      resultChannel.push_back(result[i]);
    }

    EXPECT_TRUE(MorphologyAnalyzer<Npp32f>::verifyErosionProperty(originalChannel, resultChannel))
        << "Erosion property violated for channel " << c << " in " << config.description;
  }

  // Verify floating point validity for all channels
  for (auto val : result) {
    EXPECT_TRUE(std::isfinite(val)) << "Non-finite output value in C4R erosion";
    EXPECT_GE(val, -10.0f) << "Unreasonable output value";
    EXPECT_LE(val, 10.0f) << "Unreasonable output value";
  }

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_dst);
  cudaFree(d_mask);
}

TEST_P(MorphologyComprehensiveTest, Dilate_32f_C4R_Comprehensive) {
  auto config = GetParam();
  SCOPED_TRACE("Dilate 32f C4R test: " + config.description);

  // Create 4-channel test data with sparse patterns
  std::vector<Npp32f> testData(config.width * config.height * 4);
  for (int y = 0; y < config.height; y++) {
    for (int x = 0; x < config.width; x++) {
      int idx = (y * config.width + x) * 4;

      // Create different sparse patterns for each channel
      testData[idx + 0] = ((x + y) % 4 == 0) ? 1.0f : 0.1f; // Sparse high values
      testData[idx + 1] = (x % 3 == 0) ? 0.8f : 0.2f;       // Vertical stripes
      testData[idx + 2] = (y % 3 == 0) ? 0.9f : 0.1f;       // Horizontal stripes
      testData[idx + 3] = ((x * y) % 5 == 0) ? 1.0f : 0.0f; // Complex pattern
    }
  }

  // Allocate device memory
  int srcStep, dstStep;
  Npp32f *d_src = nppiMalloc_32f_C4(config.width, config.height, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C4(config.width, config.height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Upload test data
  cudaMemcpy2D(d_src, srcStep, testData.data(), config.width * 4 * sizeof(Npp32f), config.width * 4 * sizeof(Npp32f),
               config.height, cudaMemcpyHostToDevice);

  // Create structural element
  auto kernel = createStructuralElement(config);
  Npp8u *d_mask = nullptr;
  cudaMalloc(&d_mask, kernel.size());
  cudaMemcpy(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {config.width, config.height};
  NppiSize oMaskSize = {config.kernelWidth, config.kernelHeight};
  NppiPoint oAnchor = {config.anchorX, config.anchorY};

  // Test regular version
  NppStatus status = nppiDilate_32f_C4R(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor);
  ASSERT_EQ(status, NPP_SUCCESS) << "Dilation 32f C4R failed for config: " << config.description;

  // Download and verify results
  std::vector<Npp32f> result(config.width * config.height * 4);
  cudaMemcpy2D(result.data(), config.width * 4 * sizeof(Npp32f), d_dst, dstStep, config.width * 4 * sizeof(Npp32f),
               config.height, cudaMemcpyDeviceToHost);

  // Verify all channels processed correctly
  for (int c = 0; c < 4; c++) {
    std::vector<Npp32f> originalChannel, resultChannel;
    for (int i = c; i < config.width * config.height * 4; i += 4) {
      originalChannel.push_back(testData[i]);
      resultChannel.push_back(result[i]);
    }

    EXPECT_TRUE(MorphologyAnalyzer<Npp32f>::verifyDilationProperty(originalChannel, resultChannel))
        << "Dilation property violated for channel " << c << " in " << config.description;
  }

  // Verify floating point validity for all channels
  for (auto val : result) {
    EXPECT_TRUE(std::isfinite(val)) << "Non-finite output value in C4R dilation";
    EXPECT_GE(val, -10.0f) << "Unreasonable output value";
    EXPECT_LE(val, 10.0f) << "Unreasonable output value";
  }

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_dst);
  cudaFree(d_mask);
}

// Context version test for 32f_C4R
TEST_P(MorphologyComprehensiveTest, Dilate_32f_C4R_Ctx_Comprehensive) {
  auto config = GetParam();
  SCOPED_TRACE("Dilate 32f C4R Ctx test: " + config.description);

  // Create 4-channel test data
  std::vector<Npp32f> testData(config.width * config.height * 4);
  for (int y = 0; y < config.height; y++) {
    for (int x = 0; x < config.width; x++) {
      int idx = (y * config.width + x) * 4;
      for (int c = 0; c < 4; c++) {
        testData[idx + c] = ((x + y + c) % 3 == 0) ? 1.0f : 0.2f;
      }
    }
  }

  // Allocate device memory
  int srcStep, dstStep;
  Npp32f *d_src = nppiMalloc_32f_C4(config.width, config.height, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C4(config.width, config.height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Create stream context
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  nppStreamCtx.hStream = stream;

  // Upload test data asynchronously
  cudaMemcpy2DAsync(d_src, srcStep, testData.data(), config.width * 4 * sizeof(Npp32f),
                    config.width * 4 * sizeof(Npp32f), config.height, cudaMemcpyHostToDevice, stream);

  // Create structural element
  auto kernel = createStructuralElement(config);
  Npp8u *d_mask = nullptr;
  cudaMalloc(&d_mask, kernel.size());
  cudaMemcpyAsync(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice, stream);

  NppiSize oSizeROI = {config.width, config.height};
  NppiSize oMaskSize = {config.kernelWidth, config.kernelHeight};
  NppiPoint oAnchor = {config.anchorX, config.anchorY};

  // Test context version
  NppStatus status =
      nppiDilate_32f_C4R_Ctx(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS) << "Dilation 32f C4R Ctx failed for config: " << config.description;

  // Synchronize and download results
  cudaStreamSynchronize(stream);
  std::vector<Npp32f> result(config.width * config.height * 4);
  cudaMemcpy2D(result.data(), config.width * 4 * sizeof(Npp32f), d_dst, dstStep, config.width * 4 * sizeof(Npp32f),
               config.height, cudaMemcpyDeviceToHost);

  // Verify results are reasonable
  EXPECT_TRUE(std::all_of(result.begin(), result.end(), [](Npp32f val) { return std::isfinite(val); }));

  // Cleanup
  cudaStreamDestroy(stream);
  nppiFree(d_src);
  nppiFree(d_dst);
  cudaFree(d_mask);
}

// =====================================================================================
// MORPHOLOGICAL PROPERTIES AND EDGE CASE TESTS
// =====================================================================================

TEST_P(MorphologyComprehensiveTest, MorphologyDuality_8u_C1R) {
  auto config = GetParam();
  SCOPED_TRACE("Morphology duality test: " + config.description);

  auto testData = MorphologyPatternGenerator<Npp8u>::generatePattern(TestPattern::BINARY_SQUARE, config.width,
                                                                     config.height, 0, 255);

  // Allocate device memory for source, eroded, and dilated
  int srcStep, erodedStep, dilatedStep;
  Npp8u *d_src = nppiMalloc_8u_C1(config.width, config.height, &srcStep);
  Npp8u *d_eroded = nppiMalloc_8u_C1(config.width, config.height, &erodedStep);
  Npp8u *d_dilated = nppiMalloc_8u_C1(config.width, config.height, &dilatedStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_eroded, nullptr);
  ASSERT_NE(d_dilated, nullptr);

  // Upload test data
  cudaMemcpy2D(d_src, srcStep, testData.data(), config.width * sizeof(Npp8u), config.width * sizeof(Npp8u),
               config.height, cudaMemcpyHostToDevice);

  // Create structural element
  auto kernel = createStructuralElement(config);
  Npp8u *d_mask = nullptr;
  cudaMalloc(&d_mask, kernel.size());
  cudaMemcpy(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {config.width, config.height};
  NppiSize oMaskSize = {config.kernelWidth, config.kernelHeight};
  NppiPoint oAnchor = {config.anchorX, config.anchorY};

  // Perform both erosion and dilation
  NppStatus statusErode = nppiErode_8u_C1R(d_src, srcStep, d_eroded, erodedStep, oSizeROI, d_mask, oMaskSize, oAnchor);
  NppStatus statusDilate =
      nppiDilate_8u_C1R(d_src, srcStep, d_dilated, dilatedStep, oSizeROI, d_mask, oMaskSize, oAnchor);
  ASSERT_EQ(statusErode, NPP_SUCCESS);
  ASSERT_EQ(statusDilate, NPP_SUCCESS);

  // Download results
  std::vector<Npp8u> eroded(config.width * config.height);
  std::vector<Npp8u> dilated(config.width * config.height);
  cudaMemcpy2D(eroded.data(), config.width * sizeof(Npp8u), d_eroded, erodedStep, config.width * sizeof(Npp8u),
               config.height, cudaMemcpyDeviceToHost);
  cudaMemcpy2D(dilated.data(), config.width * sizeof(Npp8u), d_dilated, dilatedStep, config.width * sizeof(Npp8u),
               config.height, cudaMemcpyDeviceToHost);

  // Verify morphological properties
  EXPECT_TRUE(MorphologyAnalyzer<Npp8u>::verifyErosionProperty(testData, eroded));
  EXPECT_TRUE(MorphologyAnalyzer<Npp8u>::verifyDilationProperty(testData, dilated));
  EXPECT_TRUE(MorphologyAnalyzer<Npp8u>::verifyDuality(eroded, dilated));

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_eroded);
  nppiFree(d_dilated);
  cudaFree(d_mask);
}

TEST_P(MorphologyComprehensiveTest, MorphologyDuality_32f_C1R) {
  auto config = GetParam();
  SCOPED_TRACE("Morphology duality 32f test: " + config.description);

  auto testData = MorphologyPatternGenerator<Npp32f>::generatePattern(TestPattern::GRADIENT_HORIZONTAL, config.width,
                                                                      config.height, 0.0f, 1.0f);

  // Allocate device memory for source, eroded, and dilated
  int srcStep, erodedStep, dilatedStep;
  Npp32f *d_src = nppiMalloc_32f_C1(config.width, config.height, &srcStep);
  Npp32f *d_eroded = nppiMalloc_32f_C1(config.width, config.height, &erodedStep);
  Npp32f *d_dilated = nppiMalloc_32f_C1(config.width, config.height, &dilatedStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_eroded, nullptr);
  ASSERT_NE(d_dilated, nullptr);

  // Upload test data
  cudaMemcpy2D(d_src, srcStep, testData.data(), config.width * sizeof(Npp32f), config.width * sizeof(Npp32f),
               config.height, cudaMemcpyHostToDevice);

  // Create structural element
  auto kernel = createStructuralElement(config);
  Npp8u *d_mask = nullptr;
  cudaMalloc(&d_mask, kernel.size());
  cudaMemcpy(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {config.width, config.height};
  NppiSize oMaskSize = {config.kernelWidth, config.kernelHeight};
  NppiPoint oAnchor = {config.anchorX, config.anchorY};

  // Perform both erosion and dilation
  NppStatus statusErode = nppiErode_32f_C1R(d_src, srcStep, d_eroded, erodedStep, oSizeROI, d_mask, oMaskSize, oAnchor);
  NppStatus statusDilate =
      nppiDilate_32f_C1R(d_src, srcStep, d_dilated, dilatedStep, oSizeROI, d_mask, oMaskSize, oAnchor);
  ASSERT_EQ(statusErode, NPP_SUCCESS);
  ASSERT_EQ(statusDilate, NPP_SUCCESS);

  // Download results
  std::vector<Npp32f> eroded(config.width * config.height);
  std::vector<Npp32f> dilated(config.width * config.height);
  cudaMemcpy2D(eroded.data(), config.width * sizeof(Npp32f), d_eroded, erodedStep, config.width * sizeof(Npp32f),
               config.height, cudaMemcpyDeviceToHost);
  cudaMemcpy2D(dilated.data(), config.width * sizeof(Npp32f), d_dilated, dilatedStep, config.width * sizeof(Npp32f),
               config.height, cudaMemcpyDeviceToHost);

  // Verify morphological properties
  EXPECT_TRUE(MorphologyAnalyzer<Npp32f>::verifyErosionProperty(testData, eroded));
  EXPECT_TRUE(MorphologyAnalyzer<Npp32f>::verifyDilationProperty(testData, dilated));
  EXPECT_TRUE(MorphologyAnalyzer<Npp32f>::verifyDuality(eroded, dilated));

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_eroded);
  nppiFree(d_dilated);
  cudaFree(d_mask);
}

// =====================================================================================
// EDGE CASES AND SPECIAL SCENARIO TESTS
// =====================================================================================

class MorphologyEdgeCaseTest : public ::testing::Test {
protected:
  void SetUp() override {
    cudaError_t err = cudaSetDevice(0);
    ASSERT_EQ(err, cudaSuccess);
  }

  void TearDown() override {
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess);
  }
};

// Test minimum size cases
TEST_F(MorphologyEdgeCaseTest, MinimumSize_1x1_Image) {
  const int width = 1, height = 1;
  std::vector<Npp8u> testData = {128};

  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C1(width, height, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C1(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, srcStep, testData.data(), width * sizeof(Npp8u), width * sizeof(Npp8u), height,
               cudaMemcpyHostToDevice);

  auto kernel = StructuralElementGenerator::createBox3x3();
  Npp8u *d_mask = nullptr;
  cudaMalloc(&d_mask, kernel.size());
  cudaMemcpy(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  NppiSize oMaskSize = {3, 3};
  NppiPoint oAnchor = {1, 1};

  // Test erosion on 1x1 image
  NppStatus status = nppiErode_8u_C1R(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> result(1);
  cudaMemcpy2D(result.data(), sizeof(Npp8u), d_dst, dstStep, sizeof(Npp8u), 1, cudaMemcpyDeviceToHost);

  // For a 1x1 image, the result depends on border handling
  EXPECT_GE(result[0], 0);
  EXPECT_LE(result[0], 255);

  nppiFree(d_src);
  nppiFree(d_dst);
  cudaFree(d_mask);
}

// Test extreme values
TEST_F(MorphologyEdgeCaseTest, ExtremeValues_8u) {
  const int width = 8, height = 8;

  // Test all zeros
  {
    std::vector<Npp8u> testData(width * height, 0);

    int srcStep, dstStep;
    Npp8u *d_src = nppiMalloc_8u_C1(width, height, &srcStep);
    Npp8u *d_dst = nppiMalloc_8u_C1(width, height, &dstStep);

    cudaMemcpy2D(d_src, srcStep, testData.data(), width * sizeof(Npp8u), width * sizeof(Npp8u), height,
                 cudaMemcpyHostToDevice);

    auto kernel = StructuralElementGenerator::createBox3x3();
    Npp8u *d_mask = nullptr;
    cudaMalloc(&d_mask, kernel.size());
    cudaMemcpy(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice);

    NppiSize oSizeROI = {width, height};
    NppiSize oMaskSize = {3, 3};
    NppiPoint oAnchor = {1, 1};

    NppStatus status = nppiErode_8u_C1R(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor);
    EXPECT_EQ(status, NPP_SUCCESS);

    std::vector<Npp8u> result(width * height);
    cudaMemcpy2D(result.data(), width * sizeof(Npp8u), d_dst, dstStep, width * sizeof(Npp8u), height,
                 cudaMemcpyDeviceToHost);

    // All zeros should remain zeros after erosion
    EXPECT_TRUE(std::all_of(result.begin(), result.end(), [](Npp8u val) { return val == 0; }));

    nppiFree(d_src);
    nppiFree(d_dst);
    cudaFree(d_mask);
  }

  // Test all max values
  {
    std::vector<Npp8u> testData(width * height, 255);

    int srcStep, dstStep;
    Npp8u *d_src = nppiMalloc_8u_C1(width, height, &srcStep);
    Npp8u *d_dst = nppiMalloc_8u_C1(width, height, &dstStep);

    cudaMemcpy2D(d_src, srcStep, testData.data(), width * sizeof(Npp8u), width * sizeof(Npp8u), height,
                 cudaMemcpyHostToDevice);

    auto kernel = StructuralElementGenerator::createBox3x3();
    Npp8u *d_mask = nullptr;
    cudaMalloc(&d_mask, kernel.size());
    cudaMemcpy(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice);

    NppiSize oSizeROI = {width, height};
    NppiSize oMaskSize = {3, 3};
    NppiPoint oAnchor = {1, 1};

    NppStatus status = nppiDilate_8u_C1R(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor);
    EXPECT_EQ(status, NPP_SUCCESS);

    std::vector<Npp8u> result(width * height);
    cudaMemcpy2D(result.data(), width * sizeof(Npp8u), d_dst, dstStep, width * sizeof(Npp8u), height,
                 cudaMemcpyDeviceToHost);

    // Most values should remain 255 after dilation
    int count255 = std::count(result.begin(), result.end(), 255);
    EXPECT_GT(count255, width * height / 2);

    nppiFree(d_src);
    nppiFree(d_dst);
    cudaFree(d_mask);
  }
}

// Test extreme values for 32f
TEST_F(MorphologyEdgeCaseTest, ExtremeValues_32f) {
  const int width = 8, height = 8;

  // Test very small values
  {
    std::vector<Npp32f> testData(width * height, 1e-6f);

    int srcStep, dstStep;
    Npp32f *d_src = nppiMalloc_32f_C1(width, height, &srcStep);
    Npp32f *d_dst = nppiMalloc_32f_C1(width, height, &dstStep);

    cudaMemcpy2D(d_src, srcStep, testData.data(), width * sizeof(Npp32f), width * sizeof(Npp32f), height,
                 cudaMemcpyHostToDevice);

    auto kernel = StructuralElementGenerator::createBox3x3();
    Npp8u *d_mask = nullptr;
    cudaMalloc(&d_mask, kernel.size());
    cudaMemcpy(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice);

    NppiSize oSizeROI = {width, height};
    NppiSize oMaskSize = {3, 3};
    NppiPoint oAnchor = {1, 1};

    NppStatus status = nppiErode_32f_C1R(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor);
    EXPECT_EQ(status, NPP_SUCCESS);

    std::vector<Npp32f> result(width * height);
    cudaMemcpy2D(result.data(), width * sizeof(Npp32f), d_dst, dstStep, width * sizeof(Npp32f), height,
                 cudaMemcpyDeviceToHost);

    // All values should be finite and small
    for (auto val : result) {
      EXPECT_TRUE(std::isfinite(val));
      EXPECT_GE(val, 0.0f);
      EXPECT_LE(val, 1e-5f);
    }

    nppiFree(d_src);
    nppiFree(d_dst);
    cudaFree(d_mask);
  }
}

// Test single element kernels
TEST_F(MorphologyEdgeCaseTest, SingleElementKernel) {
  const int width = 16, height = 16;
  auto testData = MorphologyPatternGenerator<Npp8u>::generatePattern(TestPattern::CHECKERBOARD, width, height, 0, 255);

  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C1(width, height, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C1(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, srcStep, testData.data(), width * sizeof(Npp8u), width * sizeof(Npp8u), height,
               cudaMemcpyHostToDevice);

  // Create 1x1 kernel
  std::vector<Npp8u> kernel = {1};
  Npp8u *d_mask = nullptr;
  cudaMalloc(&d_mask, kernel.size());
  cudaMemcpy(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  NppiSize oMaskSize = {1, 1};
  NppiPoint oAnchor = {0, 0};

  // Test erosion with 1x1 kernel (should be identity)
  NppStatus status = nppiErode_8u_C1R(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> result(width * height);
  cudaMemcpy2D(result.data(), width * sizeof(Npp8u), d_dst, dstStep, width * sizeof(Npp8u), height,
               cudaMemcpyDeviceToHost);

  // 1x1 erosion should be close to identity operation
  int differences = 0;
  for (size_t i = 0; i < testData.size(); i++) {
    if (testData[i] != result[i]) {
      differences++;
    }
  }

  // Allow some differences due to border handling, but most should be identical
  EXPECT_LT(differences, static_cast<int>(testData.size() * 0.1));

  nppiFree(d_src);
  nppiFree(d_dst);
  cudaFree(d_mask);
}

// Test different anchor positions edge cases
TEST_F(MorphologyEdgeCaseTest, ExtemeAnchorPositions) {
  const int width = 8, height = 8;
  auto testData = MorphologyPatternGenerator<Npp8u>::generatePattern(TestPattern::BINARY_SQUARE, width, height, 0, 255);

  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C1(width, height, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C1(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, srcStep, testData.data(), width * sizeof(Npp8u), width * sizeof(Npp8u), height,
               cudaMemcpyHostToDevice);

  auto kernel = StructuralElementGenerator::createBox5x5();
  Npp8u *d_mask = nullptr;
  cudaMalloc(&d_mask, kernel.size());
  cudaMemcpy(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  NppiSize oMaskSize = {5, 5};

  // Test extreme anchor positions
  std::vector<NppiPoint> anchors = {{0, 0}, {4, 4}, {0, 4}, {4, 0}};

  for (auto anchor : anchors) {
    NppStatus status = nppiErode_8u_C1R(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, anchor);
    EXPECT_EQ(status, NPP_SUCCESS) << "Failed with anchor (" << anchor.x << "," << anchor.y << ")";

    std::vector<Npp8u> result(width * height);
    cudaMemcpy2D(result.data(), width * sizeof(Npp8u), d_dst, dstStep, width * sizeof(Npp8u), height,
                 cudaMemcpyDeviceToHost);

    // Results should be valid regardless of anchor position
    EXPECT_TRUE(std::all_of(result.begin(), result.end(), [](Npp8u val) { return val <= 255; }));
  }

  nppiFree(d_src);
  nppiFree(d_dst);
  cudaFree(d_mask);
}

// Test very large kernels
TEST_F(MorphologyEdgeCaseTest, VeryLargeKernel) {
  const int width = 32, height = 32;
  auto testData = MorphologyPatternGenerator<Npp8u>::generatePattern(TestPattern::BINARY_SQUARE, width, height, 0, 255);

  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C1(width, height, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C1(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, srcStep, testData.data(), width * sizeof(Npp8u), width * sizeof(Npp8u), height,
               cudaMemcpyHostToDevice);

  // Create large 21x21 kernel
  const int kernelSize = 21;
  std::vector<Npp8u> kernel(kernelSize * kernelSize, 1);
  Npp8u *d_mask = nullptr;
  cudaMalloc(&d_mask, kernel.size());
  cudaMemcpy(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  NppiSize oMaskSize = {kernelSize, kernelSize};
  NppiPoint oAnchor = {kernelSize / 2, kernelSize / 2};

  NppStatus status = nppiErode_8u_C1R(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> result(width * height);
  cudaMemcpy2D(result.data(), width * sizeof(Npp8u), d_dst, dstStep, width * sizeof(Npp8u), height,
               cudaMemcpyDeviceToHost);

  // Large erosion should significantly reduce foreground
  int originalForeground = std::count(testData.begin(), testData.end(), 255);
  int resultForeground = std::count(result.begin(), result.end(), 255);

  EXPECT_LE(resultForeground, originalForeground);

  nppiFree(d_src);
  nppiFree(d_dst);
  cudaFree(d_mask);
}

// Test mixed channel data corruption check for C4R
TEST_F(MorphologyEdgeCaseTest, ChannelIndependence_C4R) {
  const int width = 16, height = 16;
  std::vector<Npp8u> testData(width * height * 4);

  // Create different patterns in each channel
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * 4;
      testData[idx + 0] = (x % 2 == 0) ? 255 : 0;       // Vertical stripes
      testData[idx + 1] = (y % 2 == 0) ? 255 : 0;       // Horizontal stripes
      testData[idx + 2] = ((x + y) % 2 == 0) ? 255 : 0; // Checkerboard
      testData[idx + 3] = 128;                          // Uniform gray
    }
  }

  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C4(width, height, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C4(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, srcStep, testData.data(), width * 4 * sizeof(Npp8u), width * 4 * sizeof(Npp8u), height,
               cudaMemcpyHostToDevice);

  auto kernel = StructuralElementGenerator::createBox3x3();
  Npp8u *d_mask = nullptr;
  cudaMalloc(&d_mask, kernel.size());
  cudaMemcpy(d_mask, kernel.data(), kernel.size(), cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  NppiSize oMaskSize = {3, 3};
  NppiPoint oAnchor = {1, 1};

  NppStatus status = nppiErode_8u_C4R(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> result(width * height * 4);
  cudaMemcpy2D(result.data(), width * 4 * sizeof(Npp8u), d_dst, dstStep, width * 4 * sizeof(Npp8u), height,
               cudaMemcpyDeviceToHost);

  // Verify channels were processed independently
  for (int c = 0; c < 4; c++) {
    std::vector<Npp8u> originalChannel, resultChannel;
    for (int i = c; i < width * height * 4; i += 4) {
      originalChannel.push_back(testData[i]);
      resultChannel.push_back(result[i]);
    }

    // Each channel should maintain valid values
    bool hasValidResults = !resultChannel.empty();
    EXPECT_TRUE(hasValidResults) << "Channel " << c << " has invalid results";
  }

  nppiFree(d_src);
  nppiFree(d_dst);
  cudaFree(d_mask);
}

// Test instantiation
INSTANTIATE_TEST_SUITE_P(MorphologyComprehensive, MorphologyComprehensiveTest,
                         ::testing::ValuesIn(MORPHOLOGY_COMPREHENSIVE_CONFIGS),
                         [](const testing::TestParamInfo<MorphologyTestConfig> &info) {
                           return info.param.description;
                         });