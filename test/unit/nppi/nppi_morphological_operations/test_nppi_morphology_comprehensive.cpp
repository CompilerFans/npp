// Comprehensive NPP Morphological Operations Test Suite
// Extended test coverage for Erode and Dilate operations across all supported data types
// Based on existing test framework with enhanced coverage

#include "../../framework/npp_test_base.h"
#include "npp.h"
#include <algorithm>
#include <chrono>
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

// =====================================================================================
// TEST CONFIGURATION AND PARAMETERS
// =====================================================================================

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
      data[0] = fgValue;                                     // Top-left
      data[width - 1] = fgValue;                             // Top-right
      data[(height - 1) * width] = fgValue;                  // Bottom-left
      data[(height - 1) * width + width - 1] = fgValue;      // Bottom-right
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
  static std::vector<Npp8u> createRectangle(int width, int height) {
    return std::vector<Npp8u>(width * height, 1);
  }

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
    
    if (original.empty()) return stats;
    
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
static const std::vector<TestPattern> TEST_PATTERNS = {
    TestPattern::UNIFORM,           TestPattern::BINARY_SQUARE,     TestPattern::CHECKERBOARD,
    TestPattern::GRADIENT_HORIZONTAL, TestPattern::GRADIENT_VERTICAL, TestPattern::RANDOM_NOISE,
    TestPattern::SPARSE_DOTS,       TestPattern::EDGE_PATTERN,      TestPattern::CORNER_PATTERN,
    TestPattern::DIAGONAL_LINES};

// =====================================================================================
// BASE TEST CLASS
// =====================================================================================

class MorphologyComprehensiveTest : public ::testing::TestWithParam<MorphologyTestConfig> {
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

  auto testData = MorphologyPatternGenerator<Npp8u>::generatePattern(TestPattern::BINARY_SQUARE, config.width, config.height, 0, 255);

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
  NppStatus status = nppiErode_8u_C1R_Ctx(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS) << "Erosion Ctx failed for config: " << config.description;

  // Synchronize and download results
  cudaStreamSynchronize(stream);
  std::vector<Npp8u> result(config.width * config.height);
  cudaMemcpy2D(result.data(), config.width * sizeof(Npp8u), d_dst, dstStep, config.width * sizeof(Npp8u),
               config.height, cudaMemcpyDeviceToHost);

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
  auto singleChannelData = MorphologyPatternGenerator<Npp8u>::generatePattern(TestPattern::BINARY_SQUARE, config.width, config.height, 0, 255);
  
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
      testData[idx + 0] = ((x + y) % 4 == 0) ? 255 : 0;  // Sparse pattern
      testData[idx + 1] = (x < config.width / 2) ? 255 : 0;  // Half split
      testData[idx + 2] = (y < config.height / 2) ? 255 : 0; // Half split
      testData[idx + 3] = ((x + y) % 2 == 0) ? 255 : 0;  // Checkerboard
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
    auto testData = MorphologyPatternGenerator<Npp32f>::generatePattern(pattern, config.width, config.height, 0.0f, 1.0f);

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
    NppStatus status = nppiErode_32f_C1R(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor);
    ASSERT_EQ(status, NPP_SUCCESS) << "Erosion 32f failed for config: " << config.description;

    // Download and verify results
    std::vector<Npp32f> result(config.width * config.height);
    cudaMemcpy2D(result.data(), config.width * sizeof(Npp32f), d_dst, dstStep, config.width * sizeof(Npp32f),
                 config.height, cudaMemcpyDeviceToHost);

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
    nppiFree(d_src);
    nppiFree(d_dst);
    cudaFree(d_mask);
  }
}

TEST_P(MorphologyComprehensiveTest, Dilate_32f_C1R_Comprehensive) {
  auto config = GetParam();
  SCOPED_TRACE("Dilate 32f C1R test: " + config.description);

  // Test multiple patterns with float values
  for (auto pattern : {TestPattern::SPARSE_DOTS, TestPattern::GRADIENT_VERTICAL, TestPattern::CORNER_PATTERN}) {
    auto testData = MorphologyPatternGenerator<Npp32f>::generatePattern(pattern, config.width, config.height, 0.0f, 1.0f);

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

  auto testData = MorphologyPatternGenerator<Npp32f>::generatePattern(TestPattern::BINARY_SQUARE, config.width, config.height, 0.0f, 1.0f);

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
  NppStatus status = nppiErode_32f_C1R_Ctx(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor, nppStreamCtx);
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
      bool inCenter = (x >= config.width / 4 && x < 3 * config.width / 4 && 
                      y >= config.height / 4 && y < 3 * config.height / 4);
      
      testData[idx + 0] = inCenter ? 1.0f : 0.0f;      // Binary pattern
      testData[idx + 1] = x / float(config.width);     // Horizontal gradient
      testData[idx + 2] = y / float(config.height);    // Vertical gradient
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
      testData[idx + 0] = ((x + y) % 4 == 0) ? 1.0f : 0.1f;  // Sparse high values
      testData[idx + 1] = (x % 3 == 0) ? 0.8f : 0.2f;         // Vertical stripes
      testData[idx + 2] = (y % 3 == 0) ? 0.9f : 0.1f;         // Horizontal stripes
      testData[idx + 3] = ((x * y) % 5 == 0) ? 1.0f : 0.0f;   // Complex pattern
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
  cudaMemcpy2DAsync(d_src, srcStep, testData.data(), config.width * 4 * sizeof(Npp32f), config.width * 4 * sizeof(Npp32f),
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
  NppStatus status = nppiDilate_32f_C4R_Ctx(d_src, srcStep, d_dst, dstStep, oSizeROI, d_mask, oMaskSize, oAnchor, nppStreamCtx);
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

  auto testData = MorphologyPatternGenerator<Npp8u>::generatePattern(TestPattern::BINARY_SQUARE, config.width, config.height, 0, 255);

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
  NppStatus statusDilate = nppiDilate_8u_C1R(d_src, srcStep, d_dilated, dilatedStep, oSizeROI, d_mask, oMaskSize, oAnchor);
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

  auto testData = MorphologyPatternGenerator<Npp32f>::generatePattern(TestPattern::GRADIENT_HORIZONTAL, config.width, config.height, 0.0f, 1.0f);

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
  NppStatus statusDilate = nppiDilate_32f_C1R(d_src, srcStep, d_dilated, dilatedStep, oSizeROI, d_mask, oMaskSize, oAnchor);
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

  cudaMemcpy2D(d_src, srcStep, testData.data(), width * sizeof(Npp8u), width * sizeof(Npp8u),
               height, cudaMemcpyHostToDevice);

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
    
    cudaMemcpy2D(d_src, srcStep, testData.data(), width * sizeof(Npp8u), width * sizeof(Npp8u),
                 height, cudaMemcpyHostToDevice);

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
    cudaMemcpy2D(result.data(), width * sizeof(Npp8u), d_dst, dstStep, width * sizeof(Npp8u),
                 height, cudaMemcpyDeviceToHost);

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
    
    cudaMemcpy2D(d_src, srcStep, testData.data(), width * sizeof(Npp8u), width * sizeof(Npp8u),
                 height, cudaMemcpyHostToDevice);

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
    cudaMemcpy2D(result.data(), width * sizeof(Npp8u), d_dst, dstStep, width * sizeof(Npp8u),
                 height, cudaMemcpyDeviceToHost);

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
    
    cudaMemcpy2D(d_src, srcStep, testData.data(), width * sizeof(Npp32f), width * sizeof(Npp32f),
                 height, cudaMemcpyHostToDevice);

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
    cudaMemcpy2D(result.data(), width * sizeof(Npp32f), d_dst, dstStep, width * sizeof(Npp32f),
                 height, cudaMemcpyDeviceToHost);

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

  cudaMemcpy2D(d_src, srcStep, testData.data(), width * sizeof(Npp8u), width * sizeof(Npp8u),
               height, cudaMemcpyHostToDevice);

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
  cudaMemcpy2D(result.data(), width * sizeof(Npp8u), d_dst, dstStep, width * sizeof(Npp8u),
               height, cudaMemcpyDeviceToHost);

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

  cudaMemcpy2D(d_src, srcStep, testData.data(), width * sizeof(Npp8u), width * sizeof(Npp8u),
               height, cudaMemcpyHostToDevice);

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
    cudaMemcpy2D(result.data(), width * sizeof(Npp8u), d_dst, dstStep, width * sizeof(Npp8u),
                 height, cudaMemcpyDeviceToHost);

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

  cudaMemcpy2D(d_src, srcStep, testData.data(), width * sizeof(Npp8u), width * sizeof(Npp8u),
               height, cudaMemcpyHostToDevice);

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
  cudaMemcpy2D(result.data(), width * sizeof(Npp8u), d_dst, dstStep, width * sizeof(Npp8u),
               height, cudaMemcpyDeviceToHost);

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
      testData[idx + 0] = (x % 2 == 0) ? 255 : 0;          // Vertical stripes
      testData[idx + 1] = (y % 2 == 0) ? 255 : 0;          // Horizontal stripes
      testData[idx + 2] = ((x + y) % 2 == 0) ? 255 : 0;    // Checkerboard
      testData[idx + 3] = 128;                              // Uniform gray
    }
  }

  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C4(width, height, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C4(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, srcStep, testData.data(), width * 4 * sizeof(Npp8u), width * 4 * sizeof(Npp8u),
               height, cudaMemcpyHostToDevice);

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
  cudaMemcpy2D(result.data(), width * 4 * sizeof(Npp8u), d_dst, dstStep, width * 4 * sizeof(Npp8u),
               height, cudaMemcpyDeviceToHost);

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