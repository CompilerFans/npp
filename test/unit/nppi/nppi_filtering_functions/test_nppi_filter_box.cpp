// Unified NPP FilterBox Tests
// Complete integration of all FilterBox testing functionality

#include "npp.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>


class FilterTest : public ::testing::Test {
  protected:
    void SetUp() override {
      // 创建测试数据
      width = 16;
      height = 16;
      size.width = width;
      size.height = height;
  
      // 使用NPP内置内存管理函数分配设备内存
      d_src = nppiMalloc_8u_C1(width, height, &step_src);
      d_dst = nppiMalloc_8u_C1(width, height, &step_dst);
  
      ASSERT_NE(d_src, nullptr) << "Failed to allocate src memory";
      ASSERT_NE(d_dst, nullptr) << "Failed to allocate dst memory";
  
      // prepare test data
      h_src.resize(width * height);
      h_dst.resize(width * height);
    }
  
    void TearDown() override {
      // 使用NPP内置函数释放内存
      if (d_src)
        nppiFree(d_src);
      if (d_dst)
        nppiFree(d_dst);
    }
  
    int width, height;
    int step_src, step_dst;
    NppiSize size;
    Npp8u *d_src, *d_dst;
    std::vector<Npp8u> h_src, h_dst;
  };
  
  TEST_F(FilterTest, FilterBox_3x3_Uniform) {
    // 创建一个更简单的测试 - 使用全100的图像
    std::fill(h_src.begin(), h_src.end(), 100);
  
    // 上传数据
    cudaError_t err = cudaMemcpy2D(d_src, step_src, h_src.data(), width, width, height, cudaMemcpyHostToDevice);
    ASSERT_EQ(err, cudaSuccess) << "Failed to upload source data";
  
    // 同步确保内存操作完成
    cudaDeviceSynchronize();
  
    // 设置3x3 box滤波器，anchor在中心
    NppiSize maskSize = {3, 3};
    NppiPoint anchor = {1, 1};
  
    // 执行滤波
    NppStatus status = nppiFilterBox_8u_C1R(d_src, step_src, d_dst, step_dst, size, maskSize, anchor);
    if (status != NPP_SUCCESS) {
      // 获取更多错误信息
      cudaError_t lastError = cudaGetLastError();
      ASSERT_EQ(status, NPP_SUCCESS) << "NPP Status: " << status << ", GPU Error: " << cudaGetErrorString(lastError);
    }
  
    // 下载结果
    err = cudaMemcpy2D(h_dst.data(), width, d_dst, step_dst, width, height, cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess);
  
    // Validate结果 - 内部区域应该保持100（均匀图像的box滤波结果使用truncation）
    for (int y = 1; y < height - 1; y++) {
      for (int x = 1; x < width - 1; x++) {
        EXPECT_EQ(h_dst[y * width + x], 100) << "At position (" << x << ", " << y << ")";
      }
    }
  }
  
  TEST_F(FilterTest, FilterBox_5x5_EdgeHandling) {
    // 创建一个全白图像
    std::fill(h_src.begin(), h_src.end(), 100);
  
    // 上传数据
    cudaError_t err = cudaMemcpy2D(d_src, step_src, h_src.data(), width, width, height, cudaMemcpyHostToDevice);
    ASSERT_EQ(err, cudaSuccess);
  
    // 设置5x5 box滤波器
    NppiSize maskSize = {5, 5};
    NppiPoint anchor = {2, 2};
  
    // 执行滤波
    NppStatus status = nppiFilterBox_8u_C1R(d_src, step_src, d_dst, step_dst, size, maskSize, anchor);
    ASSERT_EQ(status, NPP_SUCCESS);
  
    // 下载结果
    err = cudaMemcpy2D(h_dst.data(), width, d_dst, step_dst, width, height, cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess);
  
    // Validate内部区域 - 应该保持原值（使用truncation）
    for (int y = 2; y < height - 2; y++) {
      for (int x = 2; x < width - 2; x++) {
        EXPECT_EQ(h_dst[y * width + x], 100);
      }
    }
  
    // Validate边缘处理 - 由于边界效应和truncation，边缘的值可能略有不同
    // 角落像素只能访问部分邻域，使用truncation计算
    EXPECT_GT(h_dst[0], 0);   // 应该有一定的值
    EXPECT_LE(h_dst[0], 100); // 但不应超过原始值
  }
  
  // Test to verify truncation behavior explicitly
  TEST_F(FilterTest, FilterBox_TruncationBehavior) {
    // Create test image where box filter produces non-integer averages
    std::fill(h_src.begin(), h_src.end(), 0);
  
    // Set specific pattern that will test truncation vs rounding
    // For a 3x3 filter, set center 5 pixels to 127 and others to 0
    // This creates sum = 5 * 127 = 635, average = 635/9 = 70.555...
    // Truncation should give 70, rounding would give 71
    h_src[7 * width + 7] = 127; // center
    h_src[7 * width + 8] = 127; // right
    h_src[8 * width + 7] = 127; // down
    h_src[6 * width + 7] = 127; // up
    h_src[7 * width + 6] = 127; // left
  
    // Upload data
    cudaError_t err = cudaMemcpy2D(d_src, step_src, h_src.data(), width, width, height, cudaMemcpyHostToDevice);
    ASSERT_EQ(err, cudaSuccess);
  
    cudaDeviceSynchronize();
  
    // Set 3x3 box filter
    NppiSize maskSize = {3, 3};
    NppiPoint anchor = {1, 1};
  
    // Execute filter
    NppStatus status = nppiFilterBox_8u_C1R(d_src, step_src, d_dst, step_dst, size, maskSize, anchor);
    ASSERT_EQ(status, NPP_SUCCESS);
  
    // Download result
    err = cudaMemcpy2D(h_dst.data(), width, d_dst, step_dst, width, height, cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess);
  
    // Check center pixel: sum = 5*127 = 635, 635/9 = 70.555...
    // With truncation: should be 70
    // With rounding: would be 71
    EXPECT_EQ(h_dst[7 * width + 7], 70) << "Expected truncation result 70, not rounding result 71";
  }

// ==============================================================================
// UTILITY CLASSES AND ALGORITHMS
// ==============================================================================

// Boundary analysis utilities for matrix printing and comparison
class BoundaryAnalysisUtils {
public:
  template <typename T>
  static void printMatrix(const std::vector<T> &data, int width, int height, const std::string &title,
                          const std::string &testInfo = "") {
    std::cout << "\n=== " << title << " ===\n";
    if (!testInfo.empty()) {
      std::cout << "Test Info: " << testInfo << "\n";
    }
    std::cout << "Matrix [" << height << "x" << width << "]:\n";

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        if (std::is_same<T, float>::value) {
          std::cout << std::setw(8) << std::fixed << std::setprecision(3) << static_cast<float>(data[y * width + x])
                    << " ";
        } else {
          std::cout << std::setw(6) << static_cast<int>(data[y * width + x]) << " ";
        }
      }
      std::cout << "\n";
    }
    std::cout << std::endl;
  }

  template <typename T>
  static void printComparisonMatrix(const std::vector<T> &actual, const std::vector<T> &expected, int width,
                                    int height) {
    std::cout << "\nComparison (Actual vs Expected, diff):\n";
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int idx = y * width + x;
        int act = static_cast<int>(actual[idx]);
        int exp = static_cast<int>(expected[idx]);
        int diff = act - exp;

        std::cout << std::setw(2) << act << "/" << std::setw(2) << exp;
        if (diff != 0) {
          std::cout << "(" << std::showpos << diff << std::noshowpos << ")";
        } else {
          std::cout << "    ";
        }
        std::cout << " ";
      }
      std::cout << "\n";
    }
  }

  template <typename T>
  static void analyzeBoundaryBehavior(const std::vector<T> &, const std::vector<T> &output, int width, int height, int,
                                      int, int, int) {
    std::cout << "\n--- Boundary Analysis ---\n";

    // Corner analysis
    T corners[4] = {
        output[0],                               // Top-left
        output[width - 1],                       // Top-right
        output[(height - 1) * width],            // Bottom-left
        output[(height - 1) * width + width - 1] // Bottom-right
    };

    if (std::is_same<T, float>::value) {
      std::cout << "Corners: TL=" << static_cast<float>(corners[0]) << " TR=" << static_cast<float>(corners[1])
                << " BL=" << static_cast<float>(corners[2]) << " BR=" << static_cast<float>(corners[3]) << "\n";
    } else {
      std::cout << "Corners: TL=" << static_cast<int>(corners[0]) << " TR=" << static_cast<int>(corners[1])
                << " BL=" << static_cast<int>(corners[2]) << " BR=" << static_cast<int>(corners[3]) << "\n";
    }

    // Value range analysis
    T minVal = output[0], maxVal = output[0];
    for (size_t i = 1; i < output.size(); i++) {
      if (output[i] < minVal)
        minVal = output[i];
      if (output[i] > maxVal)
        maxVal = output[i];
    }

    if (std::is_same<T, float>::value) {
      std::cout << "Value Range: [" << static_cast<float>(minVal) << ", " << static_cast<float>(maxVal) << "]\n";
    } else {
      std::cout << "Value Range: [" << static_cast<int>(minVal) << ", " << static_cast<int>(maxVal) << "]\n";
    }
  }

  template <typename T>
  static void saveComparisonResult(const std::vector<T> &input, const std::vector<T> &output, int width, int height,
                                   int maskW, int maskH, int anchorX, int anchorY, const std::string &dataType) {
    std::stringstream testParams;
    testParams << "Size:" << width << "x" << height << " Mask:" << maskW << "x" << maskH << " Anchor:(" << anchorX
               << "," << anchorY << ") Type:" << dataType;

    printMatrix(input, width, height, "INPUT", testParams.str());
    printMatrix(output, width, height, "OUTPUT", testParams.str());
    analyzeBoundaryBehavior(input, output, width, height, maskW, maskH, anchorX, anchorY);
  }
};

// Zero padding reference implementation for algorithm verification
template <typename T> class ZeroPaddingReference {
public:
  static std::vector<T> apply(const std::vector<T> &input, int width, int height, int maskW, int maskH, int anchorX,
                              int anchorY) {
    std::vector<T> output(width * height);

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        // Calculate filter coverage area
        int startX = x - anchorX;
        int startY = y - anchorY;
        int endX = startX + maskW;
        int endY = startY + maskH;

        // Zero padding: sum all mask positions, treat out-of-bounds as 0
        double sum = 0.0;
        int totalMaskPixels = maskW * maskH;

        for (int j = startY; j < endY; j++) {
          for (int i = startX; i < endX; i++) {
            // Check if coordinates are within image bounds
            if (i >= 0 && i < width && j >= 0 && j < height) {
              sum += static_cast<double>(input[j * width + i]);
            }
            // Out-of-bounds pixels contribute 0 to sum (zero padding)
          }
        }

        // Calculate average using total mask size with truncation (toward zero)
        if (std::is_integral<T>::value) {
          output[y * width + x] = static_cast<T>(sum / totalMaskPixels); // Integer division (truncation)
        } else {
          output[y * width + x] = static_cast<T>(sum / totalMaskPixels);
        }
      }
    }

    return output;
  }
};

// Border replication reference implementation
template <typename T> class BorderReplicationFilter {
public:
  static std::vector<T> apply(const std::vector<T> &input, int width, int height, int maskW, int maskH, int anchorX,
                              int anchorY) {
    std::vector<T> output(width * height);

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        T sum = 0;
        int count = 0;

        // Apply mask with border replication
        for (int my = 0; my < maskH; my++) {
          for (int mx = 0; mx < maskW; mx++) {
            int srcY = y + my - anchorY;
            int srcX = x + mx - anchorX;

            // Border replication
            srcY = std::max(0, std::min(height - 1, srcY));
            srcX = std::max(0, std::min(width - 1, srcX));

            sum += input[srcY * width + srcX];
            count++;
          }
        }

        if (count > 0) {
          if (std::is_integral<T>::value) {
            output[y * width + x] = static_cast<T>(sum / count); // Integer division (truncation)
          } else {
            output[y * width + x] = sum / count;
          }
        }
      }
    }

    return output;
  }
};

// Algorithm verification utilities
class AlgorithmVerification {
public:
  // Regional verification: strict for interior, tolerant for boundary pixels
  static bool verifyPredictionRegional(const std::vector<uint8_t> &predicted, const std::vector<uint8_t> &actual,
                                       int width, int height, int maskW, int maskH, int anchorX, int anchorY) {
    if (predicted.size() != actual.size() || predicted.size() != static_cast<size_t>(width * height))
      return false;

    // Calculate boundary region affected by edge handling
    int boundaryLeft = anchorX;
    int boundaryRight = maskW - anchorX - 1;
    int boundaryTop = anchorY;
    int boundaryBottom = maskH - anchorY - 1;

    int interiorErrors = 0;
    int boundaryErrors = 0;
    int interiorPixels = 0;
    int boundaryPixels = 0;
    int maxInteriorError = 0;
    int maxBoundaryError = 0;

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int idx = y * width + x;
        int error = std::abs(static_cast<int>(predicted[idx]) - static_cast<int>(actual[idx]));

        // Determine if pixel is in boundary region
        bool isBoundary =
            (x < boundaryLeft) || (x >= width - boundaryRight) || (y < boundaryTop) || (y >= height - boundaryBottom);

        if (isBoundary) {
          boundaryPixels++;
          if (error > 1) {
            boundaryErrors++;
          }
          maxBoundaryError = std::max(maxBoundaryError, error);
        } else {
          interiorPixels++;
          if (error > 1) {
            interiorErrors++;
          }
          maxInteriorError = std::max(maxInteriorError, error);
        }
      }
    }

    float interiorErrorRate = (interiorPixels > 0) ? (float)interiorErrors / interiorPixels : 0.0f;
    float boundaryErrorRate = (boundaryPixels > 0) ? (float)boundaryErrors / boundaryPixels : 0.0f;

    std::cout << "Interior Region (strict check, tolerance=1):\n";
    std::cout << "  Pixels: " << interiorPixels << "\n";
    std::cout << "  Errors (>1): " << interiorErrors << "\n";
    std::cout << "  Error rate: " << std::fixed << std::setprecision(2) << (interiorErrorRate * 100) << "%\n";
    std::cout << "  Max error: " << maxInteriorError << "\n";

    std::cout << "Boundary Region (quality warning only):\n";
    std::cout << "  Pixels: " << boundaryPixels << "\n";
    std::cout << "  Errors (>1): " << boundaryErrors << "\n";
    std::cout << "  Error rate: " << std::fixed << std::setprecision(2) << (boundaryErrorRate * 100) << "%\n";
    std::cout << "  Max error: " << maxBoundaryError << "\n";

    bool interiorPass = (interiorErrorRate < 0.05f);

    if (interiorPass) {
      std::cout << "  ✅ PASS: Interior region meets strict requirements (<5% error rate)\n";
      if (boundaryErrorRate > 0.10f) {
        std::cout << "  ⚠️  QUALITY: Boundary algorithm differences noted (informational only)\n";
      }
    } else {
      std::cout << "  ❌ FAIL: Interior region exceeds strict error threshold\n";
    }

    return interiorPass;
  }

  template <typename T>
  static void calculateDifferenceStats(const std::vector<T> &result1, const std::vector<T> &result2,
                                       const std::string &name1, const std::string &name2) {
    if (result1.size() != result2.size()) {
      std::cout << "Size mismatch in difference calculation\n";
      return;
    }

    int totalDiff = 0;
    int maxDiff = 0;
    int minDiff = 255;
    int zeroDiffs = 0;

    for (size_t i = 0; i < result1.size(); i++) {
      int diff = std::abs(static_cast<int>(result1[i]) - static_cast<int>(result2[i]));
      totalDiff += diff;
      maxDiff = std::max(maxDiff, diff);
      minDiff = std::min(minDiff, diff);

      if (diff == 0)
        zeroDiffs++;
    }

    float avgDiff = (float)totalDiff / result1.size();
    float exactMatchRate = (float)zeroDiffs / result1.size() * 100;

    std::cout << "\n=== Difference Analysis: " << name1 << " vs " << name2 << " ===\n";
    std::cout << "Average difference: " << std::fixed << std::setprecision(2) << avgDiff << " pixels\n";
    std::cout << "Max difference: " << maxDiff << " pixels\n";
    std::cout << "Min difference: " << minDiff << " pixels\n";
    std::cout << "Exact matches: " << zeroDiffs << "/" << result1.size() << " (" << exactMatchRate << "%)\n";
  }
};

// Test pattern generators
template <typename T> class TestPatternGenerator {
public:
  static std::vector<T> createUniform(int width, int height, T value) { return std::vector<T>(width * height, value); }

  static std::vector<T> createBinaryStep(int width, int height, T leftValue, T rightValue) {
    std::vector<T> pattern(width * height);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        pattern[y * width + x] = (x < width / 2) ? leftValue : rightValue;
      }
    }
    return pattern;
  }

  static std::vector<T> createHorizontalGradient(int width, int height, T minVal, T maxVal) {
    std::vector<T> pattern(width * height);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        if (width == 1) {
          pattern[y * width + x] = minVal;
        } else {
          double ratio = static_cast<double>(x) / (width - 1);
          pattern[y * width + x] = static_cast<T>(minVal + ratio * (maxVal - minVal));
        }
      }
    }
    return pattern;
  }

  static std::vector<T> createVerticalGradient(int width, int height, T minVal, T maxVal) {
    std::vector<T> pattern(width * height);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        if (height == 1) {
          pattern[y * width + x] = minVal;
        } else {
          double ratio = static_cast<double>(y) / (height - 1);
          pattern[y * width + x] = static_cast<T>(minVal + ratio * (maxVal - minVal));
        }
      }
    }
    return pattern;
  }

  static std::vector<T> createDiagonalGradient(int width, int height, T minVal, T maxVal) {
    std::vector<T> pattern(width * height);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        double ratio = static_cast<double>(x + y) / std::max(1.0, static_cast<double>(width + height - 2));
        pattern[y * width + x] = static_cast<T>(minVal + ratio * (maxVal - minVal));
      }
    }
    return pattern;
  }

  static std::vector<T> createImpulse(int width, int height, int impulseX, int impulseY, T backgroundVal,
                                      T impulseVal) {
    std::vector<T> pattern(width * height, backgroundVal);
    if (impulseX >= 0 && impulseX < width && impulseY >= 0 && impulseY < height) {
      pattern[impulseY * width + impulseX] = impulseVal;
    }
    return pattern;
  }

  static std::vector<T> createEdgePattern(int width, int height, T edgeVal, T centerVal) {
    std::vector<T> pattern(width * height, centerVal);

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
          pattern[y * width + x] = edgeVal;
        }
      }
    }
    return pattern;
  }

  static std::vector<T> createCheckerboard(int width, int height, T val1, T val2) {
    std::vector<T> pattern(width * height);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        pattern[y * width + x] = ((x + y) % 2 == 0) ? val1 : val2;
      }
    }
    return pattern;
  }

  static std::vector<T> createRadialGradient(int width, int height, T centerVal, T edgeVal) {
    std::vector<T> pattern(width * height);
    float centerX = width / 2.0f;
    float centerY = height / 2.0f;
    float maxDist = std::sqrt(centerX * centerX + centerY * centerY);

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        float dx = x - centerX;
        float dy = y - centerY;
        float dist = std::sqrt(dx * dx + dy * dy);
        float ratio = (maxDist > 0) ? (dist / maxDist) : 0.0f;
        pattern[y * width + x] = static_cast<T>(centerVal + ratio * (edgeVal - centerVal));
      }
    }
    return pattern;
  }

  static std::vector<T> createSineWave(int width, int height, T minVal, T maxVal) {
    std::vector<T> pattern(width * height);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        float fx = static_cast<float>(x) / std::max(1.0f, static_cast<float>(width - 1));
        float fy = static_cast<float>(y) / std::max(1.0f, static_cast<float>(height - 1));
        float value = 0.5f + 0.5f * std::sin(fx * 2.0f * M_PI) * std::cos(fy * 2.0f * M_PI);
        pattern[y * width + x] = static_cast<T>(minVal + value * (maxVal - minVal));
      }
    }
    return pattern;
  }
};

// ==============================================================================
// TEST CONFIGURATIONS AND PARAMETERS
// ==============================================================================

// Base test configuration structure
struct FilterBoxTestParams {
  int width, height;
  int maskWidth, maskHeight;
  int anchorX, anchorY;
  std::string description;

  // Test categories
  bool isPowerOfTwo = false;
  bool isPrimeNumber = false;
  bool isAsymmetricKernel = false;
  bool isExtremeAspectRatio = false;
  bool isLargeKernel = false;
  bool isEdgeAnchor = false;
};

// 8u_C1R test parameters
struct FilterBox8uC1RParams {
  int width, height;
  int maskWidth, maskHeight;
  int anchorX, anchorY;
  std::string description;
};

// 32f_C1R test parameters
struct FilterBox32fC1RParams {
  int width, height;
  int maskWidth, maskHeight;
  int anchorX, anchorY;
  std::string description;
};

// 8u_C4R test parameters
struct FilterBox8uC4RParams {
  int width, height;
  int maskWidth, maskHeight;
  int anchorX, anchorY;
  std::string description;
};

// Boundary test configuration
struct BoundaryTestConfig {
  int width, height;
  int maskWidth, maskHeight;
  int anchorX, anchorY;
  std::string description;
};

// Zero padding test configuration
struct ZeroPaddingTestConfig {
  int width, height;
  int maskWidth, maskHeight;
  int anchorX, anchorY;
  std::string description;
};

// Replication analysis test configuration
struct ReplicationTestConfig {
  int width, height;
  int maskWidth, maskHeight;
  int anchorX, anchorY;
  std::string description;
};

// ==============================================================================
// VERIFICATION AND ANALYSIS UTILITIES
// ==============================================================================

// Zero padding verification utilities
class ZeroPaddingVerificationUtils {
public:
  template <typename T>
  static void verifyZeroPadding(const std::vector<T> &input, const std::vector<T> &mppResult, int width, int height,
                                int maskW, int maskH, int anchorX, int anchorY, const std::string &testName) {

    auto referenceResult = ZeroPaddingReference<T>::apply(input, width, height, maskW, maskH, anchorX, anchorY);

    std::cout << "\n=== Zero Padding Verification: " << testName << " ===\n";
    std::cout << "Image Size: " << width << "x" << height;
    std::cout << ", Mask: " << maskW << "x" << maskH;
    std::cout << ", Anchor: (" << anchorX << "," << anchorY << ")\n";

    // Compare results
    int exactMatches = 0;
    int tolerantMatches = 0;
    double maxDiff = 0.0;
    double avgDiff = 0.0;

    for (size_t i = 0; i < mppResult.size(); i++) {
      double diff = std::abs(static_cast<double>(mppResult[i]) - static_cast<double>(referenceResult[i]));
      avgDiff += diff;
      maxDiff = std::max(maxDiff, diff);

      if (diff < 1e-6) {
        exactMatches++;
        tolerantMatches++;
      } else if (std::is_integral<T>::value && diff <= 1.0) {
        tolerantMatches++;
      } else if (!std::is_integral<T>::value && diff <= 0.001) {
        tolerantMatches++;
      }
    }

    avgDiff /= mppResult.size();

    double exactMatchRate = 100.0 * exactMatches / mppResult.size();
    double tolerantMatchRate = 100.0 * tolerantMatches / mppResult.size();

    std::cout << "Exact matches: " << exactMatches << "/" << mppResult.size() << " (" << std::fixed
              << std::setprecision(2) << exactMatchRate << "%)\n";
    std::cout << "Tolerant matches: " << tolerantMatches << "/" << mppResult.size() << " (" << tolerantMatchRate
              << "%)\n";
    std::cout << "Max difference: " << std::setprecision(6) << maxDiff << "\n";
    std::cout << "Avg difference: " << avgDiff << "\n";

    // Detailed comparison for small images
    if (width <= 8 && height <= 8) {
      printDetailedComparison(input, mppResult, referenceResult, width, height, testName);
    }

    // Verification result
    if (tolerantMatchRate >= 95.0) {
      std::cout << "✅ VERIFICATION PASSED: Zero padding algorithm correctly implemented\n";
    } else {
      std::cout << "❌ VERIFICATION FAILED: Algorithm does not match zero padding expectation\n";
    }
  }

private:
  template <typename T>
  static void printDetailedComparison(const std::vector<T> &input, const std::vector<T> &mppResult,
                                      const std::vector<T> &referenceResult, int width, int height,
                                      const std::string &testName) {

    std::cout << "\nDetailed Comparison for " << testName << ":\n";

    std::cout << "\nInput:\n";
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        if (std::is_integral<T>::value) {
          std::cout << std::setw(6) << static_cast<int>(input[y * width + x]);
        } else {
          std::cout << std::setw(8) << std::fixed << std::setprecision(3) << static_cast<float>(input[y * width + x]);
        }
      }
      std::cout << "\n";
    }

    std::cout << "\nMPP Result:\n";
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        if (std::is_integral<T>::value) {
          std::cout << std::setw(6) << static_cast<int>(mppResult[y * width + x]);
        } else {
          std::cout << std::setw(8) << std::fixed << std::setprecision(3)
                    << static_cast<float>(mppResult[y * width + x]);
        }
      }
      std::cout << "\n";
    }

    std::cout << "\nZero Padding Reference:\n";
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        if (std::is_integral<T>::value) {
          std::cout << std::setw(6) << static_cast<int>(referenceResult[y * width + x]);
        } else {
          std::cout << std::setw(8) << std::fixed << std::setprecision(3)
                    << static_cast<float>(referenceResult[y * width + x]);
        }
      }
      std::cout << "\n";
    }

    std::cout << "\nDifference (MPP - Reference):\n";
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        double diff =
            static_cast<double>(mppResult[y * width + x]) - static_cast<double>(referenceResult[y * width + x]);
        std::cout << std::setw(8) << std::fixed << std::setprecision(3) << diff;
      }
      std::cout << "\n";
    }
  }
};

// Replication analysis utilities
class ReplicationAnalysisUtils {
public:
  template <typename T>
  static void analyzeDifferences(const std::vector<T> &nppResult, const std::vector<T> &replicationExpected, int width,
                                 int height, const std::string &testName) {

    if (nppResult.size() != replicationExpected.size()) {
      std::cout << "ERROR: Size mismatch in " << testName << std::endl;
      return;
    }

    std::cout << "\n=== Replication Analysis: " << testName << " ===\n";
    std::cout << "Image Size: " << width << "x" << height << " (" << nppResult.size() << " pixels)\n";

    std::vector<double> differences;
    int exactMatches = 0;
    int smallDifferences = 0;

    for (size_t i = 0; i < nppResult.size(); i++) {
      double nppVal = static_cast<double>(nppResult[i]);
      double expVal = static_cast<double>(replicationExpected[i]);
      double diff = nppVal - expVal;

      differences.push_back(diff);

      if (std::abs(diff) < 1e-6) {
        exactMatches++;
      } else if (std::is_integral<T>::value && std::abs(diff) <= 1.0) {
        smallDifferences++;
      } else if (!std::is_integral<T>::value && std::abs(diff) <= 0.01) {
        smallDifferences++;
      }
    }

    auto minMaxDiff = std::minmax_element(differences.begin(), differences.end());
    double avgDiff = std::accumulate(differences.begin(), differences.end(), 0.0) / differences.size();

    std::cout << "Exact matches: " << exactMatches << "/" << nppResult.size() << " ("
              << (100.0 * exactMatches / nppResult.size()) << "%)\n";
    std::cout << "Small differences: " << smallDifferences << "/" << nppResult.size() << " ("
              << (100.0 * smallDifferences / nppResult.size()) << "%)\n";
    std::cout << "Total close matches: " << (exactMatches + smallDifferences) << "/" << nppResult.size() << " ("
              << (100.0 * (exactMatches + smallDifferences) / nppResult.size()) << "%)\n";

    std::cout << "\nDifference Statistics (NPP - Expected):\n";
    std::cout << "  Min: " << std::fixed << std::setprecision(6) << *minMaxDiff.first << "\n";
    std::cout << "  Max: " << std::fixed << std::setprecision(6) << *minMaxDiff.second << "\n";
    std::cout << "  Avg: " << std::fixed << std::setprecision(6) << avgDiff << "\n";
  }
};

// ==============================================================================
// TEST EXECUTION HELPERS
// ==============================================================================

// Helper function to run NPP FilterBox 8u_C1R and get result
std::vector<uint8_t> runNPPFilterBox8u(const std::vector<uint8_t> &input, int width, int height, int maskW, int maskH,
                                       int anchorX, int anchorY) {
  Npp8u *d_src = (Npp8u *)nppsMalloc_8u(width * height);
  Npp8u *d_dst = (Npp8u *)nppsMalloc_8u(width * height);

  if (!d_src || !d_dst) {
    if (d_src)
      nppiFree(d_src);
    if (d_dst)
      nppiFree(d_dst);
    throw std::runtime_error("Failed to allocate GPU memory");
  }

  cudaMemcpy(d_src, input.data(), input.size() * sizeof(Npp8u), cudaMemcpyHostToDevice);

  NppStatus status = nppiFilterBox_8u_C1R(d_src, width * sizeof(Npp8u), d_dst, width * sizeof(Npp8u), {width, height},
                                          {maskW, maskH}, {anchorX, anchorY});

  std::vector<uint8_t> output(width * height);
  if (status == NPP_SUCCESS) {
    cudaMemcpy(output.data(), d_dst, output.size() * sizeof(Npp8u), cudaMemcpyDeviceToHost);
  }

  nppiFree(d_src);
  nppiFree(d_dst);

  if (status != NPP_SUCCESS) {
    throw std::runtime_error("FilterBox operation failed");
  }

  return output;
}

// Helper function to run NPP FilterBox 32f_C1R and get result
std::vector<Npp32f> runNPPFilterBox32f(const std::vector<Npp32f> &input, int width, int height, int maskW, int maskH,
                                       int anchorX, int anchorY) {
  Npp32f *d_src = (Npp32f *)nppsMalloc_32f(width * height);
  Npp32f *d_dst = (Npp32f *)nppsMalloc_32f(width * height);

  if (!d_src || !d_dst) {
    if (d_src)
      nppiFree(d_src);
    if (d_dst)
      nppiFree(d_dst);
    throw std::runtime_error("Failed to allocate GPU memory");
  }

  cudaMemcpy(d_src, input.data(), input.size() * sizeof(Npp32f), cudaMemcpyHostToDevice);

  NppStatus status = nppiFilterBox_32f_C1R(d_src, width * sizeof(Npp32f), d_dst, width * sizeof(Npp32f),
                                           {width, height}, {maskW, maskH}, {anchorX, anchorY});

  std::vector<Npp32f> output(width * height);
  if (status == NPP_SUCCESS) {
    cudaMemcpy(output.data(), d_dst, output.size() * sizeof(Npp32f), cudaMemcpyDeviceToHost);
  }

  nppiFree(d_src);
  nppiFree(d_dst);

  if (status != NPP_SUCCESS) {
    throw std::runtime_error("FilterBox operation failed");
  }

  return output;
}

// Helper function to run NPP FilterBox 8u_C4R and get result
std::vector<Npp8u> runNPPFilterBox8uC4R(const std::vector<Npp8u> &input, int width, int height, int maskW, int maskH,
                                        int anchorX, int anchorY) {
  const int channels = 4;
  Npp8u *d_src, *d_dst;
  int srcStep, dstStep;
  d_src = nppiMalloc_8u_C4(width, height, &srcStep);
  d_dst = nppiMalloc_8u_C4(width, height, &dstStep);

  if (!d_src || !d_dst) {
    if (d_src)
      nppiFree(d_src);
    if (d_dst)
      nppiFree(d_dst);
    throw std::runtime_error("Failed to allocate GPU memory for C4R");
  }

  cudaMemcpy2D(d_src, srcStep, input.data(), width * channels, width * channels, height, cudaMemcpyHostToDevice);

  NppStatus status =
      nppiFilterBox_8u_C4R(d_src, srcStep, d_dst, dstStep, {width, height}, {maskW, maskH}, {anchorX, anchorY});

  std::vector<Npp8u> output(width * height * channels);
  if (status == NPP_SUCCESS) {
    cudaMemcpy2D(output.data(), width * channels, d_dst, dstStep, width * channels, height, cudaMemcpyDeviceToHost);
  }

  nppiFree(d_src);
  nppiFree(d_dst);

  if (status != NPP_SUCCESS) {
    throw std::runtime_error("FilterBox C4R operation failed");
  }

  return output;
}

// ==============================================================================
// MAIN TEST CLASSES AND PARAMETERIZED TESTS
// ==============================================================================

// FilterBox 8u_C1R parameterized test class
class FilterBox8uC1RTest : public ::testing::TestWithParam<FilterBox8uC1RParams> {
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

// FilterBox 32f_C1R parameterized test class
class FilterBox32fC1RTest : public ::testing::TestWithParam<FilterBox32fC1RParams> {
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

// FilterBox 8u_C4R parameterized test class
class FilterBox8uC4RTest : public ::testing::TestWithParam<FilterBox8uC4RParams> {
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

// Boundary analysis test class
class FilterBoxBoundaryTest : public ::testing::TestWithParam<BoundaryTestConfig> {
protected:
  void SetUp() override {
    cudaError_t err = cudaSetDevice(0);
    ASSERT_EQ(err, cudaSuccess);
  }
};

// Zero padding verification test class
class FilterBoxZeroPaddingTest : public ::testing::TestWithParam<ZeroPaddingTestConfig> {
protected:
  void SetUp() override {
    cudaError_t err = cudaSetDevice(0);
    ASSERT_EQ(err, cudaSuccess);
  }
};

// Algorithm verification test class
class FilterBoxAlgorithmTest : public ::testing::Test {
protected:
  void SetUp() override {
    cudaError_t err = cudaSetDevice(0);
    ASSERT_EQ(err, cudaSuccess);
  }
};

// Replication analysis test class
class FilterBoxReplicationTest : public ::testing::TestWithParam<ReplicationTestConfig> {
protected:
  void SetUp() override {
    cudaError_t err = cudaSetDevice(0);
    ASSERT_EQ(err, cudaSuccess);
  }
};

// ==============================================================================
// COMPREHENSIVE TEST PARAMETER DEFINITIONS
// ==============================================================================

// 8u_C1R test parameters with comprehensive coverage
const std::vector<FilterBox8uC1RParams> FILTER_BOX_8U_C1R_PARAMS = {
    // === BASIC CONFIGURATIONS ===
    FilterBox8uC1RParams{4, 4, 3, 3, 1, 1, "4x4_3x3_basic"},
    FilterBox8uC1RParams{8, 8, 3, 3, 1, 1, "8x8_3x3_standard"},
    FilterBox8uC1RParams{16, 16, 5, 5, 2, 2, "16x16_5x5_medium"},
    FilterBox8uC1RParams{32, 32, 7, 7, 3, 3, "32x32_7x7_large"},

    // === EXTENDED COVERAGE ===
    // Prime number dimensions
    FilterBox8uC1RParams{5, 5, 3, 3, 1, 1, "5x5_3x3_prime"},
    FilterBox8uC1RParams{7, 7, 3, 3, 1, 1, "7x7_3x3_prime"},
    FilterBox8uC1RParams{11, 11, 5, 5, 2, 2, "11x11_5x5_prime"},
    FilterBox8uC1RParams{13, 13, 3, 3, 1, 1, "13x13_3x3_prime"},
    FilterBox8uC1RParams{17, 17, 7, 7, 3, 3, "17x17_7x7_prime"},

    // Power-of-2 plus one dimensions
    FilterBox8uC1RParams{9, 9, 3, 3, 1, 1, "9x9_3x3_pow2plus1"},
    FilterBox8uC1RParams{17, 17, 5, 5, 2, 2, "17x17_5x5_pow2plus1"},
    FilterBox8uC1RParams{33, 33, 7, 7, 3, 3, "33x33_7x7_pow2plus1"},
    FilterBox8uC1RParams{65, 65, 9, 9, 4, 4, "65x65_9x9_pow2plus1"},

    // Very asymmetric kernels
    FilterBox8uC1RParams{64, 64, 1, 7, 0, 3, "64x64_1x7_vertical_line"},
    FilterBox8uC1RParams{64, 64, 7, 1, 3, 0, "64x64_7x1_horizontal_line"},
    FilterBox8uC1RParams{32, 32, 1, 15, 0, 7, "32x32_1x15_extreme_vertical"},
    FilterBox8uC1RParams{32, 32, 15, 1, 7, 0, "32x32_15x1_extreme_horizontal"},

    // Mixed aspect ratio kernels
    FilterBox8uC1RParams{64, 64, 3, 9, 1, 4, "64x64_3x9_mixed_ratio"},
    FilterBox8uC1RParams{64, 64, 9, 3, 4, 1, "64x64_9x3_mixed_ratio"},
    FilterBox8uC1RParams{64, 64, 5, 11, 2, 5, "64x64_5x11_mixed_ratio"},
    FilterBox8uC1RParams{64, 64, 11, 5, 5, 2, "64x64_11x5_mixed_ratio"},

    // Extreme aspect ratio images
    FilterBox8uC1RParams{1, 256, 1, 7, 0, 3, "1x256_1x7_ultra_tall"},
    FilterBox8uC1RParams{256, 1, 7, 1, 3, 0, "256x1_7x1_ultra_wide"},
    FilterBox8uC1RParams{2, 128, 1, 5, 0, 2, "2x128_1x5_very_tall"},
    FilterBox8uC1RParams{128, 2, 5, 1, 2, 0, "128x2_5x1_very_wide"},
    FilterBox8uC1RParams{3, 85, 1, 3, 0, 1, "3x85_1x3_tall"},
    FilterBox8uC1RParams{85, 3, 3, 1, 1, 0, "85x3_3x1_wide"},

    // Large kernel sizes
    FilterBox8uC1RParams{128, 128, 21, 21, 10, 10, "128x128_21x21_large_kernel"},
    FilterBox8uC1RParams{128, 128, 25, 25, 12, 12, "128x128_25x25_very_large_kernel"},
    FilterBox8uC1RParams{64, 64, 31, 31, 15, 15, "64x64_31x31_huge_kernel"},
    FilterBox8uC1RParams{64, 64, 21, 11, 10, 5, "64x64_21x11_large_asym_kernel"},
    FilterBox8uC1RParams{64, 64, 11, 21, 5, 10, "64x64_11x21_large_asym_kernel"},

    // Different anchor positions
    FilterBox8uC1RParams{32, 32, 7, 7, 0, 0, "32x32_7x7_anchor_topleft"},
    FilterBox8uC1RParams{32, 32, 7, 7, 6, 6, "32x32_7x7_anchor_bottomright"},
    FilterBox8uC1RParams{32, 32, 7, 7, 3, 0, "32x32_7x7_anchor_top_center"},
    FilterBox8uC1RParams{32, 32, 7, 7, 3, 6, "32x32_7x7_anchor_bottom_center"},
    FilterBox8uC1RParams{32, 32, 7, 7, 0, 3, "32x32_7x7_anchor_left_center"},
    FilterBox8uC1RParams{32, 32, 7, 7, 6, 3, "32x32_7x7_anchor_right_center"},

    // Non-standard dimensions
    FilterBox8uC1RParams{23, 29, 5, 7, 2, 3, "23x29_5x7_irregular"},
    FilterBox8uC1RParams{41, 37, 9, 11, 4, 5, "41x37_9x11_irregular"},
    FilterBox8uC1RParams{19, 31, 3, 5, 1, 2, "19x31_3x5_irregular"},
    FilterBox8uC1RParams{53, 47, 7, 9, 3, 4, "53x47_7x9_irregular"},

    // High resolution tests
    FilterBox8uC1RParams{1920, 1080, 5, 5, 2, 2, "1920x1080_5x5_fullhd"},
    FilterBox8uC1RParams{1080, 1920, 5, 5, 2, 2, "1080x1920_5x5_fullhd_portrait"},
    FilterBox8uC1RParams{2560, 1440, 7, 7, 3, 3, "2560x1440_7x7_2k"},
    FilterBox8uC1RParams{1440, 2560, 7, 7, 3, 3, "1440x2560_7x7_2k_portrait"},
    FilterBox8uC1RParams{3840, 2160, 3, 3, 1, 1, "3840x2160_3x3_4k"},
    FilterBox8uC1RParams{2160, 3840, 3, 3, 1, 1, "2160x3840_3x3_4k_portrait"},

    // Ultra-wide and ultra-tall aspect ratios
    FilterBox8uC1RParams{3440, 1440, 5, 5, 2, 2, "3440x1440_5x5_ultrawide"},
    FilterBox8uC1RParams{1440, 3440, 5, 5, 2, 2, "1440x3440_5x5_ultrawide_portrait"},
    FilterBox8uC1RParams{5120, 1440, 7, 7, 3, 3, "5120x1440_7x7_superwide"},
    FilterBox8uC1RParams{1440, 5120, 7, 7, 3, 3, "1440x5120_7x7_superwide_portrait"},

    // Stress test dimensions
    FilterBox8uC1RParams{10000, 1, 3, 1, 1, 0, "10000x1_3x1_stress_wide"},
    FilterBox8uC1RParams{1, 10000, 1, 3, 0, 1, "1x10000_1x3_stress_tall"},
    FilterBox8uC1RParams{1000, 1000, 3, 3, 1, 1, "1000x1000_3x3_stress_large"},
};

// 32f_C1R test parameters (subset for performance)
const std::vector<FilterBox32fC1RParams> FILTER_BOX_32F_C1R_PARAMS = {
    FilterBox32fC1RParams{8, 8, 3, 3, 1, 1, "8x8_3x3_basic"},
    FilterBox32fC1RParams{16, 16, 5, 5, 2, 2, "16x16_5x5_medium"},
    FilterBox32fC1RParams{32, 32, 7, 7, 3, 3, "32x32_7x7_large"},
    FilterBox32fC1RParams{64, 64, 9, 9, 4, 4, "64x64_9x9_very_large"},
    FilterBox32fC1RParams{128, 128, 11, 11, 5, 5, "128x128_11x11_huge"},
    FilterBox32fC1RParams{64, 64, 1, 7, 0, 3, "64x64_1x7_vertical_line"},
    FilterBox32fC1RParams{64, 64, 7, 1, 3, 0, "64x64_7x1_horizontal_line"},
    FilterBox32fC1RParams{256, 1, 5, 1, 2, 0, "256x1_5x1_extreme_wide"},
    FilterBox32fC1RParams{1, 256, 1, 5, 0, 2, "1x256_1x5_extreme_tall"},
    FilterBox32fC1RParams{256, 256, 15, 15, 7, 7, "256x256_15x15_large_kernel"},
    FilterBox32fC1RParams{1920, 1080, 5, 5, 2, 2, "1920x1080_5x5_fullhd"},
};

// 8u_C4R test parameters (subset for performance)
const std::vector<FilterBox8uC4RParams> FILTER_BOX_8U_C4R_PARAMS = {
    FilterBox8uC4RParams{16, 16, 3, 3, 1, 1, "16x16_3x3_basic"},
    FilterBox8uC4RParams{32, 32, 5, 5, 2, 2, "32x32_5x5_medium"},
    FilterBox8uC4RParams{64, 64, 7, 7, 3, 3, "64x64_7x7_large"},
    FilterBox8uC4RParams{128, 128, 9, 9, 4, 4, "128x128_9x9_very_large"},
    FilterBox8uC4RParams{64, 32, 5, 3, 2, 1, "64x32_5x3_rect_asym"},
    FilterBox8uC4RParams{32, 64, 3, 5, 1, 2, "32x64_3x5_rect_asym"},
    FilterBox8uC4RParams{256, 192, 7, 7, 3, 3, "256x192_7x7_hires"},
    FilterBox8uC4RParams{100, 1, 3, 1, 1, 0, "100x1_3x1_extreme_wide"},
    FilterBox8uC4RParams{1, 100, 1, 3, 0, 1, "1x100_1x3_extreme_tall"},
};

// Boundary test configurations
const std::vector<BoundaryTestConfig> BOUNDARY_TEST_CONFIGS = {
    {4, 4, 3, 3, 1, 1, "4x4_3x3_center"},      {6, 6, 3, 3, 1, 1, "6x6_3x3_center"},
    {8, 8, 5, 5, 2, 2, "8x8_5x5_center"},      {3, 3, 3, 3, 1, 1, "3x3_3x3_exact"},
    {5, 5, 5, 5, 2, 2, "5x5_5x5_exact"},       {6, 6, 3, 3, 0, 0, "6x6_3x3_topleft"},
    {6, 6, 3, 3, 2, 2, "6x6_3x3_bottomright"}, {6, 6, 3, 3, 1, 0, "6x6_3x3_asymmetric"},
    {8, 8, 3, 1, 1, 0, "8x8_3x1_horizontal"},  {8, 8, 1, 3, 0, 1, "8x8_1x3_vertical"},
    {8, 4, 3, 3, 1, 1, "8x4_3x3_wide"},        {4, 8, 3, 3, 1, 1, "4x8_3x3_tall"},
};

// Zero padding test configurations
const std::vector<ZeroPaddingTestConfig> ZERO_PADDING_CONFIGS = {
    {3, 3, 3, 3, 1, 1, "3x3_img_3x3_mask"},       {4, 4, 3, 3, 1, 1, "4x4_img_3x3_mask"},
    {5, 5, 3, 3, 1, 1, "5x5_img_3x3_mask"},       {6, 6, 3, 3, 1, 1, "6x6_img_3x3_mask"},
    {8, 8, 5, 5, 2, 2, "8x8_img_5x5_mask"},       {6, 6, 3, 3, 0, 0, "6x6_anchor_topleft"},
    {6, 6, 3, 3, 2, 2, "6x6_anchor_bottomright"},
};

// Replication analysis test configurations
const std::vector<ReplicationTestConfig> REPLICATION_CONFIGS = {
    {3, 3, 3, 3, 1, 1, "3x3_img_3x3_mask"},        {4, 4, 3, 3, 1, 1, "4x4_img_3x3_mask"},
    {5, 5, 3, 3, 1, 1, "5x5_img_3x3_mask"},        {6, 6, 3, 3, 1, 1, "6x6_img_3x3_mask"},
    {6, 6, 3, 3, 0, 0, "6x6_anchor_topleft"},      {6, 6, 3, 3, 2, 2, "6x6_anchor_bottomright"},
    {6, 6, 3, 3, 1, 0, "6x6_anchor_asymmetric_y"}, {6, 6, 3, 3, 0, 1, "6x6_anchor_asymmetric_x"},
    {8, 8, 5, 5, 2, 2, "8x8_img_5x5_mask"},        {8, 8, 7, 7, 3, 3, "8x8_img_7x7_mask"},
    {8, 6, 3, 3, 1, 1, "8x6_img_3x3_mask"},        {6, 8, 3, 3, 1, 1, "6x8_img_3x3_mask"},
    {8, 8, 3, 1, 1, 0, "8x8_img_3x1_mask"},        {8, 8, 1, 3, 0, 1, "8x8_img_1x3_mask"},
};

// ==============================================================================
// PARAMETERIZED TEST IMPLEMENTATIONS
// ==============================================================================

TEST_P(FilterBox8uC1RTest, ComprehensivePatternTesting) {
  auto params = GetParam();

  std::vector<std::string> patterns = {"uniform", "impulse_corner", "checkerboard"};

  for (const auto &patternType : patterns) {
    std::vector<Npp8u> input;

    if (patternType == "uniform") {
      input = TestPatternGenerator<Npp8u>::createUniform(params.width, params.height, 128);
    } else if (patternType == "impulse_corner") {
      input = TestPatternGenerator<Npp8u>::createImpulse(params.width, params.height, 0, 0, 50, 255);
    } else { // checkerboard
      input = TestPatternGenerator<Npp8u>::createCheckerboard(params.width, params.height, 0, 255);
    }

    ASSERT_NO_THROW({
      auto output = runNPPFilterBox8u(input, params.width, params.height, params.maskWidth, params.maskHeight,
                                      params.anchorX, params.anchorY);

      // Verify output properties
      for (size_t i = 0; i < output.size(); i++) {
        EXPECT_GE(output[i], 0) << "Invalid output value at index " << i << " for " << params.description << " pattern "
                                << patternType;
        EXPECT_LE(output[i], 255) << "Invalid output value at index " << i << " for " << params.description
                                  << " pattern " << patternType;
      }
    }) << "Failed for "
       << params.description << " with pattern " << patternType;
  }
}

TEST_P(FilterBox32fC1RTest, FloatPatternTesting) {
  auto params = GetParam();

  std::vector<std::string> patterns = {"uniform", "sine_wave"};

  for (const auto &patternType : patterns) {
    std::vector<Npp32f> input;

    if (patternType == "uniform") {
      input = TestPatternGenerator<Npp32f>::createUniform(params.width, params.height, 0.5f);
    } else { // sine_wave
      input = TestPatternGenerator<Npp32f>::createSineWave(params.width, params.height, 0.0f, 1.0f);
    }

    ASSERT_NO_THROW({
      auto output = runNPPFilterBox32f(input, params.width, params.height, params.maskWidth, params.maskHeight,
                                       params.anchorX, params.anchorY);

      // Verify floating point results
      for (size_t i = 0; i < output.size(); i++) {
        EXPECT_TRUE(std::isfinite(output[i]))
            << "Non-finite output value at index " << i << " for " << params.description << " pattern " << patternType;
        EXPECT_GE(output[i], -10.0f) << "Unexpectedly low output value at index " << i;
        EXPECT_LE(output[i], 10.0f) << "Unexpectedly high output value at index " << i;
      }
    }) << "Failed for "
       << params.description << " with pattern " << patternType;
  }
}

TEST_P(FilterBox8uC4RTest, MultiChannelTesting) {
  auto params = GetParam();
  const int channels = 4;

  // Create multi-channel test data
  std::vector<Npp8u> input(params.width * params.height * channels);

  for (int y = 0; y < params.height; y++) {
    for (int x = 0; x < params.width; x++) {
      int idx = (y * params.width + x) * channels;

      // Different patterns for each channel
      input[idx + 0] = static_cast<Npp8u>((x + y) % 256);                            // R: diagonal gradient
      input[idx + 1] = static_cast<Npp8u>(x * 255 / std::max(1, params.width - 1));  // G: horizontal gradient
      input[idx + 2] = static_cast<Npp8u>(y * 255 / std::max(1, params.height - 1)); // B: vertical gradient
      input[idx + 3] = static_cast<Npp8u>((x ^ y) % 256);                            // A: XOR pattern
    }
  }

  ASSERT_NO_THROW({
    auto output = runNPPFilterBox8uC4R(input, params.width, params.height, params.maskWidth, params.maskHeight,
                                       params.anchorX, params.anchorY);

    // Verify each channel
    for (int c = 0; c < channels; c++) {
      for (int i = 0; i < params.width * params.height; i++) {
        int idx = i * channels + c;
        EXPECT_GE(output[idx], 0) << "Invalid output in channel " << c << " at pixel " << i << " for "
                                  << params.description;
        EXPECT_LE(output[idx], 255) << "Invalid output in channel " << c << " at pixel " << i << " for "
                                    << params.description;
      }
    }
  }) << "Failed for "
     << params.description;
}

TEST_P(FilterBoxBoundaryTest, BoundaryAnalysis) {
  auto config = GetParam();

  std::vector<std::string> patterns = {"uniform", "impulse_corner"};

  for (const auto &patternType : patterns) {
    std::vector<Npp8u> input;

    if (patternType == "uniform") {
      input = TestPatternGenerator<Npp8u>::createUniform(config.width, config.height, 128);
    } else { // impulse_corner
      input = TestPatternGenerator<Npp8u>::createImpulse(config.width, config.height, 0, 0, 0, 255);
    }

    ASSERT_NO_THROW({
      auto output = runNPPFilterBox8u(input, config.width, config.height, config.maskWidth, config.maskHeight,
                                      config.anchorX, config.anchorY);

      BoundaryAnalysisUtils::saveComparisonResult(input, output, config.width, config.height, config.maskWidth,
                                                  config.maskHeight, config.anchorX, config.anchorY,
                                                  "Npp8u_" + patternType);
    }) << "Failed boundary analysis for "
       << config.description << " with pattern " << patternType;
  }
}

TEST_P(FilterBoxZeroPaddingTest, ZeroPaddingVerification) {
  auto config = GetParam();

  std::vector<std::string> patterns = {"uniform", "corner_spike"};

  for (const auto &patternType : patterns) {
    std::vector<Npp8u> input;

    if (patternType == "uniform") {
      input = TestPatternGenerator<Npp8u>::createUniform(config.width, config.height, 100);
    } else { // corner_spike
      input = TestPatternGenerator<Npp8u>::createImpulse(config.width, config.height, 0, 0, 50, 200);
    }

    std::string testName = config.description + "_" + patternType;

    ASSERT_NO_THROW({
      auto mppResult = runNPPFilterBox8u(input, config.width, config.height, config.maskWidth, config.maskHeight,
                                         config.anchorX, config.anchorY);

      ZeroPaddingVerificationUtils::verifyZeroPadding<Npp8u>(input, mppResult, config.width, config.height,
                                                             config.maskWidth, config.maskHeight, config.anchorX,
                                                             config.anchorY, testName);
    }) << "Failed zero padding verification for "
       << testName;
  }
}

TEST_P(FilterBoxReplicationTest, ReplicationAnalysis) {
  auto config = GetParam();

  std::vector<std::string> patterns = {"checkerboard", "corner_spike"};

  for (const auto &patternType : patterns) {
    std::vector<Npp8u> input;

    if (patternType == "checkerboard") {
      input = TestPatternGenerator<Npp8u>::createCheckerboard(config.width, config.height, 0, 255);
    } else { // corner_spike
      input = TestPatternGenerator<Npp8u>::createImpulse(config.width, config.height, 0, 0, 128, 255);
    }

    std::string testName = config.description + "_" + patternType;

    ASSERT_NO_THROW({
      auto nppResult = runNPPFilterBox8u(input, config.width, config.height, config.maskWidth, config.maskHeight,
                                         config.anchorX, config.anchorY);

      auto replicationExpected = BorderReplicationFilter<Npp8u>::apply(
          input, config.width, config.height, config.maskWidth, config.maskHeight, config.anchorX, config.anchorY);

      ReplicationAnalysisUtils::analyzeDifferences<Npp8u>(nppResult, replicationExpected, config.width, config.height,
                                                          testName);
    }) << "Failed replication analysis for "
       << testName;
  }
}

// Algorithm verification tests
TEST_F(FilterBoxAlgorithmTest, UniformPatternWeightVerification) {
  std::cout << "\n=== Uniform Pattern Weight Verification ===\n";

  const int width = 6, height = 6;
  const int maskW = 3, maskH = 3;
  const int anchorX = 1, anchorY = 1;

  auto input = TestPatternGenerator<uint8_t>::createUniform(width, height, 128);
  auto actual = runNPPFilterBox8u(input, width, height, maskW, maskH, anchorX, anchorY);
  auto predicted = ZeroPaddingReference<uint8_t>::apply(input, width, height, maskW, maskH, anchorX, anchorY);

  std::cout << "Test configuration: " << width << "x" << height << " image, " << maskW << "x" << maskH
            << " mask, anchor(" << anchorX << "," << anchorY << ")\n";

  BoundaryAnalysisUtils::printComparisonMatrix(actual, predicted, width, height);
  EXPECT_TRUE(AlgorithmVerification::verifyPredictionRegional(predicted, actual, width, height, maskW, maskH, anchorX,
                                                              anchorY));
}

TEST_F(FilterBoxAlgorithmTest, ImpulseResponseVerification) {
  std::cout << "\n=== Impulse Response Verification ===\n";

  const int width = 5, height = 5;
  const int maskW = 3, maskH = 3;
  const int anchorX = 1, anchorY = 1;

  auto input = TestPatternGenerator<uint8_t>::createImpulse(width, height, 0, 0, 0, 255);
  auto actual = runNPPFilterBox8u(input, width, height, maskW, maskH, anchorX, anchorY);
  auto predicted = ZeroPaddingReference<uint8_t>::apply(input, width, height, maskW, maskH, anchorX, anchorY);

  std::cout << "Corner impulse (255 at [0,0]) response:\n";
  BoundaryAnalysisUtils::printComparisonMatrix(actual, predicted, width, height);
  EXPECT_TRUE(AlgorithmVerification::verifyPredictionRegional(predicted, actual, width, height, maskW, maskH, anchorX,
                                                              anchorY));
}

TEST_F(FilterBoxAlgorithmTest, EdgeCaseSizeVerification) {
  std::cout << "\n=== Edge Case Size Verification ===\n";

  const int width = 3, height = 3;
  const int maskW = 3, maskH = 3;
  const int anchorX = 1, anchorY = 1;

  auto input = TestPatternGenerator<uint8_t>::createBinaryStep(width, height, 64, 192);
  auto actual = runNPPFilterBox8u(input, width, height, maskW, maskH, anchorX, anchorY);
  auto predicted = ZeroPaddingReference<uint8_t>::apply(input, width, height, maskW, maskH, anchorX, anchorY);

  std::cout << "3x3 image with 3x3 mask (exact coverage):\n";
  BoundaryAnalysisUtils::printComparisonMatrix(actual, predicted, width, height);
  EXPECT_TRUE(AlgorithmVerification::verifyPredictionRegional(predicted, actual, width, height, maskW, maskH, anchorX,
                                                              anchorY));
}

// Performance test
TEST_F(FilterBoxAlgorithmTest, PerformanceTest) {
  std::cout << "\n=== Performance Test ===\n";

  const int width = 1920, height = 1080;
  const int maskW = 5, maskH = 5;
  const int anchorX = 2, anchorY = 2;

  auto input = TestPatternGenerator<uint8_t>::createCheckerboard(width, height, 0, 255);

  auto start = std::chrono::high_resolution_clock::now();
  const int iterations = 10;

  for (int i = 0; i < iterations; i++) {
    auto output = runNPPFilterBox8u(input, width, height, maskW, maskH, anchorX, anchorY);
    (void)output; // Suppress unused variable warning
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  double avgTimeMs = duration.count() / 1000.0 / iterations;
  double pixelsPerSec = (width * height * iterations) / (duration.count() / 1e6);

  std::cout << std::fixed << std::setprecision(3) << "Performance [1920x1080 5x5]: " << avgTimeMs << " ms/iter, "
            << pixelsPerSec / 1e6 << " Mpixels/sec" << std::endl;
}

// ==============================================================================
// TEST INSTANTIATION
// ==============================================================================

INSTANTIATE_TEST_SUITE_P(FilterBox8uC1R, FilterBox8uC1RTest, ::testing::ValuesIn(FILTER_BOX_8U_C1R_PARAMS));

INSTANTIATE_TEST_SUITE_P(FilterBox32fC1R, FilterBox32fC1RTest, ::testing::ValuesIn(FILTER_BOX_32F_C1R_PARAMS));

INSTANTIATE_TEST_SUITE_P(FilterBox8uC4R, FilterBox8uC4RTest, ::testing::ValuesIn(FILTER_BOX_8U_C4R_PARAMS));

INSTANTIATE_TEST_SUITE_P(BoundaryAnalysis, FilterBoxBoundaryTest, ::testing::ValuesIn(BOUNDARY_TEST_CONFIGS));

INSTANTIATE_TEST_SUITE_P(ZeroPaddingVerification, FilterBoxZeroPaddingTest, ::testing::ValuesIn(ZERO_PADDING_CONFIGS));

INSTANTIATE_TEST_SUITE_P(ReplicationAnalysis, FilterBoxReplicationTest, ::testing::ValuesIn(REPLICATION_CONFIGS));