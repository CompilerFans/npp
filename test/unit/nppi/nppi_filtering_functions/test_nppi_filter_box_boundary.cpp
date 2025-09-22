#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "npp.h"

// Utility class for matrix printing and analysis
class BoundaryAnalysisUtils {
public:
  // Print 2D matrix with formatted output
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

  // Analyze boundary behavior patterns
  template <typename T>
  static void analyzeBoundaryBehavior(const std::vector<T> & /* input */, const std::vector<T> &output, int width,
                                      int height, int /* maskW */, int /* maskH */, int /* anchorX */,
                                      int /* anchorY */) {
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

    // Edge analysis
    if (width > 2 && height > 2) {
      int midY = height / 2;
      int midX = width / 2;

      if (std::is_same<T, float>::value) {
        std::cout << "Edges: Left=" << static_cast<float>(output[midY * width])
                  << " Right=" << static_cast<float>(output[midY * width + width - 1])
                  << " Top=" << static_cast<float>(output[midX])
                  << " Bottom=" << static_cast<float>(output[(height - 1) * width + midX]) << "\n";
        std::cout << "Center: " << static_cast<float>(output[midY * width + midX]) << "\n";
      } else {
        std::cout << "Edges: Left=" << static_cast<int>(output[midY * width])
                  << " Right=" << static_cast<int>(output[midY * width + width - 1])
                  << " Top=" << static_cast<int>(output[midX])
                  << " Bottom=" << static_cast<int>(output[(height - 1) * width + midX]) << "\n";
        std::cout << "Center: " << static_cast<int>(output[midY * width + midX]) << "\n";
      }
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

  // Save matrix comparison results
  template <typename T>
  static void saveComparisonResult(const std::vector<T> &input, const std::vector<T> &output, int width, int height,
                                   int maskW, int maskH, int anchorX, int anchorY, const std::string &dataType) {
    std::stringstream testParams;
    testParams << "Size:" << width << "x" << height << " Mask:" << maskW << "x" << maskH << " Anchor:(" << anchorX
               << "," << anchorY << ") "
               << "Type:" << dataType;

    printMatrix(input, width, height, "INPUT", testParams.str());
    printMatrix(output, width, height, "OUTPUT", testParams.str());

    // Calculate and print boundary analysis
    analyzeBoundaryBehavior(input, output, width, height, maskW, maskH, anchorX, anchorY);
  }
};

// Test patterns generator
template <typename T> class TestPatterns {
public:
  // Uniform pattern
  static std::vector<T> createUniform(int width, int height, T value) { return std::vector<T>(width * height, value); }

  // Binary step pattern (left=val1, right=val2)
  static std::vector<T> createBinaryStep(int width, int height, T val1, T val2) {
    std::vector<T> pattern(width * height);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        pattern[y * width + x] = (x < width / 2) ? val1 : val2;
      }
    }
    return pattern;
  }

  // Horizontal gradient
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

  // Impulse at specified position
  static std::vector<T> createImpulse(int width, int height, int impulseX, int impulseY, T backgroundVal,
                                      T impulseVal) {
    std::vector<T> pattern(width * height, backgroundVal);
    if (impulseX >= 0 && impulseX < width && impulseY >= 0 && impulseY < height) {
      pattern[impulseY * width + impulseX] = impulseVal;
    }
    return pattern;
  }
};

// Test configuration structure
struct BoundaryTestConfig {
  int width, height;
  int maskWidth, maskHeight;
  int anchorX, anchorY;
  std::string description;
};

// Common test configurations
static const std::vector<BoundaryTestConfig> BOUNDARY_TEST_CONFIGS = {
    // Small images
    {4, 4, 3, 3, 1, 1, "4x4_3x3_center"},
    {6, 6, 3, 3, 1, 1, "6x6_3x3_center"},
    {8, 8, 5, 5, 2, 2, "8x8_5x5_center"},

    // Edge cases
    {3, 3, 3, 3, 1, 1, "3x3_3x3_exact"},
    {5, 5, 5, 5, 2, 2, "5x5_5x5_exact"},

    // Different anchor positions
    {6, 6, 3, 3, 0, 0, "6x6_3x3_topleft"},
    {6, 6, 3, 3, 2, 2, "6x6_3x3_bottomright"},
    {6, 6, 3, 3, 1, 0, "6x6_3x3_asymmetric"},

    // Non-square masks
    {8, 8, 3, 1, 1, 0, "8x8_3x1_horizontal"},
    {8, 8, 1, 3, 0, 1, "8x8_1x3_vertical"},

    // Rectangular images
    {8, 4, 3, 3, 1, 1, "8x4_3x3_wide"},
    {4, 8, 3, 3, 1, 1, "4x8_3x3_tall"},
};

// Test function for 8u data type
void testFilterBox8uBoundary(const BoundaryTestConfig &config, const std::string &patternName,
                             const std::vector<Npp8u> &inputPattern) {
  // Allocate device memory
  Npp8u *d_src = (Npp8u *)nppsMalloc_8u(config.width * config.height);
  Npp8u *d_dst = (Npp8u *)nppsMalloc_8u(config.width * config.height);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy to device
  cudaMemcpy(d_src, inputPattern.data(), inputPattern.size() * sizeof(Npp8u), cudaMemcpyHostToDevice);

  // Apply filter
  NppStatus status = nppiFilterBox_8u_C1R(d_src, config.width * sizeof(Npp8u), d_dst, config.width * sizeof(Npp8u),
                                          {config.width, config.height}, {config.maskWidth, config.maskHeight},
                                          {config.anchorX, config.anchorY});
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp8u> output(config.width * config.height);
  cudaMemcpy(output.data(), d_dst, output.size() * sizeof(Npp8u), cudaMemcpyDeviceToHost);

  // Save comparison result
  BoundaryAnalysisUtils::saveComparisonResult(inputPattern, output, config.width, config.height, config.maskWidth,
                                              config.maskHeight, config.anchorX, config.anchorY,
                                              "Npp8u_" + patternName);

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Test function for 32f data type
void testFilterBox32fBoundary(const BoundaryTestConfig &config, const std::string &patternName,
                              const std::vector<Npp32f> &inputPattern) {
  // Allocate device memory
  Npp32f *d_src = (Npp32f *)nppsMalloc_32f(config.width * config.height);
  Npp32f *d_dst = (Npp32f *)nppsMalloc_32f(config.width * config.height);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy to device
  cudaMemcpy(d_src, inputPattern.data(), inputPattern.size() * sizeof(Npp32f), cudaMemcpyHostToDevice);

  // Apply filter
  NppStatus status = nppiFilterBox_32f_C1R(d_src, config.width * sizeof(Npp32f), d_dst, config.width * sizeof(Npp32f),
                                           {config.width, config.height}, {config.maskWidth, config.maskHeight},
                                           {config.anchorX, config.anchorY});
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp32f> output(config.width * config.height);
  cudaMemcpy(output.data(), d_dst, output.size() * sizeof(Npp32f), cudaMemcpyDeviceToHost);

  // Save comparison result
  BoundaryAnalysisUtils::saveComparisonResult(inputPattern, output, config.width, config.height, config.maskWidth,
                                              config.maskHeight, config.anchorX, config.anchorY,
                                              "Npp32f_" + patternName);

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Test class for 8u boundary analysis
class FilterBoxBoundary8uTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize CUDA context
  }
};

// Test class for 32f boundary analysis
class FilterBoxBoundary32fTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize CUDA context
  }
};

// Test cases for 8u data type
TEST_F(FilterBoxBoundary8uTest, ComprehensiveBoundaryAnalysis) {
  std::cout << "\n" << std::string(100, '#') << "\n";
  std::cout << "NVIDIA NPP FilterBox Boundary Analysis - 8u Data Type\n";
  std::cout << std::string(100, '#') << "\n";

  for (const auto &config : BOUNDARY_TEST_CONFIGS) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "Testing Configuration: " << config.description << "\n";
    std::cout << std::string(80, '=') << "\n";

    // Test 1: Uniform pattern
    {
      auto input = TestPatterns<Npp8u>::createUniform(config.width, config.height, 128);
      testFilterBox8uBoundary(config, "Uniform", input);
    }

    // Test 2: Binary step pattern
    {
      auto input = TestPatterns<Npp8u>::createBinaryStep(config.width, config.height, 0, 255);
      testFilterBox8uBoundary(config, "BinaryStep", input);
    }

    // Test 3: Horizontal gradient
    {
      auto input = TestPatterns<Npp8u>::createHorizontalGradient(config.width, config.height, 0, 255);
      testFilterBox8uBoundary(config, "HorizontalGradient", input);
    }

    // Test 4: Corner impulse
    {
      auto input = TestPatterns<Npp8u>::createImpulse(config.width, config.height, 0, 0, 0, 255);
      testFilterBox8uBoundary(config, "CornerImpulse", input);
    }
  }
}

// Test cases for 32f data type
TEST_F(FilterBoxBoundary32fTest, ComprehensiveBoundaryAnalysis) {
  std::cout << "\n" << std::string(100, '#') << "\n";
  std::cout << "NVIDIA NPP FilterBox Boundary Analysis - 32f Data Type\n";
  std::cout << std::string(100, '#') << "\n";

  for (const auto &config : BOUNDARY_TEST_CONFIGS) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "Testing Configuration: " << config.description << "\n";
    std::cout << std::string(80, '=') << "\n";

    // Test 1: Uniform pattern
    {
      auto input = TestPatterns<Npp32f>::createUniform(config.width, config.height, 0.5f);
      testFilterBox32fBoundary(config, "Uniform", input);
    }

    // Test 2: Binary step pattern
    {
      auto input = TestPatterns<Npp32f>::createBinaryStep(config.width, config.height, 0.0f, 1.0f);
      testFilterBox32fBoundary(config, "BinaryStep", input);
    }

    // Test 3: Horizontal gradient
    {
      auto input = TestPatterns<Npp32f>::createHorizontalGradient(config.width, config.height, 0.0f, 1.0f);
      testFilterBox32fBoundary(config, "HorizontalGradient", input);
    }

    // Test 4: Corner impulse
    {
      auto input = TestPatterns<Npp32f>::createImpulse(config.width, config.height, 0, 0, 0.0f, 1.0f);
      testFilterBox32fBoundary(config, "CornerImpulse", input);
    }
  }
}