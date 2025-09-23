// NPP FilterBox vs FilterBoxBorder Comparison Analysis
// Compare nppiFilterBox with different NppiBorderType modes to identify the actual border handling

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "npp.h"

// Comparison analysis utilities
class BorderModeAnalysisUtils {
public:
  // Calculate similarity between two result vectors
  template <typename T>
  static double calculateSimilarity(const std::vector<T> &result1, const std::vector<T> &result2) {
    if (result1.size() != result2.size())
      return 0.0;

    int exactMatches = 0;
    double totalDiff = 0.0;

    for (size_t i = 0; i < result1.size(); i++) {
      double diff = std::abs(static_cast<double>(result1[i]) - static_cast<double>(result2[i]));
      if (diff < 1e-6) {
        exactMatches++;
      }
      totalDiff += diff;
    }

    double exactMatchRate = static_cast<double>(exactMatches) / result1.size();
    double avgDiff = totalDiff / result1.size();

    // Similarity score: higher is better (0-100%)
    double similarity = exactMatchRate * 100.0;
    if (exactMatchRate < 1.0) {
      // Factor in average difference for non-exact matches
      double maxPossibleDiff = std::is_integral<T>::value ? 255.0 : 1.0;
      double diffScore = std::max(0.0, (maxPossibleDiff - avgDiff) / maxPossibleDiff * 100.0);
      similarity = exactMatchRate * 70.0 + (1.0 - exactMatchRate) * diffScore * 0.3;
    }

    return similarity;
  }

  // Print detailed comparison results
  template <typename T>
  static void printDetailedComparison(const std::vector<T> &filterBoxResult,
                                      const std::map<std::string, std::vector<T>> &borderResults, int width, int height,
                                      const std::string &testName) {

    std::cout << "\n=== Border Mode Comparison: " << testName << " ===\n";
    std::cout << "Image Size: " << width << "x" << height << "\n\n";

    // Calculate similarities
    std::vector<std::pair<double, std::string>> similarities;
    for (const auto &pair : borderResults) {
      double sim = calculateSimilarity(filterBoxResult, pair.second);
      similarities.push_back({sim, pair.first});
    }

    // Sort by similarity (descending)
    std::sort(similarities.begin(), similarities.end(), std::greater<std::pair<double, std::string>>());

    std::cout << "Similarity Rankings:\n";
    for (size_t i = 0; i < similarities.size(); i++) {
      std::cout << i + 1 << ". " << std::setw(20) << std::left << similarities[i].second << ": " << std::fixed
                << std::setprecision(2) << similarities[i].first << "%\n";
    }

    // Show detailed matrix comparison for small images
    if (width <= 6 && height <= 6) {
      printMatrixComparison(filterBoxResult, borderResults, width, height);
    }

    // Analysis summary
    std::cout << "\nAnalysis Summary:\n";
    if (similarities[0].first > 95.0) {
      std::cout << "STRONG MATCH: nppiFilterBox likely uses " << similarities[0].second << " border mode\n";
    } else if (similarities[0].first > 80.0) {
      std::cout << "PROBABLE MATCH: nppiFilterBox probably uses " << similarities[0].second << " border mode\n";
    } else if (similarities[0].first > 50.0) {
      std::cout << "WEAK MATCH: nppiFilterBox may be similar to " << similarities[0].second << " border mode\n";
    } else {
      std::cout << "NO CLEAR MATCH: nppiFilterBox uses a different or custom border handling\n";
    }

    // Show top differences
    if (similarities[0].first < 100.0) {
      showTopDifferences(filterBoxResult, borderResults.at(similarities[0].second), width, height,
                         similarities[0].second);
    }
  }

private:
  // Print matrix comparison for small images
  template <typename T>
  static void printMatrixComparison(const std::vector<T> &filterBoxResult,
                                    const std::map<std::string, std::vector<T>> &borderResults, int width, int height) {

    std::cout << "\nMatrix Comparison:\n";

    // nppiFilterBox result
    std::cout << "\nnppiFilterBox Result:\n";
    printMatrix(filterBoxResult, width, height);

    // Show top 2 matching border modes
    std::vector<std::pair<double, std::string>> topModes;
    for (const auto &pair : borderResults) {
      double sim = calculateSimilarity(filterBoxResult, pair.second);
      topModes.push_back({sim, pair.first});
    }
    std::sort(topModes.begin(), topModes.end(), std::greater<>());

    for (int i = 0; i < std::min(2, static_cast<int>(topModes.size())); i++) {
      std::cout << "\n"
                << topModes[i].second << " Result (Similarity: " << std::fixed << std::setprecision(1)
                << topModes[i].first << "%):\n";
      printMatrix(borderResults.at(topModes[i].second), width, height);

      if (topModes[i].first < 100.0) {
        std::cout << "Difference (" << topModes[i].second << " - FilterBox):\n";
        printDifferenceMatrix(borderResults.at(topModes[i].second), filterBoxResult, width, height);
      }
    }
  }

  // Print single matrix
  template <typename T> static void printMatrix(const std::vector<T> &data, int width, int height) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        if (std::is_integral<T>::value) {
          std::cout << std::setw(6) << static_cast<int>(data[y * width + x]);
        } else {
          std::cout << std::setw(8) << std::fixed << std::setprecision(3) << static_cast<float>(data[y * width + x]);
        }
      }
      std::cout << "\n";
    }
  }

  // Print difference matrix
  template <typename T>
  static void printDifferenceMatrix(const std::vector<T> &data1, const std::vector<T> &data2, int width, int height) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        double diff = static_cast<double>(data1[y * width + x]) - static_cast<double>(data2[y * width + x]);
        std::cout << std::setw(8) << std::fixed << std::setprecision(3) << diff;
      }
      std::cout << "\n";
    }
  }

  // Show positions with largest differences
  template <typename T>
  static void showTopDifferences(const std::vector<T> &filterBoxResult, const std::vector<T> &borderResult, int width,
                                 int height, const std::string &borderMode) {

    std::vector<std::tuple<double, int, int, T, T>> differences; // diff, x, y, filterBox, border

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int idx = y * width + x;
        double diff = std::abs(static_cast<double>(filterBoxResult[idx]) - static_cast<double>(borderResult[idx]));
        if (diff > 1e-6) {
          differences.push_back({diff, x, y, filterBoxResult[idx], borderResult[idx]});
        }
      }
    }

    if (!differences.empty()) {
      std::sort(differences.begin(), differences.end(), std::greater<>());

      std::cout << "\nTop differences vs " << borderMode << ":\n";
      std::cout << "Position  Difference  FilterBox  " << borderMode << "\n";

      int showCount = std::min(5, static_cast<int>(differences.size()));
      for (int i = 0; i < showCount; i++) {
        auto &diff = differences[i];
        std::cout << "(" << std::get<1>(diff) << "," << std::get<2>(diff) << ")     ";
        std::cout << std::setw(8) << std::fixed << std::setprecision(3) << std::get<0>(diff) << "    ";
        if (std::is_integral<T>::value) {
          std::cout << std::setw(6) << static_cast<int>(std::get<3>(diff)) << "      ";
          std::cout << std::setw(6) << static_cast<int>(std::get<4>(diff)) << "\n";
        } else {
          std::cout << std::setw(8) << std::fixed << std::setprecision(3) << static_cast<float>(std::get<3>(diff))
                    << "  ";
          std::cout << std::setw(8) << std::fixed << std::setprecision(3) << static_cast<float>(std::get<4>(diff))
                    << "\n";
        }
      }
    }
  }
};

// Test configuration
struct BorderComparisonConfig {
  int width, height;
  int maskWidth, maskHeight;
  int anchorX, anchorY;
  std::string description;
};

// Test configurations for border comparison
static const std::vector<BorderComparisonConfig> BORDER_COMPARISON_CONFIGS = {
    {3, 3, 3, 3, 1, 1, "3x3_img_3x3_mask"}, {4, 4, 3, 3, 1, 1, "4x4_img_3x3_mask"},
    {5, 5, 3, 3, 1, 1, "5x5_img_3x3_mask"}, {6, 6, 3, 3, 1, 1, "6x6_img_3x3_mask"},
    {8, 8, 3, 3, 1, 1, "8x8_img_3x3_mask"}, {6, 6, 5, 5, 2, 2, "6x6_img_5x5_mask"},
    {8, 8, 5, 5, 2, 2, "8x8_img_5x5_mask"},
};

// Border modes to test
static const std::vector<std::pair<NppiBorderType, std::string>> BORDER_MODES = {{NPP_BORDER_REPLICATE, "REPLICATE"},
                                                                                 {NPP_BORDER_WRAP, "WRAP"},
                                                                                 {NPP_BORDER_MIRROR, "MIRROR"},
                                                                                 {NPP_BORDER_CONSTANT, "CONSTANT"}};

// Test patterns for border mode comparison
template <typename T> class BorderComparisonPatterns {
public:
  // Edge emphasis pattern for clear border effects
  static std::vector<T> createEdgePattern(int width, int height, T edgeVal, T centerVal) {
    std::vector<T> pattern(width * height, centerVal);

    // Set border pixels
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
          pattern[y * width + x] = edgeVal;
        }
      }
    }
    return pattern;
  }

  // Gradient pattern to test smoothness
  static std::vector<T> createGradientPattern(int width, int height, T minVal, T maxVal) {
    std::vector<T> pattern(width * height);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        double ratioX = width > 1 ? static_cast<double>(x) / (width - 1) : 0.0;
        pattern[y * width + x] = static_cast<T>(minVal + ratioX * (maxVal - minVal));
      }
    }
    return pattern;
  }

  // Corner spike for extreme testing
  static std::vector<T> createCornerSpike(int width, int height, T background, T spike) {
    std::vector<T> pattern(width * height, background);
    if (width > 0 && height > 0) {
      pattern[0] = spike; // Top-left corner
    }
    return pattern;
  }

  // Checkerboard for alternating patterns
  static std::vector<T> createCheckerboard(int width, int height, T val1, T val2) {
    std::vector<T> pattern(width * height);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        pattern[y * width + x] = ((x + y) % 2 == 0) ? val1 : val2;
      }
    }
    return pattern;
  }
};

// Run border mode comparison for 8u data type
void runBorderComparison8u(const BorderComparisonConfig &config, const std::string &patternName,
                           const std::vector<Npp8u> &input) {
  std::string testName = config.description + "_" + patternName;

  // Get nppiFilterBox result
  Npp8u *d_src = (Npp8u *)nppsMalloc_8u(input.size());
  Npp8u *d_dst = (Npp8u *)nppsMalloc_8u(input.size());
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy(d_src, input.data(), input.size() * sizeof(Npp8u), cudaMemcpyHostToDevice);

  NppStatus status = nppiFilterBox_8u_C1R(d_src, config.width * sizeof(Npp8u), d_dst, config.width * sizeof(Npp8u),
                                          {config.width, config.height}, {config.maskWidth, config.maskHeight},
                                          {config.anchorX, config.anchorY});
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> filterBoxResult(input.size());
  cudaMemcpy(filterBoxResult.data(), d_dst, input.size() * sizeof(Npp8u), cudaMemcpyDeviceToHost);

  // Get nppiFilterBoxBorder results for different border modes
  std::map<std::string, std::vector<Npp8u>> borderResults;

  for (const auto &borderMode : BORDER_MODES) {
    std::vector<Npp8u> borderResult(input.size());

    // Reset device memory
    cudaMemcpy(d_src, input.data(), input.size() * sizeof(Npp8u), cudaMemcpyHostToDevice);
    cudaMemset(d_dst, 0, input.size() * sizeof(Npp8u));

    status = nppiFilterBoxBorder_8u_C1R(d_src, config.width * sizeof(Npp8u), {config.width, config.height}, {0, 0},
                                        d_dst, config.width * sizeof(Npp8u), {config.width, config.height},
                                        {config.maskWidth, config.maskHeight}, {config.anchorX, config.anchorY},
                                        borderMode.first);

    if (status == NPP_SUCCESS) {
      cudaMemcpy(borderResult.data(), d_dst, input.size() * sizeof(Npp8u), cudaMemcpyDeviceToHost);
      borderResults[borderMode.second] = borderResult;
    } else {
      std::cout << "Warning: " << borderMode.second << " mode failed with status " << status << "\n";
    }
  }

  nppiFree(d_src);
  nppiFree(d_dst);

  // Analyze and print results
  BorderModeAnalysisUtils::printDetailedComparison<Npp8u>(filterBoxResult, borderResults, config.width, config.height,
                                                          testName);
}

// Run border mode comparison for 32f data type
void runBorderComparison32f(const BorderComparisonConfig &config, const std::string &patternName,
                            const std::vector<Npp32f> &input) {
  std::string testName = config.description + "_" + patternName;

  // Get nppiFilterBox result
  Npp32f *d_src = (Npp32f *)nppsMalloc_32f(input.size());
  Npp32f *d_dst = (Npp32f *)nppsMalloc_32f(input.size());
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy(d_src, input.data(), input.size() * sizeof(Npp32f), cudaMemcpyHostToDevice);

  NppStatus status = nppiFilterBox_32f_C1R(d_src, config.width * sizeof(Npp32f), d_dst, config.width * sizeof(Npp32f),
                                           {config.width, config.height}, {config.maskWidth, config.maskHeight},
                                           {config.anchorX, config.anchorY});
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> filterBoxResult(input.size());
  cudaMemcpy(filterBoxResult.data(), d_dst, input.size() * sizeof(Npp32f), cudaMemcpyDeviceToHost);

  // Get nppiFilterBoxBorder results for different border modes
  std::map<std::string, std::vector<Npp32f>> borderResults;

  for (const auto &borderMode : BORDER_MODES) {
    std::vector<Npp32f> borderResult(input.size());

    // Reset device memory
    cudaMemcpy(d_src, input.data(), input.size() * sizeof(Npp32f), cudaMemcpyHostToDevice);
    cudaMemset(d_dst, 0, input.size() * sizeof(Npp32f));

    status = nppiFilterBoxBorder_32f_C1R(d_src, config.width * sizeof(Npp32f), {config.width, config.height}, {0, 0},
                                         d_dst, config.width * sizeof(Npp32f), {config.width, config.height},
                                         {config.maskWidth, config.maskHeight}, {config.anchorX, config.anchorY},
                                         borderMode.first);

    if (status == NPP_SUCCESS) {
      cudaMemcpy(borderResult.data(), d_dst, input.size() * sizeof(Npp32f), cudaMemcpyDeviceToHost);
      borderResults[borderMode.second] = borderResult;
    } else {
      std::cout << "Warning: " << borderMode.second << " mode failed with status " << status << "\n";
    }
  }

  nppiFree(d_src);
  nppiFree(d_dst);

  // Analyze and print results
  BorderModeAnalysisUtils::printDetailedComparison<Npp32f>(filterBoxResult, borderResults, config.width, config.height,
                                                           testName);
}

// Test class
class FilterBoxBorderComparisonTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize CUDA context
  }
};

// Main test for 8u border mode comparison
TEST_F(FilterBoxBorderComparisonTest, BorderModeComparison8u) {
  std::cout << "\n" << std::string(120, '#') << "\n";
  std::cout << "NPP FilterBox vs FilterBoxBorder Mode Comparison - 8u Data Type\n";
  std::cout << std::string(120, '#') << "\n";

  for (const auto &config : BORDER_COMPARISON_CONFIGS) {
    std::cout << "\n" << std::string(100, '=') << "\n";
    std::cout << "Configuration: " << config.description << "\n";
    std::cout << "Image: " << config.width << "x" << config.height << ", Mask: " << config.maskWidth << "x"
              << config.maskHeight;
    std::cout << ", Anchor: (" << config.anchorX << "," << config.anchorY << ")\n";
    std::cout << std::string(100, '=') << "\n";

    // Test with different patterns
    {
      auto input = BorderComparisonPatterns<Npp8u>::createEdgePattern(config.width, config.height, 255, 0);
      runBorderComparison8u(config, "EdgePattern", input);
    }

    {
      auto input = BorderComparisonPatterns<Npp8u>::createGradientPattern(config.width, config.height, 0, 255);
      runBorderComparison8u(config, "GradientPattern", input);
    }

    {
      auto input = BorderComparisonPatterns<Npp8u>::createCornerSpike(config.width, config.height, 128, 255);
      runBorderComparison8u(config, "CornerSpike", input);
    }

    {
      auto input = BorderComparisonPatterns<Npp8u>::createCheckerboard(config.width, config.height, 0, 255);
      runBorderComparison8u(config, "Checkerboard", input);
    }
  }
}

// Main test for 32f border mode comparison
TEST_F(FilterBoxBorderComparisonTest, BorderModeComparison32f) {
  std::cout << "\n" << std::string(120, '#') << "\n";
  std::cout << "NPP FilterBox vs FilterBoxBorder Mode Comparison - 32f Data Type\n";
  std::cout << std::string(120, '#') << "\n";

  for (const auto &config : BORDER_COMPARISON_CONFIGS) {
    std::cout << "\n" << std::string(100, '=') << "\n";
    std::cout << "Configuration: " << config.description << "\n";
    std::cout << "Image: " << config.width << "x" << config.height << ", Mask: " << config.maskWidth << "x"
              << config.maskHeight;
    std::cout << ", Anchor: (" << config.anchorX << "," << config.anchorY << ")\n";
    std::cout << std::string(100, '=') << "\n";

    // Test with different patterns
    {
      auto input = BorderComparisonPatterns<Npp32f>::createEdgePattern(config.width, config.height, 1.0f, 0.0f);
      runBorderComparison32f(config, "EdgePattern", input);
    }

    {
      auto input = BorderComparisonPatterns<Npp32f>::createGradientPattern(config.width, config.height, 0.0f, 1.0f);
      runBorderComparison32f(config, "GradientPattern", input);
    }

    {
      auto input = BorderComparisonPatterns<Npp32f>::createCornerSpike(config.width, config.height, 0.5f, 1.0f);
      runBorderComparison32f(config, "CornerSpike", input);
    }

    {
      auto input = BorderComparisonPatterns<Npp32f>::createCheckerboard(config.width, config.height, 0.0f, 1.0f);
      runBorderComparison32f(config, "Checkerboard", input);
    }
  }
}