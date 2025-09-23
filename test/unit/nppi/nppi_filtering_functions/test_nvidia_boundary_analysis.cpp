// Deep analysis of NVIDIA NPP FilterBox boundary processing characteristics
// This test analyzes NVIDIA NPP's boundary handling algorithm to find patterns

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <vector>

#include "npp.h"

class NVIDIABoundaryAnalyzer {
public:
  struct PixelAnalysis {
    int x, y;
    uint8_t input;
    uint8_t output;
    bool isBoundary;
    int distanceFromEdge;
    std::vector<uint8_t> filterWindow;
    int validPixelsInWindow;
  };

  // Analyze a single test case to understand boundary behavior
  static std::vector<PixelAnalysis> analyzeBoundaryBehavior(const std::vector<uint8_t> &input, int width, int height,
                                                            int maskW, int maskH, int anchorX, int anchorY) {

    // Run NVIDIA NPP FilterBox
    auto nppResult = runNPPFilterBox(input, width, height, maskW, maskH, anchorX, anchorY);

    std::vector<PixelAnalysis> analysis;

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        PixelAnalysis pixel;
        pixel.x = x;
        pixel.y = y;
        pixel.input = input[y * width + x];
        pixel.output = nppResult[y * width + x];

        // Calculate distance from edge
        int distToLeft = x;
        int distToRight = width - 1 - x;
        int distToTop = y;
        int distToBottom = height - 1 - y;
        pixel.distanceFromEdge = std::min({distToLeft, distToRight, distToTop, distToBottom});

        // Determine if affected by boundary
        int filterLeft = x - anchorX;
        int filterRight = x - anchorX + maskW - 1;
        int filterTop = y - anchorY;
        int filterBottom = y - anchorY + maskH - 1;

        pixel.isBoundary = (filterLeft < 0) || (filterRight >= width) || (filterTop < 0) || (filterBottom >= height);

        // Extract filter window values and count valid pixels
        pixel.validPixelsInWindow = 0;
        for (int my = 0; my < maskH; my++) {
          for (int mx = 0; mx < maskW; mx++) {
            int px = x - anchorX + mx;
            int py = y - anchorY + my;

            if (px >= 0 && px < width && py >= 0 && py < height) {
              pixel.filterWindow.push_back(input[py * width + px]);
              pixel.validPixelsInWindow++;
            } else {
              pixel.filterWindow.push_back(0); // Mark out-of-bounds
            }
          }
        }

        analysis.push_back(pixel);
      }
    }

    return analysis;
  }

  // Analyze edge replication hypothesis
  static void analyzeEdgeReplication(const std::vector<PixelAnalysis> &analysis, int width, int height, int maskW,
                                     int maskH) {
    std::cout << "\n=== Edge Replication Analysis ===\n";

    for (const auto &pixel : analysis) {
      if (!pixel.isBoundary)
        continue;

      // Test different replication strategies
      std::vector<uint8_t> replicated = pixel.filterWindow;

      // Strategy 1: Nearest pixel replication
      for (size_t i = 0; i < replicated.size(); i++) {
        if (replicated[i] == 0) { // Out-of-bounds pixel
          int my = i / maskW;
          int mx = i % maskW;
          int px = pixel.x - (maskW / 2) + mx;
          int py = pixel.y - (maskH / 2) + my;

          // Clamp to nearest valid pixel
          px = std::max(0, std::min(width - 1, px));
          py = std::max(0, std::min(height - 1, py));

          // This would be the replicated value (but we don't have easy access to input here)
          // For now, mark as special value
          replicated[i] = 255; // Placeholder
        }
      }
    }
  }

  // Analyze weighted average hypothesis
  static void analyzeWeightedAverage(const std::vector<PixelAnalysis> &analysis) {
    std::cout << "\n=== Weighted Average Analysis ===\n";

    for (const auto &pixel : analysis) {
      if (!pixel.isBoundary)
        continue;

      // Calculate what output would be with different strategies

      // Strategy 1: Average of valid pixels only
      int sumValid = 0;
      for (uint8_t val : pixel.filterWindow) {
        if (val != 0 || pixel.input == 0) { // 0 might be valid input value
          sumValid += val;
        }
      }
      int avgValid = (pixel.validPixelsInWindow > 0) ? sumValid / pixel.validPixelsInWindow : 0;

      // Strategy 2: Zero padding (what our MPP does)
      int sumZeroPad = 0;
      for (uint8_t val : pixel.filterWindow) {
        sumZeroPad += val;
      }
      int totalPixels = static_cast<int>(pixel.filterWindow.size());
      int avgZeroPad = (totalPixels > 0) ? sumZeroPad / totalPixels : 0;

      // Compare with actual NVIDIA output
      int diffFromValidAvg = std::abs(pixel.output - avgValid);
      int diffFromZeroPad = std::abs(pixel.output - avgZeroPad);

      if (pixel.distanceFromEdge <= 1) { // Focus on edge pixels
        std::cout << "Pixel (" << pixel.x << "," << pixel.y << "): "
                  << "NPP=" << (int)pixel.output << ", ValidAvg=" << avgValid << " (diff=" << diffFromValidAvg << ")"
                  << ", ZeroPad=" << avgZeroPad << " (diff=" << diffFromZeroPad << ")"
                  << ", ValidCount=" << pixel.validPixelsInWindow << "/" << totalPixels << "\n";
      }
    }
  }

  // Find patterns in boundary processing
  static void findBoundaryPatterns(const std::vector<PixelAnalysis> &analysis, int /* width */, int /* height */,
                                   int maskW, int maskH) {
    std::cout << "\n=== Boundary Pattern Analysis ===\n";

    // Group pixels by distance from edge
    std::map<int, std::vector<PixelAnalysis>> byDistance;
    for (const auto &pixel : analysis) {
      byDistance[pixel.distanceFromEdge].push_back(pixel);
    }

    for (const auto &[distance, pixels] : byDistance) {
      if (distance > 3)
        continue; // Focus on edge regions

      std::cout << "\nDistance " << distance << " from edge (" << pixels.size() << " pixels):\n";

      // Analyze patterns
      int totalDiffFromZeroPad = 0;
      int totalDiffFromValidAvg = 0;
      int count = 0;

      for (const auto &pixel : pixels) {
        // Calculate expected values
        int sumValid = 0;
        for (uint8_t val : pixel.filterWindow) {
          if (val != 0 || pixel.input == 0) {
            sumValid += val;
          }
        }
        int avgValid = (pixel.validPixelsInWindow > 0) ? sumValid / pixel.validPixelsInWindow : 0;

        int sumZeroPad = 0;
        for (uint8_t val : pixel.filterWindow) {
          sumZeroPad += val;
        }
        int totalPixels = maskW * maskH;
        int avgZeroPad = (totalPixels > 0) ? sumZeroPad / totalPixels : 0;

        totalDiffFromValidAvg += std::abs(pixel.output - avgValid);
        totalDiffFromZeroPad += std::abs(pixel.output - avgZeroPad);
        count++;
      }

      if (count > 0) {
        double avgDiffValidAvg = (double)totalDiffFromValidAvg / count;
        double avgDiffZeroPad = (double)totalDiffFromZeroPad / count;

        std::cout << "  Avg diff from ValidAvg: " << std::fixed << std::setprecision(2) << avgDiffValidAvg << "\n";
        std::cout << "  Avg diff from ZeroPad: " << avgDiffZeroPad << "\n";

        if (avgDiffValidAvg < avgDiffZeroPad * 0.8) {
          std::cout << "  → Pattern suggests: Closer to ValidAvg (boundary clipping/replication)\n";
        } else if (avgDiffZeroPad < avgDiffValidAvg * 0.8) {
          std::cout << "  → Pattern suggests: Closer to ZeroPad\n";
        } else {
          std::cout << "  → Pattern suggests: Mixed or other strategy\n";
        }
      }
    }
  }

private:
  static std::vector<uint8_t> runNPPFilterBox(const std::vector<uint8_t> &input, int width, int height, int maskW,
                                              int maskH, int anchorX, int anchorY) {
    // Allocate device memory
    Npp8u *d_src = (Npp8u *)nppsMalloc_8u(width * height);
    Npp8u *d_dst = (Npp8u *)nppsMalloc_8u(width * height);

    // Copy to device
    cudaMemcpy(d_src, input.data(), input.size() * sizeof(Npp8u), cudaMemcpyHostToDevice);

    // Apply filter
    NppStatus status = nppiFilterBox_8u_C1R(d_src, width * sizeof(Npp8u), d_dst, width * sizeof(Npp8u), {width, height},
                                            {maskW, maskH}, {anchorX, anchorY});

    if (status != NPP_SUCCESS) {
      nppiFree(d_src);
      nppiFree(d_dst);
      throw std::runtime_error("NPP FilterBox failed");
    }

    // Copy result back
    std::vector<uint8_t> output(width * height);
    cudaMemcpy(output.data(), d_dst, output.size() * sizeof(Npp8u), cudaMemcpyDeviceToHost);

    nppiFree(d_src);
    nppiFree(d_dst);

    return output;
  }
};

class NVIDIABoundaryAnalysisTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize CUDA context if needed
  }
};

// Test with uniform input to understand boundary scaling
TEST_F(NVIDIABoundaryAnalysisTest, UniformInputBoundaryAnalysis) {
  std::cout << "\n" << std::string(80, '=') << "\n";
  std::cout << "NVIDIA NPP Boundary Analysis - Uniform Input\n";
  std::cout << std::string(80, '=') << "\n";

  const int width = 6, height = 6;
  const int maskW = 3, maskH = 3;
  const int anchorX = 1, anchorY = 1;

  // Use uniform value 128 to clearly see boundary effects
  std::vector<uint8_t> input(width * height, 128);

  auto analysis = NVIDIABoundaryAnalyzer::analyzeBoundaryBehavior(input, width, height, maskW, maskH, anchorX, anchorY);

  NVIDIABoundaryAnalyzer::analyzeWeightedAverage(analysis);
  NVIDIABoundaryAnalyzer::findBoundaryPatterns(analysis, width, height, maskW, maskH);
}

// Test with edge pattern to understand boundary replication
TEST_F(NVIDIABoundaryAnalysisTest, EdgePatternBoundaryAnalysis) {
  std::cout << "\n" << std::string(80, '=') << "\n";
  std::cout << "NVIDIA NPP Boundary Analysis - Edge Pattern\n";
  std::cout << std::string(80, '=') << "\n";

  const int width = 6, height = 6;
  const int maskW = 3, maskH = 3;
  const int anchorX = 1, anchorY = 1;

  // Create edge pattern: high values at edges, low in center
  std::vector<uint8_t> input(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
        input[y * width + x] = 200; // Edge pixels
      } else {
        input[y * width + x] = 50; // Interior pixels
      }
    }
  }

  auto analysis = NVIDIABoundaryAnalyzer::analyzeBoundaryBehavior(input, width, height, maskW, maskH, anchorX, anchorY);

  NVIDIABoundaryAnalyzer::analyzeWeightedAverage(analysis);
  NVIDIABoundaryAnalyzer::findBoundaryPatterns(analysis, width, height, maskW, maskH);
}

// Test with gradient to understand smooth boundary handling
TEST_F(NVIDIABoundaryAnalysisTest, GradientBoundaryAnalysis) {
  std::cout << "\n" << std::string(80, '=') << "\n";
  std::cout << "NVIDIA NPP Boundary Analysis - Gradient Pattern\n";
  std::cout << std::string(80, '=') << "\n";

  const int width = 8, height = 6;
  const int maskW = 3, maskH = 3;
  const int anchorX = 1, anchorY = 1;

  // Create horizontal gradient
  std::vector<uint8_t> input(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      input[y * width + x] = (x * 255) / (width - 1);
    }
  }

  auto analysis = NVIDIABoundaryAnalyzer::analyzeBoundaryBehavior(input, width, height, maskW, maskH, anchorX, anchorY);

  NVIDIABoundaryAnalyzer::analyzeWeightedAverage(analysis);
  NVIDIABoundaryAnalyzer::findBoundaryPatterns(analysis, width, height, maskW, maskH);
}

// Test corner cases with small images
TEST_F(NVIDIABoundaryAnalysisTest, SmallImageCornerCases) {
  std::cout << "\n" << std::string(80, '=') << "\n";
  std::cout << "NVIDIA NPP Boundary Analysis - Small Image Corner Cases\n";
  std::cout << std::string(80, '=') << "\n";

  // Test 3x3 image with 3x3 filter (all pixels are boundary pixels)
  const int width = 3, height = 3;
  const int maskW = 3, maskH = 3;
  const int anchorX = 1, anchorY = 1;

  // Use step function
  std::vector<uint8_t> input = {100, 100, 200, 100, 100, 200, 100, 100, 200};

  auto analysis = NVIDIABoundaryAnalyzer::analyzeBoundaryBehavior(input, width, height, maskW, maskH, anchorX, anchorY);

  std::cout << "All pixels are boundary pixels in this configuration:\n";
  NVIDIABoundaryAnalyzer::analyzeWeightedAverage(analysis);
  NVIDIABoundaryAnalyzer::findBoundaryPatterns(analysis, width, height, maskW, maskH);
}