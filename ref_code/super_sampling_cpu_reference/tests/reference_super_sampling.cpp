#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <npp.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
  cudaError_t err = call; \
  if (err != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
    exit(1); \
  } \
} while(0)

#define CHECK_NPP(call) do { \
  NppStatus status = call; \
  if (status != NPP_SUCCESS) { \
    std::cerr << "NPP error: " << status << std::endl; \
    exit(1); \
  } \
} while(0)

// Reference CPU implementation that closely matches NVIDIA NPP
template<typename T>
class SuperSamplingCPU {
public:
  static void resize(const T* pSrc, int nSrcStep, int srcWidth, int srcHeight,
                     T* pDst, int nDstStep, int dstWidth, int dstHeight,
                     int channels = 1) {

    float scaleX = (float)srcWidth / dstWidth;
    float scaleY = (float)srcHeight / dstHeight;

    for (int dy = 0; dy < dstHeight; dy++) {
      for (int dx = 0; dx < dstWidth; dx++) {

        // For each channel
        for (int c = 0; c < channels; c++) {
          float result = interpolatePixel(pSrc, nSrcStep, srcWidth, srcHeight,
                                         dx, dy, scaleX, scaleY, c, channels);

          // Convert to output type with proper rounding
          T* dstRow = (T*)((char*)pDst + dy * nDstStep);
          dstRow[dx * channels + c] = convertToInt(result);
        }
      }
    }
  }

private:
  // Core interpolation logic
  static float interpolatePixel(const T* pSrc, int nSrcStep,
                                int srcWidth, int srcHeight,
                                int dx, int dy,
                                float scaleX, float scaleY,
                                int channel, int channels) {

    // Calculate sampling region center in source space
    float srcCenterX = (dx + 0.5f) * scaleX;
    float srcCenterY = (dy + 0.5f) * scaleY;

    // Sampling region bounds
    float xMin = srcCenterX - scaleX * 0.5f;
    float xMax = srcCenterX + scaleX * 0.5f;
    float yMin = srcCenterY - scaleY * 0.5f;
    float yMax = srcCenterY + scaleY * 0.5f;

    // CRITICAL: Use ceil for min, floor for max
    int xMinInt = (int)std::ceil(xMin);
    int xMaxInt = (int)std::floor(xMax);
    int yMinInt = (int)std::ceil(yMin);
    int yMaxInt = (int)std::floor(yMax);

    // Clamp to valid source region
    xMinInt = std::max(0, xMinInt);
    xMaxInt = std::min(srcWidth, xMaxInt);
    yMinInt = std::max(0, yMinInt);
    yMaxInt = std::min(srcHeight, yMaxInt);

    // CRITICAL: Edge weights must align with ceil/floor
    float wxMin = std::ceil(xMin) - xMin;
    float wxMax = xMax - std::floor(xMax);
    float wyMin = std::ceil(yMin) - yMin;
    float wyMax = yMax - std::floor(yMax);

    // Clamp weights to [0, 1]
    wxMin = std::min(1.0f, std::max(0.0f, wxMin));
    wxMax = std::min(1.0f, std::max(0.0f, wxMax));
    wyMin = std::min(1.0f, std::max(0.0f, wyMin));
    wyMax = std::min(1.0f, std::max(0.0f, wyMax));

    // Accumulate weighted sum
    float sum = 0.0f;

    // Process top edge (y = yMinInt - 1)
    if (wyMin > 0.0f && yMinInt > 0) {
      int y = yMinInt - 1;
      const T* srcRow = (const T*)((const char*)pSrc + y * nSrcStep);

      // Top-left corner
      if (wxMin > 0.0f && xMinInt > 0) {
        int x = xMinInt - 1;
        sum += srcRow[x * channels + channel] * wxMin * wyMin;
      }

      // Top edge
      for (int x = xMinInt; x < xMaxInt; x++) {
        sum += srcRow[x * channels + channel] * wyMin;
      }

      // Top-right corner
      if (wxMax > 0.0f && xMaxInt < srcWidth) {
        int x = xMaxInt;
        sum += srcRow[x * channels + channel] * wxMax * wyMin;
      }
    }

    // Process middle rows
    for (int y = yMinInt; y < yMaxInt; y++) {
      const T* srcRow = (const T*)((const char*)pSrc + y * nSrcStep);

      // Left edge
      if (wxMin > 0.0f && xMinInt > 0) {
        int x = xMinInt - 1;
        sum += srcRow[x * channels + channel] * wxMin;
      }

      // Center pixels (fully covered)
      for (int x = xMinInt; x < xMaxInt; x++) {
        sum += srcRow[x * channels + channel];
      }

      // Right edge
      if (wxMax > 0.0f && xMaxInt < srcWidth) {
        int x = xMaxInt;
        sum += srcRow[x * channels + channel] * wxMax;
      }
    }

    // Process bottom edge (y = yMaxInt)
    if (wyMax > 0.0f && yMaxInt < srcHeight) {
      int y = yMaxInt;
      const T* srcRow = (const T*)((const char*)pSrc + y * nSrcStep);

      // Bottom-left corner
      if (wxMin > 0.0f && xMinInt > 0) {
        int x = xMinInt - 1;
        sum += srcRow[x * channels + channel] * wxMin * wyMax;
      }

      // Bottom edge
      for (int x = xMinInt; x < xMaxInt; x++) {
        sum += srcRow[x * channels + channel] * wyMax;
      }

      // Bottom-right corner
      if (wxMax > 0.0f && xMaxInt < srcWidth) {
        int x = xMaxInt;
        sum += srcRow[x * channels + channel] * wxMax * wyMax;
      }
    }

    // Normalize by total sampling area
    float totalWeight = scaleX * scaleY;
    return sum / totalWeight;
  }

  // CRITICAL: Use +0.5 rounding, not banker's rounding
  static T convertToInt(float value) {
    return (T)(value + 0.5f);
  }
};

// Test statistics
struct TestStats {
  int totalPixels;
  int exactMatches;
  int within1;
  int within2;
  int maxDiff;
  double avgDiff;
  double avgAbsDiff;

  void compute(const std::vector<unsigned char>& cpu,
               const std::vector<unsigned char>& npp) {
    totalPixels = cpu.size();
    exactMatches = 0;
    within1 = 0;
    within2 = 0;
    maxDiff = 0;
    double sumDiff = 0.0;
    double sumAbsDiff = 0.0;

    for (size_t i = 0; i < cpu.size(); i++) {
      int diff = (int)cpu[i] - (int)npp[i];
      int absDiff = std::abs(diff);

      if (absDiff == 0) exactMatches++;
      if (absDiff <= 1) within1++;
      if (absDiff <= 2) within2++;

      maxDiff = std::max(maxDiff, absDiff);
      sumDiff += diff;
      sumAbsDiff += absDiff;
    }

    avgDiff = sumDiff / totalPixels;
    avgAbsDiff = sumAbsDiff / totalPixels;
  }

  void print(const char* testName) const {
    std::cout << "\n=== " << testName << " ===\n";
    std::cout << "Total pixels: " << totalPixels << "\n";
    std::cout << "Exact matches: " << exactMatches << " ("
              << (100.0 * exactMatches / totalPixels) << "%)\n";
    std::cout << "Within ±1: " << within1 << " ("
              << (100.0 * within1 / totalPixels) << "%)\n";
    std::cout << "Within ±2: " << within2 << " ("
              << (100.0 * within2 / totalPixels) << "%)\n";
    std::cout << "Max difference: " << maxDiff << "\n";
    std::cout << "Avg difference: " << std::fixed << std::setprecision(3)
              << avgDiff << "\n";
    std::cout << "Avg abs difference: " << avgAbsDiff << "\n";

    if (exactMatches == totalPixels) {
      std::cout << "✓ PERFECT MATCH!\n";
    } else if (within1 == totalPixels) {
      std::cout << "✓ EXCELLENT (all within ±1)\n";
    } else if (within2 == totalPixels) {
      std::cout << "✓ GOOD (all within ±2)\n";
    } else {
      std::cout << "⚠ Some pixels differ by more than 2\n";
    }
  }
};

void runTest(const char* testName, int srcW, int srcH, int dstW, int dstH,
             bool useGradient = true) {

  // Create test data
  std::vector<unsigned char> srcData(srcW * srcH);
  if (useGradient) {
    for (int y = 0; y < srcH; y++) {
      for (int x = 0; x < srcW; x++) {
        srcData[y * srcW + x] = (x * 255) / (srcW - 1);
      }
    }
  } else {
    // Checkerboard
    for (int y = 0; y < srcH; y++) {
      for (int x = 0; x < srcW; x++) {
        srcData[y * srcW + x] = ((x/4 + y/4) % 2) ? 255 : 0;
      }
    }
  }

  // NPP resize
  unsigned char *d_src, *d_dst;
  CHECK_CUDA(cudaMalloc(&d_src, srcW * srcH));
  CHECK_CUDA(cudaMalloc(&d_dst, dstW * dstH));
  CHECK_CUDA(cudaMemcpy(d_src, srcData.data(), srcW * srcH, cudaMemcpyHostToDevice));

  NppiSize srcSize = {srcW, srcH};
  NppiRect srcROI = {0, 0, srcW, srcH};
  NppiSize dstSize = {dstW, dstH};
  NppiRect dstROI = {0, 0, dstW, dstH};

  CHECK_NPP(nppiResize_8u_C1R(d_src, srcW, srcSize, srcROI,
                              d_dst, dstW, dstSize, dstROI,
                              NPPI_INTER_SUPER));

  std::vector<unsigned char> nppResult(dstW * dstH);
  CHECK_CUDA(cudaMemcpy(nppResult.data(), d_dst, dstW * dstH, cudaMemcpyDeviceToHost));

  // CPU reference resize
  std::vector<unsigned char> cpuResult(dstW * dstH);
  SuperSamplingCPU<unsigned char>::resize(
    srcData.data(), srcW, srcW, srcH,
    cpuResult.data(), dstW, dstW, dstH, 1
  );

  // Compute and print statistics
  TestStats stats;
  stats.compute(cpuResult, nppResult);

  std::cout << "\nTest: " << testName;
  std::cout << " (" << srcW << "x" << srcH << " -> " << dstW << "x" << dstH << ")";
  std::cout << " scale=" << std::setprecision(3) << ((float)srcW/dstW) << "x\n";
  stats.print(testName);

  // Show first mismatch if any
  if (stats.exactMatches < stats.totalPixels) {
    for (int i = 0; i < dstW * dstH; i++) {
      if (cpuResult[i] != nppResult[i]) {
        int dx = i % dstW;
        int dy = i / dstW;
        std::cout << "\nFirst mismatch at (" << dx << "," << dy << "):\n";
        std::cout << "  CPU: " << (int)cpuResult[i] << "\n";
        std::cout << "  NPP: " << (int)nppResult[i] << "\n";
        std::cout << "  Diff: " << ((int)cpuResult[i] - (int)nppResult[i]) << "\n";
        break;
      }
    }
  }

  CHECK_CUDA(cudaFree(d_src));
  CHECK_CUDA(cudaFree(d_dst));
}

int main() {
  std::cout << "=== CPU Reference Implementation Validation ===\n";
  std::cout << "Comparing CPU reference vs NVIDIA NPP resize (SUPER mode)\n\n";

  std::cout << "Algorithm details:\n";
  std::cout << "  - Bounds: ceil(xMin), floor(xMax)\n";
  std::cout << "  - Weights: ceil(xMin)-xMin, xMax-floor(xMax)\n";
  std::cout << "  - Final rounding: (int)(value + 0.5f)\n";
  std::cout << std::string(70, '=') << "\n";

  // Integer scales (should be perfect or near-perfect)
  runTest("Integer 2x downscale", 256, 256, 128, 128);
  runTest("Integer 4x downscale", 256, 256, 64, 64);
  runTest("Integer 8x downscale", 512, 512, 64, 64);

  // Common fractional scales
  runTest("Fractional 1.5x", 90, 90, 60, 60);
  runTest("Fractional 2.5x", 250, 250, 100, 100);
  runTest("Fractional 3.3x", 330, 330, 100, 100);

  // Problematic fractional scales
  runTest("Fractional 1.333x (128/96)", 128, 128, 96, 96);
  runTest("Fractional 1.667x (500/300)", 500, 500, 300, 300);

  // Large images
  runTest("Large 2x (1024x768)", 1024, 768, 512, 384);

  // Different patterns
  runTest("Checkerboard 4x", 256, 256, 64, 64, false);

  // Extreme downscaling
  runTest("Extreme 16x", 512, 512, 32, 32);
  runTest("Extreme 32x", 1024, 1024, 32, 32);

  std::cout << "\n" << std::string(70, '=') << "\n";
  std::cout << "Summary:\n";
  std::cout << "- Perfect match (100%): Implementation is bit-exact\n";
  std::cout << "- Within ±1: Excellent, differences due to float precision\n";
  std::cout << "- Within ±2: Good, acceptable for most applications\n";
  std::cout << "- Larger differences: May indicate algorithm mismatch\n";
  std::cout << std::string(70, '=') << "\n";

  return 0;
}
