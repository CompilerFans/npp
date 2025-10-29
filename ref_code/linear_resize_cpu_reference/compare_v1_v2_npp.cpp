#include <iostream>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <vector>
#include <npp.h>
#include "linear_interpolation_cpu.h"     // V1: NPP-matching (hybrid algorithm)
#include "linear_interpolation_v2.h"      // V2: mpp-style (standard bilinear)

// Test result structure
struct TestResult {
    std::string testName;
    int v1MatchCount;
    int v1TotalPixels;
    float v1MatchPercent;
    int v1MaxDiff;
    int v2MatchCount;
    int v2TotalPixels;
    float v2MatchPercent;
    int v2MaxDiff;
    int v1v2DiffCount;
    int v1v2MaxDiff;
};

// Compare two images and return statistics
void compareImages(const uint8_t* img1, const uint8_t* img2, int width, int height, int channels,
                   int& matchCount, int& totalPixels, int& maxDiff) {
    matchCount = 0;
    totalPixels = width * height * channels;
    maxDiff = 0;

    for (int i = 0; i < totalPixels; i++) {
        int diff = std::abs((int)img1[i] - (int)img2[i]);
        if (diff == 0) {
            matchCount++;
        }
        if (diff > maxDiff) {
            maxDiff = diff;
        }
    }
}

// Print detailed pixel comparison for small images
void printDetailedComparison(const std::string& name,
                            const uint8_t* nppResult,
                            const uint8_t* v1Result,
                            const uint8_t* v2Result,
                            int width, int height, int channels) {
    if (width > 8 || height > 8) {
        std::cout << "  (Image too large for detailed display)\n";
        return;
    }

    std::cout << "  Detailed pixel comparison:\n";
    std::cout << "  Format: [NPP | V1 | V2]\n\n";

    for (int y = 0; y < height; y++) {
        std::cout << "  Row " << y << ": ";
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                int idx = (y * width + x) * channels + c;
                std::cout << "[" << std::setw(3) << (int)nppResult[idx]
                         << "|" << std::setw(3) << (int)v1Result[idx]
                         << "|" << std::setw(3) << (int)v2Result[idx] << "] ";
            }
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// Run a single test case
TestResult runTest(const std::string& testName,
                  const uint8_t* srcData, int srcWidth, int srcHeight,
                  int dstWidth, int dstHeight, int channels) {
    TestResult result;
    result.testName = testName;

    // Allocate destination buffers
    int srcStep = srcWidth * channels;
    int dstStep = dstWidth * channels;
    int dstSize = dstHeight * dstStep;

    uint8_t* nppResult = new uint8_t[dstSize];
    uint8_t* v1Result = new uint8_t[dstSize];
    uint8_t* v2Result = new uint8_t[dstSize];

    memset(nppResult, 0, dstSize);
    memset(v1Result, 0, dstSize);
    memset(v2Result, 0, dstSize);

    // Allocate device memory for NPP
    uint8_t* d_src = nullptr;
    uint8_t* d_dst = nullptr;
    cudaMalloc(&d_src, srcHeight * srcStep);
    cudaMalloc(&d_dst, dstHeight * dstStep);
    cudaMemcpy(d_src, srcData, srcHeight * srcStep, cudaMemcpyHostToDevice);

    // Run NPP resize
    NppiSize srcSize = {srcWidth, srcHeight};
    NppiSize dstSize_npp = {dstWidth, dstHeight};
    NppiRect srcROI = {0, 0, srcWidth, srcHeight};
    NppiRect dstROI = {0, 0, dstWidth, dstHeight};

    NppStatus status;
    if (channels == 1) {
        status = nppiResize_8u_C1R(d_src, srcStep, srcSize, srcROI,
                                   d_dst, dstStep, dstSize_npp, dstROI,
                                   NPPI_INTER_LINEAR);
    } else {
        status = nppiResize_8u_C3R(d_src, srcStep, srcSize, srcROI,
                                   d_dst, dstStep, dstSize_npp, dstROI,
                                   NPPI_INTER_LINEAR);
    }

    cudaMemcpy(nppResult, d_dst, dstHeight * dstStep, cudaMemcpyDeviceToHost);
    cudaFree(d_src);
    cudaFree(d_dst);

    // Run V1 (NPP-matching hybrid algorithm)
    LinearInterpolationCPU<uint8_t>::resize(srcData, srcStep, srcWidth, srcHeight,
                                            v1Result, dstStep, dstWidth, dstHeight,
                                            channels);

    // Run V2 (mpp-style standard bilinear)
    LinearInterpolationV2::resize(srcData, srcStep, srcWidth, srcHeight,
                                  v2Result, dstStep, dstWidth, dstHeight,
                                  channels);

    // Compare V1 vs NPP
    compareImages(nppResult, v1Result, dstWidth, dstHeight, channels,
                 result.v1MatchCount, result.v1TotalPixels, result.v1MaxDiff);
    result.v1MatchPercent = (float)result.v1MatchCount / result.v1TotalPixels * 100.0f;

    // Compare V2 vs NPP
    compareImages(nppResult, v2Result, dstWidth, dstHeight, channels,
                 result.v2MatchCount, result.v2TotalPixels, result.v2MaxDiff);
    result.v2MatchPercent = (float)result.v2MatchCount / result.v2TotalPixels * 100.0f;

    // Compare V1 vs V2
    int v1v2Total;
    compareImages(v1Result, v2Result, dstWidth, dstHeight, channels,
                 result.v1v2DiffCount, v1v2Total, result.v1v2MaxDiff);
    result.v1v2DiffCount = v1v2Total - result.v1v2DiffCount; // Convert match to diff count

    // Print results
    std::cout << "\n=== " << testName << " ===\n";
    std::cout << "Source: " << srcWidth << "x" << srcHeight << " -> Destination: " << dstWidth << "x" << dstHeight;
    float scaleX = (float)dstWidth / srcWidth;
    float scaleY = (float)dstHeight / srcHeight;
    std::cout << " (scale: " << scaleX << "x, " << scaleY << "x)\n";

    std::cout << "\nV1 (NPP-matching) vs NPP:\n";
    std::cout << "  Match: " << result.v1MatchCount << "/" << result.v1TotalPixels
              << " (" << std::fixed << std::setprecision(1) << result.v1MatchPercent << "%)\n";
    std::cout << "  Max diff: " << result.v1MaxDiff << "\n";

    std::cout << "\nV2 (mpp-style) vs NPP:\n";
    std::cout << "  Match: " << result.v2MatchCount << "/" << result.v2TotalPixels
              << " (" << std::fixed << std::setprecision(1) << result.v2MatchPercent << "%)\n";
    std::cout << "  Max diff: " << result.v2MaxDiff << "\n";

    std::cout << "\nV1 vs V2:\n";
    std::cout << "  Different pixels: " << result.v1v2DiffCount << "/" << result.v1TotalPixels
              << " (" << std::fixed << std::setprecision(1)
              << ((float)result.v1v2DiffCount / result.v1TotalPixels * 100.0f) << "%)\n";
    std::cout << "  Max diff: " << result.v1v2MaxDiff << "\n";

    // Print detailed comparison for small images
    if (dstWidth <= 8 && dstHeight <= 8) {
        printDetailedComparison(testName, nppResult, v1Result, v2Result,
                               dstWidth, dstHeight, channels);
    }

    delete[] nppResult;
    delete[] v1Result;
    delete[] v2Result;

    return result;
}

// Create test pattern: simple gradient
void createGradientPattern(uint8_t* data, int width, int height, int channels) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                data[(y * width + x) * channels + c] = (x * 255 / (width - 1));
            }
        }
    }
}

// Create test pattern: checkerboard
void createCheckerboardPattern(uint8_t* data, int width, int height, int channels) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            uint8_t value = ((x + y) % 2) ? 255 : 0;
            for (int c = 0; c < channels; c++) {
                data[(y * width + x) * channels + c] = value;
            }
        }
    }
}

int main() {
    std::cout << "========================================\n";
    std::cout << "Linear Interpolation Comparison Test\n";
    std::cout << "V1: NPP-matching (hybrid algorithm)\n";
    std::cout << "V2: mpp-style (standard bilinear)\n";
    std::cout << "========================================\n";

    std::vector<TestResult> results;

    // Test 1: Integer upscale 2x (grayscale)
    {
        int srcWidth = 4, srcHeight = 4;
        int dstWidth = 8, dstHeight = 8;
        uint8_t srcData[16];
        createGradientPattern(srcData, srcWidth, srcHeight, 1);
        results.push_back(runTest("Integer 2x upscale (gray)", srcData, srcWidth, srcHeight,
                                 dstWidth, dstHeight, 1));
    }

    // Test 2: Integer downscale 0.5x (grayscale)
    {
        int srcWidth = 4, srcHeight = 4;
        int dstWidth = 2, dstHeight = 2;
        uint8_t srcData[16];
        createGradientPattern(srcData, srcWidth, srcHeight, 1);
        results.push_back(runTest("Integer 0.5x downscale (gray)", srcData, srcWidth, srcHeight,
                                 dstWidth, dstHeight, 1));
    }

    // Test 3: Non-integer upscale 1.5x (grayscale)
    {
        int srcWidth = 2, srcHeight = 2;
        int dstWidth = 3, dstHeight = 3;
        uint8_t srcData[4] = {0, 100, 200, 255};
        results.push_back(runTest("Non-integer 1.5x upscale (gray)", srcData, srcWidth, srcHeight,
                                 dstWidth, dstHeight, 1));
    }

    // Test 4: Integer upscale 4x (grayscale)
    {
        int srcWidth = 2, srcHeight = 2;
        int dstWidth = 8, dstHeight = 8;
        uint8_t srcData[4] = {0, 100, 200, 255};
        results.push_back(runTest("Integer 4x upscale (gray)", srcData, srcWidth, srcHeight,
                                 dstWidth, dstHeight, 1));
    }

    // Test 5: Integer downscale 0.25x (grayscale)
    {
        int srcWidth = 4, srcHeight = 4;
        int dstWidth = 1, dstHeight = 1;
        uint8_t srcData[16];
        createCheckerboardPattern(srcData, srcWidth, srcHeight, 1);
        results.push_back(runTest("Integer 0.25x downscale (gray)", srcData, srcWidth, srcHeight,
                                 dstWidth, dstHeight, 1));
    }

    // Test 6: Non-integer upscale 2.5x (grayscale)
    {
        int srcWidth = 4, srcHeight = 4;
        int dstWidth = 10, dstHeight = 10;
        uint8_t srcData[16];
        createGradientPattern(srcData, srcWidth, srcHeight, 1);
        results.push_back(runTest("Non-integer 2.5x upscale (gray)", srcData, srcWidth, srcHeight,
                                 dstWidth, dstHeight, 1));
    }

    // Test 7: Non-integer downscale 0.67x (grayscale)
    {
        int srcWidth = 6, srcHeight = 6;
        int dstWidth = 4, dstHeight = 4;
        uint8_t srcData[36];
        createGradientPattern(srcData, srcWidth, srcHeight, 1);
        results.push_back(runTest("Non-integer 0.67x downscale (gray)", srcData, srcWidth, srcHeight,
                                 dstWidth, dstHeight, 1));
    }

    // Test 8: Integer upscale 2x (RGB)
    {
        int srcWidth = 4, srcHeight = 4;
        int dstWidth = 8, dstHeight = 8;
        uint8_t srcData[48];
        createGradientPattern(srcData, srcWidth, srcHeight, 3);
        results.push_back(runTest("Integer 2x upscale (RGB)", srcData, srcWidth, srcHeight,
                                 dstWidth, dstHeight, 3));
    }

    // Print summary
    std::cout << "\n========================================\n";
    std::cout << "SUMMARY\n";
    std::cout << "========================================\n\n";

    std::cout << std::left << std::setw(35) << "Test Case"
              << std::setw(15) << "V1 Match"
              << std::setw(15) << "V2 Match"
              << std::setw(15) << "V1-V2 Diff" << "\n";
    std::cout << std::string(80, '-') << "\n";

    for (const auto& r : results) {
        std::cout << std::left << std::setw(35) << r.testName
                  << std::setw(15) << (std::to_string((int)r.v1MatchPercent) + "%")
                  << std::setw(15) << (std::to_string((int)r.v2MatchPercent) + "%")
                  << std::setw(15) << std::to_string(r.v1v2DiffCount) << "\n";
    }

    std::cout << "\n========================================\n";
    std::cout << "KEY OBSERVATIONS\n";
    std::cout << "========================================\n\n";

    std::cout << "1. V1 (NPP-matching) uses hybrid algorithm:\n";
    std::cout << "   - Bilinear for upscale\n";
    std::cout << "   - Floor method for downscale\n";
    std::cout << "   - Should match NPP 100% for integer scales\n\n";

    std::cout << "2. V2 (mpp-style) uses standard bilinear:\n";
    std::cout << "   - Always bilinear interpolation\n";
    std::cout << "   - Better quality for downscale\n";
    std::cout << "   - May differ from NPP in downscale cases\n\n";

    std::cout << "3. Expected differences:\n";
    std::cout << "   - Integer upscale: V1 ≈ V2 (both use bilinear)\n";
    std::cout << "   - Integer downscale: V1 ≠ V2 (floor vs bilinear)\n";
    std::cout << "   - Non-integer: Both differ from NPP's proprietary algorithm\n\n";

    return 0;
}
