/**
 * Validation Test: CPU Reference vs NVIDIA NPP Linear Interpolation
 *
 * Purpose: Validate CPU reference implementation accuracy
 */

#include <cuda_runtime.h>
#include <npp.h>
#include <nppi_geometry_transforms.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include "linear_interpolation_cpu.h"

struct TestCase {
    int srcWidth, srcHeight;
    int dstWidth, dstHeight;
    int channels;
    const char* description;
};

void runValidation(const TestCase& test) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Test: " << test.description << std::endl;
    std::cout << test.srcWidth << "x" << test.srcHeight << " -> "
              << test.dstWidth << "x" << test.dstHeight
              << " (" << test.channels << " channels)" << std::endl;

    float scaleX = (float)test.dstWidth / test.srcWidth;
    float scaleY = (float)test.dstHeight / test.srcHeight;
    std::cout << "Scale: " << scaleX << "x (width), " << scaleY << "x (height)" << std::endl;

    if (scaleX > 1.0f || scaleY > 1.0f) {
        std::cout << "Type: UPSCALE" << std::endl;
    } else if (scaleX < 1.0f || scaleY < 1.0f) {
        std::cout << "Type: DOWNSCALE" << std::endl;
    } else {
        std::cout << "Type: 1:1 (no scaling)" << std::endl;
    }

    // Create source image
    std::vector<Npp8u> srcData(test.srcWidth * test.srcHeight * test.channels);
    for (int y = 0; y < test.srcHeight; y++) {
        for (int x = 0; x < test.srcWidth; x++) {
            for (int c = 0; c < test.channels; c++) {
                int idx = (y * test.srcWidth + x) * test.channels + c;
                // Create varied pattern
                srcData[idx] = (Npp8u)((x * 37 + y * 23 + c * 17) % 256);
            }
        }
    }

    // Allocate device memory
    Npp8u* d_src;
    Npp8u* d_dst;
    size_t srcStep = test.srcWidth * test.channels * sizeof(Npp8u);
    size_t dstStep = test.dstWidth * test.channels * sizeof(Npp8u);

    cudaMalloc(&d_src, test.srcHeight * srcStep);
    cudaMalloc(&d_dst, test.dstHeight * dstStep);
    cudaMemcpy(d_src, srcData.data(), test.srcHeight * srcStep, cudaMemcpyHostToDevice);

    // NPP resize
    NppiSize srcSize = {test.srcWidth, test.srcHeight};
    NppiSize dstSize = {test.dstWidth, test.dstHeight};
    NppiRect srcROI = {0, 0, test.srcWidth, test.srcHeight};
    NppiRect dstROI = {0, 0, test.dstWidth, test.dstHeight};

    NppStatus status;
    if (test.channels == 1) {
        status = nppiResize_8u_C1R(d_src, srcStep, srcSize, srcROI,
                                   d_dst, dstStep, dstSize, dstROI,
                                   NPPI_INTER_LINEAR);
    } else if (test.channels == 3) {
        status = nppiResize_8u_C3R(d_src, srcStep, srcSize, srcROI,
                                   d_dst, dstStep, dstSize, dstROI,
                                   NPPI_INTER_LINEAR);
    } else {
        status = nppiResize_8u_C4R(d_src, srcStep, srcSize, srcROI,
                                   d_dst, dstStep, dstSize, dstROI,
                                   NPPI_INTER_LINEAR);
    }

    if (status != NPP_SUCCESS) {
        std::cerr << "NPP Error: " << status << std::endl;
        cudaFree(d_src);
        cudaFree(d_dst);
        return;
    }

    // Get NPP result
    std::vector<Npp8u> nppResult(test.dstWidth * test.dstHeight * test.channels);
    cudaMemcpy(nppResult.data(), d_dst, test.dstHeight * dstStep, cudaMemcpyDeviceToHost);

    // CPU reference
    std::vector<Npp8u> cpuResult(test.dstWidth * test.dstHeight * test.channels);
    LinearInterpolationCPU<Npp8u>::resize(
        srcData.data(), srcStep,
        test.srcWidth, test.srcHeight,
        cpuResult.data(), dstStep,
        test.dstWidth, test.dstHeight,
        test.channels
    );

    // Compare results
    int perfectMatch = 0;
    int within1 = 0;
    int within2 = 0;
    int maxDiff = 0;
    long long totalDiff = 0;
    int totalPixels = test.dstWidth * test.dstHeight * test.channels;

    std::vector<int> diffHistogram(256, 0);

    for (int i = 0; i < totalPixels; i++) {
        int diff = std::abs((int)nppResult[i] - (int)cpuResult[i]);
        diffHistogram[diff]++;
        maxDiff = std::max(maxDiff, diff);
        totalDiff += diff;

        if (diff == 0) perfectMatch++;
        if (diff <= 1) within1++;
        if (diff <= 2) within2++;
    }

    // Print statistics
    std::cout << "\n=== Accuracy Statistics ===" << std::endl;
    std::cout << "Total pixels: " << totalPixels << std::endl;
    std::cout << "Perfect match (diff=0): " << perfectMatch << " ("
              << (100.0 * perfectMatch / totalPixels) << "%)" << std::endl;
    std::cout << "Within ±1: " << within1 << " ("
              << (100.0 * within1 / totalPixels) << "%)" << std::endl;
    std::cout << "Within ±2: " << within2 << " ("
              << (100.0 * within2 / totalPixels) << "%)" << std::endl;
    std::cout << "Max difference: " << maxDiff << std::endl;
    std::cout << "Average difference: " << (double)totalDiff / totalPixels << std::endl;

    // Show difference histogram (for differences > 0)
    if (maxDiff > 0) {
        std::cout << "\nDifference histogram:" << std::endl;
        for (int d = 0; d <= std::min(5, maxDiff); d++) {
            if (diffHistogram[d] > 0) {
                std::cout << "  diff=" << d << ": " << diffHistogram[d]
                         << " (" << (100.0 * diffHistogram[d] / totalPixels) << "%)" << std::endl;
            }
        }
        if (maxDiff > 5) {
            std::cout << "  diff>5: " << (totalPixels - perfectMatch - within1 - within2) << std::endl;
        }
    }

    // Show sample comparisons
    std::cout << "\n=== Sample Pixel Comparison ===" << std::endl;
    int samples[][2] = {{0, 0}, {test.dstWidth-1, 0}, {0, test.dstHeight-1},
                        {test.dstWidth-1, test.dstHeight-1}, {test.dstWidth/2, test.dstHeight/2}};
    const char* names[] = {"Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right", "Center"};

    for (int i = 0; i < 5; i++) {
        int x = samples[i][0];
        int y = samples[i][1];
        if (x >= test.dstWidth || y >= test.dstHeight) continue;

        int baseIdx = (y * test.dstWidth + x) * test.channels;
        std::cout << names[i] << " (" << x << "," << y << "): ";
        std::cout << "NPP=[";
        for (int c = 0; c < test.channels; c++) {
            std::cout << (int)nppResult[baseIdx + c];
            if (c < test.channels - 1) std::cout << ",";
        }
        std::cout << "] CPU=[";
        for (int c = 0; c < test.channels; c++) {
            std::cout << (int)cpuResult[baseIdx + c];
            if (c < test.channels - 1) std::cout << ",";
        }
        std::cout << "]" << std::endl;
    }

    // Final verdict
    std::cout << "\n=== Verdict ===" << std::endl;
    if (perfectMatch == totalPixels) {
        std::cout << "✓ PERFECT MATCH (100% identical)" << std::endl;
    } else if (within1 == totalPixels) {
        std::cout << "✓ EXCELLENT (100% within ±1)" << std::endl;
    } else if (within2 == totalPixels) {
        std::cout << "✓ GOOD (100% within ±2)" << std::endl;
    } else if ((double)within1 / totalPixels > 0.99) {
        std::cout << "✓ ACCEPTABLE (>99% within ±1)" << std::endl;
    } else {
        std::cout << "⚠ NEEDS IMPROVEMENT" << std::endl;
    }

    cudaFree(d_src);
    cudaFree(d_dst);
}

int main() {
    std::cout << "CPU Reference Validation for Linear Interpolation" << std::endl;
    std::cout << "Comparing CPU implementation against NVIDIA NPP" << std::endl;

    std::vector<TestCase> testCases = {
        // Upscaling - integer ratios
        {4, 4, 8, 8, 1, "Upscale 2x - Grayscale"},
        {4, 4, 8, 8, 3, "Upscale 2x - RGB"},
        {2, 2, 8, 8, 1, "Upscale 4x - Grayscale"},
        {3, 3, 12, 12, 3, "Upscale 4x - RGB"},

        // Upscaling - fractional ratios
        {4, 4, 6, 6, 1, "Upscale 1.5x - Grayscale"},
        {4, 4, 10, 10, 3, "Upscale 2.5x - RGB"},

        // Downscaling - integer ratios
        {8, 8, 4, 4, 1, "Downscale 0.5x - Grayscale"},
        {8, 8, 4, 4, 3, "Downscale 0.5x - RGB"},
        {16, 16, 4, 4, 1, "Downscale 0.25x - Grayscale"},

        // Downscaling - fractional ratios
        {8, 8, 6, 6, 3, "Downscale 0.75x - RGB"},
        {10, 10, 4, 4, 1, "Downscale 0.4x - Grayscale"},

        // Non-uniform scaling
        {4, 8, 8, 4, 1, "Non-uniform (2x width, 0.5x height)"},
        {8, 4, 4, 8, 3, "Non-uniform (0.5x width, 2x height)"},
        {6, 8, 8, 6, 3, "Non-uniform (1.33x width, 0.75x height)"},

        // Edge cases
        {1, 8, 1, 4, 1, "1D vertical downscale"},
        {8, 1, 4, 1, 3, "1D horizontal downscale"},
    };

    int totalTests = testCases.size();
    int perfectTests = 0;
    int excellentTests = 0;

    for (const auto& test : testCases) {
        runValidation(test);

        // Track results (this is simplified - would need to capture results properly)
    }

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "OVERALL SUMMARY" << std::endl;
    std::cout << "Total tests: " << totalTests << std::endl;
    std::cout << "Run all tests above to evaluate accuracy" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    return 0;
}
