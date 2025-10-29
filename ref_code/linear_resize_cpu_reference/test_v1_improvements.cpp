#include <iostream>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <vector>
#include <npp.h>
#include "linear_interpolation_cpu.h"             // V1 original
#include "linear_interpolation_v1_improved.h"     // V1 improved

struct ComparisonResult {
    std::string testName;
    int originalMatch;
    int improvedMatch;
    int totalPixels;
    int originalMaxDiff;
    int improvedMaxDiff;
    float originalMatchPercent;
    float improvedMatchPercent;
};

void compareImplementations(const std::string& testName,
                           const uint8_t* srcData, int srcW, int srcH,
                           int dstW, int dstH, int channels,
                           ComparisonResult& result) {
    result.testName = testName;
    result.totalPixels = dstW * dstH * channels;

    int srcStep = srcW * channels;
    int dstStep = dstW * channels;

    // Allocate buffers
    uint8_t* nppResult = new uint8_t[dstH * dstStep];
    uint8_t* v1OrigResult = new uint8_t[dstH * dstStep];
    uint8_t* v1ImpvResult = new uint8_t[dstH * dstStep];

    // NPP
    uint8_t *d_src, *d_dst;
    cudaMalloc(&d_src, srcH * srcStep);
    cudaMalloc(&d_dst, dstH * dstStep);
    cudaMemcpy(d_src, srcData, srcH * srcStep, cudaMemcpyHostToDevice);

    NppiSize srcSize = {srcW, srcH};
    NppiSize dstSize = {dstW, dstH};
    NppiRect srcROI = {0, 0, srcW, srcH};
    NppiRect dstROI = {0, 0, dstW, dstH};

    if (channels == 1) {
        nppiResize_8u_C1R(d_src, srcStep, srcSize, srcROI,
                         d_dst, dstStep, dstSize, dstROI,
                         NPPI_INTER_LINEAR);
    } else {
        nppiResize_8u_C3R(d_src, srcStep, srcSize, srcROI,
                         d_dst, dstStep, dstSize, dstROI,
                         NPPI_INTER_LINEAR);
    }

    cudaMemcpy(nppResult, d_dst, dstH * dstStep, cudaMemcpyDeviceToHost);

    // V1 Original
    LinearInterpolationCPU<uint8_t>::resize(srcData, srcStep, srcW, srcH,
                                            v1OrigResult, dstStep, dstW, dstH,
                                            channels);

    // V1 Improved
    LinearInterpolationV1Improved<uint8_t>::resize(srcData, srcStep, srcW, srcH,
                                                   v1ImpvResult, dstStep, dstW, dstH,
                                                   channels);

    // Compare
    result.originalMatch = 0;
    result.improvedMatch = 0;
    result.originalMaxDiff = 0;
    result.improvedMaxDiff = 0;

    for (int i = 0; i < result.totalPixels; i++) {
        int origDiff = std::abs((int)nppResult[i] - (int)v1OrigResult[i]);
        int impvDiff = std::abs((int)nppResult[i] - (int)v1ImpvResult[i]);

        if (origDiff == 0) result.originalMatch++;
        if (impvDiff == 0) result.improvedMatch++;

        if (origDiff > result.originalMaxDiff) result.originalMaxDiff = origDiff;
        if (impvDiff > result.improvedMaxDiff) result.improvedMaxDiff = impvDiff;
    }

    result.originalMatchPercent = (float)result.originalMatch / result.totalPixels * 100.0f;
    result.improvedMatchPercent = (float)result.improvedMatch / result.totalPixels * 100.0f;

    // Print results
    std::cout << "\n=== " << testName << " ===\n";
    std::cout << "Source: " << srcW << "x" << srcH << " -> Dst: " << dstW << "x" << dstH
             << " (scale: " << (float)dstW/srcW << "x, " << (float)dstH/srcH << "x)\n\n";

    std::cout << "V1 Original:\n";
    std::cout << "  Match: " << result.originalMatch << "/" << result.totalPixels
             << " (" << std::fixed << std::setprecision(1) << result.originalMatchPercent << "%)\n";
    std::cout << "  Max diff: " << result.originalMaxDiff << "\n\n";

    std::cout << "V1 Improved:\n";
    std::cout << "  Match: " << result.improvedMatch << "/" << result.totalPixels
             << " (" << std::fixed << std::setprecision(1) << result.improvedMatchPercent << "%)\n";
    std::cout << "  Max diff: " << result.improvedMaxDiff << "\n\n";

    float improvement = result.improvedMatchPercent - result.originalMatchPercent;
    std::cout << "Improvement: ";
    if (improvement > 0) {
        std::cout << "+" << improvement << "% ✓\n";
    } else if (improvement < 0) {
        std::cout << improvement << "% ✗\n";
    } else {
        std::cout << "No change\n";
    }

    // Show sample pixels for small images
    if (dstW <= 8 && dstH <= 8 && channels == 1) {
        std::cout << "\nSample pixels [NPP | Orig | Impv]:\n";
        for (int y = 0; y < std::min(4, dstH); y++) {
            std::cout << "  Row " << y << ": ";
            for (int x = 0; x < std::min(8, dstW); x++) {
                int idx = y * dstW + x;
                std::cout << "[" << std::setw(3) << (int)nppResult[idx]
                         << "|" << std::setw(3) << (int)v1OrigResult[idx]
                         << "|" << std::setw(3) << (int)v1ImpvResult[idx] << "] ";
            }
            std::cout << "\n";
        }
    }

    cudaFree(d_src);
    cudaFree(d_dst);
    delete[] nppResult;
    delete[] v1OrigResult;
    delete[] v1ImpvResult;
}

int main() {
    std::cout << "========================================\n";
    std::cout << "V1 Original vs V1 Improved Comparison\n";
    std::cout << "========================================\n";

    std::vector<ComparisonResult> results;

    // Test 1: 0.25x downscale (the worst case in original)
    {
        int srcW = 16, srcH = 16;
        int dstW = 4, dstH = 4;
        uint8_t srcData[256];
        for (int i = 0; i < 256; i++) srcData[i] = i;

        ComparisonResult r;
        compareImplementations("0.25x Downscale (16x16->4x4)", srcData, srcW, srcH,
                              dstW, dstH, 1, r);
        results.push_back(r);
    }

    // Test 2: 0.5x downscale (should stay perfect)
    {
        int srcW = 8, srcH = 8;
        int dstW = 4, dstH = 4;
        uint8_t srcData[64];
        for (int i = 0; i < 64; i++) srcData[i] = (i * 4) % 256;

        ComparisonResult r;
        compareImplementations("0.5x Downscale (8x8->4x4)", srcData, srcW, srcH,
                              dstW, dstH, 1, r);
        results.push_back(r);
    }

    // Test 3: Non-uniform 2x width, 0.5x height
    {
        int srcW = 4, srcH = 8;
        int dstW = 8, dstH = 4;
        uint8_t srcData[32];
        for (int y = 0; y < srcH; y++) {
            for (int x = 0; x < srcW; x++) {
                srcData[y * srcW + x] = (x * 255 / (srcW - 1));
            }
        }

        ComparisonResult r;
        compareImplementations("Non-uniform (4x8->8x4, 2x×0.5x)", srcData, srcW, srcH,
                              dstW, dstH, 1, r);
        results.push_back(r);
    }

    // Test 4: 0.75x downscale
    {
        int srcW = 8, srcH = 8;
        int dstW = 6, dstH = 6;
        uint8_t srcData[64];
        for (int y = 0; y < srcH; y++) {
            for (int x = 0; x < srcW; x++) {
                srcData[y * srcW + x] = (x * 255 / (srcW - 1));
            }
        }

        ComparisonResult r;
        compareImplementations("0.75x Downscale (8x8->6x6)", srcData, srcW, srcH,
                              dstW, dstH, 1, r);
        results.push_back(r);
    }

    // Test 5: 2x upscale (should stay perfect)
    {
        int srcW = 4, srcH = 4;
        int dstW = 8, dstH = 8;
        uint8_t srcData[16];
        for (int i = 0; i < 16; i++) srcData[i] = (i * 17);

        ComparisonResult r;
        compareImplementations("2x Upscale (4x4->8x8)", srcData, srcW, srcH,
                              dstW, dstH, 1, r);
        results.push_back(r);
    }

    // Test 6: Non-uniform 0.5x width, 2x height
    {
        int srcW = 8, srcH = 4;
        int dstW = 4, dstH = 8;
        uint8_t srcData[32];
        for (int y = 0; y < srcH; y++) {
            for (int x = 0; x < srcW; x++) {
                srcData[y * srcW + x] = (x * 255 / (srcW - 1));
            }
        }

        ComparisonResult r;
        compareImplementations("Non-uniform (8x4->4x8, 0.5x×2x)", srcData, srcW, srcH,
                              dstW, dstH, 1, r);
        results.push_back(r);
    }

    // Summary
    std::cout << "\n========================================\n";
    std::cout << "SUMMARY\n";
    std::cout << "========================================\n\n";

    std::cout << std::left << std::setw(35) << "Test Case"
             << std::setw(15) << "Original"
             << std::setw(15) << "Improved"
             << std::setw(15) << "Change" << "\n";
    std::cout << std::string(80, '-') << "\n";

    int totalImproved = 0;
    int totalWorse = 0;
    int totalSame = 0;

    for (const auto& r : results) {
        std::cout << std::left << std::setw(35) << r.testName
                 << std::setw(15) << (std::to_string((int)r.originalMatchPercent) + "%")
                 << std::setw(15) << (std::to_string((int)r.improvedMatchPercent) + "%");

        float change = r.improvedMatchPercent - r.originalMatchPercent;
        if (change > 0) {
            std::cout << "+" << std::fixed << std::setprecision(1) << change << "% ✓";
            totalImproved++;
        } else if (change < 0) {
            std::cout << std::fixed << std::setprecision(1) << change << "% ✗";
            totalWorse++;
        } else {
            std::cout << "No change";
            totalSame++;
        }
        std::cout << "\n";
    }

    std::cout << "\n" << std::string(80, '-') << "\n";
    std::cout << "Tests improved: " << totalImproved << "\n";
    std::cout << "Tests worse: " << totalWorse << "\n";
    std::cout << "Tests unchanged: " << totalSame << "\n";

    return 0;
}
