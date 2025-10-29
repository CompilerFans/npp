#include <iostream>
#include <iomanip>
#include <vector>
#include <cstring>
#include <cmath>
#include <npp.h>
#include "linear_interpolation_cpu.h"

struct TestCase {
    std::string name;
    int srcW, srcH;
    int dstW, dstH;
    int channels;
    float scaleX, scaleY;
};

struct TestResult {
    std::string name;
    int matchCount;
    int totalPixels;
    float matchPercent;
    int maxDiff;
    float avgDiff;
    float scaleX, scaleY;
};

void runTest(const TestCase& test, TestResult& result) {
    result.name = test.name;
    result.scaleX = test.scaleX;
    result.scaleY = test.scaleY;

    int srcStep = test.srcW * test.channels;
    int dstStep = test.dstW * test.channels;
    int srcSize = test.srcH * srcStep;
    int dstSize = test.dstH * dstStep;

    // Create gradient pattern
    uint8_t* srcData = new uint8_t[srcSize];
    for (int y = 0; y < test.srcH; y++) {
        for (int x = 0; x < test.srcW; x++) {
            for (int c = 0; c < test.channels; c++) {
                srcData[(y * test.srcW + x) * test.channels + c] =
                    (x * 255 / (test.srcW - 1));
            }
        }
    }

    uint8_t* nppResult = new uint8_t[dstSize];
    uint8_t* cpuResult = new uint8_t[dstSize];

    // NPP
    uint8_t *d_src, *d_dst;
    cudaMalloc(&d_src, srcSize);
    cudaMalloc(&d_dst, dstSize);
    cudaMemcpy(d_src, srcData, srcSize, cudaMemcpyHostToDevice);

    NppiSize srcSz = {test.srcW, test.srcH};
    NppiSize dstSz = {test.dstW, test.dstH};
    NppiRect srcROI = {0, 0, test.srcW, test.srcH};
    NppiRect dstROI = {0, 0, test.dstW, test.dstH};

    if (test.channels == 1) {
        nppiResize_8u_C1R(d_src, srcStep, srcSz, srcROI,
                         d_dst, dstStep, dstSz, dstROI, NPPI_INTER_LINEAR);
    } else {
        nppiResize_8u_C3R(d_src, srcStep, srcSz, srcROI,
                         d_dst, dstStep, dstSz, dstROI, NPPI_INTER_LINEAR);
    }

    cudaMemcpy(nppResult, d_dst, dstSize, cudaMemcpyDeviceToHost);

    // CPU
    LinearInterpolationCPU<uint8_t>::resize(srcData, srcStep, test.srcW, test.srcH,
                                            cpuResult, dstStep, test.dstW, test.dstH,
                                            test.channels);

    // Compare
    result.totalPixels = test.dstW * test.dstH * test.channels;
    result.matchCount = 0;
    result.maxDiff = 0;
    int totalDiff = 0;

    for (int i = 0; i < result.totalPixels; i++) {
        int diff = std::abs((int)nppResult[i] - (int)cpuResult[i]);
        if (diff == 0) result.matchCount++;
        if (diff > result.maxDiff) result.maxDiff = diff;
        totalDiff += diff;
    }

    result.matchPercent = (float)result.matchCount / result.totalPixels * 100.0f;
    result.avgDiff = (float)totalDiff / result.totalPixels;

    cudaFree(d_src);
    cudaFree(d_dst);
    delete[] srcData;
    delete[] nppResult;
    delete[] cpuResult;
}

int main() {
    std::cout << "========================================\n";
    std::cout << "Comprehensive Linear Resize Test Suite\n";
    std::cout << "========================================\n\n";

    std::vector<TestCase> tests;

    // Category 1: Integer upscales
    tests.push_back({"2x Upscale", 4, 4, 8, 8, 1, 2.0f, 2.0f});
    tests.push_back({"3x Upscale", 4, 4, 12, 12, 1, 3.0f, 3.0f});
    tests.push_back({"4x Upscale", 4, 4, 16, 16, 1, 4.0f, 4.0f});
    tests.push_back({"5x Upscale", 4, 4, 20, 20, 1, 5.0f, 5.0f});
    tests.push_back({"8x Upscale", 2, 2, 16, 16, 1, 8.0f, 8.0f});

    // Category 2: Integer downscales
    tests.push_back({"0.5x Downscale", 8, 8, 4, 4, 1, 0.5f, 0.5f});
    tests.push_back({"0.33x Downscale (1/3)", 12, 12, 4, 4, 1, 0.33f, 0.33f});
    tests.push_back({"0.25x Downscale", 16, 16, 4, 4, 1, 0.25f, 0.25f});
    tests.push_back({"0.2x Downscale (1/5)", 20, 20, 4, 4, 1, 0.2f, 0.2f});
    tests.push_back({"0.125x Downscale", 16, 16, 2, 2, 1, 0.125f, 0.125f});

    // Category 3: Non-integer upscales
    tests.push_back({"1.5x Upscale", 4, 4, 6, 6, 1, 1.5f, 1.5f});
    tests.push_back({"2.5x Upscale", 4, 4, 10, 10, 1, 2.5f, 2.5f});
    tests.push_back({"3.5x Upscale", 4, 4, 14, 14, 1, 3.5f, 3.5f});
    tests.push_back({"1.33x Upscale (4/3)", 6, 6, 8, 8, 1, 1.33f, 1.33f});
    tests.push_back({"1.25x Upscale (5/4)", 8, 8, 10, 10, 1, 1.25f, 1.25f});

    // Category 4: Non-integer downscales
    tests.push_back({"0.75x Downscale (3/4)", 8, 8, 6, 6, 1, 0.75f, 0.75f});
    tests.push_back({"0.67x Downscale (2/3)", 9, 9, 6, 6, 1, 0.67f, 0.67f});
    tests.push_back({"0.6x Downscale (3/5)", 10, 10, 6, 6, 1, 0.6f, 0.6f});
    tests.push_back({"0.8x Downscale (4/5)", 10, 10, 8, 8, 1, 0.8f, 0.8f});
    tests.push_back({"0.875x Downscale (7/8)", 8, 8, 7, 7, 1, 0.875f, 0.875f});
    tests.push_back({"0.4x Downscale (2/5)", 10, 10, 4, 4, 1, 0.4f, 0.4f});

    // Category 5: Non-uniform scaling
    tests.push_back({"2x×0.5x Non-uniform", 4, 8, 8, 4, 1, 2.0f, 0.5f});
    tests.push_back({"0.5x×2x Non-uniform", 8, 4, 4, 8, 1, 0.5f, 2.0f});
    tests.push_back({"3x×0.5x Non-uniform", 4, 8, 12, 4, 1, 3.0f, 0.5f});
    tests.push_back({"0.5x×3x Non-uniform", 8, 4, 4, 12, 1, 0.5f, 3.0f});
    tests.push_back({"2x×0.75x Non-uniform", 4, 8, 8, 6, 1, 2.0f, 0.75f});
    tests.push_back({"0.75x×2x Non-uniform", 8, 4, 6, 8, 1, 0.75f, 2.0f});
    tests.push_back({"1.5x×0.67x Non-uniform", 6, 9, 9, 6, 1, 1.5f, 0.67f});

    // Category 6: Edge cases
    tests.push_back({"Very small (2x2->4x4)", 2, 2, 4, 4, 1, 2.0f, 2.0f});
    tests.push_back({"Very small (3x3->6x6)", 3, 3, 6, 6, 1, 2.0f, 2.0f});
    tests.push_back({"Large (16x16->32x32)", 16, 16, 32, 32, 1, 2.0f, 2.0f});
    tests.push_back({"Large downscale (32x32->8x8)", 32, 32, 8, 8, 1, 0.25f, 0.25f});
    tests.push_back({"Extreme 16x", 2, 2, 32, 32, 1, 16.0f, 16.0f});
    tests.push_back({"Extreme 0.0625x", 32, 32, 2, 2, 1, 0.0625f, 0.0625f});

    // Category 7: RGB
    tests.push_back({"2x Upscale RGB", 4, 4, 8, 8, 3, 2.0f, 2.0f});
    tests.push_back({"0.5x Downscale RGB", 8, 8, 4, 4, 3, 0.5f, 0.5f});
    tests.push_back({"1.5x Upscale RGB", 4, 4, 6, 6, 3, 1.5f, 1.5f});
    tests.push_back({"0.75x Downscale RGB", 8, 8, 6, 6, 3, 0.75f, 0.75f});

    std::vector<TestResult> results;

    std::cout << "Running " << tests.size() << " test cases...\n\n";

    for (const auto& test : tests) {
        TestResult result;
        runTest(test, result);
        results.push_back(result);

        std::cout << std::left << std::setw(35) << test.name
                 << " [" << test.srcW << "x" << test.srcH << "->" << test.dstW << "x" << test.dstH << "] "
                 << std::fixed << std::setprecision(1)
                 << std::setw(6) << result.matchPercent << "% "
                 << "(max:" << std::setw(3) << result.maxDiff << ") ";

        if (result.matchPercent == 100.0f) {
            std::cout << "✓ Perfect";
        } else if (result.matchPercent >= 90.0f) {
            std::cout << "○ Excellent";
        } else if (result.matchPercent >= 50.0f) {
            std::cout << "△ Good";
        } else if (result.matchPercent >= 25.0f) {
            std::cout << "▽ Fair";
        } else {
            std::cout << "✗ Poor";
        }
        std::cout << "\n";
    }

    // Summary by category
    std::cout << "\n========================================\n";
    std::cout << "CATEGORY SUMMARY\n";
    std::cout << "========================================\n\n";

    auto printCategory = [&](const std::string& name, int start, int end) {
        int perfect = 0, excellent = 0, good = 0, fair = 0, poor = 0;
        float totalMatch = 0.0f;
        int maxDiffInCategory = 0;

        for (int i = start; i < end && i < results.size(); i++) {
            if (results[i].matchPercent == 100.0f) perfect++;
            else if (results[i].matchPercent >= 90.0f) excellent++;
            else if (results[i].matchPercent >= 50.0f) good++;
            else if (results[i].matchPercent >= 25.0f) fair++;
            else poor++;

            totalMatch += results[i].matchPercent;
            if (results[i].maxDiff > maxDiffInCategory) {
                maxDiffInCategory = results[i].maxDiff;
            }
        }

        int count = end - start;
        if (count > results.size() - start) count = results.size() - start;

        std::cout << name << " (" << count << " tests):\n";
        std::cout << "  Perfect (100%):  " << perfect << "\n";
        std::cout << "  Excellent (>90%): " << excellent << "\n";
        std::cout << "  Good (>50%):     " << good << "\n";
        std::cout << "  Fair (>25%):     " << fair << "\n";
        std::cout << "  Poor (<25%):     " << poor << "\n";
        std::cout << "  Avg match:       " << std::fixed << std::setprecision(1)
                 << (totalMatch / count) << "%\n";
        std::cout << "  Max diff:        " << maxDiffInCategory << "\n\n";
    };

    printCategory("Integer Upscales", 0, 5);
    printCategory("Integer Downscales", 5, 10);
    printCategory("Non-integer Upscales", 10, 15);
    printCategory("Non-integer Downscales", 15, 21);
    printCategory("Non-uniform Scaling", 21, 28);
    printCategory("Edge Cases", 28, 34);
    printCategory("RGB Tests", 34, 38);

    // Overall statistics
    std::cout << "========================================\n";
    std::cout << "OVERALL STATISTICS\n";
    std::cout << "========================================\n\n";

    int totalPerfect = 0, totalExcellent = 0, totalGood = 0, totalFair = 0, totalPoor = 0;
    float overallMatch = 0.0f;
    int overallMaxDiff = 0;

    for (const auto& r : results) {
        if (r.matchPercent == 100.0f) totalPerfect++;
        else if (r.matchPercent >= 90.0f) totalExcellent++;
        else if (r.matchPercent >= 50.0f) totalGood++;
        else if (r.matchPercent >= 25.0f) totalFair++;
        else totalPoor++;

        overallMatch += r.matchPercent;
        if (r.maxDiff > overallMaxDiff) overallMaxDiff = r.maxDiff;
    }

    std::cout << "Total tests:      " << results.size() << "\n";
    std::cout << "Perfect (100%):   " << totalPerfect << " ("
             << std::fixed << std::setprecision(1)
             << (float)totalPerfect/results.size()*100 << "%)\n";
    std::cout << "Excellent (>90%): " << totalExcellent << " ("
             << (float)totalExcellent/results.size()*100 << "%)\n";
    std::cout << "Good (>50%):      " << totalGood << " ("
             << (float)totalGood/results.size()*100 << "%)\n";
    std::cout << "Fair (>25%):      " << totalFair << " ("
             << (float)totalFair/results.size()*100 << "%)\n";
    std::cout << "Poor (<25%):      " << totalPoor << " ("
             << (float)totalPoor/results.size()*100 << "%)\n\n";
    std::cout << "Average match:    " << (overallMatch / results.size()) << "%\n";
    std::cout << "Max diff overall: " << overallMaxDiff << "\n";

    // Problem areas
    std::cout << "\n========================================\n";
    std::cout << "PROBLEM AREAS (Match < 50%)\n";
    std::cout << "========================================\n\n";

    for (const auto& r : results) {
        if (r.matchPercent < 50.0f) {
            std::cout << std::left << std::setw(35) << r.name
                     << std::fixed << std::setprecision(1)
                     << std::setw(6) << r.matchPercent << "% "
                     << "(max diff: " << r.maxDiff << ", "
                     << "avg: " << std::setprecision(2) << r.avgDiff << ")\n";
        }
    }

    return 0;
}
