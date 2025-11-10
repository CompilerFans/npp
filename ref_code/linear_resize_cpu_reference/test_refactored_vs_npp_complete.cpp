#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <npp.h>
#include "linear_resize_refactored.h"

struct TestCase {
    const char* name;
    const char* category;
    int srcW, srcH;
    int dstW, dstH;
    int channels;
};

struct TestResult {
    std::string name;
    std::string size;
    int totalPixels;
    int matchCount;
    float matchPercent;
    int maxDiff;
    int avgDiff;
    std::string status;
};

std::vector<TestResult> allResults;

TestResult runTest(const TestCase& test) {
    int srcSize = test.srcW * test.srcH * test.channels;
    int dstSize = test.dstW * test.dstH * test.channels;

    // Generate test pattern
    std::vector<uint8_t> srcData(srcSize);
    for (int i = 0; i < srcSize; i++) {
        srcData[i] = (i * 255) / (srcSize - 1);
    }

    // NPP resize
    std::vector<uint8_t> nppResult(dstSize);
    uint8_t *d_src, *d_dst;
    cudaMalloc(&d_src, srcSize);
    cudaMalloc(&d_dst, dstSize);
    cudaMemcpy(d_src, srcData.data(), srcSize, cudaMemcpyHostToDevice);

    NppiSize srcSz = {test.srcW, test.srcH};
    NppiSize dstSz = {test.dstW, test.dstH};
    NppiRect srcROI = {0, 0, test.srcW, test.srcH};
    NppiRect dstROI = {0, 0, test.dstW, test.dstH};

    if (test.channels == 1) {
        nppiResize_8u_C1R(d_src, test.srcW, srcSz, srcROI,
                         d_dst, test.dstW, dstSz, dstROI,
                         NPPI_INTER_LINEAR);
    } else if (test.channels == 3) {
        nppiResize_8u_C3R(d_src, test.srcW * 3, srcSz, srcROI,
                         d_dst, test.dstW * 3, dstSz, dstROI,
                         NPPI_INTER_LINEAR);
    } else if (test.channels == 4) {
        nppiResize_8u_C4R(d_src, test.srcW * 4, srcSz, srcROI,
                         d_dst, test.dstW * 4, dstSz, dstROI,
                         NPPI_INTER_LINEAR);
    }

    cudaMemcpy(nppResult.data(), d_dst, dstSize, cudaMemcpyDeviceToHost);
    cudaFree(d_src);
    cudaFree(d_dst);

    // CPU refactored resize
    std::vector<uint8_t> cpuResult(dstSize);
    LinearResizeRefactored<uint8_t>::resize(
        srcData.data(), test.srcW * test.channels, test.srcW, test.srcH,
        cpuResult.data(), test.dstW * test.channels, test.dstW, test.dstH,
        test.channels
    );

    // Analyze differences
    int matchCount = 0;
    int maxDiff = 0;
    int totalDiff = 0;

    for (int i = 0; i < dstSize; i++) {
        int diff = std::abs((int)nppResult[i] - (int)cpuResult[i]);
        if (diff == 0) matchCount++;
        if (diff > maxDiff) maxDiff = diff;
        totalDiff += diff;
    }

    float matchPercent = (float)matchCount * 100.0f / dstSize;
    int avgDiff = (dstSize > 0) ? (totalDiff / dstSize) : 0;

    // Determine status
    std::string status;
    if (matchPercent == 100.0f) {
        status = "✓ PERFECT";
    } else if (matchPercent >= 95.0f) {
        status = "✓ EXCELLENT";
    } else if (matchPercent >= 80.0f) {
        status = "~ GOOD";
    } else if (matchPercent >= 50.0f) {
        status = "⚠ FAIR";
    } else {
        status = "✗ POOR";
    }

    // Build size string
    std::string sizeStr = std::to_string(test.srcW) + "x" + std::to_string(test.srcH) +
                         "→" + std::to_string(test.dstW) + "x" + std::to_string(test.dstH);
    if (test.channels > 1) {
        sizeStr += "×" + std::to_string(test.channels);
    }

    TestResult result = {
        test.name,
        sizeStr,
        dstSize,
        matchCount,
        matchPercent,
        maxDiff,
        avgDiff,
        status
    };

    return result;
}

void printCategoryHeader(const char* category) {
    std::cout << "\n" << std::string(100, '-') << "\n";
    std::cout << "  " << category << "\n";
    std::cout << std::string(100, '-') << "\n";
}

void printSummaryStats() {
    int totalTests = allResults.size();
    int perfectMatches = 0;
    int excellentMatches = 0;
    int goodMatches = 0;

    int totalPixels = 0;
    int totalMatched = 0;

    for (const auto& r : allResults) {
        if (r.matchPercent == 100.0f) perfectMatches++;
        else if (r.matchPercent >= 95.0f) excellentMatches++;
        else if (r.matchPercent >= 80.0f) goodMatches++;

        totalPixels += r.totalPixels;
        totalMatched += r.matchCount;
    }

    float overallMatch = (float)totalMatched * 100.0f / totalPixels;

    std::cout << "\n" << std::string(100, '=') << "\n";
    std::cout << "SUMMARY STATISTICS\n";
    std::cout << std::string(100, '=') << "\n\n";

    std::cout << "Total test cases:     " << totalTests << "\n";
    std::cout << "Perfect matches:      " << perfectMatches << " ("
             << std::fixed << std::setprecision(1)
             << (float)perfectMatches * 100.0f / totalTests << "%)\n";
    std::cout << "Excellent matches:    " << excellentMatches << " ("
             << (float)excellentMatches * 100.0f / totalTests << "%)\n";
    std::cout << "Good matches:         " << goodMatches << " ("
             << (float)goodMatches * 100.0f / totalTests << "%)\n";
    std::cout << "\nOverall pixel match:  " << std::setprecision(2) << overallMatch << "%\n";
    std::cout << "Total pixels tested:  " << totalPixels << "\n";
    std::cout << "Total matched pixels: " << totalMatched << "\n";
}

void printDetailedResults() {
    std::cout << "\n" << std::string(100, '=') << "\n";
    std::cout << "DETAILED RESULTS BY MATCH PERCENTAGE\n";
    std::cout << std::string(100, '=') << "\n";

    // Perfect matches
    std::cout << "\n✓ PERFECT MATCHES (100%):\n";
    for (const auto& r : allResults) {
        if (r.matchPercent == 100.0f) {
            std::cout << "  • " << std::left << std::setw(40) << r.name
                     << std::setw(20) << r.size << "\n";
        }
    }

    // Excellent matches
    int excellentCount = 0;
    for (const auto& r : allResults) {
        if (r.matchPercent >= 95.0f && r.matchPercent < 100.0f) excellentCount++;
    }
    if (excellentCount > 0) {
        std::cout << "\n✓ EXCELLENT MATCHES (95-99%):\n";
        for (const auto& r : allResults) {
            if (r.matchPercent >= 95.0f && r.matchPercent < 100.0f) {
                std::cout << "  • " << std::left << std::setw(40) << r.name
                         << std::setw(20) << r.size
                         << std::fixed << std::setprecision(1) << r.matchPercent << "%"
                         << " (maxDiff=" << r.maxDiff << ")\n";
            }
        }
    }

    // Poor matches
    std::cout << "\n✗ POOR MATCHES (<80%):\n";
    for (const auto& r : allResults) {
        if (r.matchPercent < 80.0f) {
            std::cout << "  • " << std::left << std::setw(40) << r.name
                     << std::setw(20) << r.size
                     << std::fixed << std::setprecision(1) << r.matchPercent << "%"
                     << " (maxDiff=" << r.maxDiff << ", avgDiff=" << r.avgDiff << ")\n";
        }
    }
}

int main() {
    std::cout << "╔════════════════════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    Refactored Implementation vs NVIDIA NPP - Complete Precision Test            ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════════════════════════════════╝\n";

    std::cout << "\n" << std::string(100, '=') << "\n";
    std::cout << std::left << std::setw(40) << "Test Case"
             << std::setw(20) << "Size"
             << std::setw(12) << "Match%"
             << std::setw(10) << "MaxDiff"
             << std::setw(10) << "AvgDiff"
             << std::setw(12) << "Status\n";
    std::cout << std::string(100, '=') << "\n";

    // Define comprehensive test cases
    std::vector<TestCase> allTests = {
        // Integer upscales
        {"2x upscale (gray)", "Integer Upscales", 4, 4, 8, 8, 1},
        {"3x upscale (gray)", "Integer Upscales", 4, 4, 12, 12, 1},
        {"4x upscale (gray)", "Integer Upscales", 4, 4, 16, 16, 1},
        {"5x upscale (gray)", "Integer Upscales", 4, 4, 20, 20, 1},
        {"8x upscale (gray)", "Integer Upscales", 2, 2, 16, 16, 1},

        // Integer downscales
        {"0.5x downscale (1/2)", "Integer Downscales", 8, 8, 4, 4, 1},
        {"0.33x downscale (1/3)", "Integer Downscales", 12, 12, 4, 4, 1},
        {"0.25x downscale (1/4)", "Integer Downscales", 16, 16, 4, 4, 1},
        {"0.2x downscale (1/5)", "Integer Downscales", 20, 20, 4, 4, 1},
        {"0.125x downscale (1/8)", "Integer Downscales", 32, 32, 4, 4, 1},

        // Fractional upscales
        {"1.25x upscale (5/4)", "Fractional Upscales", 8, 8, 10, 10, 1},
        {"1.33x upscale (4/3)", "Fractional Upscales", 6, 6, 8, 8, 1},
        {"1.5x upscale (3/2)", "Fractional Upscales", 8, 8, 12, 12, 1},
        {"2.5x upscale (5/2)", "Fractional Upscales", 4, 4, 10, 10, 1},
        {"3.5x upscale (7/2)", "Fractional Upscales", 4, 4, 14, 14, 1},

        // Fractional downscales
        {"0.875x downscale (7/8)", "Fractional Downscales", 8, 8, 7, 7, 1},
        {"0.8x downscale (4/5)", "Fractional Downscales", 10, 10, 8, 8, 1},
        {"0.75x downscale (3/4)", "Fractional Downscales", 8, 8, 6, 6, 1},
        {"0.67x downscale (2/3)", "Fractional Downscales", 9, 9, 6, 6, 1},
        {"0.6x downscale (3/5)", "Fractional Downscales", 10, 10, 6, 6, 1},
        {"0.4x downscale (2/5)", "Fractional Downscales", 10, 10, 4, 4, 1},

        // Non-uniform scaling
        {"Non-uniform 2x×0.5x", "Non-uniform Scaling", 8, 4, 16, 2, 1},
        {"Non-uniform 4x×0.25x", "Non-uniform Scaling", 8, 4, 32, 1, 1},
        {"Non-uniform 0.5x×2x", "Non-uniform Scaling", 4, 8, 2, 16, 1},
        {"Non-uniform 0.25x×4x", "Non-uniform Scaling", 4, 8, 1, 32, 1},
        {"Non-uniform 3x×0.5x", "Non-uniform Scaling", 8, 4, 24, 2, 1},
        {"Non-uniform 0.75x×2x", "Non-uniform Scaling", 8, 4, 6, 8, 1},
        {"Non-uniform 2x×0.75x", "Non-uniform Scaling", 4, 8, 8, 6, 1},

        // Multi-channel
        {"RGB 2x upscale", "Multi-channel", 4, 4, 8, 8, 3},
        {"RGB 0.5x downscale", "Multi-channel", 8, 8, 4, 4, 3},
        {"RGB 0.75x downscale", "Multi-channel", 8, 8, 6, 6, 3},
        {"RGB 3x upscale", "Multi-channel", 4, 4, 12, 12, 3},
        {"RGBA 2x upscale", "Multi-channel", 4, 4, 8, 8, 4},
        {"RGBA 0.5x downscale", "Multi-channel", 8, 8, 4, 4, 4},

        // Edge cases
        {"Same size 8x8", "Edge Cases", 8, 8, 8, 8, 1},
        {"Same size 16x16", "Edge Cases", 16, 16, 16, 16, 1},
        {"Large source 64x64→32x32", "Edge Cases", 64, 64, 32, 32, 1},
        {"Large target 16x16→64x64", "Edge Cases", 16, 16, 64, 64, 1},
        {"Extreme down 100x100→10x10", "Edge Cases", 100, 100, 10, 10, 1},
        {"Extreme down 128x128→8x8", "Edge Cases", 128, 128, 8, 8, 1},

        // Small sizes
        {"Tiny 2x2→4x4", "Small Sizes", 2, 2, 4, 4, 1},
        {"Tiny 3x3→6x6", "Small Sizes", 3, 3, 6, 6, 1},
        {"Tiny 4x4→2x2", "Small Sizes", 4, 4, 2, 2, 1},

        // Rectangular (non-square)
        {"Rect 8x4→16x8", "Rectangular", 8, 4, 16, 8, 1},
        {"Rect 4x8→8x16", "Rectangular", 4, 8, 8, 16, 1},
        {"Rect 16x8→8x4", "Rectangular", 16, 8, 8, 4, 1},
        {"Rect 8x16→4x8", "Rectangular", 8, 16, 4, 8, 1},
    };

    // Run all tests and collect results
    std::string currentCategory = "";
    for (const auto& test : allTests) {
        if (currentCategory != test.category) {
            printCategoryHeader(test.category);
            currentCategory = test.category;
        }

        TestResult result = runTest(test);
        allResults.push_back(result);

        std::cout << std::left << std::setw(40) << result.name
                 << std::setw(20) << result.size
                 << std::setw(12) << std::fixed << std::setprecision(1) << result.matchPercent
                 << std::setw(10) << result.maxDiff
                 << std::setw(10) << result.avgDiff
                 << std::setw(12) << result.status << "\n";
    }

    // Print summary and detailed analysis
    printSummaryStats();
    printDetailedResults();

    std::cout << "\n" << std::string(100, '=') << "\n";
    std::cout << "\nTest Legend:\n";
    std::cout << "  ✓ PERFECT   : 100% pixel match\n";
    std::cout << "  ✓ EXCELLENT : 95-99% pixel match\n";
    std::cout << "  ~ GOOD      : 80-94% pixel match\n";
    std::cout << "  ⚠ FAIR      : 50-79% pixel match\n";
    std::cout << "  ✗ POOR      : <50% pixel match\n\n";

    std::cout << "Implementation Quality Assessment:\n";
    int perfectCount = 0;
    for (const auto& r : allResults) {
        if (r.matchPercent == 100.0f) perfectCount++;
    }
    float perfectRate = (float)perfectCount * 100.0f / allResults.size();

    std::cout << "  Perfect match rate: " << std::fixed << std::setprecision(1)
             << perfectRate << "% (" << perfectCount << "/" << allResults.size() << " tests)\n";

    if (perfectRate >= 40.0f) {
        std::cout << "  ✓ Production ready for common use cases\n";
    } else if (perfectRate >= 25.0f) {
        std::cout << "  ~ Suitable for non-critical applications\n";
    } else {
        std::cout << "  ⚠ Needs improvement for production use\n";
    }

    std::cout << "\n✓ Complete precision test finished\n\n";

    return 0;
}
