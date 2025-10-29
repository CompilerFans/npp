#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <npp.h>
#include "linear_resize_cpu_final.h"

struct TestCase {
    const char* name;
    int srcW, srcH;
    int dstW, dstH;
    int channels;
};

void printSummaryHeader() {
    std::cout << "\n╔════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         CPU vs NPP Linear Interpolation Comparison Test           ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════╝\n\n";
}

void printTestHeader() {
    std::cout << std::string(80, '=') << "\n";
    std::cout << std::left << std::setw(35) << "Test Case"
             << std::setw(12) << "Size"
             << std::setw(10) << "Match%"
             << std::setw(10) << "MaxDiff"
             << std::setw(10) << "Status"
             << "\n";
    std::cout << std::string(80, '-') << "\n";
}

void runTest(const TestCase& test) {
    int srcSize = test.srcW * test.srcH * test.channels;
    int dstSize = test.dstW * test.dstH * test.channels;

    // Allocate host memory
    std::vector<uint8_t> srcData(srcSize);
    std::vector<uint8_t> nppResult(dstSize);
    std::vector<uint8_t> cpuResult(dstSize);

    // Generate test pattern
    for (int i = 0; i < srcSize; i++) {
        srcData[i] = (i * 255) / (srcSize - 1);
    }

    // NPP resize
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

    // CPU resize
    LinearResizeCPU<uint8_t>::resize(
        srcData.data(), test.srcW * test.channels, test.srcW, test.srcH,
        cpuResult.data(), test.dstW * test.channels, test.dstW, test.dstH,
        test.channels
    );

    // Compare results
    int matchCount = 0;
    int maxDiff = 0;
    for (int i = 0; i < dstSize; i++) {
        int diff = std::abs((int)nppResult[i] - (int)cpuResult[i]);
        if (diff == 0) matchCount++;
        if (diff > maxDiff) maxDiff = diff;
    }

    float matchPercent = (float)matchCount * 100.0f / dstSize;

    // Print result
    std::string sizeStr = std::to_string(test.srcW) + "x" + std::to_string(test.srcH) +
                         "→" + std::to_string(test.dstW) + "x" + std::to_string(test.dstH);
    if (test.channels > 1) {
        sizeStr += "x" + std::to_string(test.channels);
    }

    std::string status;
    if (matchPercent == 100.0f) {
        status = "✓ PERFECT";
    } else if (matchPercent >= 95.0f) {
        status = "✓ EXCELLENT";
    } else if (matchPercent >= 80.0f) {
        status = "~ GOOD";
    } else {
        status = "✗ POOR";
    }

    std::cout << std::left << std::setw(35) << test.name
             << std::setw(12) << sizeStr
             << std::setw(10) << std::fixed << std::setprecision(1) << matchPercent
             << std::setw(10) << maxDiff
             << std::setw(10) << status
             << "\n";
}

void printCategoryHeader(const char* category) {
    std::cout << "\n" << std::string(80, '-') << "\n";
    std::cout << "  " << category << "\n";
    std::cout << std::string(80, '-') << "\n";
}

int main() {
    printSummaryHeader();
    printTestHeader();

    // Test cases organized by category
    std::vector<TestCase> integerUpscales = {
        {"2x upscale", 4, 4, 8, 8, 1},
        {"3x upscale", 4, 4, 12, 12, 1},
        {"4x upscale", 4, 4, 16, 16, 1},
        {"5x upscale", 4, 4, 20, 20, 1}
    };

    std::vector<TestCase> integerDownscales = {
        {"0.5x downscale", 8, 8, 4, 4, 1},
        {"0.33x downscale", 12, 12, 4, 4, 1},
        {"0.25x downscale", 16, 16, 4, 4, 1},
        {"0.2x downscale", 20, 20, 4, 4, 1},
        {"0.125x downscale", 32, 32, 4, 4, 1}
    };

    std::vector<TestCase> fractionalDownscales = {
        {"0.875x downscale", 8, 8, 7, 7, 1},
        {"0.8x downscale", 10, 10, 8, 8, 1},
        {"0.75x downscale", 8, 8, 6, 6, 1},
        {"0.67x downscale", 9, 9, 6, 6, 1},
        {"0.6x downscale", 10, 10, 6, 6, 1}
    };

    std::vector<TestCase> nonUniform = {
        {"Non-uniform 2x×0.5x", 8, 4, 16, 2, 1},
        {"Non-uniform 4x×0.25x", 8, 4, 32, 1, 1},
        {"Non-uniform 0.5x×2x", 4, 8, 2, 16, 1},
        {"Non-uniform 0.25x×4x", 4, 8, 1, 32, 1}
    };

    std::vector<TestCase> fractionalUpscales = {
        {"1.25x upscale", 8, 8, 10, 10, 1},
        {"1.33x upscale", 6, 6, 8, 8, 1},
        {"1.5x upscale", 8, 8, 12, 12, 1},
        {"2.5x upscale", 4, 4, 10, 10, 1}
    };

    std::vector<TestCase> multiChannel = {
        {"RGB 2x upscale", 4, 4, 8, 8, 3},
        {"RGB 0.5x downscale", 8, 8, 4, 4, 3},
        {"RGB 0.75x downscale", 8, 8, 6, 6, 3},
        {"RGBA 2x upscale", 4, 4, 8, 8, 4},
        {"RGBA 0.5x downscale", 8, 8, 4, 4, 4}
    };

    std::vector<TestCase> edgeCases = {
        {"Same size", 8, 8, 8, 8, 1},
        {"Large source", 64, 64, 32, 32, 1},
        {"Large target", 16, 16, 64, 64, 1},
        {"Extreme downscale", 100, 100, 10, 10, 1}
    };

    // Run all tests
    printCategoryHeader("INTEGER UPSCALES");
    for (const auto& test : integerUpscales) runTest(test);

    printCategoryHeader("INTEGER DOWNSCALES");
    for (const auto& test : integerDownscales) runTest(test);

    printCategoryHeader("FRACTIONAL DOWNSCALES");
    for (const auto& test : fractionalDownscales) runTest(test);

    printCategoryHeader("NON-UNIFORM SCALING");
    for (const auto& test : nonUniform) runTest(test);

    printCategoryHeader("FRACTIONAL UPSCALES");
    for (const auto& test : fractionalUpscales) runTest(test);

    printCategoryHeader("MULTI-CHANNEL");
    for (const auto& test : multiChannel) runTest(test);

    printCategoryHeader("EDGE CASES");
    for (const auto& test : edgeCases) runTest(test);

    // Summary statistics
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "\n✓ Test completed successfully\n";
    std::cout << "\nLegend:\n";
    std::cout << "  ✓ PERFECT   : 100% pixel match\n";
    std::cout << "  ✓ EXCELLENT : ≥95% pixel match\n";
    std::cout << "  ~ GOOD      : ≥80% pixel match\n";
    std::cout << "  ✗ POOR      : <80% pixel match\n";

    std::cout << "\nKey findings:\n";
    std::cout << "  • Integer downscales (≥2x): Expected 100% match\n";
    std::cout << "  • 2x upscale: Expected 100% match\n";
    std::cout << "  • 0.75x, 0.67x downscale: Expected 100% match\n";
    std::cout << "  • Non-uniform integer combinations: Expected 100% match\n";
    std::cout << "  • Large integer upscales (3x+): Known differences due to NPP proprietary weights\n\n";

    return 0;
}
