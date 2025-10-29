#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <npp.h>
#include "linear_resize_cpu_final.h"
#include "linear_resize_refactored.h"

struct TestCase {
    const char* name;
    int srcW, srcH, dstW, dstH, channels;
};

void runComparisonTest(const TestCase& test) {
    int srcSize = test.srcW * test.srcH * test.channels;
    int dstSize = test.dstW * test.dstH * test.channels;

    // Generate test data
    std::vector<uint8_t> srcData(srcSize);
    for (int i = 0; i < srcSize; i++) {
        srcData[i] = (i * 255) / (srcSize - 1);
    }

    // NPP result
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
                         d_dst, test.dstW, dstSz, dstROI, NPPI_INTER_LINEAR);
    } else if (test.channels == 3) {
        nppiResize_8u_C3R(d_src, test.srcW * 3, srcSz, srcROI,
                         d_dst, test.dstW * 3, dstSz, dstROI, NPPI_INTER_LINEAR);
    }

    cudaMemcpy(nppResult.data(), d_dst, dstSize, cudaMemcpyDeviceToHost);
    cudaFree(d_src);
    cudaFree(d_dst);

    // Original implementation result
    std::vector<uint8_t> originalResult(dstSize);
    LinearResizeCPU<uint8_t>::resize(
        srcData.data(), test.srcW * test.channels, test.srcW, test.srcH,
        originalResult.data(), test.dstW * test.channels, test.dstW, test.dstH,
        test.channels
    );

    // Refactored implementation result
    std::vector<uint8_t> refactoredResult(dstSize);
    LinearResizeRefactored<uint8_t>::resize(
        srcData.data(), test.srcW * test.channels, test.srcW, test.srcH,
        refactoredResult.data(), test.dstW * test.channels, test.dstW, test.dstH,
        test.channels
    );

    // Compare all three
    int nppOrigMatch = 0, nppRefMatch = 0, origRefMatch = 0;
    int maxDiffNppOrig = 0, maxDiffNppRef = 0, maxDiffOrigRef = 0;

    for (int i = 0; i < dstSize; i++) {
        int diffNppOrig = std::abs((int)nppResult[i] - (int)originalResult[i]);
        int diffNppRef = std::abs((int)nppResult[i] - (int)refactoredResult[i]);
        int diffOrigRef = std::abs((int)originalResult[i] - (int)refactoredResult[i]);

        if (diffNppOrig == 0) nppOrigMatch++;
        if (diffNppRef == 0) nppRefMatch++;
        if (diffOrigRef == 0) origRefMatch++;

        maxDiffNppOrig = std::max(maxDiffNppOrig, diffNppOrig);
        maxDiffNppRef = std::max(maxDiffNppRef, diffNppRef);
        maxDiffOrigRef = std::max(maxDiffOrigRef, diffOrigRef);
    }

    float matchOriginal = (float)nppOrigMatch * 100.0f / dstSize;
    float matchRefactored = (float)nppRefMatch * 100.0f / dstSize;
    float matchBetween = (float)origRefMatch * 100.0f / dstSize;

    // Print results
    std::string sizeStr = std::to_string(test.srcW) + "x" + std::to_string(test.srcH) +
                         "→" + std::to_string(test.dstW) + "x" + std::to_string(test.dstH);
    if (test.channels > 1) sizeStr += "x" + std::to_string(test.channels);

    std::cout << std::left << std::setw(30) << test.name
             << std::setw(15) << sizeStr
             << "Orig: " << std::setw(5) << std::fixed << std::setprecision(1) << matchOriginal << "%"
             << " Ref: " << std::setw(5) << matchRefactored << "%"
             << " Match: " << std::setw(5) << matchBetween << "%";

    if (matchBetween == 100.0f) {
        std::cout << " ✓ IDENTICAL";
    } else {
        std::cout << " ✗ DIFFER (maxDiff=" << maxDiffOrigRef << ")";
    }
    std::cout << "\n";
}

int main() {
    std::cout << "╔════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║      Refactored Implementation Verification Test                       ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════════╝\n\n";

    std::cout << "Comparing: Original vs Refactored vs NPP\n";
    std::cout << "Orig/Ref = Match% with NPP, Match = Original vs Refactored\n\n";

    std::cout << std::string(90, '=') << "\n";
    std::cout << std::left << std::setw(30) << "Test Case"
             << std::setw(15) << "Size"
             << "NPP Match Results\n";
    std::cout << std::string(90, '-') << "\n";

    std::vector<TestCase> tests = {
        // Critical test cases
        {"2x upscale", 4, 4, 8, 8, 1},
        {"0.5x downscale", 8, 8, 4, 4, 1},
        {"0.25x downscale", 16, 16, 4, 4, 1},
        {"0.67x downscale", 9, 9, 6, 6, 1},
        {"0.75x downscale", 8, 8, 6, 6, 1},

        // Non-uniform
        {"Non-uniform 2x×0.5x", 8, 4, 16, 2, 1},
        {"Non-uniform 0.5x×2x", 4, 8, 2, 16, 1},

        // Large cases
        {"3x upscale", 4, 4, 12, 12, 1},
        {"4x upscale", 4, 4, 16, 16, 1},
        {"Large downscale", 64, 64, 32, 32, 1},

        // Multi-channel
        {"RGB 2x upscale", 4, 4, 8, 8, 3},
        {"RGB 0.5x downscale", 8, 8, 4, 4, 3},

        // Edge cases
        {"Same size", 8, 8, 8, 8, 1},
        {"1.5x upscale", 8, 8, 12, 12, 1},
    };

    for (const auto& test : tests) {
        runComparisonTest(test);
    }

    std::cout << std::string(90, '=') << "\n\n";

    std::cout << "✓ All tests completed\n\n";
    std::cout << "Verification results:\n";
    std::cout << "  • Match = 100% : Refactored produces IDENTICAL results to Original\n";
    std::cout << "  • Match < 100% : Implementation differs (BUG!)\n\n";
    std::cout << "Expected: All tests should show 100% Match between Original and Refactored\n";

    return 0;
}
