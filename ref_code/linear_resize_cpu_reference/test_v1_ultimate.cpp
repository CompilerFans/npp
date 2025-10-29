#include <iostream>
#include <iomanip>
#include <cmath>
#include <npp.h>
#include "linear_interpolation_v1_ultimate.h"

void testScale(const char* name, int srcW, int srcH, int dstW, int dstH) {
    std::cout << "\n========================================\n";
    std::cout << "TEST: " << name << " (" << srcW << "x" << srcH << " -> " << dstW << "x" << dstH << ")\n";
    std::cout << "Scale: " << std::fixed << std::setprecision(2)
             << (float)dstW/srcW << "x, " << (float)dstH/srcH << "x\n";
    std::cout << "========================================\n";

    int srcSize = srcW * srcH;
    int dstSize = dstW * dstH;

    // Create gradient pattern
    uint8_t* srcData = new uint8_t[srcSize];
    for (int i = 0; i < srcSize; i++) {
        srcData[i] = (i * 255) / (srcSize - 1);
    }

    // NPP result
    uint8_t* nppResult = new uint8_t[dstSize];
    uint8_t *d_src, *d_dst;
    cudaMalloc(&d_src, srcSize);
    cudaMalloc(&d_dst, dstSize);
    cudaMemcpy(d_src, srcData, srcSize, cudaMemcpyHostToDevice);

    NppiSize srcSz = {srcW, srcH};
    NppiSize dstSz = {dstW, dstH};
    NppiRect srcROI = {0, 0, srcW, srcH};
    NppiRect dstROI = {0, 0, dstW, dstH};

    nppiResize_8u_C1R(d_src, srcW, srcSz, srcROI,
                     d_dst, dstW, dstSz, dstROI,
                     NPPI_INTER_LINEAR);

    cudaMemcpy(nppResult, d_dst, dstSize, cudaMemcpyDeviceToHost);

    // V1 Ultimate result
    uint8_t* v1Result = new uint8_t[dstSize];
    LinearInterpolationV1Ultimate<uint8_t>::resize(srcData, srcW, srcW, srcH,
                                                     v1Result, dstW, dstW, dstH, 1);

    // Compare
    int matchCount = 0;
    int maxDiff = 0;
    for (int i = 0; i < dstSize; i++) {
        int diff = std::abs((int)nppResult[i] - (int)v1Result[i]);
        if (diff == 0) matchCount++;
        if (diff > maxDiff) maxDiff = diff;
    }

    float matchPercent = (float)matchCount * 100.0f / dstSize;

    std::cout << "Results: " << matchCount << "/" << dstSize << " pixels match ("
             << std::fixed << std::setprecision(1) << matchPercent << "%)\n";
    std::cout << "Max difference: " << maxDiff << "\n";

    if (matchPercent == 100.0f) {
        std::cout << "✓ PERFECT MATCH!\n";
    } else if (matchPercent >= 95.0f) {
        std::cout << "✓ Excellent match\n";
    } else if (matchPercent >= 80.0f) {
        std::cout << "~ Good match\n";
    } else {
        std::cout << "✗ Poor match - needs investigation\n";

        // Show first few mismatches for small sizes
        if (dstW <= 8 && dstH <= 8) {
            std::cout << "\nDetailed comparison:\n";
            for (int y = 0; y < dstH; y++) {
                for (int x = 0; x < dstW; x++) {
                    int idx = y * dstW + x;
                    int npp = nppResult[idx];
                    int v1 = v1Result[idx];
                    int diff = std::abs(npp - v1);
                    std::cout << "(" << std::setw(3) << npp << ","
                             << std::setw(3) << v1 << ","
                             << std::setw(2) << diff << ") ";
                }
                std::cout << "\n";
            }
        }
    }

    cudaFree(d_src);
    cudaFree(d_dst);
    delete[] srcData;
    delete[] nppResult;
    delete[] v1Result;
}

int main() {
    std::cout << "========================================\n";
    std::cout << "V1 ULTIMATE VALIDATION\n";
    std::cout << "Testing fractional downscales with new algorithm\n";
    std::cout << "========================================\n";

    // Previously problematic fractional downscales
    std::cout << "\n=== FRACTIONAL DOWNSCALES (1.0 < scale < 2.0) ===\n";
    testScale("0.875x (7/8)", 8, 2, 7, 2);
    testScale("0.8x (4/5)", 10, 2, 8, 2);
    testScale("0.75x (3/4)", 8, 2, 6, 2);
    testScale("0.67x (2/3)", 9, 2, 6, 2);
    testScale("0.6x (3/5)", 10, 2, 6, 2);

    // Previously working cases - should still work
    std::cout << "\n=== WORKING CASES (should maintain 100%) ===\n";
    testScale("2x upscale", 4, 4, 8, 8);
    testScale("0.5x downscale", 8, 8, 4, 4);
    testScale("0.25x downscale", 16, 16, 4, 4);
    testScale("Non-uniform 2x×0.5x", 8, 4, 16, 2);

    // New test: combination of fractional and other modes
    std::cout << "\n=== MIXED MODE TESTS ===\n";
    testScale("Fractional X, upscale Y (0.75x, 2x)", 8, 4, 6, 8);
    testScale("Upscale X, fractional Y (2x, 0.75x)", 4, 8, 8, 6);
    testScale("Fractional both (0.8x, 0.75x)", 10, 8, 8, 6);

    std::cout << "\n========================================\n";
    std::cout << "VALIDATION COMPLETE\n";
    std::cout << "========================================\n";

    return 0;
}
