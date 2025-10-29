#include <iostream>
#include <iomanip>
#include <cmath>
#include <npp.h>
#include "linear_interpolation_cpu.h"

// Deep analysis of integer upscales
void analyzeIntegerUpscale(const char* name, int factor) {
    std::cout << "\n========================================\n";
    std::cout << "ANALYZING: " << factor << "x INTEGER UPSCALE\n";
    std::cout << "========================================\n\n";

    int srcW = 4, srcH = 4;
    int dstW = srcW * factor;
    int dstH = srcH * factor;

    int srcSize = srcW * srcH;
    int dstSize = dstW * dstH;

    // Create simple pattern
    uint8_t* srcData = new uint8_t[srcSize];
    for (int i = 0; i < srcSize; i++) {
        srcData[i] = (i * 255) / (srcSize - 1);
    }

    std::cout << "Source " << srcW << "x" << srcH << ":\n";
    for (int y = 0; y < srcH; y++) {
        std::cout << "  ";
        for (int x = 0; x < srcW; x++) {
            std::cout << std::setw(3) << (int)srcData[y * srcW + x] << " ";
        }
        std::cout << "\n";
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

    // V1 result
    uint8_t* v1Result = new uint8_t[dstSize];
    LinearInterpolationCPU<uint8_t>::resize(srcData, srcW, srcW, srcH,
                                            v1Result, dstW, dstW, dstH, 1);

    // Compare
    int matchCount = 0;
    int maxDiff = 0;
    for (int i = 0; i < dstSize; i++) {
        int diff = std::abs((int)nppResult[i] - (int)v1Result[i]);
        if (diff == 0) matchCount++;
        if (diff > maxDiff) maxDiff = diff;
    }

    std::cout << "\nMatch: " << matchCount << "/" << dstSize << " pixels ("
             << std::fixed << std::setprecision(1)
             << (float)matchCount * 100.0f / dstSize << "%)\n";
    std::cout << "Max difference: " << maxDiff << "\n\n";

    // Analyze first few rows in detail
    float scale = 1.0f / factor;
    std::cout << "Detailed analysis (first " << std::min(3, dstH) << " rows):\n";
    std::cout << "Format: (NPP,V1,diff)\n\n";

    for (int dy = 0; dy < std::min(3, dstH); dy++) {
        std::cout << "Row " << dy << " (srcY=" << std::fixed << std::setprecision(3)
                 << ((dy + 0.5f) * scale - 0.5f) << "):\n  ";
        for (int dx = 0; dx < dstW; dx++) {
            int idx = dy * dstW + dx;
            int npp = nppResult[idx];
            int v1 = v1Result[idx];
            int diff = std::abs(npp - v1);

            std::cout << "(" << std::setw(3) << npp << ","
                     << std::setw(3) << v1 << ","
                     << std::setw(2) << diff << ") ";
        }
        std::cout << "\n";
    }

    // Analyze specific pixels to find pattern
    std::cout << "\nPer-pixel coordinate analysis (row 0, first " << std::min(8, dstW) << " pixels):\n";
    for (int dx = 0; dx < std::min(8, dstW); dx++) {
        float srcX = (dx + 0.5f) * scale - 0.5f;
        int x0 = (int)std::floor(srcX);
        int x1 = std::min(x0 + 1, srcW - 1);
        x0 = std::max(0, x0);
        float fx = srcX - std::floor(srcX);

        int npp = nppResult[dx];
        int v1 = v1Result[dx];

        int v0_src = srcData[x0];
        int v1_src = srcData[x1];

        float bilinear = v0_src * (1.0f - fx) + v1_src * fx;

        std::cout << "dst[" << dx << "]: srcX=" << std::fixed << std::setprecision(3) << srcX
                 << ", fx=" << fx << "\n";
        std::cout << "  neighbors=[" << v0_src << "," << v1_src << "]\n";
        std::cout << "  NPP=" << npp << ", V1=" << v1 << ", diff=" << std::abs(npp - v1) << "\n";
        std::cout << "  Standard bilinear=" << std::setprecision(1) << bilinear
                 << " -> " << (int)(bilinear + 0.5f) << "\n";

        // Try to reverse engineer fx
        if (v0_src != v1_src && npp != v0_src && npp != v1_src) {
            float fx_npp = (float)(npp - v0_src) / (v1_src - v0_src);
            std::cout << "  NPP effective fx=" << std::setprecision(3) << fx_npp;
            if (fx > 0.001f) {
                float factor_ratio = fx_npp / fx;
                std::cout << " (ratio=" << std::setprecision(2) << factor_ratio << ")";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    cudaFree(d_src);
    cudaFree(d_dst);
    delete[] srcData;
    delete[] nppResult;
    delete[] v1Result;
}

int main() {
    std::cout << "========================================\n";
    std::cout << "INTEGER UPSCALE DEEP ANALYSIS\n";
    std::cout << "========================================\n";

    // Test problematic integer upscales
    analyzeIntegerUpscale("3x upscale", 3);
    analyzeIntegerUpscale("4x upscale", 4);
    analyzeIntegerUpscale("5x upscale", 5);

    // Compare with working case
    std::cout << "\n========================================\n";
    std::cout << "COMPARISON WITH WORKING CASE\n";
    std::cout << "========================================\n";
    analyzeIntegerUpscale("2x upscale [WORKING]", 2);

    return 0;
}
