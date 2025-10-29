#include <iostream>
#include <iomanip>
#include <cmath>
#include <npp.h>
#include "linear_resize_cpu_final.h"

int main() {
    std::cout << "Debugging 0.75x downscale (8x8 -> 6x6)\n\n";

    int srcW = 8, srcH = 8;
    int dstW = 6, dstH = 6;
    int srcSize = srcW * srcH;
    int dstSize = dstW * dstH;

    uint8_t* srcData = new uint8_t[srcSize];
    for (int i = 0; i < srcSize; i++) {
        srcData[i] = (i * 255) / (srcSize - 1);
    }

    std::cout << "Source 8x8:\n";
    for (int y = 0; y < srcH; y++) {
        std::cout << "  ";
        for (int x = 0; x < srcW; x++) {
            std::cout << std::setw(3) << (int)srcData[y * srcW + x] << " ";
        }
        std::cout << "\n";
    }

    // NPP
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

    // CPU
    uint8_t* cpuResult = new uint8_t[dstSize];
    LinearResizeCPU<uint8_t>::resize(srcData, srcW, srcW, srcH,
                                     cpuResult, dstW, dstW, dstH, 1);

    std::cout << "\nNPP result 6x6:\n";
    for (int y = 0; y < dstH; y++) {
        std::cout << "  ";
        for (int x = 0; x < dstW; x++) {
            std::cout << std::setw(3) << (int)nppResult[y * dstW + x] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\nCPU result 6x6:\n";
    for (int y = 0; y < dstH; y++) {
        std::cout << "  ";
        for (int x = 0; x < dstW; x++) {
            std::cout << std::setw(3) << (int)cpuResult[y * dstW + x] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\nDifference:\n";
    int matchCount = 0;
    int maxDiff = 0;
    for (int y = 0; y < dstH; y++) {
        std::cout << "  ";
        for (int x = 0; x < dstW; x++) {
            int idx = y * dstW + x;
            int diff = std::abs((int)nppResult[idx] - (int)cpuResult[idx]);
            if (diff == 0) matchCount++;
            if (diff > maxDiff) maxDiff = diff;
            std::cout << std::setw(3) << diff << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\nMatch: " << matchCount << "/" << dstSize << " ("
             << std::fixed << std::setprecision(1)
             << (float)matchCount * 100.0f / dstSize << "%)\n";
    std::cout << "Max diff: " << maxDiff << "\n";

    // Analyze scale factor
    float scaleX = (float)srcW / dstW;
    float scaleY = (float)srcH / dstH;
    std::cout << "\nScale: " << std::setprecision(4) << scaleX << " x " << scaleY << "\n";
    std::cout << "Should use fractional algorithm: " << (scaleX >= 1.0f && scaleX < 2.0f ? "YES" : "NO") << "\n";

    cudaFree(d_src);
    cudaFree(d_dst);
    delete[] srcData;
    delete[] nppResult;
    delete[] cpuResult;

    return 0;
}
