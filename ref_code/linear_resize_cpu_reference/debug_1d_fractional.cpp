#include <iostream>
#include <iomanip>
#include <cmath>
#include <npp.h>
#include "linear_interpolation_v1_ultimate.h"

// Debug specific 1D fractional downscale
void debug1DFractional(const char* name, int srcW, int dstW) {
    std::cout << "\n========================================\n";
    std::cout << "DEBUG: " << name << " (" << srcW << " -> " << dstW << ")\n";
    float scale = (float)srcW / dstW;
    std::cout << "Scale: " << std::fixed << std::setprecision(4) << scale << "\n";
    std::cout << "========================================\n\n";

    // 2 rows for proper 2D input
    int srcH = 2, dstH = 2;
    uint8_t* srcData = new uint8_t[srcW * srcH];

    // Create gradient
    for (int i = 0; i < srcW; i++) {
        srcData[i] = (i * 255) / (srcW - 1);
        srcData[srcW + i] = (i * 255) / (srcW - 1);
    }

    uint8_t* nppResult = new uint8_t[dstW * dstH];
    uint8_t* v1Result = new uint8_t[dstW * dstH];

    // NPP
    uint8_t *d_src, *d_dst;
    cudaMalloc(&d_src, srcW * srcH);
    cudaMalloc(&d_dst, dstW * dstH);
    cudaMemcpy(d_src, srcData, srcW * srcH, cudaMemcpyHostToDevice);

    NppiSize srcSz = {srcW, srcH};
    NppiSize dstSz = {dstW, dstH};
    NppiRect srcROI = {0, 0, srcW, srcH};
    NppiRect dstROI = {0, 0, dstW, dstH};

    nppiResize_8u_C1R(d_src, srcW, srcSz, srcROI,
                     d_dst, dstW, dstSz, dstROI,
                     NPPI_INTER_LINEAR);

    cudaMemcpy(nppResult, d_dst, dstW * dstH, cudaMemcpyDeviceToHost);

    // V1 Ultimate
    LinearInterpolationV1Ultimate<uint8_t>::resize(srcData, srcW, srcW, srcH,
                                                     v1Result, dstW, dstW, dstH, 1);

    std::cout << "Source: ";
    for (int i = 0; i < srcW; i++) {
        std::cout << std::setw(3) << (int)srcData[i] << " ";
    }
    std::cout << "\n\nNPP:    ";
    for (int i = 0; i < dstW; i++) {
        std::cout << std::setw(3) << (int)nppResult[i] << " ";
    }
    std::cout << "\nV1 Ult: ";
    for (int i = 0; i < dstW; i++) {
        std::cout << std::setw(3) << (int)v1Result[i] << " ";
    }
    std::cout << "\n\nPer-pixel analysis:\n";

    float threshold = 0.5f / scale;
    float attenuation = 1.0f / scale;

    std::cout << "Algorithm params: threshold=" << std::setprecision(3) << threshold
             << ", attenuation=" << attenuation << "\n\n";

    for (int dx = 0; dx < dstW; dx++) {
        float srcX = (dx + 0.5f) * scale - 0.5f;
        int x0 = (int)std::floor(srcX);
        int x1 = std::min(x0 + 1, srcW - 1);
        x0 = std::max(0, x0);
        float fx = srcX - x0;

        int npp = nppResult[dx];
        int v1 = v1Result[dx];
        int v0 = srcData[x0];
        int v1_src = srcData[x1];

        std::cout << "dst[" << dx << "]: srcX=" << std::setprecision(3) << srcX
                 << ", x0=" << x0 << ", fx=" << fx << "\n";
        std::cout << "  neighbors=[" << v0 << "," << v1_src << "]\n";
        std::cout << "  NPP=" << npp << ", V1=" << v1 << ", diff=" << std::abs(npp - v1) << "\n";

        if (fx <= threshold) {
            std::cout << "  fx <= threshold -> using FLOOR\n";
            std::cout << "  Expected: " << v0 << "\n";
        } else {
            float fx_att = fx * attenuation;
            float expected = v0 * (1.0f - fx_att) + v1_src * fx_att;
            std::cout << "  fx > threshold -> using ATTENUATED INTERP\n";
            std::cout << "  fx_attenuated=" << std::setprecision(3) << fx_att << "\n";
            std::cout << "  Expected: " << std::setprecision(1) << expected
                     << " -> " << (int)(expected + 0.5f) << "\n";
        }

        // Try to reverse engineer what NPP is doing
        if (v0 != v1_src && std::abs(npp - v1) > 0) {
            float fx_effective = (float)(npp - v0) / (v1_src - v0);
            std::cout << "  NPP effective fx=" << std::setprecision(3) << fx_effective;
            if (fx > 0.001f) {
                float actual_atten = fx_effective / fx;
                std::cout << " (atten=" << actual_atten << ")";
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
    std::cout << "1D FRACTIONAL DOWNSCALE DEBUG\n";
    std::cout << "========================================\n";

    // Cases that don't match
    debug1DFractional("0.875x (7/8)", 8, 7);
    debug1DFractional("0.8x (4/5)", 10, 8);
    debug1DFractional("0.6x (3/5)", 10, 6);

    // Cases that match perfectly
    std::cout << "\n========================================\n";
    std::cout << "COMPARISON: WORKING CASES\n";
    std::cout << "========================================\n";

    debug1DFractional("0.75x (3/4) [WORKING]", 8, 6);
    debug1DFractional("0.67x (2/3) [WORKING]", 9, 6);

    return 0;
}
